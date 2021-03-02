from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from ibug.roi_tanh_warping import roi_tanh_polar_restore, roi_tanh_polar_warp
from .rtnet import rtnet50, rtnet101, FCN

ENCODER_MAP = {
    'rtnet50': [rtnet50, 2048],  # model_func, in_channels
    'rtnet101': [rtnet101, 2048]
}
DECODER_MAP = {
    'fcn': FCN
}
TARGET_SIZE = (512, 512)

norm_mean = (0.406, 0.456, 0.485)

norm_std = (0.225, 0.224, 0.229)

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(norm_mean, norm_std)
]
)

WEIGHT = {
    'rtnet50': Path(__file__).parent / 'weights/rtnet50.torch',
    'rtnet101': Path(__file__).parent / 'weights/rtnet101.torch',
}


class SegmentationModel(nn.Module):

    def __init__(self, encoder='rtnet50', decoder='fcn', num_classes=11):
        super().__init__()
        encoder_func, in_channels = ENCODER_MAP[encoder.lower()]
        self.encoder = encoder_func()
        self.decoder = DECODER_MAP[decoder.lower()](
            in_channels=in_channels, num_classes=num_classes)

    def forward(self, x, rois):
        input_shape = x.shape[-2:]
        features = self.encoder(x, rois)['out']
        x = self.decoder(features)
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=False)
        return x


class RTNetPredictor(object):
    def __init__(self, device='cuda:0', ckpt=None, encoder='rtnet50', decoder='fcn', num_classes=11):
        self.device = device
        self.model = SegmentationModel(encoder, decoder, num_classes)
        if ckpt is None:
            ckpt = WEIGHT[encoder]
        ckpt = torch.load(ckpt, 'cpu')
        ckpt = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(ckpt, True)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def predict_img(self, img, bboxes, rgb=False):

        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            if rgb:
                img = img[:, :, ::-1]
        else:
            raise TypeError
        h, w = img.shape[:2]

        img = TRANSFORM(img).unsqueeze(0).to(self.device)

        num_faces = len(bboxes)
        bboxes_tensor = torch.tensor(
            bboxes).view(num_faces, -1).to(self.device)

        img = img.repeat(num_faces, 1, 1, 1)
        img = roi_tanh_polar_warp(
            img, bboxes_tensor, target_height=TARGET_SIZE[0], target_width=TARGET_SIZE[1], keep_aspect_ratio=True)

        logits = self.model(img, bboxes_tensor)
        mask = self.restore_warp(h, w, logits, bboxes_tensor)
        return mask

    def restore_warp(self, h, w, logits: torch.Tensor, bboxes_tensor):
        # import ipdb; ipdb.set_trace()
        logits = logits.sigmoid()
        # print(logits.argmax(-1).max())
        logits[:, 0] = 1 - logits[:, 0]  # background class
        logits = roi_tanh_polar_restore(
            logits, bboxes_tensor, w, h, keep_aspect_ratio=True
        )
        # print(logits.argmax(-1).max())
        logits[:, 0] = 1 - logits[:, 0]
        # print(logits.argmax(-1).max())
        predict = logits.cpu().argmax(1).numpy()
        return predict

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, rois: torch.Tensor):

        features = self.encoder(x, rois, return_features=True)
        x = self.decoder(features['c4'])
        features['hm'] = x
        return features
