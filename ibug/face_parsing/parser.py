from functools import partial
from pathlib import Path
from socket import IP_DROP_MEMBERSHIP
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from ibug.roi_tanh_warping import roi_tanh_polar_restore, roi_tanh_polar_warp
import ibug.roi_tanh_warping.reference_impl as ref
from .rtnet import rtnet50, rtnet101, FCN
from .resnet import Backbone, DeepLabV3Plus
from torch.nn.functional import softmax

ENCODER_MAP = {
    'rtnet50': [rtnet50, 2048],  # model_func, in_channels
    'mask-prop-rtnet50': [rtnet50, 2048],  # model_func, in_channels
    'rtnet101': [rtnet101, 2048],
}
DECODER_MAP = {
    'fcn': FCN,
    'deeplabv3plus': DeepLabV3Plus
}

WEIGHT = {
    # 'rtnet50-fcn-11': (Path(__file__).parent / 'rtnet/weights/rtnet50.torch', (0.406, 0.456, 0.485), (0.225, 0.224, 0.229), (512, 512)),
    # 'rtnet101-fcn-11': (Path(__file__).parent / 'rtnet/weights/rtnet101.torch', (0.406, 0.456, 0.485), (0.225, 0.224, 0.229), (512, 512)),
    'rtnet50-fcn-11': (Path(__file__).parent / 'rtnet/weights/rtnet50-fcn-11.torch', 0.5, 0.5, (513, 513)),
    'rtnet50-fcn-14': (Path(__file__).parent / 'rtnet/weights/rtnet50-fcn-14.torch', 0.5, 0.5, (513, 513)),
    'rtnet101-fcn-14': (Path(__file__).parent / 'rtnet/weights/rtnet101-fcn-14.torch', 0.5, 0.5, (513, 513)),
    'resnet50-fcn-14': (Path(__file__).parent / 'resnet/weights/resnet50-fcn-14.torch', 0.5, 0.5, (513, 513)),
    'mask-prop-resnet50-fcn-14': (Path(__file__).parent / 'resnet/weights/mask-prop-resnet50-fcn-14.torch', 0.5, 0.5, (513, 513)),
    'mask-prop-rtnet50-fcn-14': (Path(__file__).parent / 'rtnet/weights/mask-prop-rtnet50-fcn-14.torch', 0.5, 0.5, (513, 513)),
    'resnet50-deeplabv3plus-14': (Path(__file__).parent / 'resnet/weights/resnet50-deeplabv3plus-14.torch', 0.5, 0.5, (513, 513)),
}


class SegmentationModel(nn.Module):

    def __init__(self, encoder='rtnet50', decoder='fcn', num_classes=14, input_channel=3):
        super().__init__()
        self.num_classes=num_classes
        if 'rtnet' in encoder:
            encoder_func, in_channels = ENCODER_MAP[encoder.lower()]
            self.encoder = encoder_func(input_channel=input_channel)
        else:
            self.encoder = Backbone(encoder, input_channel=input_channel)
            in_channels = self.encoder.num_channels
        self.decoder = DECODER_MAP[decoder.lower()](
            in_channels=in_channels, num_classes=num_classes)
        self.low_level = getattr(self.decoder, 'low_level', False)

    def forward(self, x, rois):
        input_shape = x.shape[-2:]
        features = self.encoder(x, rois)
        low = features['c1']
        high = features['c4']
        if self.low_level:
            x = self.decoder(high, low)
        else:
            x = self.decoder(high)
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=False)
        return x


class FaceParser(object):
    def __init__(self, device='cuda:0', ckpt=None, encoder='rtnet50', decoder='fcn', num_classes=11, input_channel=3):
        self.device = device
        model_name = '-'.join([encoder, decoder, str(num_classes)])
        assert model_name in WEIGHT, f'{model_name} is not supported'

        pretrained_ckpt, mean, std, sz = WEIGHT[model_name]
        self.sz = sz

        self.model = SegmentationModel(encoder, decoder, num_classes, input_channel=input_channel)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        if ckpt is None:
            ckpt = pretrained_ckpt
        ckpt = torch.load(ckpt, 'cpu')
        ckpt = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(ckpt, True)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def predict_img(self, img, bboxes, rgb=False, prev_mask=None):

        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            if rgb:
                img = img[:, :, ::-1]
        else:
            raise TypeError
        h, w = img.shape[:2]

        num_faces = len(bboxes)


        
        # img = np.concatenate([img, prev_mask], axis=2)

        imgs = [ref.roi_tanh_polar_warp(img, b, *self.sz, keep_aspect_ratio=True) for b in bboxes]
        imgs = [self.transform(img) for img in imgs]
        bboxes_tensor = torch.tensor(
            bboxes).view(num_faces, -1).to(self.device)

        img = torch.stack(imgs).to(self.device)
        if prev_mask is not None:
            # prev_mask = cv2.resize(prev_mask, (w, h), interpolation=cv2.INTER_NEAREST)[..., np.newaxis]
            # import ipdb; ipdb.set_trace()
            prev_mask = (prev_mask - 0.5 * self.model.num_classes) / (0.5 * self.model.num_classes)
            prev_mask = np.stack([ref.roi_tanh_polar_warp(m, b, *self.sz, keep_aspect_ratio=True) for b, m in zip(bboxes, prev_mask)])
            prev_mask = torch.from_numpy(prev_mask).to(self.device).view(-1, 1, *self.sz)
            img = torch.concat([img, prev_mask], dim=1).float()
        logits = self.model(img, bboxes_tensor)
        mask = self.restore_warp(h, w, logits, bboxes_tensor)
        return mask

    def restore_warp(self, h, w, logits: torch.Tensor, bboxes_tensor):
        logits = softmax(logits, 1)
        logits[:, 0] = 1 - logits[:, 0]  # background class
        logits = roi_tanh_polar_restore(
            logits, bboxes_tensor, w, h, keep_aspect_ratio=True
        )
        logits[:, 0] = 1 - logits[:, 0]
        predict = logits.cpu().argmax(1).numpy()
        return predict

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, rois: torch.Tensor):
        features = self.model.encoder(x, rois, return_features=True)
        x = self.model.decoder(features['c4'])
        features['logits'] = x
        return features
