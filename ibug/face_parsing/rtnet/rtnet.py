import math
import torch
import torch.nn as nn
from ibug.roi_tanh_warping import (roi_tanh_polar_to_roi_tanh,
                                   roi_tanh_to_roi_tanh_polar)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,
            mix_padding=False, padding_modes=['replicate', 'circular']):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)

    if not mix_padding:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

    else:
        return nn.Sequential(
            MixPad2d([dilation, dilation], padding_modes),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=0, bias=False, groups=groups, dilation=dilation)
        )


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False, groups=groups)


class Bottleneck(nn.Module):
    '''
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (
            stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes,
                               kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def forward(self, x):
        x, rois = x['x'], x['rois']

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)
        return dict(x=x, rois=rois)


class MixPad2d(nn.Module):
    """Mixed padding modes for H and W dimensions

    Args:
        padding (tuple): the size of the padding for x and y, ie (pad_x, pad_y)
        modes (tuple): the padding modes for x and y, the values of each can be
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``['replicate', 'circular']``

    """
    __constants__ = ['modes', 'padding']

    def __init__(self, padding=[1, 1], modes=['replicate', 'circular']):
        super(MixPad2d, self).__init__()
        assert len(padding) == 2
        self.padding = padding  # x, y
        self.modes = modes

    def forward(self, x):
        #  (left, right, top, down) is used in nn.functional.pad
        # pad height (y axis)
        x = nn.functional.pad(
            x, (0, 0,  self.padding[1], self.padding[1]), self.modes[1])
        # pad width (x axis)
        x = nn.functional.pad(
            x, (self.padding[0], self.padding[0], 0, 0), self.modes[0])
        return x

    def extra_repr(self):
        repr_ = """Mixed Padding: \t x axis: mode: {}, padding: {},\n\t y axis mode: {}, padding: {}""".format(
            self.modes[0], self.padding[0], self.modes[1], self.padding[1])
        return repr_


class HybridBlock(nn.Module):
    expansion = 4
    # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.
    pooling_r = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None, hybrid=False):
        super(HybridBlock, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(
            inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(
            inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)
        self.hybrid = hybrid

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        self.conv_polar, self.conv_cart = None, None
        self.conv_polar = nn.Sequential(
            conv3x3(group_width, group_width, stride=stride, groups=cardinality, dilation=dilation,
                    mix_padding=True),
            norm_layer(group_width),
        )
        self.conv_cart = nn.Sequential(
            conv3x3(group_width, group_width, stride=stride, groups=cardinality, dilation=dilation,
                    mix_padding=False),
            norm_layer(group_width),
        )
        self.conv3 = nn.Conv2d(
            group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        x, rois = x['x'], x['rois']
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.conv_polar(out_a)
        if self.hybrid:
            _, _, h1, w1 = out_b.size()
            out_b = roi_tanh_polar_to_roi_tanh(
                out_b, rois, w1, h1, keep_aspect_ratio=True)
            out_b = self.conv_cart(out_b)
            _, _, h2, w2 = out_b.size()
            out_b = roi_tanh_to_roi_tanh_polar(
                out_b, rois/(h1/h2), w2, h2, keep_aspect_ratio=True)
        else:
            out_b = self.conv_cart(out_b)

        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return dict(x=out, rois=rois)


class RTNet(nn.Module):
    """ RTNet Variants Definations
    Parameters
    ----------
    block : Block
        Class for the residual block.
    layers : list of int
        Numbers of layers in each block.
    dilated : bool, default False
        Applying dilation strategy to pretrained RTNet yielding a stride-8 model.
    deep_stem : bool, default False
        Replace 7x7 conv in input stem with 3 3x3 conv.
    avg_down : bool, default False
        Use AvgPool instead of stride conv when
        downsampling in the bottleneck.
    norm_layer : object
        Normalization layer used (default: :class:`torch.nn.BatchNorm2d`).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, groups=1, bottleneck_width=32, dilated=True, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False, hybrid_stages=[True, True, True],
                 avd=False, norm_layer=nn.BatchNorm2d, zero_init_residual=True, **kwargs):
        super(RTNet, self).__init__()
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd

        if hybrid_stages is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            hybrid_stages = [False, False, False]
        if len(hybrid_stages) != 3:
            raise ValueError("hybrid_stages should be None "
                             "or a 3-element tuple, got {}".format(
                                 hybrid_stages))
        print("Hybrid stages {}".format(hybrid_stages))
        conv_layer = nn.Conv2d
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3,
                           stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3,
                           stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3,
                           stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, norm_layer=norm_layer, hybrid=hybrid_stages[0])
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer, hybrid=hybrid_stages[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer, hybrid=hybrid_stages[2])
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer, hybrid=hybrid_stages[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer, hybrid=hybrid_stages[2])
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer, hybrid=hybrid_stages[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer, hybrid=hybrid_stages[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (HybridBlock, Bottleneck)):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                    # nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    is_first=True, hybrid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=1, is_first=is_first,
                                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=2, is_first=is_first,
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if not hybrid:
                layers.append(Bottleneck(self.inplanes, planes,
                              cardinality=self.cardinality))
            else:
                layers.append(block(self.inplanes, planes,
                                    cardinality=self.cardinality,
                                    bottleneck_width=self.bottleneck_width,
                                    avd=self.avd, dilation=dilation,
                                    norm_layer=norm_layer, hybrid=hybrid))

        return nn.Sequential(*layers)

    def forward(self, x, rois, *args, **kwargs):
        _, _, H, _ = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        _, _, H_stem, _ = x.shape
        s1 = H / H_stem
        c1 = self.layer1(dict(x=x, rois=rois/s1))['x']
        s2 = H / c1.shape[2]
        c2 = self.layer2(dict(x=c1, rois=rois/s2))['x']
        s3 = H / c2.shape[2]
        c3 = self.layer3(dict(x=c2, rois=rois/s3))['x']
        s4 = H / c3.shape[2]
        c4 = self.layer4(dict(x=c3, rois=rois/s4))['x']
        return dict(c1=c1, c2=c2, c3=c3, c4=c4)


def rtnet50(pretrained=False, **kwargs):
    """Constructs a RTNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RTNet(HybridBlock, [3, 4, 6, 3],
                  deep_stem=False, stem_width=32, avg_down=False,
                  avd=False, **kwargs)
    model.num_channels = 2048
    return model


def rtnet101(pretrained=False, **kwargs):
    """Constructs a RTNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RTNet(HybridBlock, [3, 4, 23, 3],
                  deep_stem=False, stem_width=64, avg_down=False,
                  avd=False, **kwargs)
    model.num_channels = 2048
    return model


class FCN(nn.Sequential):
    def __init__(self, in_channels, num_classes, **kwargs):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        ]

        super(FCN, self).__init__(*layers)


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    rois = torch.rand(1, 4).cuda(0)
    model = rtnet101()
    model = model.cuda(0)
    print(model(images, rois).size())
