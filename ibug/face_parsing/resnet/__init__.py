# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import logging

import torch
import torch.nn.functional as F
from .resnet import *
from .decoder import *
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


_logger = logging.getLogger(__name__)


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module,  num_channels: int):
        super().__init__()

        return_layers = {"layer1": "c1", "layer2": "c2",
                         "layer3": "c3", "layer4": "c4"}
        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, images, rois=None):
        return self.body(images)

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 input_channel=3
                 ):
        if 'resnet18' in name or 'resnet34' in name:
            replace_stride_with_dilation = [False, False, False]
        else:
            replace_stride_with_dilation = [False, True, True]
        if 'mask-prop' in name:
            name = name[len("mask-prop-"):]
        backbone = globals().get(name)(
                replace_stride_with_dilation=replace_stride_with_dilation,
                input_channel=input_channel)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels)


