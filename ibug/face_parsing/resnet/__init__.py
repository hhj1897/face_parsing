"""
Backbone modules.
"""
import logging

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from .decoder import *

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

    def __init__(self, name: str):
        if 'resnet18' in name or 'resnet34' in name:
            replace_stride_with_dilation = [False, False, False]
        else:
            replace_stride_with_dilation = [False, True, True]
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=replace_stride_with_dilation)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels)
