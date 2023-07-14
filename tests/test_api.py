"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

Tests for the UPSCALE library external-facing API
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from upscale import MaskingManager, PruningManager


def test_simple():
    x = torch.rand((1, 3, 224, 224), device='cuda')
    model = resnet18().cuda()
    MaskingManager(model).importance().mask(amount=0.1)
    PruningManager(model).compute([x]).prune()