"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

Basic example to use UPSCALE
"""

import torch, torchvision
from upscale import MaskingManager, PruningManager

x = torch.rand((1, 3, 224, 224)).cuda()
model = torchvision.models.get_model('resnet18', pretrained=True).cuda()  # get any pytorch model
MaskingManager(model).importance().mask()
PruningManager(model).compute([x]).prune()