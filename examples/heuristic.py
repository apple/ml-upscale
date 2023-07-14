"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

How to customize heuristic used for pruning, in UPSCALE
"""

import torch, torchvision
from upscale import MaskingManager, PruningManager
from upscale.masking.importance import HRank, LAMP


# for most heuristics, simply pass the heuristic to `.importance(...)`
x = torch.rand((1, 3, 224, 224), device='cuda')
model = torchvision.models.get_model('resnet18', pretrained=True).cuda()  # get any pytorch model
MaskingManager(model).importance(LAMP()).mask()
PruningManager(model).compute([x]).prune()

# for HRank, we need to run several forward passes to collect feature map
# statistics

heuristic = HRank()
heuristic.register(model)
for _ in range(10):
    model(torch.rand((1, 3, 224, 224), device='cuda'))
heuristic.deregister(model)

MaskingManager(model).importance(heuristic).mask()
PruningManager(model).compute([x]).prune()