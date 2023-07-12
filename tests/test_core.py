"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

Tests for the core UPSCALE library code.
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet, densenet121

from upscale.pruning import PruningManager as PyTorchPruningManager


class SimpleArithmetic(nn.Module):
    """
    Current export pipeline should fail correctness on this example, when using
    output pruning. This is because the 0 indicator is lost after addition.
    By contrast, v2 in this file should pass, as it appropriately re-applies a
    1-0 mask to zero out pruned channels.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv2 = nn.Conv2d(10, 10, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tensor(3.0) * (x + 1.0)
        x = self.conv2(x)
        return x


class UnbiasedConv(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 10, 3, bias=False),
            nn.Conv2d(10, 10, 3, bias=False),
        )


class Conv(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 10, 3),
            nn.Conv2d(10, 10, 3),
        )


class ConvLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 10, 3)
        self.conv1 = nn.Conv2d(10, 10, 3)
        self.linear = nn.Linear(220 * 220 * 10, 5)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class ConvBNRelu(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 10, 3),
            nn.Conv2d(10, 10, 3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3),
        )


class ConvBNReluLumpy(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 5, 3),
            nn.Conv2d(5, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 10, 3),
        )


class MultipleProducers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
        self.conv3 = nn.Conv2d(10, 10, 3, padding=1)

    def forward(self, x):
        y = self.conv0(x) + self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return y


class ConcatProducers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 5, 3, padding=1)
        self.conv1 = nn.Conv2d(3, 5, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 15, 3, padding=1)
        self.conv3 = nn.Conv2d(15, 10, 3, padding=1)

    def forward(self, x):
        y = torch.cat([self.conv0(x), self.conv1(x)], dim=1)
        y = self.conv2(y)
        y = self.conv3(y)
        return y


class MultipleProducersConsumers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 5, 3, padding=1)
        self.conv1 = nn.Conv2d(3, 5, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 5, 3, padding=1)
        self.conv3 = nn.Conv2d(10, 5, 3, padding=1)
        self.conv4 = nn.Conv2d(10, 10, 3, padding=1)

    def forward(self, x):
        y = torch.cat([self.conv0(x), self.conv1(x)], dim=1)
        y = torch.cat([self.conv2(y), self.conv3(y)], dim=1)
        y = self.conv4(y)
        return y


class MultipleConsumers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv1 = nn.Conv2d(10, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
        self.conv3 = nn.Conv2d(10, 10, 3, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        y = self.conv1(x) + self.conv2(x)
        y = self.conv3(y)
        return y


class ConcatConsumers(nn.Module):
    pruned_inputs = [
        ('conv2.weight', 0),
        ('conv2.weight', 3),
        ('conv3.weight', 0),
        ('conv3.weight', 1),
        ('conv3.weight', 3),
        ('conv3.weight', 6)
    ]

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv1 = nn.Conv2d(4, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x):
        y1 = self.conv0(x)
        y2 = self.conv1(y1)
        y3 = self.conv2(torch.cat([y2, y1], dim=1))
        y = self.conv3(y3)
        return y


class Deconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 5, 3, padding=1)
        self.conv1 = nn.ConvTranspose2d(5, 10, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(10, 20, 3, padding=1)
        self.conv3 = nn.ConvTranspose2d(20, 10, 3, padding=1)

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        return y


class MultipleInputs(nn.Module):
    def __init__(self, Net=ConvBNRelu):
        super().__init__()
        self.cbr1 = Net()
        self.cbr2 = Net()
        self.conv = nn.Conv2d(20, 10, 3)

    def forward(self, x1, x2):
        y = torch.cat([self.cbr1(x1), self.cbr2(x2)], dim=1)
        return self.conv(y)


class Depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 5, 3, padding=1)
        self.conv1 = nn.Conv2d(5, 5, 3, padding=1, groups=5)
        self.conv2 = nn.Conv2d(5, 20, 3, padding=1)
        self.conv3 = nn.Conv2d(20, 10, 3, padding=1)

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        return y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(3, 5)
        self.linear1 = nn.Linear(5, 10)
        self.linear2 = nn.Linear(10, 20)
        self.linear3 = nn.Linear(20, 10)

    def forward(self, x):
        y1 = self.linear0(x)
        y2 = self.linear1(y1)
        y3 = self.linear2(y2)
        y4 = self.linear3(y3)
        return y4


class Permute(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 5, 3, padding=1)
        self.conv1 = nn.Conv2d(5, 5, 3, padding=1)

    def forward(self, x):
        y = self.conv0(x)
        y = y.permute(0, 2, 3, 1)
        y = y.permute(0, 3, 1, 2)
        y = self.conv1(y)
        return y


def check_pruning_correctness(
    net, inputs=None, input_shapes=[(1, 3, 224, 224)], threshold=1e-14, Manager=PyTorchPruningManager, **kwargs
):      
    if inputs is None:
        inputs = [torch.rand(input_shape) for input_shape in input_shapes]

    if torch.cuda.is_available():
        net = net.cuda()
        inputs = [x.cuda() for x in inputs]

    if isinstance(net, nn.Module):  # if pytorch, run high precision tests
        inputs = [xi.double() for xi in inputs]
        net = net.double()

    # compute training masks and channels to prune at export
    manager = Manager(net)
    manager.compute(inputs, **kwargs)

    # mock pruned model with training masks
    net_mock = manager.get_mock_model()
    y_mock = net_mock(*inputs)

    # actually prune the model
    manager.prune()
    y_pruned = manager(*inputs)  # use manager.forward instead of net.forward

    print((y_mock - y_pruned).abs().max())

    return y_pruned, (y_mock - y_pruned).abs().max() <= threshold


@pytest.mark.parametrize("Net, pruned_inputs, pruned_outputs, input_shapes", (
    (SimpleArithmetic, [], [('conv1.weight', 1)], [(1, 3, 224, 224)]),
    (UnbiasedConv, [], [('0.weight', 1)], [(1, 3, 224, 224)]),
    (Conv, [], [('0.weight', 1)], [(1, 3, 224, 224)]),
    (ConvLinear, [], [('conv0.weight', 1)], [(1, 3, 224, 224)]),
    (ConvBNRelu, [], [('1.weight', 1)], [(1, 3, 224, 224)]),
    (MultipleProducers, [], [('conv1.weight', 3)], [(1, 3, 224, 224)]),
    (MultipleConsumers, [], [('conv0.weight', 3)], [(1, 3, 224, 224)]),
    (MultipleInputs, [], [('cbr2.1.weight', 1), ('cbr2.4.weight', 1)], [(1, 3, 224, 224)] * 2),
    (Depthwise, [('conv2.weight', 3)], [], [(1, 3, 224, 224)]),
    (ConcatProducers, [('conv2.weight', 3)], [], [(1, 3, 224, 224)]),
    (MultipleProducersConsumers, [('conv2.weight', 3), ('conv4.weight', 3)], [], [(1, 3, 224, 224)]),
    (Deconv, [('conv2.weight', 3)], [], [(1, 3, 224, 224)]),
    (ConvBNReluLumpy, [], [('1.weight', 1)], [(1, 3, 224, 224)]),
    (UnbiasedConv, [], [('0.weight', 1)], [(1, 3, 5, 5)]),        # 'small' input size (anything >= 10 will pass)
    (MLP, [('linear2.weight', 2)], [], [(1, 3)]),
    # (Permute, [], [('conv1.weight', 2)], [(1, 3, 5, 5)]), # TODO: restore
))
def test_pruning_core_pytorch_constrained(Net, pruned_inputs, pruned_outputs, input_shapes):
    """Test core pytorch pruning implementation.

    These tests are designed to catch specific edge cases in e2e tests. For randomized tests on more
    realistic models, see `test_pruning_core_random`.
    """
    net = Net().eval()
    assert check_pruning_correctness(
        net,
        pruned_inputs=pruned_inputs,
        pruned_outputs=pruned_outputs,
        input_shapes=input_shapes,
        constrained=True)[1]


@pytest.mark.parametrize("Net, pruned_inputs", (
    (ConvBNRelu, [('1.weight', 1)]),
))
def test_pruning_core_prune_by_name(Net, pruned_inputs):
    """Test named parameters work as input as well."""
    net = Net().eval()
    assert check_pruning_correctness(net, pruned_inputs=pruned_inputs)[1]


@pytest.mark.parametrize("Net, pruned_inputs, pruned_outputs", (
    (ConcatProducers, [('conv2.weight', 3)], []),
    (ConcatConsumers, ConcatConsumers.pruned_inputs, []),
    (MultipleProducersConsumers, [('conv2.weight', 3), ('conv4.weight', 3)], []),
    (Deconv, [('conv1.weight', 3)], []),
    (resnet18, [('layer1.0.conv1.weight', 3), ('layer2.0.conv1.weight', 5)], []),
    (densenet121, [('features.denseblock1.denselayer1.conv1.weight', 1)], []),
    (alexnet, [('features.3.weight', 1)], []),
    (MultipleConsumers, [('conv1.weight', 3), ('conv2.weight', 3)], []),  # simple case, when masks are 'equal'
    (MultipleConsumers, [('conv1.weight', 3), ('conv2.weight', 3), ('conv2.weight', 2)], []),  # should prune 3, but retain 2 for subselection
))
def test_pruning_core_pytorch_unconstrained(Net, pruned_inputs, pruned_outputs):
    net = Net().eval()
    assert check_pruning_correctness(
        net,
        pruned_inputs=pruned_inputs,
        pruned_outputs=pruned_outputs)[1]
