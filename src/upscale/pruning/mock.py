"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

Utilities to mock a pruned model, with a standard PyTorch model. To do this,
applies masks during the forward pass.

You can easily run pruned inference with regular data and a generated, mock
pruned model.

    net_mock = MockPrunedModel(net)
    y = net_mock(x)
"""


import torch
import torch.nn as nn

from .resolve import PruningSpec
from .utils import get_tensor_and_modules, logger


class MockPrunedModel(nn.Module):
    """
    This model effects a mock pruned model for pruned inference.

    Apply pruning masks by pre-multiplying provided masks to inputs and
    post-multiplying masks to outputs. Allows user to emulate pruning model
    without changing any model code. This model effects a mock pruned
    inference.

    >>> import logging, sys
    >>> from .resolve import generate_pruning_spec
    >>> from .tracing import trace
    >>> x = torch.rand((1, 3, 8, 8))
    >>> class Net(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv1 = nn.Conv2d(3, 10, 3)
    ...         self.conv2 = nn.Conv2d(10, 5, 3)
    ...     def forward(self, x):
    ...         x = torch.cat([x[:, -2:], x[:, :1]], dim=1)
    ...         x = self.conv1(x)
    ...         x = torch.cat([x[:, -5:], x[:, :5]], dim=1)
    ...         x = self.conv2(x)
    ...         return x
    ...
    >>> net = Net()
    >>> traced = trace(net, [x])
    >>> spec = generate_pruning_spec(
    ...     traced, [('conv1.weight', 3)], constrained=True, net=net)
    >>> mock = MockPrunedModel(net, spec=spec)
    >>> handler = logging.StreamHandler(sys.stdout)
    >>> logger.addHandler(handler)
    >>> logger.setLevel(logging.DEBUG)
    >>> y = mock(x)
    Masked output for conv1.
    Masked input for conv2.
    >>> logger.removeHandler(handler)
    """

    def __init__(self, net: nn.Module, spec: PruningSpec):
        super().__init__()
        self.net = net
        self.spec = spec
        self.handles = []

    def register_hooks(self):
        """
        Register pre and post forward hooks to mask input and output tensors
        for each module.
        """
        if self.handles:
            return
        for tensor_id, module in get_tensor_and_modules(self.spec.tensor_to_metadata):
            module_name = module._metadata['name']

            if tensor_id in self.spec.tensor_to_inputs_masks:
                pre_hook = module.register_forward_pre_hook(
                    self.generate_pre_forward_hook(
                        module_name,
                        self.spec.tensor_to_inputs_masks[tensor_id]))
                self.handles.append(pre_hook)

            if tensor_id in self.spec.tensor_to_outputs_masks:
                post_hook = module.register_forward_hook(
                    self.generate_post_forward_hook(
                        module_name,
                        self.spec.tensor_to_outputs_masks[tensor_id]))
                self.handles.append(post_hook)

    def generate_pre_forward_hook(self, module_name, inputs_masks):
        """
        Generate a pre-forward hook to mask inputs before a layer.
        """
        def hook(module, inputs):
            new_inputs = []
            for input, mask in zip(inputs, inputs_masks):
                tensor = torch.Tensor(input)
                if len(tensor.shape) == 0:
                    new_inputs.append(tensor)
                    continue
                new_inputs.append(tensor * self.match_ndims(mask, tensor) \
                    .to(tensor.device))
            logger.debug(f"Masked input for {module_name}.")
            return tuple(new_inputs) or inputs
        return hook

    def generate_post_forward_hook(self, module_name, outputs_masks):
        """
        Generate a post-forward hook for outputs after a layer.
        """
        def hook(module, input, out):
            mask = outputs_masks[0].view(1, -1, 1, 1)
            assert out.shape[1] == mask.shape[1], (
                f'Output mask for {module_name} is invalid. Output: '
                f'{out.shape} | Mask: {mask.shape}'
            )
            out = out * self.match_ndims(mask, out).to(out.device)
            logger.debug(f"Masked output for {module_name}.")
            return out
        return hook

    def deregister_hooks(self):
        """
        Remove all hooks.
        """
        for handle in self.handles:
            handle.remove()

    @classmethod
    def match_ndims(
        cls,
        mask: torch.Tensor,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Reshape mask according to the number of dimensions in the reference
        tensor.

        If the input only has 2 dimensions, the mask should only have 2
        dimensions. If the input has 3, reshape the mask to also have 3. etc.

        Args:
            mask: 1-dimensional tensor of 1s and 0s.
            tensor: Tensor to be masked.

        Returns:
            Reshaped mask, with the same number of dims as the input tensor
        """
        shape = (1, -1) + ((1,) * (len(tensor.shape) - 2))
        return mask.view(*shape)

    def forward(self, *x) -> torch.Tensor:
        self.register_hooks()
        y = self.net(*x)
        self.deregister_hooks()
        return torch.Tensor(y)
