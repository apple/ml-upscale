"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

Tracking channel manipulations using a special tensor. Tensor used for tracking
all the input channels used, to compute every output channel.

This allows us to run the forward pass in a model normally to track tensor
changes. All tracking occurs internally, within the special `Tracer`.

To read this file, start from `trace`.
"""

import copy
from dataclasses import dataclass, field
import inspect
import itertools
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trace import Trace, flatten_trace, trace_from_n_channels, union_traces
from .utils import get_n_channels, logger


@dataclass
class Metadata:
    num_output_channels: int = -1
    non_tracers: List = field(default_factory=list)
    is_trace_reset: bool = False
    is_input: bool = False
    channel_axis: int = 1


def infinite_id_generator():
    """Infinite integer generator"""
    i = 0
    while True:
        yield i
        i += 1


def trace(
    net: nn.Module,
    inputs: List[torch.Tensor],
    key: str = 'out',
) -> 'Tracer':
    """
    Trace the network by running a forward pass with a special tensor.
    """
    assign_modules_to_params(net)
    id_generator = infinite_id_generator()
    tracers = [Tracer(input, id_generator=id_generator) for input in inputs]
    y = net(*tracers)

    # Handle models with dictionary output
    if isinstance(y, dict):
        assert key in y, 'Specify which output key, with trace(..., key=KEY)'
        for _key in y.keys():  # HACK: merge all metadata manually
            for attr in ('tensor_to_metadata', 'tensor_to_traces'):
                getattr(y[key], attr).update(getattr(y[_key], attr))
        y = y[key]

    for input in tracers:  # mark inputs
        y.tensor_to_metadata[input.tensor_id] = Metadata(is_input=True)
    return y


def assign_modules_to_params(net: nn.Module):
    """
    Assign tensor idenfitiers to all layers in the network.
    
    Op ids are assigned to parameters, so that __torch_function__ (classmethod)
    can fetch the layer that a parameter comes from.
    """
    # assign names to params
    for name, param in net.named_parameters():
        param._metadata = {'name': name}

    # assign modules to params
    for name, module in net.named_modules():
        module._metadata = {'name': name}
        for _, param in module.named_parameters():
            param._metadata['module'] = module

    # assign parent module to child
    frontier = [net]
    while frontier:
        parent = frontier.pop(0)
        for _, child in parent.named_children():
            child._metadata['parent'] = parent
            frontier.append(child)


def get_torch_function_signature(func):
    """Use torch function overrides for inspection.

    Note that inspect.signature(<torch func>) gives 'no signature found'.
    This applies to all torch functions.
    """
    overrides = torch.overrides.get_testing_overrides()
    signature = inspect.signature(overrides[func])
    return signature


def coerce(func, args, kwargs):
    """
    Coerces all positional and keyword arguments into a single
    keyword-arguments dictionary.

    >>> args = [[torch.rand(1), torch.rand(1)]]
    >>> kwargs = {'out': torch.rand(1)}
    >>> coerced = coerce(torch.cat, args, kwargs)
    >>> bool(coerced['tensors'] == args[0])  # mapped arg to kwarg
    True
    >>> bool(coerced['out'] == kwargs['out'])  # preserve kwarg
    True
    """
    # TODO: Fill in defaults without making it confusing
    unified = copy.copy(kwargs)
    signature = get_torch_function_signature(func)
    for i, (name, param) in enumerate(signature.parameters.items()):
        if i < len(args):
            unified[name] = args[i]
    return unified


class IdTensor(torch.Tensor):
    """
    Substitute for a regular tensor to id every activation.

    - Generate id'ed tensors for every operation applied.
    - Assign id's in-place to passed tensors.

    >>> x = IdTensor(torch.zeros((2, 3, 1, 1)))
    >>> x.tensor_id
    0
    >>> x.cpu().tensor_id
    1
    >>> x[:, :2].tensor_id
    2
    >>> conv = nn.Conv2d(3, 3, 1)
    >>> conv(x).tensor_id
    3
    >>> torch.mean(x).tensor_id
    4
    >>> bn = nn.BatchNorm2d(3)
    >>> _ = x.shape  # NOTE: should not affect ID
    >>> bn(x).tensor_id
    5
    >>> c = torch.tensor(3.0)
    >>> (x + c).tensor_id
    6
    """
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        return super().__new__(cls, x)

    def __init__(
        self,
        x: torch.Tensor,
        tensor_id: int = None,
        id_generator=None
    ):
        self.id_generator = id_generator or infinite_id_generator()
        self.tensor_id = (
            next(self.id_generator) if tensor_id is None else tensor_id
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types,
        args: Tuple = (),
        kwargs: Dict = {}
    ) -> 'IdTensor':
        """
        Redefine pytorch wrapper function around all tensor interactions.

        NOTE: If this function throws a TypeError, the error will propagate up
        as the following error:

            Unsupported operand type(s) for ?: 'Tracer' and 'Tracer'
        """
        out = super().__torch_function__(func, types, args, kwargs)
        if not isinstance(out, torch.Tensor):
            return out
        out = torch.Tensor(out)

        kwargs = coerce(func, args, kwargs)
        tracer = cls.get_tracers(kwargs, True)[0]
        id_generator = tracer.id_generator
        tensor_id = next(id_generator)

        out, metadata = cls.post_torch_func_hook(func, kwargs, tensor_id, out)
        return cls(out, tensor_id, id_generator, **metadata)

    @classmethod
    def post_torch_func_hook(
        cls,
        func: Callable,
        kwargs: Dict,
        tensor_id: int,
        out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Modify the original function's tensor result, and compute keyword
        arguments used to initialize the new custom tensor.

        Args:
            func: The function applied to these arguments
            kwargs: Keyword arguments passed to this function
            tensor_id: The tensor id that is assigned to the output custom
                tensor. This same id is used to mark the input non-custom
                tensors.
            out: The output of the above function called on the above
                arguments.

        Returns:
            The new output, possibly modified
            Any keyword arguments to pass to the output custom tensor's
                constructor
        """
        return out, {}

    @classmethod
    def get_tracers(
        cls,
        args: Union[Dict, List],
        tracers: bool = False
    ) -> List:
        """
        Get all custom tensors in provided arguments.

        >>> args = [[IdTensor([1]), torch.tensor(3.0), 4]]
        >>> args
        [[IdTensor([1.]), tensor(3.), 4]]
        >>> IdTensor.get_tracers(args)
        [tensor(3.)]
        >>> IdTensor.get_tracers(args, True)
        [IdTensor([1.])]
        """
        if isinstance(args, dict):
            values = [cls.get_tracers(val, tracers) for val in args.values()]
            return list(itertools.chain(*values))
        if isinstance(args, (list, tuple)):
            items = [cls.get_tracers(arg, tracers) for arg in args]
            return list(itertools.chain(*items))
        if (
            (tracers and isinstance(args, cls))
            or (not tracers and not isinstance(args, cls)
                and isinstance(args, (torch.Tensor, nn.Parameter)))
        ):
            return [args]
        return []


class Tracer(IdTensor):
    """
    Substitute for a regular tensor to trace how channels move in model.

    - Tracks the source channels for each output channel in the current tensor.
    - Saves traces for *all operations.
    - Tracks the number of output channels per operation.

    >>> x = Tracer(torch.rand((1, 3, 16, 16)))
    >>> x.tensor_trace
    [{(0, 0)}, {(0, 1)}, {(0, 2)}]
    >>> (x + x).tensor_trace
    [{(0, 0)}, {(0, 1)}, {(0, 2)}]
    >>> (torch.exp(x)).tensor_trace
    [{(0, 0)}, {(0, 1)}, {(0, 2)}]
    >>> (x[:, :2]).tensor_trace
    [{(0, 0)}, {(0, 1)}]
    >>> (x[:1]).tensor_trace  # batch dim does not affect trace
    [{(0, 0)}, {(0, 1)}, {(0, 2)}]
    >>> (x[:, :2] + x[:, -2:]).tensor_trace
    [{(0, 1), (0, 0)}, {(0, 1), (0, 2)}]
    >>> (torch.cat([x[:, :2], x[:, -2:]], dim=1)).tensor_trace
    [{(0, 0)}, {(0, 1)}, {(0, 1)}, {(0, 2)}]
    >>> (torch.mean(x)).tensor_trace  # scalar influenced by all input channels
    [{(0, 1), (0, 2), (0, 0)}]
    >>> (torch.mean(x, dim=0).tensor_trace)  # mean along batch not channel
    [{(0, 0)}, {(0, 1)}, {(0, 2)}]
    >>> y = nn.Conv2d(3, 5, 3)(x)
    >>> [channel for ( (_, channel), ) in y.tensor_trace]
    [0, 1, 2, 3, 4]
    >>> (nn.BatchNorm2d(3)(x)).tensor_trace
    [{(0, 0)}, {(0, 1)}, {(0, 2)}]
    >>> x.tensor_to_traces
    {}
    >>> y = torch.cat([x[:, -2:], x[:, :1]], dim=1)
    >>> y = nn.Conv2d(3, 1, 1)(y)
    >>> key = sorted(y.tensor_to_traces.keys())[2] # get second trace
    >>> y.tensor_to_traces[key]
    [[{(0, 1)}, {(0, 2)}], [{(0, 0)}]]
    >>> y = torch.permute(x, (0, 2, 3, 1))
    >>> y.channel_axis
    3
    >>> y[:, :2].tensor_trace  # not modifying channel axis, trace same
    [{(0, 0)}, {(0, 1)}, {(0, 2)}]
    >>> y[:, :, :, :2].tensor_trace  # modifying channel, *should affect trace
    [{(0, 0)}, {(0, 1)}]
    >>> x = Tracer(torch.rand((1, 3)))
    >>> nn.Linear(3, 4)(x).tensor_trace  # Ensure linear layer is producer
    [{(1, 0)}, {(1, 1)}, {(1, 2)}, {(1, 3)}]
    """
    def __init__(
        self,
        data,
        tensor_id: int = None,
        id_generator: Callable = None,
        tensor_trace: List = None,
        tensor_to_traces: Dict = None,
        tensor_to_metadata: Dict = None,
        channel_axis=1,
    ):
        """
        Args:
            tensor_id int: unique identifier for this tensor
            id_generator Callable: function that generates unique IDs
            tensor_trace Trace: which input channels each current output
                channel uses. Avoid using `trace` name to avoid conflict with
                `Tensor.trace` method.
            tensor_to_traces Dict[int, Trace]: which source tensor channels
                each destination tensor's input channel uses
            tensor_to_metadata: Dict[int, Dict]: Mapping from tensor id to
                metadata
            channel_axis int: dimension for channel axis
        """
        super().__init__(data, tensor_id=tensor_id, id_generator=id_generator)

        # Initialize layer trace
        if not tensor_trace:
            n_channels = data.shape[1] if len(data.shape) > 1 else 1
            tensor_trace = trace_from_n_channels(n_channels, self.tensor_id)
        self.tensor_trace = tensor_trace

        # Initialize store for all traces
        self.tensor_to_traces = tensor_to_traces or {}

        # Initialize log of all output channel counts
        self.tensor_to_metadata = tensor_to_metadata or {}

        self.channel_axis = channel_axis

    @classmethod
    def post_torch_func_hook(
        cls,
        func: Callable,
        kwargs: Dict,
        tensor_id: int,
        out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the new channel trace and update metadata."""
        tracers = cls.get_tracers(kwargs, True)  # collect all inputs
        non_tracers = cls.get_tracers(kwargs)

        # Compute new channel configuration
        # collect all original channel traces
        original_traces = [tracer.tensor_trace for tracer in tracers]
        trace, is_trace_reset, channel_axis = cls.compute_new_channels(
            tensor_id, original_traces, func, kwargs, out,
            tracers[0].channel_axis)

        # Collect new cumulative traces store
        tensor_to_traces = {}
        for tracer in tracers:
            for other_id, traces in tracer.tensor_to_traces.items():
                tensor_to_traces[other_id] = traces  # collect all traces
        tensor_to_traces[tensor_id] = original_traces  # save trace for this op

        # Add metadata about this func and its inputs/outputs
        tensor_to_metadata = {
            k: v for tracer in tracers for k,
            v in tracer.tensor_to_metadata.items()
        }
        tensor_to_metadata[tensor_id] = Metadata(
            num_output_channels=out.shape[channel_axis] if len(
                out.shape) > channel_axis else 1,
            non_tracers=non_tracers,  # prunable tensors -- params, constants
            is_trace_reset=is_trace_reset,
            channel_axis=channel_axis,
        )

        return out, {
            'tensor_trace': trace,
            'tensor_to_traces': tensor_to_traces,
            'tensor_to_metadata': tensor_to_metadata,
            'channel_axis': channel_axis
        }

    @classmethod
    def compute_new_channels(
        cls,
        tensor_id: int,
        traces: List[Trace],
        func: Callable,
        kwargs: Dict,
        out: torch.Tensor,
        channel_axis: int
    ) -> Tuple[Trace, bool, int]:
        """
        Compute effect of different operations on channels.

        This section of code will likely be bug-prone unless we come up with a
        different design. Effectively, this section of code needs to keep up
        with the different torch operation's effects on channels.

        Note this section can be tested automatically by passing d-dimensional
        tensors to every possible torch function, with a random value in the
        d-dimensional tensor set to NaN. Then, compare with which output
        channels our function claims each input channel affects.

        Returns:
            trace - Updated or new trace
            is_trace_reset - If this tensor is itself a source tensor. In other
                words, it is its own source.
            channel_axis - dimension for the channel axis

        >>> x, traces = torch.rand(1, 3, 12, 12), [trace_from_n_channels(0, 3)]
        >>> k = {'input': x, 'dim': (0, 2, 3, 1)}
        >>> Tracer.compute_new_channels(0, traces, torch.permute, k, x, 1)[2]
        3
        """
        # TODO: support view, reshape by tracking channel axes
        # TODO: support pad with a n/a source?
        is_trace_reset = False
        is_convolution = func in (
            torch.conv1d,
            torch.conv2d,
            torch.conv3d,
            torch.conv_transpose1d,
            torch.conv_transpose2d,
            torch.conv_transpose3d)
        if is_convolution and kwargs['groups'] == 1:
            # start new trace for this source convolution
            trace = trace_from_n_channels(out.shape[channel_axis], tensor_id)
            is_trace_reset = True
        elif func == F.linear:
            # start new trace for linear layer
            trace = trace_from_n_channels(out.shape[-1], tensor_id)
            is_trace_reset = True
        elif (
            func in (torch.mean, torch.sum)
            and (len(out.shape) <= channel_axis
                 or out.shape[channel_axis] <= 1)
        ):
            # union all channels for summary op
            trace = flatten_trace(traces[0])
        elif func == torch.cat and kwargs['dim'] == channel_axis:
            trace = sum(traces, [])  # stack all channels for concatenate
        # TODO: does not properly capture [..., :2]
        elif (
            func == cls.__getitem__
            and isinstance(kwargs['idx'], (list, tuple))
            and len(kwargs['idx']) > channel_axis
        ):
            # handles slices along channel axis
            trace = traces[0][kwargs['idx'][channel_axis]]
        elif func == torch.flatten:
            # TODO: more general solution for wrapping/unwrapping
            n, _, h, w = torch.Tensor(kwargs['input']).shape
            # TODO: Make a sparse version of Trace?
            flattened = [[item] * (h * w) for item in traces[0]]
            trace = Trace(n * list(itertools.chain(*flattened)))
        elif func == torch.permute:
            channel_axis = kwargs['dim'].index(channel_axis)
            trace = traces[0]
        else:
            trace = union_traces(traces)  # handles elementwise operations

        if (
            func not in (torch.mean, F.linear)
            and len(trace) != get_n_channels(out, channel_axis)
        ):
            logger.warning(
                f"Trace has length {len(trace)} but output tensor has "
                f"{get_n_channels(out, channel_axis)} channels. This *might "
                f"mean Tracer.compute_new_channels does not account for "
                f"`{func.__name__}`")

        if (
            is_convolution and
            kwargs['groups'] not in (1, torch.Tensor(kwargs['input']).shape[1])
        ):  # TODO: more general solution for wrapping/unwrapping
            logger.warning(
                'Trace does not account for grouped convolutions other than'
                ' depthwise convolutions.')
            # NOTE: This is because generate_pruned_weight assumes grouped =
            # depthwise for now. Grouped convolutions add some complex
            # inter-convolution constraints.

        return trace, is_trace_reset, channel_axis
