# Pruning Export Algorithm

Here is the basic API for pruning. Start with a normal PyTorch model and input data.

    net = resnet18()
    x = torch.rand((1, 3, 224, 224))

Given a list of pruned channels, create the pruning manager and compute pruning parameters.

    manager = PruningManager(net)
    pruned_inputs = [('layer1.1.conv1.weight', 5)]  # specifies (param name, channel id)
    manager.compute([x], pruned_inputs=pruned_inputs)

To validate the pruned masks, generate a mock pruned model. You can then run inference with this mock pruned model.

    net_mock = manager.get_mock_model()
    y_mock = net_mock(x)

You can then prune the model in place, using the generated masks.

    manager.prune()
    y_pruned = net(x)

The outputs, both the mock pruned and the actually pruned, should match with bit precision.

    assert (y_pruned == y_mock).all()

To run e2e tests and doctests for core pruning code, use

    py.test test/pruning/test_pruning_core.py --doctest-modules pruning

See `./manager.py` for a working doctest that exemplifies this.

## Definitions

- ChannelSource: where a channel comes from
- ChannelSources: sources for a channel
- trace: channelsources for a layer
- tensor_to_traces: mapping from layer to input traces
- producer: operation that produces output for a segment. Convolution or dense layer.
- consumer: operation that consumes input for a segment. Convolution or dense layer.

## File Structure

The codebase is roughly organized by 'feature' in pruning export:

- `manager.py`: High-level API for using this pruning export implementation. Our end-to-end tests all use this API.
- `trace.py`: The data structure for storing input channels used to compute an output channel -- called "traces".
- `tracing.py`: Custom tensors that actually compute and store traces. Pass these as input to trace a new model.
- `resolve.py`: Compute weight and activation indices that effect pruning. Indices stored in a 'pruning spec'.
- `reorder.py`: Graph algorithm for computing activation and weight reordering.
- `mock.py`: Utilities that use the pruning spec to emulate a pruned model's forward pass.
- `pruner.py`: Utilities that use the pruning spec to to actually prune the model in-place.
- `utils.py`: Random utilities.

See the respective files for more in-depth explanations. The files are abstracted in this way so that the core pruning
algorithm can be reused for other frameworks.