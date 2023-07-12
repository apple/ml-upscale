# Unconstrained Channel Pruning · [Paper](https://machinelearning.apple.com/research/unconstrained-channel-pruning)

**UPSCALE: Unconstrained Channel Pruning** @ [ICML 2023](https://icml.cc/virtual/2023/poster/25215)<br/>
[Alvin Wan](https://alvinwan.com), [Hanxiang Hao](https://scholar.google.com/citations?user=IMn1m2sAAAAJ&hl=en&oi=ao), [Kaushik Patnaik](https://openreview.net/profile?id=~Kaushik_Patnaik1), [Yueyang Xu](https://github.com/inSam), [Omer Hadad](https://scholar.google.com/citations?user=cHZBEjQAAAAJ&hl=en), [David Güera](https://davidguera.com), [Zhile Ren](https://jrenzhile.com), [Qi Shan](https://scholar.google.com/citations?user=0FbnKXwAAAAJ&hl=en)

By removing constraints from existing pruners, we improve ImageNet accuracy for post-training pruned models by 2.1 points on average - benefiting DenseNet (+16.9), EfficientNetV2 (+7.9), and ResNet (+6.2). Furthermore, for these unconstrained pruned models, UPSCALE improves inference speeds by up to 2x over a baseline export.

## Quick Start

Install our package.

```bash
pip install apple-upscale
```

Mask and prune channels, using the default magnitude pruner.

```python
import torchvision
from upscale import MaskingManager, PruningManager

x = torch.rand((1, 3, 224, 224)).cuda()
model = torchvision.models.get_model('resnet18', pretrained=True).cuda()  # get any pytorch model
MaskingManager(model).importance().mask()
PruningManager(model).compute([x]).prune()
```

## Customize Pruning

We provide a number of pruning heuristics out of the box:

- Magnitude ([L1](https://arxiv.org/abs/1608.08710) and [L2](https://arxiv.org/abs/1608.03665))
- [LAMP](https://arxiv.org/abs/2010.07611)
- [FPGM](https://arxiv.org/abs/1811.00250)
- [HRank](https://arxiv.org/abs/2002.10179)

You can pass the desired heuristic into the `UpscaleManager.mask` method call. You can also configure the pruning ratio in `UpscaleManager.mask`. A value of `0.25` means 25% of channels are set to zero.

```python
from upscale.importance import LAMP
MaskingManager(model).importance(LAMP()).mask(amount=0.25)
```

You can also zero out channels using any method you see fit.

```python
model.conv0.weight[:, 24] = 0
```

Then, run our export.

```python
PruningManager(model).compute([x]).prune()
```

## Advanced

You may want direct access to network segments to build a heavily-customized pruning algorithm.

```python
for segment in MaskingManager(model).segments():
    # prune each segment in the network independently
    for layer in segment.layers:
        # layers in the segment
```

## Development

> **NOTE:** See [upscale/pruning/README.md](upscale/pruning/README.md) for more details on how the core export algorithm code is organized.

Clone and setup.

```bash
git clone git@github.com:apple/ml-upscale.git
cd upscale
pip install -e .
```

Run tests.

```
py.test src tests --doctest-modules
```

## Paper

Follow the development installation instructions to have the paper code under `paper/` available.

To run the baseline unconstrained export, pass `baseline=True` to `PruningManager.prune`.

```python
PruningManager(model).compute([x]).prune(baseline=True)
```

To reproduce the paper results, run

```bash
python paper/main.py resnet18
```

Plug in any model in the `torchvision.models` namespace.

```
usage: main.py [-h] [--side {input,output} [{input,output} ...]]
               [--method {constrained,unconstrained} [{constrained,unconstrained} ...]]
               [--amount AMOUNT [AMOUNT ...]] [--epochs EPOCHS] 
               [--heuristic {l1,l2,lamp,fpgm,hrank}] [--global] [--out OUT] 
               [--force] [--latency] [--clean]
               model

positional arguments:
  model                 model to prune

options:
  -h, --help            show this help message and exit
  --side {input,output} [{input,output} ...]
                        prune which "side" -- producers, or consumers
  --method {constrained,unconstrained} [{constrained,unconstrained} ...]
                        how to handle multiple branches
  --amount AMOUNT [AMOUNT ...]
                        amounts to prune by. .6 means 60 percent pruned
  --epochs EPOCHS       number of epochs to train for
  --heuristic {l1,l2,lamp,fpgm,hrank}
                        pruning heuristic
  --global              apply heuristic globally
  --out OUT             directory to write results.csv to
  --force               force latency rerun
  --latency             measure latency locally
  --clean               clean the dataframe
```

## Citation

If you find this useful for your research, please consider citing

```
@inproceedings{wan2023upscale,
  title={UPSCALE: Unconstrained Channel Pruning},
  author={Alvin Wan and Hanxiang Hao and Kaushik Patnaik and Yueyang Xu and Omer Hadad and David Guera and Zhile Ren and Qi Shan},
  booktitle={ICML},
  year={2023}
}
```
