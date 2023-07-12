"""
For licensing see accompanying LICENSE file.
Copyright (C) 2023 Apple Inc. All Rights Reserved.

Reproduce experiments from the main paper. Provides basic implementation of
ImageNet training and validation for a number of different pruning heuristics.
"""

import argparse
import copy
from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import time
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset

from upscale.pruning import PruningManager
from upscale.pruning.tracing import trace
from upscale.pruning.pruner import get_tensor_to_module_metadata
from upscale.masking.importance import Magnitude, LAMP, HRank, FPGM
from upscale.masking.mask import MaskingManager
from utils import Latency


class ImagenetClassDataset(datasets.ImageFolder):
    """
    Pytorch Dataset for Imagenet (Classification)
    
    Args:
        is_train (bool, optional): Training if true, evaluation otherwise.
                                   Defaults to True.
    """
    def __init__(self, is_train : bool = True):
        if is_train:
            root_dir = '/tmp/imagenet-1.0.0/data/raw/training/'
            transform_fn = transforms.Compose([
                transforms.RandomResizedCrop(size=224, interpolation=Image.Resampling.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            root_dir = '/tmp/imagenet-1.0.0/data/raw/validation/'
            transform_fn = transforms.Compose([
                transforms.Resize(size=256, interpolation=Image.Resampling.BILINEAR),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        super().__init__(root_dir, transform_fn)


class ImageNetTrainer:
    def __init__(self):
        # get training dataloader
        training_ds = ImagenetClassDataset(is_train=True)
        self.training_loader = torch.utils.data.DataLoader(training_ds, batch_size=256*2, num_workers=16, shuffle=True)  # TODO: better batchsize config
    
    def train_model(self, model, num_epochs=1, num_steps=float('inf'), should_train=True):
        """train the model on the ImageNet

        :param model: type(torchvision.models)
        :param num_epochs: int
        :return: model: type(torchvision.models)
        """
        # model = deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        # model = torch.nn.DataParallel(model).cuda()
        if should_train:
            model.train()
        else:
            model.eval()
        for epoch in range(num_epochs):
            # training one epoch
            pbar = tqdm(enumerate(self.training_loader), total=num_steps if num_steps < float('inf') else len(self.training_loader))
            for i, (images, targets) in pbar:
                images = images.cuda()
                targets = targets.cuda()
                outputs = model(images)

                if i >= num_steps:
                    break
                if not should_train:
                    continue

                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Epoch {epoch}: loss {float(loss) / images.shape[0] :.4f}")
        # return model


class ImageNetValidator:
    def __init__(self):
        # get validation dataloader
        validation_ds = ImagenetClassDataset(is_train=False)
        self.validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size=64*4, num_workers=16, shuffle=False)  # TODO: better batchsize config
    
    def validate_model(self, model):
        """Validate the model on the ImageNet validation set

        :param model: type(torchvision.models)
        :return: accuracy <float>
        """
        # get model
        model.eval()
        # benchmark
        pbar = tqdm(enumerate(self.validation_loader), total=len(self.validation_loader))
        total_num = 0
        correct_num = 0
        for i, (images, targets) in pbar:
            images = images.cuda()
            targets = targets.cuda()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            total_num += len(targets)
            correct_num += torch.sum(preds == targets)
            pbar.set_description(f"Current top-1 accuracy: {correct_num / total_num :.4f}")
        return correct_num / total_num


def prune_model(model, x, is_not_ane=True, constrained=False, baseline=False, **kwargs):
    """Identify input or output channels that have been masked out completely.
    
    Use these channel indices to drop weights.
    """
    model = copy.deepcopy(model)
    mgr = PruningManager(model)
    mgr.compute([x], constrained=constrained, **kwargs)
    mgr.prune(is_not_ane=is_not_ane, baseline=baseline)
    return model


HEURISTICS = {
    'l1': partial(Magnitude, p=1),
    'l2': Magnitude,
    'lamp': LAMP,
    'fpgm': FPGM,
    'hrank': HRank,
}


def save_df(df, path):
    df = df.sort_values(by=['model', 'heuristic', 'side', 'method', 'amount'])
    df.to_csv(path, index=False)


def clean_df(df):
    for model in tqdm(df['model'].unique()):
        for amount in df['amount'].unique():
            for side in df['side'].unique():
                for method in df['method'].unique():
                    for heuristic in df['heuristic'].unique():
                        for epoch in range(5):
                            epoch_condition = (df['model'] == model) * (df['amount'] == amount) * (df['side'] == side) * (df['method'] == method) * (df['heuristic'] == heuristic) * (df['epochs'] == epoch)
                            rows = df[ epoch_condition ]
                            indices = df.index[ epoch_condition ]

                            # drop any redundant rows. drop anything without latency measurements first. could def do this more efficiently lol
                            if len(rows) > 1:
                                remove = []
                                keep, keep_row = None, None
                                for index, (_, row) in zip(indices, rows.iterrows()):
                                    if keep is None:
                                        keep = index
                                        keep_row = row
                                    elif keep_row['gpu_latency_mean'] == -1. and row['gpu_latency_mean'] != -1.:
                                        remove.append(keep)
                                        keep = index
                                        keep_row = row
                                    else:
                                        remove.append(index)
                                print(f"Keeping {keep} and discarding {remove}. Keeping: {dict(keep_row)}")
                                df = df.drop(remove)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default='resnet18', help='model to prune')
    parser.add_argument('--side', nargs='+', default=['input', 'output'], choices=('input', 'output'), help='prune which "side" -- producers, or consumers')
    parser.add_argument('--method', nargs='+', default=['unconstrained', 'constrained'], choices=('constrained', 'unconstrained'), help='how to handle multiple branches')
    parser.add_argument('--amount', nargs='+', default=[.1, .2, .3, .4, .5], type=float, help='amounts to prune by. .6 means 60 percent pruned')
    parser.add_argument('--epochs', default=0, type=int, help='number of epochs to train for')
    parser.add_argument('--heuristic', default='l2', choices=HEURISTICS.keys(), help='pruning heuristic')
    parser.add_argument('--global', dest='is_global', action='store_true', help='apply heuristic globally')
    parser.add_argument('--out', default='./', help='directory to write results.csv to')
    parser.add_argument('--force', action='store_true', help='force latency rerun')
    parser.add_argument('--latency', action='store_true', help='measure latency locally')
    parser.add_argument('--clean', action='store_true', help='clean the dataframe')
    args = parser.parse_args()

    validator = None
    trainer = None

    out = Path(args.out)
    os.makedirs(out, exist_ok=True)
    os.makedirs(out / 'checkpoints', exist_ok=True)

    if args.heuristic == 'lamp':
        assert args.is_global, 'LAMP only makes sense when applied globally (otherwise, its no different from magnitude pruning. Set --global)'

    fields = ('amount', 'epochs', 'global', 'gpu_latency_mean', 'gpu_latency_std', 'gpu_naive_latency_mean', 'gpu_naive_latency_std', 'heuristic', 'method', 'model', 'side', 'val')
    latency = Latency(backend=Latency.TORCH_PROFILER, n_iters=5, n_warmup=2)

    df = None
    if (out / 'results.csv').exists(): # and not args.force:
        df = pd.read_csv(out / 'results.csv')
        if args.clean:
            df = clean_df(df)
        save_df(df, out / 'results.csv')
    else:
        df = pd.DataFrame(columns=fields)

    for side in args.side:
        # HACK: lol
        HEURISTICS['hrank'] = partial(HRank, side=side, out=out)
        heuristic = HEURISTICS[args.heuristic]()
        for method in args.method:
            for amount in args.amount:
                print('='*20)
                print(json.dumps({
                    'amount': amount,
                    'side': side,
                    'method': method,
                    'heuristic': args.heuristic,
                    'model': args.model,
                }, indent=2))
                print('='*20)
                        
                condition = (df['model'] == args.model) * (df['amount'] == amount) * (df['side'] == side) * (df['method'] == method) * (df['heuristic'] == args.heuristic)
                rows = df[ condition ]
                indices = df.index[ condition ]
                if (
                    not args.force
                    and df is not None
                    and len(rows) > 0  # if result already gathered
                    and ((any(mean != -1.0 for mean in rows['gpu_latency_mean']) or (side == 'output' and method == 'unconstrained')) or not args.latency)  # and latency is already measured OR latency not being measured
                ):
                    continue

                path_checkpoint = out / 'checkpoints'/ f"{args.model}-{side}-{method}-{args.heuristic}-{amount}-0.pth"
                # gotta import torchvision after torch.hub.load lol https://github.com/pytorch/hub/issues/46
                # model = torch.hub.load('pytorch/vision', args.model, pretrained=True) # https://github.com/pytorch/vision/issues/7397#issuecomment-1459858748
                model = torchvision.models.get_model(args.model, pretrained=True)
                model = torch.nn.DataParallel(model).cuda()
                model._name = args.model

                if validator is None or trainer is None:
                    validator = ImageNetValidator()
                    trainer = ImageNetTrainer()

                # if path_checkpoint.exists():
                #     model = torch.load(path_checkpoint)
                # else:
                if heuristic.needs_feature_map_statistics:
                    is_hook_registered = heuristic.register(model)
                    if is_hook_registered:
                        trainer.train_model(model, num_steps=1, should_train=False)
                    heuristic.deregister(model)

                MaskingManager(model.module, side=side, method=method, is_global=args.is_global) \
                    .importance(heuristic=heuristic) \
                    .mask(amount=amount)
                
                for epoch in range(args.epochs + 1):
                    epoch_condition = condition * (df['epochs'] == epoch)
                    rows = df[ epoch_condition ]
                    indices = df.index[ epoch_condition ]
                    if len(rows) > 0:
                        data = dict(rows.iloc[0])
                    else:
                        data = {
                            'val': float(validator.validate_model(model)),
                            'side': side,
                            'method': method,
                            'amount': amount,
                            'epochs': epoch,
                            'heuristic': args.heuristic,
                            'global': args.is_global,
                            'model': args.model,
                            'gpu_latency_mean': -1.,
                            'gpu_latency_std': -1.,
                            'gpu_naive_latency_mean': -1.,
                            'gpu_naive_latency_std': -1.,
                        }

                    if not (side == 'output' and method == 'unconstrained') and args.latency and (data['gpu_latency_mean'] == -1. or args.force):  # unconstrained output pruning not supported
                        x = torch.rand((1, 3, 224, 224)).cuda()
                        start = time.time()

                        # get our latency
                        pruned = prune_model(model.module, x, constrained=(method == 'constrained'))
                        logging.info(f"Pruned in {round((time.time() - start) / 1000., 2)}s")
                        data['gpu_latency_mean'], data['gpu_latency_std'] = latency.forward(pruned, x)

                        # get naive export latency
                        if method == 'unconstrained':
                            pruned = prune_model(model.module, x, constrained=(method == 'constrained'), reorder_producer=False, reorder_consumer=False, baseline=True)
                            data['gpu_naive_latency_mean'], data['gpu_naive_latency_std'] = latency.forward(pruned, x)

                    if len(rows) > 0:
                        # update dataframe if rows already exist
                        for key, value in data.items():
                            df.loc[indices[0], key] = value
                    else:
                        # otherwise, add to dataframe
                        df.loc[len(df.index)] = data
                    save_df(df, out / 'results.csv')
                    torch.save(model, path_checkpoint)

                    if epoch < args.epochs:
                        path_checkpoint = out / 'checkpoints'/ f"{args.model}-{side}-{method}-{args.heuristic}-{amount}-{epoch + 1}.pth"
                        if path_checkpoint.exists():
                            model = torch.load(path_checkpoint)
                        trainer.train_model(model)


if __name__ == '__main__':
    main()