# -*- coding: utf-8 -*-
"""
Neural network support module.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from collections import OrderedDict

from typing import List, Optional, Sequence, Tuple
from numpy.typing import NDArray


import torch as pt
from torch import nn as tnn
from torch.utils import data as tdt
from torch import cuda as tcd
from torch import optim as top


eps = np.finfo(np.float32).eps


class DatasetPixel(tdt.Dataset):
    """Dataset class for various learned methods."""

    def __init__(self, filter_vals: NDArray, pixel_vals: Optional[NDArray] = None, pixel_weights: Optional[NDArray] = None):
        self.filter_values = pt.tensor(filter_vals, dtype=pt.float)
        self.pixel_values = pt.tensor(pixel_vals, dtype=pt.float) if pixel_vals is not None else None
        self.pixel_weights = pt.tensor(pixel_weights, dtype=pt.float) if pixel_weights is not None else None

    def __len__(self):
        return self.filter_values.shape[0]

    def __getitem__(self, idx):
        if self.pixel_values is not None:
            if self.pixel_weights is not None:
                return self.filter_values[idx, :], self.pixel_values[idx], self.pixel_weights[idx]
            else:
                return self.filter_values[idx, :], self.pixel_values[idx]
        else:
            return self.filter_values[idx, :]


class ModelNetwork(tnn.Module):
    """Neural network model class."""

    def __init__(self, layers_size: Sequence[int], linears_labels: Optional[Sequence[str]] = None):
        super(ModelNetwork, self).__init__()

        self.linears_dims = [(layers_size[ii], layers_size[ii + 1]) for ii in range(len(layers_size) - 1)]

        tmp_op_stack = OrderedDict()
        for ii, (i, o) in enumerate(self.linears_dims):
            if linears_labels is None:
                lab = f"linear_{ii}"
            else:
                lab = linears_labels[ii]
            tmp_op_stack[lab] = tnn.Linear(i, o)
            tmp_op_stack.move_to_end(lab)

            lab = f"activation_{ii}"
            # tmp_op_stack[lab] = tnn.Sigmoid()
            tmp_op_stack[lab] = tnn.GELU()
            # tmp_op_stack[lab] = tnn.ReLU()
            tmp_op_stack.move_to_end(lab)

        lab = f"flatten"
        tmp_op_stack[lab] = tnn.Flatten(0, 1)
        tmp_op_stack.move_to_end(lab)

        self.op_stack = tnn.Sequential(tmp_op_stack)

    def forward(self, x):
        return self.op_stack(x)

    def set_weights(self, weights: Sequence[NDArray], biases: Sequence[NDArray]) -> None:
        tmp_op_stack = self.op_stack.state_dict()
        for ii, (iw, ib) in enumerate(zip(weights, biases)):
            tw = tmp_op_stack[f"linear_{ii}.weight"]
            tb = tmp_op_stack[f"linear_{ii}.bias"]

            if tw.shape != iw.shape:
                raise ValueError(f"The input weights at linear {ii} (shape: {iw.shape}) should have shape: {tw.shape}")
            if tb.shape != ib.shape:
                raise ValueError(f"The input bias at linear {ii} (shape: {ib.shape}) should have shape: {tb.shape}")

            tmp_op_stack[f"linear_{ii}.weight"] = pt.tensor(iw)
            tmp_op_stack[f"linear_{ii}.bias"] = pt.tensor(ib)

        self.op_stack.load_state_dict(tmp_op_stack)

    def get_weights(self) -> Tuple[List[NDArray], List[NDArray]]:
        tmp_op_stack = self.op_stack.state_dict()
        weights = [np.array([])] * len(self.linears_dims)
        biases = [np.array([])] * len(self.linears_dims)
        for ii in range(len(self.linears_dims)):
            weights[ii] = tmp_op_stack[f"linear_{ii}.weight"].numpy()
            biases[ii] = tmp_op_stack[f"linear_{ii}.bias"].numpy()

        return weights, biases


class NeuralNetwork:
    """Neural Network proxy object."""

    layers_size: Sequence[int]

    def __init__(
        self,
        layers_size: Sequence[int],
        device: str = "cuda" if tcd.is_available() else "cpu",
        batch_size: int = 512,
    ) -> None:
        self.layers_size = layers_size

        self.device = device
        self.batch_size = batch_size
        self.model = ModelNetwork(self.layers_size).to(self.device)

    def train(self, dataset_train: tdt.Dataset, iterations: int = 10_000, dataset_test: Optional[tdt.Dataset] = None):
        dataloader_train = tdt.DataLoader(dataset_train, batch_size=self.batch_size)
        datasize_train = len(dataloader_train.dataset)
        if dataset_test is not None:
            dataloader_test = tdt.DataLoader(dataset_test, batch_size=self.batch_size)
            datasize_test = len(dataloader_test.dataset)
        loss_fn = tnn.MSELoss(reduction="none")
        learning_rate = 1e-3
        optimizer = top.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)

        self.model.train()
        for ii in range(iterations):
            loss_train = 0.0
            for bunch_train in dataloader_train:
                has_weights = len(bunch_train) == 3
                if has_weights:
                    X, y, w = bunch_train
                    w = w.to(self.device)
                else:
                    X, y = bunch_train
                X, y = X.to(self.device), y.to(self.device)

                def closure():
                    if pt.is_grad_enabled():
                        optimizer.zero_grad()

                    # Compute prediction error
                    pred_y = self.model(X)
                    loss = loss_fn(pred_y, y)

                    if has_weights:
                        loss *= w / w.mean()
                    loss = loss.sum()

                    # Backpropagation
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                loss = optimizer.step(closure=closure)

                loss_train += loss.item()
            loss_train /= datasize_train

            if ii % 100 == 0:
                if dataset_test is not None:
                    loss_test = 0.0
                    for bunch_test in dataloader_test:
                        has_weights = len(bunch_test) == 3
                        if has_weights:
                            X, y, w = bunch_test
                        else:
                            X, y = bunch_test
                        X, y = X.to(self.device), y.to(self.device)

                        pred_y = self.model(X)
                        loss = loss_fn(pred_y, y)
                        if has_weights:
                            w = w.to(self.device)
                            loss *= w / w.mean()
                        loss_test += loss.sum().item()
                    loss_test /= datasize_test
                else:
                    loss_test = np.NaN
                print(f"{ii}-{iterations} loss: {loss_train:>7f}  [test: {loss_test:>7f}]")

    def test(self, dataset: tdt.Dataset):
        dataloader = tdt.DataLoader(dataset, batch_size=self.batch_size)
        loss_fn = tnn.MSELoss(reduction="sum")

        datasize = len(dataloader.dataset)
        self.model.eval()
        test_loss = 0
        with pt.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = loss_fn(pred, y)
                test_loss += loss.item()
        test_loss /= datasize
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    def predict(self, dataset: tdt.Dataset) -> NDArray:
        dataloader = tdt.DataLoader(dataset, batch_size=self.batch_size)
        with pt.no_grad():
            return np.concatenate([self.model(X).numpy() for X in dataloader])
