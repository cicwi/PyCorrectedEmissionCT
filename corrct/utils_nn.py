# -*- coding: utf-8 -*-
"""
Neural network support module.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from collections import OrderedDict

from typing import Callable, List, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray

import torch as pt
from torch import nn as tnn
from torch.utils import data as tdt
from torch import cuda as tcd
from torch import optim as top

from tqdm import tqdm


eps = np.finfo(np.float32).eps


class DatasetPixel(tdt.Dataset):
    """Dataset class for various learned methods."""

    def __init__(self, inp_vals: NDArray, tgt_vals: Optional[NDArray] = None, tgt_wgts: Optional[NDArray] = None):
        self.inp_vals = pt.tensor(inp_vals, dtype=pt.float)
        self.tgt_vals = pt.tensor(tgt_vals, dtype=pt.float) if tgt_vals is not None else None
        self.tgt_wgts = pt.tensor(tgt_wgts, dtype=pt.float) if tgt_wgts is not None else None

    def __len__(self):
        return self.inp_vals.shape[0]

    def __getitem__(self, idx):
        if self.tgt_vals is not None:
            if self.tgt_wgts is not None:
                return self.inp_vals[idx, :], self.tgt_vals[idx], self.tgt_wgts[idx]
            else:
                return self.inp_vals[idx, :], self.tgt_vals[idx]
        else:
            return self.inp_vals[idx, :]


class Model(tnn.Module):
    """Neural network model class."""

    def __init__(self, layers_size: Sequence[int], problem_type: str = "regression"):
        super(Model, self).__init__()

        self.linears_dims = [(layers_size[ii], layers_size[ii + 1]) for ii in range(len(layers_size) - 1)]

        tmp_op_stack = OrderedDict()
        for ii, (i, o) in enumerate(self.linears_dims):
            lab = f"linear_{ii}"
            tmp_op_stack[lab] = tnn.Linear(i, o)
            tmp_op_stack.move_to_end(lab)

            if ii < (len(self.linears_dims) - 1) or problem_type == "classification":
                lab = f"activation_{ii}"
                tmp_op_stack[lab] = tnn.Sigmoid()
                # tmp_op_stack[lab] = tnn.GELU()
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
            weights[ii] = tmp_op_stack[f"linear_{ii}.weight"].numpy().copy()
            biases[ii] = tmp_op_stack[f"linear_{ii}.bias"].numpy().copy()

        return weights, biases

    def orthogonalize_layer(
        self, layer: Union[int, Sequence[int], None] = None, weight: float = 1.0, normalized: bool = False
    ) -> None:
        w, b = self.get_weights()

        if layer is None:
            layer = [*range(len(w))]
        elif not isinstance(layer, Sequence):
            layer = [layer]

        for l in layer:
            for ii in range(1, w[l].shape[0]):
                for jj in range(0, ii):
                    other_vec = w[l][jj, ...]
                    w[l][ii, ...] -= weight * other_vec * other_vec.dot(w[l][ii, ...]) / other_vec.dot(other_vec)

            if normalized:
                w[l] /= np.linalg.norm(w[l], ord=1, axis=-1, keepdims=True)

        self.set_weights(w, b)

    def normalize_layer(self, layer: int, axis: int = 0, norm: int = 2) -> None:
        w, b = self.get_weights()

        norms = np.linalg.norm(w[layer], axis=axis, keepdims=True, ord=norm)
        w[layer] /= norms / np.mean(norms)

        self.set_weights(w, b)


class TrainingInfo:
    """Training report object."""

    loss_values_trn: NDArray
    loss_values_tst: NDArray

    loss_init_tst: float

    init_weights: Tuple[Sequence[NDArray], Sequence[NDArray]]
    best_weights: Tuple[Sequence[NDArray], Sequence[NDArray]]

    def __init__(self, epochs: int, loss_tst: float, weights: Tuple[Sequence[NDArray], Sequence[NDArray]]) -> None:
        self.loss_values_trn = np.zeros(epochs)
        self.loss_values_tst = np.zeros(epochs)

        self.loss_init_tst = loss_tst

        self.init_weights = self.best_weights = weights


class NeuralNetwork:
    """Neural Network proxy object."""

    layers_size: Sequence[int]

    def __init__(
        self,
        layers_size: Sequence[int],
        device: str = "cuda" if tcd.is_available() else "cpu",
        batch_size_default: int = 2**20,
    ) -> None:
        self.layers_size = layers_size

        self.device = device
        self.batch_size_default = batch_size_default
        self.model = Model(self.layers_size).to(self.device)

    def train_adam(
        self,
        data_train: Sequence[NDArray],
        data_test: Sequence[NDArray],
        batch_size: int = 2**8,
        iterations: int = 10_000,
        encourage_orthogonal: bool = True,
    ) -> TrainingInfo:
        return self.train(
            data_train,
            data_test,
            iterations=iterations,
            batch_size=batch_size,
            optim_class=top.AdamW,
            encourage_orthogonal=encourage_orthogonal,
        )

    def train_lbfgs(
        self,
        data_train: Sequence[NDArray],
        data_test: Sequence[NDArray],
        batch_size: int = 2**20,
        iterations: int = 10_000,
        encourage_orthogonal: bool = False,
    ) -> TrainingInfo:
        return self.train(
            data_train,
            data_test,
            iterations=iterations,
            batch_size=batch_size,
            optim_class=lambda *x, **y: top.LBFGS(*x, **y, max_iter=1000, history_size=200),
            encourage_orthogonal=encourage_orthogonal,
        )

    def train(
        self,
        data_train: Sequence[NDArray],
        data_test: Sequence[NDArray],
        optim_class: Callable = top.AdamW,
        batch_size: Optional[int] = None,
        iterations: int = 10_000,
        encourage_orthogonal: bool = True,
        verbose: bool = True,
    ) -> TrainingInfo:
        dataset_train = DatasetPixel(*data_train)
        dataset_test = DatasetPixel(*data_test)

        datasize_train = len(dataset_train)
        datasize_test = len(dataset_test)

        if batch_size is None:
            batch_size = self.batch_size_default

        dataloader_train = tdt.DataLoader(dataset_train, batch_size=batch_size)
        dataloader_test = tdt.DataLoader(dataset_test, batch_size=batch_size)

        loss_fn = tnn.MSELoss(reduction="none")

        learning_rate = 1e-3
        optimizer = optim_class(self.model.parameters(), lr=learning_rate)
        scheduler = top.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, min_lr=learning_rate * 1e-2)

        best_loss_test = self._test(dataloader_test, datasize_test, loss_fn)
        best_epoch = -1

        info = TrainingInfo(epochs=iterations, loss_tst=best_loss_test, weights=self.model.get_weights())

        scheduler.step(best_loss_test)

        if verbose:
            print(f"Initial test loss: {best_loss_test:>7f}")

        for ii in tqdm(range(iterations)):
            self.model.train()
            loss_train = 0.0
            for bunch_train in dataloader_train:
                has_weights = len(bunch_train) == 3
                if has_weights:
                    X, y, w = bunch_train
                    w = w.to(self.device, non_blocking=True)
                else:
                    X, y = bunch_train
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                def closure():
                    if pt.is_grad_enabled():
                        optimizer.zero_grad()

                    # Compute prediction error
                    pred_y = self.model(X)
                    loss = loss_fn(pred_y, y)

                    if has_weights:
                        loss *= w
                    loss = loss.sum()

                    # Backpropagation
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                if encourage_orthogonal:
                    self.model.orthogonalize_layer(layer=0, weight=learning_rate * 1e-2)

                loss = optimizer.step(closure=closure)

                loss_train += loss.item()
            loss_train /= datasize_train

            curr_loss_test = self._test(dataloader_test, datasize_test, loss_fn)

            scheduler.step(curr_loss_test)

            if curr_loss_test < best_loss_test:
                best_loss_test = curr_loss_test
                best_epoch = ii
                info.best_weights = self.model.get_weights()

            info.loss_values_trn[ii] = loss_train
            info.loss_values_tst[ii] = curr_loss_test

            if verbose and ii % 20 == 0:
                grad_avg_val = np.max([param.grad.mean() for _, param in self.model.named_parameters()])
                print(f"{ii}-{iterations} loss: {loss_train:>6f}  [test: {curr_loss_test:>7f}, avg grad: {grad_avg_val:>4f}]")

        if verbose:
            print(f"Using weights from epoch {best_epoch}, with loss: {best_loss_test:>7f}")

        self.model.set_weights(*info.best_weights)

        return info

    def _test(self, dataloader: tdt.DataLoader, datasize: int, loss_fn: tnn.Module) -> float:
        self.model.eval()
        loss_test = 0.0
        with pt.inference_mode():
            for bunch_test in dataloader:
                has_weights = len(bunch_test) == 3
                if has_weights:
                    X, y, w = bunch_test
                    w = w.to(self.device, non_blocking=True)
                else:
                    X, y = bunch_test
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                pred_y = self.model(X)
                loss = loss_fn(pred_y, y)
                if has_weights:
                    loss *= w
                loss_test += loss.sum().item()

        return loss_test / datasize

    def test(self, data: Sequence[NDArray], verbose: bool = False) -> float:
        dataset_test = DatasetPixel(*data)
        dataloader_test = tdt.DataLoader(dataset_test, batch_size=self.batch_size_default)

        loss_test = self._test(dataloader_test, datasize=len(dataset_test), loss_fn=tnn.MSELoss(reduction="none"))

        if verbose:
            print(f"Test Error: \n Avg loss: {loss_test:>8f} \n")

        return loss_test

    def predict(self, data: Sequence[NDArray]) -> NDArray:
        dataset = DatasetPixel(*data)
        dataloader = tdt.DataLoader(dataset, batch_size=self.batch_size_default)
        self.model.eval()
        with pt.inference_mode():
            return np.concatenate([self.model(X).numpy() for X in dataloader])
