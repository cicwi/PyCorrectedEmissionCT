#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data fidelity classes.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from typing import Sequence, Union, Any
from numpy.typing import NDArray

from abc import ABC, abstractmethod

from . import operators

from copy import deepcopy


eps = np.finfo(np.float32).eps


NDArrayFloat = NDArray[np.floating]


def _soft_threshold(values: NDArrayFloat, threshold: Union[float, NDArrayFloat]) -> None:
    abs_values = np.abs(values)
    valid_values = abs_values > 0
    if isinstance(threshold, (float, int)) or threshold.size == 1:
        local_threshold = threshold
    else:
        local_threshold = threshold[valid_values]
    values[valid_values] *= np.fmax((abs_values[valid_values] - local_threshold) / abs_values[valid_values], 0)


class DataFidelityBase(ABC):
    """Define the DataFidelity classes interface."""

    data: Union[NDArrayFloat, None]
    sigma: Union[float, NDArrayFloat]
    background: Union[NDArrayFloat, None]

    sigma_data: Union[NDArrayFloat, None]

    __data_fidelity_name__ = ""

    def __init__(self, background: Union[float, NDArrayFloat, None] = None) -> None:
        """
        Initialize the base data-fidelity class.

        Parameters
        ----------
        background : float | NDArrayFloat | None, optional
            The data background. The default is None.
        """
        self.background = np.array(background) if background is not None else None
        self.data = None
        self.sigma = 1.0
        self.sigma_data = None

    def _slice_attr(self, attr: str, ind: Any) -> None:
        attr_val = self.__getattribute__(attr)
        if attr_val is not None and isinstance(attr_val, np.ndarray) and attr_val.size > 1:
            self.__setattr__(attr, attr_val[ind])

    def __getitem__(self, ind: Any) -> "DataFidelityBase":
        """
        Slice the norm and all its attributes.

        Parameters
        ----------
        ind : Any
            Slicing indices.

        Returns
        -------
        DataFidelityBase
            The sliced norm.
        """
        new_self = deepcopy(self)
        for attr in self.__dict__.keys():
            new_self._slice_attr(attr, ind)
        return new_self

    def info(self) -> str:
        """
        Return the data-fidelity info.

        Returns
        -------
        str
            Data fidelity info string.
        """
        if self.background is not None:
            if np.array(self.background).size > 1:
                bckgrnd_str = "(B:<array>)"
            else:
                bckgrnd_str = "(B:%g)" % self.background
        else:
            bckgrnd_str = ""
        return self.__data_fidelity_name__ + bckgrnd_str

    def upper(self) -> str:
        """
        Return the upper case name of the data-fidelity.

        Returns
        -------
        str
            Upper case string name of the data-fidelity.
        """
        return self.info().upper()

    def lower(self) -> str:
        """
        Return the lower case name of the data-fidelity.

        Returns
        -------
        str
            Lower case string name of the data-fidelity.
        """
        return self.info().lower()

    def assign_data(self, data: Union[float, NDArrayFloat, None] = None, sigma: Union[float, NDArrayFloat] = 1.0) -> None:
        """Initialize the data bias, and sigma of the data term.

        Parameters
        ----------
        data : Union[float, NDArrayFloat, None], optional
            The data bias, by default None
        sigma : Union[float, NDArrayFloat], optional
            The sigma, by default 1.0
        """
        self.data = np.array(data) if data is not None else None
        self.sigma = sigma
        self.sigma_data = self._compute_sigma_data()

    def compute_residual(self, proj_primal: NDArrayFloat, mask: Union[NDArrayFloat, None] = None) -> NDArrayFloat:
        """Compute the residual in the dual domain.

        Parameters
        ----------
        proj_primal : NDArrayFloat
            Projection of the primal solution
        mask : Union[NDArrayFloat, None], optional
            Mask of the dual domain, by default None

        Returns
        -------
        NDArrayFloat
            The residual
        """
        if self.background is not None:
            proj_primal = proj_primal + self.background

        if self.data is not None:
            residual = self.data - proj_primal
        else:
            residual = proj_primal.copy()

        if mask is not None:
            residual *= mask
        return residual

    @abstractmethod
    def compute_residual_norm(self, dual: NDArrayFloat) -> float:
        """Compute the norm of the residual.

        Parameters
        ----------
        dual : NDArrayFloat
            The residual in the dual domain.

        Returns
        -------
        float
            The residual norm.
        """

    def _compute_sigma_data(self):
        if self.data is None:
            return None
        else:
            return self.sigma * self.data

    def compute_data_dual_dot(self, dual: NDArrayFloat, mask: Union[NDArrayFloat, None] = None) -> float:
        """Compute the dot product of the data bias and the dual solution.

        Parameters
        ----------
        dual : NDArrayFloat
            The dual solution.
        mask : Union[NDArrayFloat, None], optional
            Mask of the dual domain, by default None

        Returns
        -------
        float
            The dot product between the data bias and the dual solution
        """
        if self.data is not None:
            if mask is not None:
                dual = dual * mask

            return np.dot(dual.flatten(), self.data.flatten())
        else:
            return 0.0

    def initialize_dual(self) -> NDArrayFloat:
        """Initialize the dual domain solution.

        Returns
        -------
        NDArrayFloat
            A zero array with the dimensions of the dual domain.
        """
        return np.zeros_like(self.data)

    def update_dual(self, dual: NDArrayFloat, proj_primal: NDArrayFloat) -> None:
        """Update the dual solution.

        Parameters
        ----------
        dual : NDArrayFloat
            The current dual solution
        proj_primal : NDArrayFloat
            The projected primal solution
        """
        if self.background is None:
            dual += proj_primal * self.sigma
        else:
            dual += (proj_primal + self.background) * self.sigma

    @abstractmethod
    def apply_proximal(self, dual: NDArrayFloat) -> None:
        """Apply the proximal in the dual domain.

        Parameters
        ----------
        dual : NDArrayFloat
            The dual solution
        """

    @abstractmethod
    def compute_primal_dual_gap(
        self, proj_primal: NDArrayFloat, dual: NDArrayFloat, mask: Union[NDArrayFloat, None] = None
    ) -> float:
        """Compute the primal-dual gap of the current solution.

        Parameters
        ----------
        proj_primal : NDArrayFloat
            The projected primal solution (in the dual domain)
        dual : NDArrayFloat
            The dual solution
        mask : Union[NDArrayFloat, None], optional
            Mask in the dual domain, by default None

        Returns
        -------
        float
            The primal-dual gap
        """


class DataFidelity_l2(DataFidelityBase):
    """l2-norm data-fidelity class."""

    __data_fidelity_name__ = "l2"

    def __init__(self, background: Union[float, NDArrayFloat, None] = None) -> None:
        super().__init__(background=background)

    def assign_data(self, data: Union[float, NDArrayFloat, None] = None, sigma: Union[float, NDArrayFloat] = 1.0) -> None:
        super().assign_data(data=data, sigma=sigma)
        self.sigma1 = 1 / (1 + sigma)

    def compute_residual_norm(self, dual: NDArrayFloat) -> float:
        return float(np.linalg.norm(dual.flatten(), ord=2) ** 2)

    def apply_proximal(self, dual: NDArrayFloat) -> None:
        if self.data is not None and self.sigma_data is not None:
            dual -= self.sigma_data
        dual *= self.sigma1

    def compute_primal_dual_gap(
        self, proj_primal: NDArrayFloat, dual: NDArrayFloat, mask: Union[NDArrayFloat, None] = None
    ) -> float:
        return float(
            np.linalg.norm(self.compute_residual(proj_primal, mask), ord=2) + np.linalg.norm(dual, ord=2)
        ) / 2 + self.compute_data_dual_dot(dual)


class DataFidelity_wl2(DataFidelity_l2):
    """Weighted l2-norm data-fidelity class."""

    __data_fidelity_name__ = "wl2"

    def __init__(self, weights: Union[float, NDArrayFloat], background: Union[float, NDArrayFloat, None] = None) -> None:
        super().__init__(background=background)
        self.weights = np.array(weights)

    def assign_data(self, data: Union[float, NDArrayFloat, None], sigma: Union[float, NDArrayFloat] = 1.0):
        super().assign_data(data=data, sigma=sigma)
        if isinstance(self.sigma, np.ndarray):
            dtype = self.sigma.dtype
        else:
            dtype = type(self.sigma)
        invalid_weights = (self.weights == 0).astype(dtype)
        self.sigma1 = 1 / (1 + sigma / (self.weights + invalid_weights)) * (1 - invalid_weights)

    def compute_residual(self, proj_primal, mask: Union[float, NDArrayFloat, None] = None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        if self.data is not None:
            residual = (self.data - proj_primal) * self.weights
        else:
            residual = proj_primal * self.weights
        if mask is not None:
            residual *= mask
        return residual

    def compute_residual_norm(self, dual: Union[float, NDArrayFloat]) -> float:
        valid_weights = self.weights != 0
        if isinstance(dual, np.ndarray):
            dual = dual[valid_weights]
        weights = self.weights[valid_weights]
        return float(np.linalg.norm((dual / np.sqrt(weights)).flatten(), ord=2) ** 2)


class DataFidelity_l2b(DataFidelity_l2):
    """l2-norm ball data-fidelity class."""

    __data_fidelity_name__ = "l2b"

    def __init__(self, local_error: Union[float, NDArrayFloat], background: Union[float, NDArrayFloat, None] = None):
        super().__init__(background=background)
        self.local_error = local_error

    def assign_data(self, data: Union[float, NDArrayFloat, None], sigma: Union[float, NDArrayFloat] = 1.0):
        self.sigma_error = sigma * self.local_error
        self.sigma_sqrt_error = sigma * np.sqrt(self.local_error)
        super().assign_data(data=data, sigma=sigma)
        self.sigma1 = 1 / (1 + self.sigma_error)

    def compute_residual(self, proj_primal: NDArrayFloat, mask: Union[NDArrayFloat, None] = None) -> NDArrayFloat:
        residual = super().compute_residual(proj_primal, mask)
        _soft_threshold(residual, self.sigma_sqrt_error)
        return residual

    def apply_proximal(self, dual: NDArrayFloat) -> None:
        if self.data is not None and self.sigma_data is not None:
            dual -= self.sigma_data
        _soft_threshold(dual, self.sigma_sqrt_error)
        dual *= self.sigma1

    def compute_primal_dual_gap(
        self, proj_primal: NDArrayFloat, dual: NDArrayFloat, mask: Union[NDArrayFloat, None] = None
    ) -> float:
        return float(
            np.linalg.norm(self.compute_residual(proj_primal, mask), ord=2)
            + np.linalg.norm(np.sqrt(self.local_error) * dual, ord=2)
        ) / 2 + self.compute_data_dual_dot(dual)


class DataFidelity_Huber(DataFidelityBase):
    """Huber-norm data-fidelity class. Given a parameter a: l2-norm for x < a, and l1-norm for x > a."""

    __data_fidelity_name__ = "Hub"

    def __init__(self, local_error, background=None, l2_axis=None):
        super().__init__(background=background)
        self.local_error = local_error
        self.l2_axis = l2_axis

    def assign_data(self, data, sigma=1.0):
        self.one_sigma_error = 1 / (1 + sigma * self.local_error)
        super().assign_data(data=data, sigma=sigma)

    def compute_residual_norm(self, dual):
        l2_points = dual <= self.local_error
        l1_points = 1 - l2_points
        return np.linalg.norm(dual[l2_points].flatten(), ord=2) ** 2 + np.linalg.norm(dual[l1_points].flatten(), ord=1)

    def apply_proximal(self, dual):
        if self.data is not None and self.sigma_data is not None:
            dual -= self.sigma_data

        dual *= self.one_sigma_error

        if self.l2_axis is None:
            dual /= np.fmax(1, np.abs(dual))
        else:
            dual_dir_norm_l2 = np.linalg.norm(dual, ord=2, axis=self.l2_axis, keepdims=True)
            dual /= np.fmax(1, dual_dir_norm_l2)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        return (
            np.linalg.norm(self.compute_residual(proj_primal, mask), ord=2)
            + self.compute_data_dual_dot(dual)
            + self.local_error * np.linalg.norm(dual, ord=2)
        )


class DataFidelity_l1(DataFidelityBase):
    """l1-norm data-fidelity class."""

    __data_fidelity_name__ = "l1"

    def __init__(self, background=None):
        super().__init__(background=background)

    def _get_inner_norm(self, dual):
        return np.abs(dual)

    def _apply_threshold(self, dual):
        pass

    def apply_proximal(self, dual, weight=1.0):
        if self.data is not None:
            dual -= self.sigma_data
        self._apply_threshold(dual)
        dual_inner_norm = self._get_inner_norm(dual)
        dual /= np.fmax(dual_inner_norm, weight)
        dual *= weight

    def compute_residual_norm(self, dual):
        dual = dual.copy()
        self._apply_threshold(dual)
        dual_inner_norm = self._get_inner_norm(dual)
        return np.linalg.norm(dual_inner_norm, ord=1)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background

        residual = self.compute_residual(proj_primal, mask)
        self._apply_threshold(residual)
        residual_inner_norm = self._get_inner_norm(residual)
        return np.linalg.norm(residual_inner_norm, ord=1) + self.compute_data_dual_dot(dual)


class DataFidelity_l12(DataFidelity_l1):
    """l12-norm data-fidelity class."""

    __data_fidelity_name__ = "l12"

    def __init__(self, background=None, l2_axis=0):
        super().__init__(background=background)
        self.l2_axis = l2_axis

    def _get_inner_norm(self, dual):
        return np.linalg.norm(dual, ord=2, axis=self.l2_axis, keepdims=True)


class DataFidelity_l1b(DataFidelity_l1):
    """l1-norm ball data-fidelity class."""

    __data_fidelity_name__ = "l1b"

    def __init__(self, local_error, background=None):
        super().__init__(background=background)
        self.local_error = local_error

    def assign_data(self, data, sigma=1.0):
        self.sigma_error = sigma * self.local_error
        super().assign_data(data=data, sigma=sigma)

    def _apply_threshold(self, dual):
        _soft_threshold(dual, self.local_error)


class DataFidelity_KL(DataFidelityBase):
    """KullbackLeibler data-fidelity class."""

    __data_fidelity_name__ = "KL"

    def _compute_sigma_data(self):
        if self.data is None:
            return None
        else:
            return 4 * self.sigma * np.fmax(self.data, 0.0)

    def apply_proximal(self, dual):
        if self.sigma_data is not None:
            dual[:] = (1 + dual[:] - np.sqrt((dual[:] - 1) ** 2 + self.sigma_data[:])) / 2
        else:
            dual[:] = (1 + dual[:] - np.sqrt((dual[:] - 1) ** 2)) / 2

    def compute_residual(self, proj_primal, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        # we take the Moreau envelope here, and apply the proximal to it
        residual = np.fmax(proj_primal, eps) * self.sigma

        self.apply_proximal(residual)

        if mask is not None:
            residual *= mask
        return -residual

    def compute_residual_norm(self, dual):
        return np.linalg.norm(dual.flatten(), ord=1)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background

        if self.data is not None:
            data_nn = np.fmax(self.data, eps)
            proj_primal_nn = np.fmax(proj_primal, eps)
            residual = proj_primal_nn - data_nn * (1 - np.log(data_nn) + np.log(proj_primal_nn))
        else:
            residual = np.copy(proj_primal)
        if mask is not None:
            residual *= mask

        return np.linalg.norm(residual, ord=1)


class DataFidelity_ln(DataFidelityBase):
    """nuclear-norm data-fidelity class."""

    __data_fidelity_name__ = "ln"

    def __init__(self, background=None, ln_axes: Sequence[int] = (1, -1), spectral_norm: DataFidelityBase = DataFidelity_l1()):
        super().__init__(background=background)
        self.ln_axes = ln_axes
        self.spectral_norm = spectral_norm
        self.use_fallback = False

    def apply_proximal(self, dual):
        dual_tmp = dual.copy()

        if self.sigma_data is not None:
            # If we have a bias term, we interpret it as an addition to the rows in the SVD decomposition.
            # Performing this operation before the transpose is a waste of computation, but it simplifies the logic.
            dual_tmp = np.concatenate((dual_tmp, self.sigma_data), axis=self.ln_axes[0])

        if self.use_fallback:
            t_range = [*range(len(dual_tmp.shape))]
            t_range.append(t_range.pop(self.ln_axes[0]))
            t_range.append(t_range.pop(self.ln_axes[1]))
            dual_tmp = np.transpose(dual_tmp, t_range)

            U, s_p, Vt = np.linalg.svd(dual_tmp, full_matrices=False)

            self.spectral_norm.apply_proximal(s_p)

            dual_tmp = np.matmul(U, s_p[..., None] * Vt)
            dual_tmp = np.transpose(dual_tmp, np.argsort(t_range))
        else:
            op_svd = operators.TransformSVD(dual_tmp.shape, axes_rows=self.ln_axes[0], axes_cols=self.ln_axes[1])
            s_p = op_svd(dual_tmp)

            self.spectral_norm.apply_proximal(s_p)

            dual_tmp = op_svd.T(s_p)

        if self.data is not None:
            # We now strip the bias data, to make sure that we don't change dimensionality.
            # dual_tmp = dual_tmp[..., : dual_tmp.shape[-2] - 1 :, :]
            dual_tmp = np.take(dual_tmp, np.arange(dual_tmp.shape[self.ln_axes[0]] - 1), axis=self.ln_axes[0])

        dual[:] = dual_tmp[:]

    def compute_residual_norm(self, dual):
        op_svd = operators.TransformSVD(dual.shape, axes_rows=self.ln_axes[0], axes_cols=self.ln_axes[1])
        s_p = op_svd(dual)
        return np.linalg.norm(s_p, ord=1)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background

        residual = self.compute_residual(proj_primal, mask)

        return self.compute_residual_norm(residual) + self.compute_data_dual_dot(dual)
