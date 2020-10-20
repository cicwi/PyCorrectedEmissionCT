#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data fidelity classes.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from typing import Sequence
import numpy as np

from abc import ABC, abstractmethod

from . import operators


eps = np.finfo(np.float32).eps


class DataFidelityBase(ABC):
    """
    Initialize the base data-fidelity class.

    This class that defines the object interface.

    Parameters
    ----------
    background : Optional[Union[float, ArrayLike]], optional
        The data background. The default is None.
    """

    __data_fidelity_name__ = ""

    def __init__(self, background=None):
        self.background = background
        self.data = None
        self.sigma = None

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

    def assign_data(self, data=None, sigma=1.0):
        self.data = data
        self.sigma = sigma
        if self.data is not None:
            self._compute_sigma_data()
        else:
            self.sigma_data = None

    def compute_residual(self, proj_primal, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        if self.data is not None:
            residual = self.data - proj_primal
        else:
            residual = np.copy(proj_primal)
        if mask is not None:
            residual *= mask
        return residual

    @abstractmethod
    def compute_residual_norm(self, dual):
        raise NotImplementedError()

    def _compute_sigma_data(self) -> None:
        self.sigma_data = self.sigma * self.data

    @staticmethod
    def _soft_threshold(values, threshold) -> None:
        abs_values = np.abs(values)
        valid_values = abs_values > 0
        if isinstance(threshold, (float, int)) or threshold.size == 1:
            local_threshold = threshold
        else:
            local_threshold = threshold[valid_values]
        values[valid_values] *= np.fmax((abs_values[valid_values] - local_threshold) / abs_values[valid_values], 0)

    def compute_data_dual_dot(self, dual, mask=None):
        if self.data is not None:
            return np.dot(dual.flatten(), self.data.flatten())
        else:
            return 0

    def initialize_dual(self):
        return np.zeros_like(self.data)

    def update_dual(self, dual, proj_primal) -> None:
        if self.background is None:
            dual += proj_primal * self.sigma
        else:
            dual += (proj_primal + self.background) * self.sigma

    @abstractmethod
    def apply_proximal(self, dual) -> None:
        raise NotImplementedError()

    @abstractmethod
    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        raise NotImplementedError()


class DataFidelity_l2(DataFidelityBase):
    """l2-norm data-fidelity class."""

    __data_fidelity_name__ = "l2"

    def __init__(self, background=None):
        super().__init__(background=background)

    def assign_data(self, data, sigma=1.0):
        super().assign_data(data=data, sigma=sigma)
        self.sigma1 = 1 / (1 + sigma)

    def compute_residual_norm(self, dual):
        return np.linalg.norm(dual.flatten(), ord=2) ** 2

    def apply_proximal(self, dual):
        if self.data is not None:
            dual -= self.sigma_data
        dual *= self.sigma1

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        return (
            np.linalg.norm(self.compute_residual(proj_primal, mask), ord=2) + np.linalg.norm(dual, ord=2)
        ) / 2 + self.compute_data_dual_dot(dual)


class DataFidelity_wl2(DataFidelity_l2):
    """Weighted l2-norm data-fidelity class."""

    __data_fidelity_name__ = "wl2"

    def __init__(self, weights, background=None):
        super().__init__(background=background)
        self.weights = weights

    def assign_data(self, data, sigma=1.0):
        super().assign_data(data=data, sigma=sigma)
        self.sigma1 = 1 / (1 + sigma / self.weights)

    def compute_residual(self, proj_primal, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        if self.data is not None:
            residual = (self.data - proj_primal) * self.weights
        else:
            residual = proj_primal * self.weights
        if mask is not None:
            residual *= mask
        return residual

    def compute_residual_norm(self, dual):
        return np.linalg.norm((dual / np.sqrt(self.weights)).flatten(), ord=2) ** 2


class DataFidelity_l2b(DataFidelity_l2):
    """l2-norm ball data-fidelity class."""

    __data_fidelity_name__ = "l2b"

    def __init__(self, local_error, background=None):
        super().__init__(background=background)
        self.local_error = local_error

    def assign_data(self, data, sigma=1.0):
        self.sigma_error = sigma * self.local_error
        self.sigma_sqrt_error = sigma * np.sqrt(self.local_error)
        super().assign_data(data=data, sigma=sigma)
        self.sigma1 = 1 / (1 + self.sigma_error)

    def compute_residual(self, proj_primal, mask=None):
        residual = super().compute_residual(proj_primal, mask)
        self._soft_threshold(residual, self.sigma_sqrt_error)
        return residual

    def apply_proximal(self, dual):
        if self.data is not None:
            dual -= self.sigma_data
        self._soft_threshold(dual, self.sigma_sqrt_error)
        dual *= self.sigma1

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        return (
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
        if self.data is not None:
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

    def apply_proximal(self, dual, weight=1):
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
        self._soft_threshold(dual, self.local_error)


class DataFidelity_KL(DataFidelityBase):
    """KullbackLeibler data-fidelity class."""

    __data_fidelity_name__ = "KL"

    def _compute_sigma_data(self):
        self.sigma_data = 4 * self.sigma * np.fmax(self.data, 0)

    def apply_proximal(self, dual):
        if self.data is not None:
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

        if self.data is not None:
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
        temp_dual = np.linalg.norm(dual, ord=2, axis=self.l2_axis)
        return np.linalg.norm(temp_dual.flatten(), ord=1)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        return np.linalg.norm(
            np.linalg.norm(self.compute_residual(proj_primal, mask), ord=2, axis=self.l2_axis), ord=1
        ) + self.compute_data_dual_dot(dual)
