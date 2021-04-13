#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solvers for the tomographic reconstruction problem.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import numpy.random

import scipy.sparse as sps
import scipy.ndimage as spimg

import copy as cp

from . import operators

try:
    import pywt

    has_pywt = True
    use_swtn = pywt.version.version >= "1.0.2"
except ImportError:
    has_pywt = False
    use_swtn = False
    print("WARNING - pywt was not found")

from tqdm import tqdm


eps = np.finfo(np.float32).eps


# ---- Data Fidelity terms ----


class DataFidelityBase(object):
    """Base data-fidelity class that defines the object interface."""

    __data_fidelity_name__ = ""

    def __init__(self, background=None):
        self.background = background

    def info(self):
        if self.background is not None:
            if np.array(self.background).size > 1:
                bckgrnd_str = "(B:<array>)"
            else:
                bckgrnd_str = "(B:%g)" % self.background
        else:
            bckgrnd_str = ""
        return self.__data_fidelity_name__ + bckgrnd_str

    def upper(self):
        return self.info().upper()

    def lower(self):
        return self.info().lower()

    def assign_data(self, data=None, sigma=1):
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

    def compute_residual_norm(self, dual):
        raise NotImplementedError()

    def _compute_sigma_data(self):
        self.sigma_data = self.sigma * self.data

    @staticmethod
    def _soft_threshold(values, threshold):
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

    def update_dual(self, dual, proj_primal):
        if self.background is None:
            dual += proj_primal * self.sigma
        else:
            dual += (proj_primal + self.background) * self.sigma

    def apply_proximal(self, dual):
        raise NotImplementedError()

    def compute_update_primal(self, dual):
        return dual

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        raise NotImplementedError()


class DataFidelity_l2(DataFidelityBase):
    """l2-norm data-fidelity class."""

    __data_fidelity_name__ = "l2"

    def __init__(self, background=None):
        super().__init__(background=background)

    def assign_data(self, data, sigma=1):
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

    def assign_data(self, data, sigma=1):
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

    def compute_update_primal(self, dual):
        return dual * self.weights


class DataFidelity_l2b(DataFidelity_l2):
    """l2-norm ball data-fidelity class."""

    __data_fidelity_name__ = "l2b"

    def __init__(self, local_error, background=None):
        super().__init__(background=background)
        self.local_error = local_error

    def assign_data(self, data, sigma=1):
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

    def assign_data(self, data, sigma=1):
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

    def apply_proximal(self, dual):
        if self.data is not None:
            dual -= self.sigma_data
        dual /= np.fmax(1, np.abs(dual))

    def compute_residual_norm(self, dual):
        return np.linalg.norm(dual.flatten(), ord=1)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        return np.linalg.norm(self.compute_residual(proj_primal, mask), ord=1) + self.compute_data_dual_dot(dual)


class DataFidelity_l12(DataFidelityBase):
    """l12-norm data-fidelity class."""

    __data_fidelity_name__ = "l12"

    def __init__(self, background=None, l2_axis=0):
        super().__init__(background=background)
        self.l2_axis = l2_axis

    def apply_proximal(self, dual):
        if self.data is not None:
            dual -= self.sigma_data
        dual_dir_norm_l2 = np.linalg.norm(dual, ord=2, axis=self.l2_axis, keepdims=True)
        dual /= np.fmax(1, dual_dir_norm_l2)

    def compute_residual_norm(self, dual):
        temp_dual = np.linalg.norm(dual, ord=2, axis=self.l2_axis)
        return np.linalg.norm(temp_dual.flatten(), ord=1)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        return np.linalg.norm(
            np.linalg.norm(self.compute_residual(proj_primal, mask), ord=2, axis=self.l2_axis), ord=1
        ) + self.compute_data_dual_dot(dual)


class DataFidelity_l1b(DataFidelity_l1):
    """l1-norm ball data-fidelity class."""

    __data_fidelity_name__ = "l1b"

    def __init__(self, local_error, background=None):
        super().__init__(background=background)
        self.local_error = local_error

    def assign_data(self, data, sigma=1):
        self.sigma_error = sigma * self.local_error
        super().assign_data(data=data, sigma=sigma)

    def apply_proximal(self, dual):
        if self.data is not None:
            dual -= self.sigma_data
        self._soft_threshold(dual, self.sigma_error)
        dual /= np.fmax(1, np.abs(dual))

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        return np.linalg.norm(
            np.fmax(np.abs(self.compute_residual(proj_primal, mask)) - self.local_error, 0), ord=1
        ) + self.compute_data_dual_dot(dual)


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


# ---- Regularizers ----


class BaseRegularizer(object):
    """Base regularizer class that defines the Regularizer object interface."""

    __reg_name__ = ""

    def __init__(self, weight, norm):
        self.weight = weight
        self.dtype = None
        self.op = None
        self.norm = norm

    def info(self):
        return self.__reg_name__ + "(w:%g" % self.weight + ")"

    def upper(self):
        return self.__reg_name__.upper()

    def lower(self):
        return self.__reg_name__.lower()

    def initialize_sigma_tau(self, primal):
        raise NotImplementedError()

    def initialize_dual(self):
        return np.zeros(self.op.adj_shape, dtype=self.dtype)

    def update_dual(self, dual, primal):
        raise NotImplementedError()

    def apply_proximal(self, dual):
        self.norm.apply_proximal(dual)

    def compute_update_primal(self, dual):
        raise NotImplementedError()


class Regularizer_Grad(BaseRegularizer):
    """Total Variation (TV) regularizer. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "TV"

    def __init__(self, weight, ndims=2, axes=None, norm=DataFidelity_l12()):
        super().__init__(weight=weight, norm=norm)

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

    def initialize_sigma_tau(self, primal):
        self.dtype = primal.dtype
        self.op = operators.TransformGradient(primal.shape, axes=self.axes)

        self.sigma = 0.5
        self.norm.assign_data(None, sigma=self.sigma)

        return self.weight * 2 * self.ndims

    def update_dual(self, dual, primal):
        dual += self.sigma * self.op(primal)

    def compute_update_primal(self, dual):
        return self.weight * self.op.T(dual)


class Regularizer_TV2D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 2D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "TV2D"

    def __init__(self, weight, axes=None, norm=DataFidelity_l12()):
        super().__init__(weight=weight, ndims=2, axes=axes, norm=norm)


class Regularizer_TV3D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 3D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "TV3D"

    def __init__(self, weight, axes=None, norm=DataFidelity_l12()):
        super().__init__(weight=weight, ndims=3, axes=axes, norm=norm)


class Regularizer_HubTV2D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 2D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "HubTV2D"

    def __init__(self, weight, huber_size, axes=None):
        super().__init__(weight=weight, ndims=2, axes=axes, norm=DataFidelity_Huber(huber_size, l2_axis=0))


class Regularizer_HubTV3D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 3D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "HubTV3D"

    def __init__(self, weight, huber_size, axes=None):
        super().__init__(weight=weight, ndims=3, axes=axes, norm=DataFidelity_Huber(huber_size, l2_axis=0))


class Regularizer_Smooth2D(Regularizer_Grad):
    """It can be used to promote smooth reconstructions."""

    __reg_name__ = "Smooth2D"

    def __init__(self, weight, axes=None, norm=DataFidelity_l2()):
        super().__init__(weight=weight, ndims=2, axes=axes, norm=norm)


class Regularizer_Smooth3D(Regularizer_Grad):
    """It can be used to promote smooth reconstructions."""

    __reg_name__ = "Smooth3D"

    def __init__(self, weight, axes=None, norm=DataFidelity_l2()):
        super().__init__(weight=weight, ndims=3, axes=axes, norm=norm)


class Regularizer_lap(BaseRegularizer):
    """Laplacian regularizer. It can be used to promote smooth reconstructions."""

    __reg_name__ = "lap"

    def __init__(self, weight, ndims=2, axes=None):
        super().__init__(weight=weight, norm=DataFidelity_l1())

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

    def initialize_sigma_tau(self, primal):
        self.dtype = primal.dtype
        self.op = operators.TransformLaplacian(primal.shape, axes=self.axes)

        self.sigma = 0.25
        self.norm.assign_data(None, sigma=self.sigma)

        return self.weight * 4 * self.ndims

    def update_dual(self, dual, primal):
        dual += self.sigma * self.op(primal)

    def compute_update_primal(self, dual):
        return self.weight * self.op.T(dual)


class Regularizer_lap2D(Regularizer_lap):
    """Laplacian regularizer in 2D. It can be used to promote smooth reconstructions."""

    __reg_name__ = "lap2D"

    def __init__(self, weight):
        Regularizer_lap.__init__(self, weight=weight, ndims=2)


class Regularizer_lap3D(Regularizer_lap):
    """Laplacian regularizer in 3D. It can be used to promote smooth reconstructions."""

    __reg_name__ = "lap3D"

    def __init__(self, weight):
        Regularizer_lap.__init__(self, weight=weight, ndims=3)


class Regularizer_l1(BaseRegularizer):
    """l1-norm regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "l1"

    def __init__(self, weight):
        super().__init__(weight=weight, norm=DataFidelity_l1())

    def initialize_sigma_tau(self, primal):
        self.dtype = primal.dtype
        self.op = operators.TransformIdentity(primal.shape)

        self.norm.assign_data(None, sigma=1)

        return self.weight

    def update_dual(self, dual, primal):
        dual += primal

    def compute_update_primal(self, dual):
        return self.weight * dual


class Regularizer_swl(BaseRegularizer):
    """Base stationary wavelet regularizer. It can be used to promote sparse reconstructions in the wavelet domain."""

    __reg_name__ = "swl"

    def info(self):
        return self.__reg_name__ + "(t:" + self.wavelet + "-l:%d" % self.level + "-w:%g" % self.weight + ")"

    def __init__(
        self, weight, wavelet, level, ndims=2, axes=None, pad_on_demand="constant", normalized=False, norm=DataFidelity_l1()
    ):
        if not has_pywt:
            raise ValueError("Cannot use l1wl regularizer because pywavelets is not installed.")
        if not use_swtn:
            raise ValueError("Cannot use l1wl regularizer because pywavelets is too old (<1.0.2).")
        super().__init__(weight=weight, norm=norm)
        self.wavelet = wavelet
        self.level = level
        self.normalized = normalized

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

        self.pad_on_demand = pad_on_demand

        self.scaling_func_mult = 2 ** np.arange(self.level, 0, -1)
        self.scaling_func_mult = np.tile(self.scaling_func_mult[:, None], [1, (2 ** self.ndims) - 1])
        self.scaling_func_mult = np.concatenate(([self.scaling_func_mult[0, 0]], self.scaling_func_mult.flatten()))

    def initialize_sigma_tau(self, primal):
        self.dtype = primal.dtype
        self.op = operators.TransformStationaryWavelet(
            primal.shape,
            wavelet=self.wavelet,
            level=self.level,
            axes=self.axes,
            pad_on_demand=self.pad_on_demand,
            normalized=self.normalized,
        )

        if self.normalized:
            self.sigma = 1
            self.norm.assign_data(None, sigma=self.sigma)

            return self.weight * self.scaling_func_mult.size
        else:
            self.sigma = np.reshape(1 / self.scaling_func_mult, [-1] + [1] * len(self.op.dir_shape))
            self.norm.assign_data(None, sigma=self.sigma)

            tau = np.ones_like(self.scaling_func_mult) * ((2 ** self.ndims) - 1)
            tau[0] += 1
            return self.weight * np.sum(tau / self.scaling_func_mult)

    def update_dual(self, dual, primal):
        upd = self.op(primal)
        if not self.normalized:
            upd *= self.sigma
        dual += upd

    def compute_update_primal(self, dual):
        return self.weight * self.op.T(dual)


class Regularizer_l1swl(Regularizer_swl):
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "l1swl"

    def __init__(self, weight, wavelet, level, ndims=2, axes=None, pad_on_demand="constant", normalized=False):
        super().__init__(
            weight,
            wavelet,
            level,
            ndims=ndims,
            axes=axes,
            pad_on_demand=pad_on_demand,
            normalized=normalized,
            norm=DataFidelity_l1(),
        )


class Regularizer_Hub_swl(Regularizer_swl):
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "Hubswl"

    def __init__(
        self, weight, wavelet, level, ndims=2, axes=None, pad_on_demand="constant", normalized=False, huber_size=None
    ):
        super().__init__(
            weight,
            wavelet,
            level,
            ndims=ndims,
            axes=axes,
            pad_on_demand=pad_on_demand,
            normalized=normalized,
            norm=DataFidelity_Huber(huber_size),
        )


class Regularizer_dwl(BaseRegularizer):
    """Base decimated wavelet regularizer. It can be used to promote sparse reconstructions in the wavelet domain."""

    __reg_name__ = "dwl"

    def info(self):
        return self.__reg_name__ + "(t:" + self.wavelet + "-l:%d" % self.level + "-w:%g" % self.weight + ")"

    def __init__(self, weight, wavelet, level, ndims=2, axes=None, pad_on_demand="constant", norm=DataFidelityBase()):
        if not has_pywt:
            raise ValueError("Cannot use l1wl regularizer because pywavelets is not installed.")
        if not use_swtn:
            raise ValueError("Cannot use l1wl regularizer because pywavelets is too old (<1.0.2).")
        super().__init__(weight=weight, norm=norm)
        self.wavelet = wavelet
        self.level = level

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

        self.pad_on_demand = pad_on_demand

        self.scaling_func_mult = 2 ** np.arange(self.level, 0, -1)

    def initialize_sigma_tau(self, primal):
        self.dtype = primal.dtype
        self.op = operators.TransformDecimatedWavelet(
            primal.shape, wavelet=self.wavelet, level=self.level, axes=self.axes, pad_on_demand=self.pad_on_demand
        )

        self.sigma = [np.ones(self.op.sub_band_shapes[0], self.dtype) * self.scaling_func_mult[0]]
        for ii_l in range(self.level):
            d = {}
            for label in self.op.sub_band_shapes[ii_l + 1].keys():
                d[label] = np.ones(self.op.sub_band_shapes[ii_l + 1][label], self.dtype) * self.scaling_func_mult[ii_l]
            self.sigma.append(d)
        self.sigma, _ = pywt.coeffs_to_array(self.sigma, axes=self.axes)
        self.norm.assign_data(None, sigma=self.sigma)

        tau = np.ones_like(self.scaling_func_mult) * ((2 ** self.ndims) - 1)
        tau[0] += 1
        return self.weight * np.sum(tau / self.scaling_func_mult)

    def update_dual(self, dual, primal):
        dual += self.op(primal) * self.sigma

    def compute_update_primal(self, dual):
        return self.weight * self.op.T(dual)


class Regularizer_l1dwl(Regularizer_dwl):
    """l1-norm decimated wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "l1dwl"

    def __init__(self, weight, wavelet, level, ndims=2, axes=None, pad_on_demand="constant"):
        super().__init__(weight, wavelet, level, ndims=ndims, axes=axes, pad_on_demand=pad_on_demand, norm=DataFidelity_l1())


class Regularizer_Hub_dwl(Regularizer_dwl):
    """l1-norm decimated wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "Hubdwl"

    def __init__(self, weight, wavelet, level, ndims=2, axes=None, pad_on_demand="constant", huber_size=None):
        super().__init__(
            weight, wavelet, level, ndims=ndims, axes=axes, pad_on_demand=pad_on_demand, norm=DataFidelity_Huber(huber_size)
        )


class BaseRegularizer_med(BaseRegularizer):
    """Median filter regularizer base class. It can be used to promote filtered reconstructions."""

    __reg_name__ = "med"

    def info(self):
        return self.__reg_name__ + "(s:%s" % np.array(self.filt_size) + "-w:%g" % self.weight + ")"

    def __init__(self, weight, filt_size=3, norm=DataFidelityBase()):
        super().__init__(weight=weight, norm=norm)
        self.filt_size = filt_size

    def initialize_sigma_tau(self, primal):
        self.dtype = primal.dtype
        self.op = operators.TransformIdentity(primal.shape)
        self.norm.assign_data(None, sigma=1)

        return self.weight

    def update_dual(self, dual, primal):
        dual += primal - spimg.median_filter(primal, self.filt_size)

    def compute_update_primal(self, dual):
        return self.weight * dual


class Regularizer_l1med(BaseRegularizer_med):
    """l1-norm median filter regularizer. It can be used to promote filtered reconstructions."""

    __reg_name__ = "l1med"

    def __init__(self, weight, filt_size=3):
        BaseRegularizer_med.__init__(self, weight, filt_size=filt_size, norm=DataFidelity_l1())


class Regularizer_l2med(BaseRegularizer_med):
    """l2-norm median filter regularizer. It can be used to promote filtered reconstructions."""

    __reg_name__ = "l2med"

    def __init__(self, weight, filt_size=3):
        BaseRegularizer_med.__init__(self, weight, filt_size=filt_size, norm=DataFidelity_l2())


class Regularizer_fft(BaseRegularizer):
    """Fourier regularizer. It can be used to promote sparse reconstructions in the Fourier domain."""

    __reg_name__ = "fft"

    def __init__(self, weight, ndims=2, axes=None, mask="exp", norm=DataFidelity_l12()):
        super().__init__(weight=weight, norm=norm)

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

        self.mask = mask

    def initialize_sigma_tau(self, primal):
        self.dtype = primal.dtype
        self.op = operators.TransformFourier(primal.shape, axes=self.axes)

        if isinstance(self.mask, str):
            coords = [np.fft.fftfreq(s) for s in self.op.adj_shape[self.axes]]
            coords = np.array(np.meshgrid(*coords, indexing="ij"))

            if self.mask.lower() == "delta":
                self.sigma = 1 - np.all(coords == 0, axis=0)
            elif self.mask.lower() == "exp":
                self.sigma = 1 - np.exp(-np.sqrt(np.sum(coords ** 2, axis=0)) * 12)
            elif self.mask.lower() == "exp2":
                self.sigma = 1 - np.exp(-np.sum(coords ** 2, axis=0) * 36)
            else:
                raise ValueError('Unknown FFT mask: %s. Options are: "delta", "exp". and "exp2".' % self.mask)

            new_shape = np.ones_like(self.op.adj_shape)
            new_shape[self.axes] = self.op.adj_shape[self.axes]
            self.sigma = np.reshape(self.sigma, new_shape)
        else:
            self.sigma = 1

        self.norm.assign_data(None, sigma=self.sigma)

        return self.weight

    def update_dual(self, dual, primal):
        dual += self.op(primal) * self.sigma

    def compute_update_primal(self, dual):
        return self.weight * self.op.T(dual)


# ---- Constraints ----


class Constraint_LowerLimit(BaseRegularizer):
    """Lower limit constraint. It can be used to promote reconstructions in certain regions of solution space."""

    __reg_name__ = "lowlim"

    def info(self):
        return self.__reg_name__ + "(l:%g" % self.limit + ")"

    def __init__(self, limit, norm=DataFidelityBase()):
        super().__init__(weight=1, norm=norm)
        self.limit = limit

    def initialize_sigma_tau(self, primal):
        self.dtype = primal.dtype
        self.op = operators.TransformIdentity(primal.shape)

        self.norm.assign_data(self.limit, sigma=1)

        return self.weight

    def update_dual(self, dual, primal):
        dual += primal

    def apply_proximal(self, dual):
        dual[dual > self.limit] = self.limit
        self.norm.apply_proximal(dual)

    def compute_update_primal(self, dual):
        return self.weight * dual


class Constraint_UpperLimit(BaseRegularizer):
    """Upper limit constraint. It can be used to promote reconstructions in certain regions of solution space."""

    __reg_name__ = "uplim"

    def info(self):
        return self.__reg_name__ + "(l:%g" % self.limit + ")"

    def __init__(self, limit, norm=DataFidelityBase()):
        super().__init__(weight=1, norm=norm)
        self.limit = limit

    def initialize_sigma_tau(self, primal):
        self.dtype = primal.dtype
        self.op = operators.TransformIdentity(primal.shape)

        self.norm.assign_data(self.limit, sigma=1)

        return self.weight

    def update_dual(self, dual, primal):
        dual += primal

    def apply_proximal(self, dual):
        dual[dual < self.limit] = self.limit
        self.norm.apply_proximal(dual)

    def compute_update_primal(self, dual):
        return self.weight * dual


# ---- Solvers ----


class Solver(object):
    """Base solver class."""

    def __init__(self, verbose=False, relaxation=1, tolerance=None, data_term_test=DataFidelity_l2()):
        self.verbose = verbose
        self.relaxation = relaxation
        self.tolerance = tolerance
        self.data_term_test = cp.deepcopy(data_term_test)

    def info(self):
        return type(self).__name__

    def upper(self):
        return type(self).__name__.upper()

    def lower(self):
        return type(self).__name__.lower()

    @staticmethod
    def _initialize_data_operators(A, At):
        if At is None:
            if isinstance(A, np.ndarray):
                At = A.transpose((1, 0))
            elif isinstance(A, sps.dia_matrix) or isinstance(A, sps.linalg.LinearOperator):
                At = A.transpose()

        if isinstance(At, np.ndarray) or isinstance(At, sps.dia_matrix):
            At_m = At
            At = At_m.dot
        if isinstance(A, np.ndarray) or isinstance(A, sps.dia_matrix):
            A_m = A
            A = A_m.dot
        return (A, At)

    @staticmethod
    def _initialize_regularizer(regularizer):
        if regularizer is None:
            return []
        elif isinstance(regularizer, BaseRegularizer):
            return [regularizer]
        elif isinstance(regularizer, (list, tuple)):
            check_regs_ok = [isinstance(r, BaseRegularizer) for r in regularizer]
            if not np.all(check_regs_ok):
                raise ValueError(
                    "The following regularizers are not derived from the BaseRegularizer class: %s"
                    % np.array(np.arange(len(check_regs_ok))[np.array(check_regs_ok, dtype=np.bool)])
                )
            else:
                return list(regularizer)
        else:
            raise ValueError("Unknown regularizer type.")

    @staticmethod
    def _initialize_b_masks(b, b_mask, b_test_mask):
        if b_test_mask is not None:
            if b_mask is None:
                b_mask = np.ones_like(b)
            # As we are being passed a test residual pixel mask, we need
            # to make sure to mask those pixels out from the reconstruction.
            # At the same time, we need to remove any masked pixel from the test count.
            b_mask, b_test_mask = b_mask * (1 - b_test_mask), b_test_mask * b_mask
        return (b_mask, b_test_mask)


class Sart(Solver):
    """Solver class implementing the Simultaneous Algebraic Reconstruction Technique (SART) algorithm."""

    def __call__(  # noqa: C901
        self, A, b, iterations, A_num_rows, x0=None, At=None, lower_limit=None, upper_limit=None, x_mask=None, b_mask=None
    ):
        """
        """
        # Back-projection diagonal re-scaling
        b_ones = np.ones_like(b)
        if b_mask is not None:
            b_ones *= b_mask
        tau = [At(b_ones[ii, :], ii) for ii in range(A_num_rows)]
        tau = np.abs(np.stack(tau, axis=0))
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = self.relaxation / tau

        # Forward-projection diagonal re-scaling
        x_ones = np.ones(tau.shape[1:], dtype=tau.dtype)
        if x_mask is not None:
            x_ones *= x_mask
        sigma = [A(x_ones, ii) for ii in range(A_num_rows)]
        sigma = np.abs(np.stack(sigma, axis=0))
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        if x0 is None:
            x0 = np.zeros_like(x_ones)
        x = x0

        if self.tolerance is not None:
            res_norm_0 = np.linalg.norm((b * b_mask).flatten())
            res_norm_rel = np.ones((iterations,)) * self.tolerance
        else:
            res_norm_rel = None

        rows_sequence = np.random.permutation(A_num_rows)

        if self.verbose:
            algo_info = "- Performing %s iterations: " % self.upper()

            def progressbar(x):
                return tqdm(x, desc=algo_info)
        else:
            def progressbar(x):
                return x

        for ii in progressbar(range(iterations)):

            for ii_a in rows_sequence:
                res = A(x, ii_a) - b[ii_a, ...]
                if b_mask is not None:
                    res *= b_mask[ii_a, ...]

                x -= At(res * sigma[ii_a, ...], ii_a) * tau[ii_a, ...]

                if lower_limit is not None:
                    x = np.fmax(x, lower_limit)
                if upper_limit is not None:
                    x = np.fmin(x, upper_limit)
                if x_mask is not None:
                    x *= x_mask

            if self.tolerance is not None:
                res = np.empty_like(b)
                for ii_a in rows_sequence:
                    res[..., ii_a, :] = A(x, ii_a) - b[..., ii_a, :]
                if b_mask is not None:
                    res *= b_mask
                res_norm_rel[ii] = np.linalg.norm(res) / res_norm_0

                if self.tolerance > res_norm_rel[ii]:
                    break

        return (x, res_norm_rel)


class Sirt(Solver):
    """Solver class implementing the Simultaneous Iterative Reconstruction Technique (SIRT) algorithm."""

    def __init__(
        self,
        verbose=False,
        tolerance=None,
        relaxation=1.95,
        data_term="l2",
        regularizer=None,
        data_term_test=DataFidelity_l2(),
    ):
        super().__init__(verbose=verbose, tolerance=tolerance, relaxation=relaxation, data_term_test=data_term_test)
        self.data_term = self._initialize_data_fidelity_function(data_term)
        self.regularizer = self._initialize_regularizer(regularizer)

    def _initialize_data_fidelity_function(self, data_term):
        if isinstance(data_term, str):
            if data_term.lower() == "l2":
                return DataFidelity_l2()
            else:
                raise ValueError('Unknown data term: "%s", only accepted terms are: "l2".' % data_term)
        elif isinstance(data_term, (DataFidelity_l2, DataFidelity_KL)):
            return data_term
        else:
            raise ValueError('Unsupported data term: "%s", only accepted terms are "l2"-based.' % data_term.info())

    def info(self):
        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        return Solver.info(self) + "-" + self.data_term.info() + reg_info

    def __call__(  # noqa: C901
        self,
        A,
        b,
        iterations,
        x0=None,
        At=None,
        lower_limit=None,
        upper_limit=None,
        x_mask=None,
        b_mask=None,
        b_test_mask=None,
    ):
        """
        """
        (A, At) = self._initialize_data_operators(A, At)

        (b_mask, b_test_mask) = self._initialize_b_masks(b, b_mask, b_test_mask)

        # Back-projection diagonal re-scaling
        tau = np.ones_like(b)
        if b_mask is not None:
            tau *= b_mask
        tau = np.abs(At(tau))
        for reg in self.regularizer:
            tau += reg.initialize_sigma_tau(tau)
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = self.relaxation / tau

        # Forward-projection diagonal re-scaling
        sigma = np.ones_like(tau)
        if x_mask is not None:
            sigma *= x_mask
        sigma = np.abs(A(sigma))
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        if x0 is None:
            x0 = At(b * sigma) * tau
        x = x0

        self.data_term.assign_data(b, sigma)

        if b_test_mask is not None:
            if self.data_term_test.background != self.data_term.background:
                print("WARNING - the data_term and and data_term_test should have the same background. Making them equal.")
                self.data_term_test.background = self.data_term.background
            self.data_term_test.assign_data(b, sigma)

            res_test_0 = self.data_term_test.compute_residual(0, mask=b_test_mask)
            res_test_norm_0 = self.data_term_test.compute_residual_norm(res_test_0)
            res_test_norm_rel = np.ones((iterations,))
        else:
            res_test_norm_rel = None

        if self.tolerance is not None:
            res_0 = self.data_term.compute_residual(0, mask=b_mask)
            res_norm_0 = self.data_term.compute_residual_norm(res_0)
            res_norm_rel = np.ones((iterations,)) * self.tolerance
        else:
            res_norm_rel = None

        if self.verbose:
            reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
            algo_info = "- Performing %s-%s%s iterations: " % (self.upper(), self.data_term.upper(), reg_info)

            def progressbar(x):
                return tqdm(x, desc=algo_info)
        else:
            def progressbar(x):
                return x

        for ii in progressbar(range(iterations)):
            Ax = A(x)
            res = self.data_term.compute_residual(Ax, mask=b_mask)

            if b_test_mask is not None:
                res_test = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                res_test_norm_rel[ii] = self.data_term_test.compute_residual_norm(res_test.flatten()) / res_test_norm_0

            if self.tolerance is not None:
                res_norm_rel[ii] = self.data_term.compute_residual_norm(res.flatten()) / res_norm_0
                if self.tolerance > res_norm_rel[ii]:
                    break

            q = [reg.initialize_dual() for reg in self.regularizer]
            for q_r, reg in zip(q, self.regularizer):
                reg.update_dual(q_r, x)
                reg.apply_proximal(q_r)

            upd = At(self.data_term.compute_update_primal(res) * sigma)
            for q_r, reg in zip(q, self.regularizer):
                upd -= reg.compute_update_primal(q_r)
            x += upd * tau

            if lower_limit is not None or upper_limit is not None:
                x = x.clip(lower_limit, upper_limit)
            if x_mask is not None:
                x *= x_mask

        return (x, (res_norm_rel, res_test_norm_rel, ii))


class CP(Solver):
    """Solver class implementing the primal-dual algorithm from Chambolle and Pock.

    It allows to specify two types of data fidelity terms: l2-norm and
    Kulback-Leibler. The first assumes Gaussian noise and the second Poisson
    noise as dominant sources of noise in the data.
    This solver also allows to specify a chosen regularizer, from the ones
    based on the BaseRegularizer interface.
    """

    def __init__(
        self,
        verbose=False,
        tolerance=None,
        relaxation=0.95,
        data_term="l2",
        regularizer=None,
        data_term_test=DataFidelity_l2(),
    ):
        super().__init__(verbose=verbose, tolerance=tolerance, relaxation=relaxation, data_term_test=data_term_test)
        self.data_term = self._initialize_data_fidelity_function(data_term)
        self.regularizer = self._initialize_regularizer(regularizer)

    def info(self):
        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        return Solver.info(self) + "-" + self.data_term.info() + reg_info

    @staticmethod
    def _initialize_data_fidelity_function(data_term):
        if isinstance(data_term, str):
            if data_term.lower() == "l2":
                return DataFidelity_l2()
            if data_term.lower() == "l1":
                return DataFidelity_l1()
            if data_term.lower() == "kl":
                return DataFidelity_KL()
            else:
                raise ValueError('Unknown data term: "%s", accepted terms are: "l2" | "l1" | "kl".' % data_term)
        else:
            return data_term

    def power_method(self, A, At, b, iterations=5):
        x = np.random.rand(*b.shape).astype(b.dtype)
        x /= np.linalg.norm(x)
        x = At(x)

        x_norm = np.linalg.norm(x)
        L = x_norm

        for ii in range(iterations):
            x /= x_norm
            x = At(A(x))

            x_norm = np.linalg.norm(x)
            L = np.sqrt(x_norm)

        return (L, x.shape, x.dtype)

    def _get_data_sigma_tau_unpreconditioned(self, A, At, b):
        (L, x_shape, x_dtype) = self.power_method(A, At, b)
        tau = L

        dummy_x = np.empty(x_shape, dtype=x_dtype)
        for reg in self.regularizer:
            tau += reg.initialize_sigma_tau(dummy_x)

        tau = self.relaxation / tau
        sigma = 0.95 / L
        return (x_shape, x_dtype, sigma, tau)

    def __call__(  # noqa: C901
        self,
        A,
        b,
        iterations,
        x0=None,
        At=None,
        upper_limit=None,
        lower_limit=None,
        x_mask=None,
        b_mask=None,
        b_test_mask=None,
        precondition=False,
    ):
        """
        """
        (A, At) = self._initialize_data_operators(A, At)
        if precondition:
            try:
                At_abs = At.absolute()
                A_abs = A.absolute()
            except AttributeError:
                print(A, At)
                print("WARNING: Turning off preconditioning because system matrix does not support absolute")
                precondition = False

        (b_mask, b_test_mask) = self._initialize_b_masks(b, b_mask, b_test_mask)

        if precondition:
            tau = np.ones_like(b)
            if b_mask is not None:
                tau *= b_mask
            tau = np.abs(At_abs(tau))
            for reg in self.regularizer:
                tau += reg.initialize_sigma_tau(tau)
            tau[(tau / np.max(tau)) < 1e-5] = 1
            tau = self.relaxation / tau

            x_shape = tau.shape
            x_dtype = tau.dtype

            sigma = np.ones_like(tau)
            if x_mask is not None:
                sigma *= x_mask
            sigma = np.abs(A_abs(sigma))
            sigma[(sigma / np.max(sigma)) < 1e-5] = 1
            sigma = 0.95 / sigma
        else:
            (x_shape, x_dtype, sigma, tau) = self._get_data_sigma_tau_unpreconditioned(A, At, b)

        if x0 is None:
            x0 = np.zeros(x_shape, dtype=x_dtype)
        x = x0
        x_relax = x

        self.data_term.assign_data(b, sigma)
        p = self.data_term.initialize_dual()

        q = [reg.initialize_dual() for reg in self.regularizer]

        if b_test_mask is not None:
            if self.data_term_test.background != self.data_term.background:
                print("WARNING - the data_term and and data_term_test should have the same background. Making them equal.")
                self.data_term_test.background = self.data_term.background
            self.data_term_test.assign_data(b, sigma)

            res_test_0 = self.data_term_test.compute_residual(0, mask=b_test_mask)
            res_test_norm_0 = self.data_term_test.compute_residual_norm(res_test_0)
            res_test_norm_rel = np.ones((iterations,))
        else:
            res_test_norm_rel = None

        if self.tolerance is not None:
            res_0 = self.data_term.compute_residual(0, mask=b_mask)
            res_norm_0 = self.data_term.compute_residual_norm(res_0)
            res_norm_rel = np.ones((iterations,)) * self.tolerance
        else:
            res_norm_rel = None

        if self.verbose:
            reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
            algo_info = "- Performing %s-%s%s iterations: " % (self.upper(), self.data_term.upper(), reg_info)

            def progressbar(x):
                return tqdm(x, desc=algo_info)
        else:
            def progressbar(x):
                return x

        for ii in progressbar(range(iterations)):
            Ax_rlx = A(x_relax)
            self.data_term.update_dual(p, Ax_rlx)
            self.data_term.apply_proximal(p)

            if b_mask is not None:
                p *= b_mask

            for q_r, reg in zip(q, self.regularizer):
                reg.update_dual(q_r, x_relax)
                reg.apply_proximal(q_r)

            upd = At(p)
            for q_r, reg in zip(q, self.regularizer):
                upd += reg.compute_update_primal(q_r)
            x_new = x - upd * tau

            if lower_limit is not None or upper_limit is not None:
                x_new = x_new.clip(lower_limit, upper_limit)
            if x_mask is not None:
                x_new *= x_mask

            x_relax = x_new + (x_new - x)
            x = x_new

            if b_test_mask is not None or self.tolerance is not None:
                Ax = A(x)

            if b_test_mask is not None:
                res_test = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                res_test_norm_rel[ii] = self.data_term_test.compute_residual_norm(res_test.flatten()) / res_test_norm_0

            if self.tolerance is not None:
                res = self.data_term.compute_residual(Ax, mask=b_mask)
                res_norm_rel[ii] = self.data_term.compute_residual_norm(res.flatten()) / res_norm_0
                if self.tolerance > res_norm_rel[ii]:
                    break

        return (x, (res_norm_rel, res_test_norm_rel, ii))
