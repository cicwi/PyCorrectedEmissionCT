#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regularizers module.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import numpy.random

import scipy.ndimage as spimg

from . import operators
from . import data_terms

try:
    import pywt

    has_pywt = True
    use_swtn = pywt.version.version >= "1.0.2"
except ImportError:
    has_pywt = False
    use_swtn = False
    print("WARNING - pywt was not found")


# ---- Data Fidelity terms ----


DataFidelityBase = data_terms.DataFidelityBase
DataFidelity_l2 = data_terms.DataFidelity_l2
DataFidelity_wl2 = data_terms.DataFidelity_wl2
DataFidelity_l2b = data_terms.DataFidelity_l2b
DataFidelity_l12 = data_terms.DataFidelity_l12
DataFidelity_l1 = data_terms.DataFidelity_l1
DataFidelity_l1b = data_terms.DataFidelity_l1b
DataFidelity_Huber = data_terms.DataFidelity_Huber
DataFidelity_KL = data_terms.DataFidelity_KL


# ---- Regularizers ----


class BaseRegularizer(object):
    """Base regularizer class that defines the Regularizer object interface."""

    __reg_name__ = ""

    def __init__(self, weight, norm):
        self.weight = np.array(weight)
        self.dtype = None
        self.op = None
        self.norm = norm

    def info(self):
        return self.__reg_name__ + "(w:%g" % self.weight.max() + ")"

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

    __reg_name__ = "grad"

    def __init__(self, weight, ndims=2, axes=None, norm=DataFidelity_l12()):
        super().__init__(weight=weight, norm=norm)

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.intp)
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


class Regularizer_smooth2D(Regularizer_Grad):
    """It can be used to promote smooth reconstructions."""

    __reg_name__ = "smooth2D"

    def __init__(self, weight, axes=None, norm=DataFidelity_l2()):
        super().__init__(weight=weight, ndims=2, axes=axes, norm=norm)


class Regularizer_smooth3D(Regularizer_Grad):
    """It can be used to promote smooth reconstructions."""

    __reg_name__ = "smooth3D"

    def __init__(self, weight, axes=None, norm=DataFidelity_l2()):
        super().__init__(weight=weight, ndims=3, axes=axes, norm=norm)


class Regularizer_lap(BaseRegularizer):
    """Laplacian regularizer. It can be used to promote smooth reconstructions."""

    __reg_name__ = "lap"

    def __init__(self, weight, ndims=2, axes=None):
        super().__init__(weight=weight, norm=DataFidelity_l1())

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.intp)
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
        return self.__reg_name__ + "(t:" + self.wavelet + "-l:%d" % self.level + "-w:%g" % self.weight.max() + ")"

    def __init__(
        self,
        weight,
        wavelet,
        level,
        ndims=2,
        axes=None,
        pad_on_demand="constant",
        normalized=False,
        min_approx=False,
        norm=DataFidelity_l1(),
    ):
        if not has_pywt:
            raise ValueError("Cannot use wavelet regularizer because pywavelets is not installed.")
        if not use_swtn:
            raise ValueError("Cannot use stationary wavelet regularizer because pywavelets is too old (<1.0.2).")
        super().__init__(weight=weight, norm=norm)
        self.wavelet = wavelet
        self.level = level
        self.normalized = normalized
        self.min_approx = min_approx

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.intp)
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
        if not self.min_approx:
            dual[0, ...] = 0

    def compute_update_primal(self, dual):
        return self.weight * self.op.T(dual)


class Regularizer_l1swl(Regularizer_swl):
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "l1swl"

    def __init__(
        self, weight, wavelet, level, ndims=2, axes=None, pad_on_demand="constant", normalized=False, min_approx=False
    ):
        super().__init__(
            weight,
            wavelet,
            level,
            ndims=ndims,
            axes=axes,
            pad_on_demand=pad_on_demand,
            normalized=normalized,
            min_approx=min_approx,
            norm=DataFidelity_l1(),
        )


class Regularizer_Hub_swl(Regularizer_swl):
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "Hubswl"

    def __init__(
        self,
        weight,
        wavelet,
        level,
        ndims=2,
        axes=None,
        pad_on_demand="constant",
        normalized=False,
        min_approx=False,
        huber_size=None,
    ):
        super().__init__(
            weight,
            wavelet,
            level,
            ndims=ndims,
            axes=axes,
            pad_on_demand=pad_on_demand,
            normalized=normalized,
            min_approx=min_approx,
            norm=DataFidelity_Huber(huber_size),
        )


class Regularizer_dwl(BaseRegularizer):
    """Base decimated wavelet regularizer. It can be used to promote sparse reconstructions in the wavelet domain."""

    __reg_name__ = "dwl"

    def info(self):
        return self.__reg_name__ + "(t:" + self.wavelet + "-l:%d" % self.level + "-w:%g" % self.weight.max() + ")"

    def __init__(
        self, weight, wavelet, level, ndims=2, axes=None, pad_on_demand="constant", min_approx=False, norm=DataFidelityBase()
    ):
        if not has_pywt:
            raise ValueError("Cannot use wavelet regularizer because pywavelets is not installed.")
        super().__init__(weight=weight, norm=norm)
        self.wavelet = wavelet
        self.level = level
        self.min_approx = min_approx

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.intp)
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
        dual += self.sigma * self.op(primal)
        if not self.min_approx:
            slices = [slice(0, x) for x in self.op.sub_band_shapes[0]]
            dual[tuple(slices)] = 0

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
        return self.__reg_name__ + "(s:%s" % np.array(self.filt_size) + "-w:%g" % self.weight.max() + ")"

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
            axes = np.arange(-ndims, 0, dtype=np.intp)
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
        dual += self.sigma * self.op(primal)

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
