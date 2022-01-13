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

from abc import ABC, abstractmethod

try:
    import pywt

    has_pywt = True
    use_swtn = pywt.version.version >= "1.0.2"
except ImportError:
    has_pywt = False
    use_swtn = False
    print("WARNING - pywt was not found")


from typing import Union, Sequence, Optional
from numpy.typing import ArrayLike


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


class BaseRegularizer(ABC):
    """
    Initilizie a base regularizer class, that defines the Regularizer object interface.

    Parameters
    ----------
    weight : Union[float, ArrayLike]
        The weight of the regularizer.
    norm : DataFidelityBase
        The norm of the regularizer minimization.
    """

    __reg_name__ = ""

    def __init__(self, weight: Union[float, ArrayLike], norm: data_terms.DataFidelityBase):
        self.weight = np.array(weight)
        self.dtype = None
        self.op = None
        self.norm = norm

    def info(self) -> str:
        """
        Return the regularizer info.

        Returns
        -------
        str
            Regularizer info string.
        """
        return self.__reg_name__ + "(w:%g" % self.weight.max() + ")"

    def upper(self) -> str:
        """
        Return the upper case name of the regularizer.

        Returns
        -------
        str
            Upper case string name of the regularizer.
        """
        return self.__reg_name__.upper()

    def lower(self) -> str:
        """
        Return the lower case name of the regularizer.

        Returns
        -------
        str
            Lower case string name of the regularizer.
        """
        return self.__reg_name__.lower()

    @abstractmethod
    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
        """
        Initialize the internal state, operator, and sigma. It then returns the tau.

        Parameters
        ----------
        primal : ArrayLike
            The primal vector.

        Returns
        -------
        Union[float, ArrayLike]
            The tau to be used in the SIRT or PDHG algorithm.
        """
        raise NotImplementedError()

    def initialize_dual(self) -> ArrayLike:
        """
        Return the initialized dual.

        Returns
        -------
        ArrayLike
            Initialized (zero) dual.
        """
        return np.zeros(self.op.adj_shape, dtype=self.dtype)

    def update_dual(self, dual: ArrayLike, primal: ArrayLike) -> None:
        """
        Update the dual in-place.

        Parameters
        ----------
        dual : ArrayLike
            Current stat of the dual.
        primal : ArrayLike
            Primal or over-relaxation of the primal.
        """
        dual += self.sigma * self.op(primal)

    def apply_proximal(self, dual: ArrayLike) -> None:
        """
        Apply the proximal operator to the dual in-place.

        Parameters
        ----------
        dual : ArrayLike
            The dual to be applied the proximal on.
        """
        if isinstance(self.norm, DataFidelity_l1):
            self.norm.apply_proximal(dual, self.weight)
        else:
            self.norm.apply_proximal(dual)

    def compute_update_primal(self, dual: ArrayLike) -> ArrayLike:
        """
        Compute the partial update of a primal term, from this regularizer.

        Parameters
        ----------
        dual : ArrayLike
            The dual associated to this regularizer.

        Returns
        -------
        upd : ArrayLike
            The update to the primal.
        """
        upd = self.op.T(dual)
        if not isinstance(self.norm, DataFidelity_l1):
            upd *= self.weight
        return upd


class Regularizer_Grad(BaseRegularizer):
    """Gradient regularizer.

    When used with l1-norms, it promotes piece-wise constant reconstructions.
    When used with l2-norm, it promotes smooth reconstructions.

    Parameters
    ----------
    weight : Union[float, ArrayLike]
        The weight of the regularizer.
    ndims : int, optional
        The number of dimensions. The default is 2.
    axes : Sequence, optional
        The axes over which it computes the gradient. If None, it uses the last 2. The default is None.
    norm : DataFidelityBase, optional
        The norm of the regularizer minimization. The default is DataFidelity_l12().
    """

    __reg_name__ = "grad"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        ndims: int = 2,
        axes: Optional[Sequence[int]] = None,
        norm: DataFidelityBase = DataFidelity_l12(),
    ):
        super().__init__(weight=weight, norm=norm)

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
        self.dtype = primal.dtype
        self.op = operators.TransformGradient(primal.shape, axes=self.axes)

        self.sigma = 0.5
        self.norm.assign_data(None, sigma=self.sigma)

        tau = 2 * self.ndims
        if not isinstance(self.norm, DataFidelity_l1):
            tau *= self.weight
        return tau


class Regularizer_TV2D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 2D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "TV2D"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        axes: Optional[Sequence[int]] = None,
        norm: DataFidelityBase = DataFidelity_l12(),
    ):
        super().__init__(weight=weight, ndims=2, axes=axes, norm=norm)


class Regularizer_TV3D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 3D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "TV3D"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        axes: Optional[Sequence[int]] = None,
        norm: DataFidelityBase = DataFidelity_l12(),
    ):
        super().__init__(weight=weight, ndims=3, axes=axes, norm=norm)


class Regularizer_HubTV2D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 2D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "HubTV2D"

    def __init__(self, weight: Union[float, ArrayLike], huber_size: int, axes: Optional[Sequence[int]] = None):
        super().__init__(weight=weight, ndims=2, axes=axes, norm=DataFidelity_Huber(huber_size, l2_axis=0))


class Regularizer_HubTV3D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 3D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "HubTV3D"

    def __init__(self, weight: Union[float, ArrayLike], huber_size: int, axes: Optional[Sequence[int]] = None):
        super().__init__(weight=weight, ndims=3, axes=axes, norm=DataFidelity_Huber(huber_size, l2_axis=0))


class Regularizer_smooth2D(Regularizer_Grad):
    """It can be used to promote smooth reconstructions."""

    __reg_name__ = "smooth2D"

    def __init__(
        self, weight: Union[float, ArrayLike], axes: Optional[Sequence[int]] = None, norm: DataFidelityBase = DataFidelity_l2()
    ):
        super().__init__(weight=weight, ndims=2, axes=axes, norm=norm)


class Regularizer_smooth3D(Regularizer_Grad):
    """It can be used to promote smooth reconstructions."""

    __reg_name__ = "smooth3D"

    def __init__(
        self, weight: Union[float, ArrayLike], axes: Optional[Sequence[int]] = None, norm: DataFidelityBase = DataFidelity_l2()
    ):
        super().__init__(weight=weight, ndims=3, axes=axes, norm=norm)


class Regularizer_lap(BaseRegularizer):
    """Laplacian regularizer. It can be used to promote smooth reconstructions."""

    __reg_name__ = "lap"

    def __init__(self, weight: Union[float, ArrayLike], ndims: int = 2, axes: Optional[Sequence[int]] = None):
        super().__init__(weight=weight, norm=DataFidelity_l1())

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
        self.dtype = primal.dtype
        self.op = operators.TransformLaplacian(primal.shape, axes=self.axes)

        self.sigma = 0.25
        self.norm.assign_data(None, sigma=self.sigma)

        return 4 * self.ndims


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

    def __init__(self, weight: Union[float, ArrayLike], norm: DataFidelityBase = DataFidelity_l1()):
        super().__init__(weight=weight, norm=norm)

    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
        self.dtype = primal.dtype
        self.op = operators.TransformIdentity(primal.shape)

        self.norm.assign_data(None, sigma=1)

        return 1

    def update_dual(self, dual: ArrayLike, primal: ArrayLike) -> None:
        dual += primal


class Regularizer_swl(BaseRegularizer):
    """Base stationary wavelet regularizer. It can be used to promote sparse reconstructions in the wavelet domain."""

    __reg_name__ = "swl"

    def info(self) -> str:
        """
        Return the regularizer info.

        Returns
        -------
        str
            Regularizer info string.
        """
        return self.__reg_name__ + "(t:" + self.wavelet + "-l:%d" % self.level + "-w:%g" % self.weight.max() + ")"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Optional[Sequence[int]] = None,
        pad_on_demand: str = "constant",
        normalized: bool = False,
        min_approx: bool = True,
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
            axes = np.arange(-ndims, 0, dtype=int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

        self.pad_on_demand = pad_on_demand

    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
        self.dtype = primal.dtype
        self.op = operators.TransformStationaryWavelet(
            primal.shape,
            wavelet=self.wavelet,
            level=self.level,
            axes=self.axes,
            pad_on_demand=self.pad_on_demand,
            normalized=self.normalized,
        )

        filt_bank_l1norm = np.linalg.norm(self.op.w.filter_bank, ord=1, axis=-1)
        lo_dec_mult = filt_bank_l1norm[0] ** self.ndims
        lo_rec_mult = filt_bank_l1norm[2] ** self.ndims

        self.dec_func_mult = (
            self.op.wlet_dec_filter_mult[None, :] * (lo_dec_mult ** np.arange(self.level - 1, -1, -1))[:, None]
        )
        self.dec_func_mult = np.concatenate(([lo_dec_mult ** self.level], self.dec_func_mult.flatten()))

        self.rec_func_mult = (
            self.op.wlet_rec_filter_mult[None, :] * (lo_rec_mult ** np.arange(self.level - 1, -1, -1))[:, None]
        )
        self.rec_func_mult = np.concatenate(([lo_rec_mult ** self.level], self.rec_func_mult.flatten()))

        # self.dec_func_mult = 2 ** np.arange(self.level, 0, -1)
        # self.dec_func_mult = np.tile(self.dec_func_mult[:, None], [1, (2 ** self.ndims) - 1])
        # self.dec_func_mult = np.concatenate(([self.dec_func_mult[0, 0]], self.dec_func_mult.flatten()))

        if self.normalized:
            self.sigma = 1
            self.norm.assign_data(None, sigma=self.sigma)

            tau = self.dec_func_mult.size
        else:
            self.sigma = np.reshape(1 / self.dec_func_mult, [-1] + [1] * len(self.op.dir_shape))
            self.norm.assign_data(None, sigma=self.sigma)

            tau = np.ones_like(self.rec_func_mult) * ((2 ** self.ndims) - 1)
            tau[0] += 1
            tau = np.sum(tau / self.rec_func_mult)

        if not isinstance(self.norm, DataFidelity_l1):
            tau *= self.weight
        return tau

    def update_dual(self, dual: ArrayLike, primal: ArrayLike) -> None:
        upd = self.op(primal)
        if not self.normalized:
            upd *= self.sigma
        dual += upd
        if not self.min_approx:
            dual[0, ...] = 0


class Regularizer_l1swl(Regularizer_swl):
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions in the wavelet domain."""

    __reg_name__ = "l1swl"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Optional[Sequence[int]] = None,
        pad_on_demand: str = "constant",
        normalized: bool = False,
        min_approx: bool = True,
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
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions in the wavelet domain."""

    __reg_name__ = "Hubswl"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Optional[Sequence[int]] = None,
        pad_on_demand: str = "constant",
        normalized: bool = False,
        min_approx: bool = True,
        huber_size: Optional[int] = None,
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

    def info(self) -> str:
        """
        Return the regularizer info.

        Returns
        -------
        str
            Regularizer info string.
        """
        return self.__reg_name__ + "(t:" + self.wavelet + "-l:%d" % self.level + "-w:%g" % self.weight.max() + ")"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Optional[Sequence[int]] = None,
        pad_on_demand: str = "constant",
        min_approx: bool = True,
        norm: DataFidelityBase = DataFidelity_l1(),
    ):
        if not has_pywt:
            raise ValueError("Cannot use wavelet regularizer because pywavelets is not installed.")
        super().__init__(weight=weight, norm=norm)
        self.wavelet = wavelet
        self.level = level
        self.min_approx = min_approx

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

        self.pad_on_demand = pad_on_demand

    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
        self.dtype = primal.dtype
        self.op = operators.TransformDecimatedWavelet(
            primal.shape, wavelet=self.wavelet, level=self.level, axes=self.axes, pad_on_demand=self.pad_on_demand
        )

        filt_bank_l1norm = np.linalg.norm(self.op.w.filter_bank, ord=1, axis=-1)
        lo_dec_mult = filt_bank_l1norm[0] ** self.ndims
        lo_rec_mult = filt_bank_l1norm[2] ** self.ndims

        self.dec_func_mult = (
            self.op.wlet_dec_filter_mult[None, :] * (lo_dec_mult ** np.arange(self.level - 1, -1, -1))[:, None]
        )
        self.dec_func_mult = np.concatenate(([lo_dec_mult ** self.level], self.dec_func_mult.flatten()))

        self.rec_func_mult = (
            self.op.wlet_rec_filter_mult[None, :] * (lo_rec_mult ** np.arange(self.level - 1, -1, -1))[:, None]
        )
        self.rec_func_mult = np.concatenate(([lo_rec_mult ** self.level], self.rec_func_mult.flatten()))

        # self.dec_func_mult = 2 ** np.arange(self.level, 0, -1)
        # self.rec_func_mult = self.dec_func_mult

        self.sigma = [np.ones(self.op.sub_band_shapes[0], self.dtype) * self.dec_func_mult[0]]
        count = 0
        for ii_l in range(self.level):
            d = {}
            for label in self.op.sub_band_shapes[ii_l + 1].keys():
                # d[label] = np.ones(self.op.sub_band_shapes[ii_l + 1][label], self.dtype) * self.dec_func_mult[ii_l]
                d[label] = np.ones(self.op.sub_band_shapes[ii_l + 1][label], self.dtype) * self.dec_func_mult[count]
                count += 1
            self.sigma.append(d)
        self.sigma, _ = pywt.coeffs_to_array(self.sigma, axes=self.axes)
        self.norm.assign_data(None, sigma=self.sigma)

        tau = np.ones_like(self.rec_func_mult) * ((2 ** self.ndims) - 1)
        tau[0] += 1
        tau = np.sum(tau / self.rec_func_mult)

        if not isinstance(self.norm, DataFidelity_l1):
            tau *= self.weight
        return tau

    def update_dual(self, dual: ArrayLike, primal: ArrayLike) -> None:
        super().update_dual(dual, primal)
        if not self.min_approx:
            slices = [slice(0, x) for x in self.op.sub_band_shapes[0]]
            dual[tuple(slices)] = 0


class Regularizer_l1dwl(Regularizer_dwl):
    """l1-norm decimated wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "l1dwl"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Optional[Sequence[int]] = None,
        pad_on_demand: str = "constant",
    ):
        super().__init__(weight, wavelet, level, ndims=ndims, axes=axes, pad_on_demand=pad_on_demand, norm=DataFidelity_l1())


class Regularizer_Hub_dwl(Regularizer_dwl):
    """l1-norm decimated wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "Hubdwl"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Optional[Sequence[int]] = None,
        pad_on_demand: str = "constant",
        huber_size: Optional[int] = None,
    ):
        super().__init__(
            weight, wavelet, level, ndims=ndims, axes=axes, pad_on_demand=pad_on_demand, norm=DataFidelity_Huber(huber_size)
        )


class BaseRegularizer_med(BaseRegularizer):
    """Median filter regularizer base class. It can be used to promote filtered reconstructions."""

    __reg_name__ = "med"

    def info(self) -> str:
        """
        Return the regularizer info.

        Returns
        -------
        str
            Regularizer info string.
        """
        return self.__reg_name__ + "(s:%s" % np.array(self.filt_size) + "-w:%g" % self.weight.max() + ")"

    def __init__(self, weight: Union[float, ArrayLike], filt_size: int = 3, norm: DataFidelityBase = DataFidelity_l1()):
        super().__init__(weight=weight, norm=norm)
        self.filt_size = filt_size

    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
        self.dtype = primal.dtype
        self.op = operators.TransformIdentity(primal.shape)
        self.norm.assign_data(None, sigma=1)

        if not isinstance(self.norm, DataFidelity_l1):
            return self.weight
        else:
            return 1

    def update_dual(self, dual: ArrayLike, primal: ArrayLike) -> None:
        dual += primal - spimg.median_filter(primal, self.filt_size)


class Regularizer_l1med(BaseRegularizer_med):
    """l1-norm median filter regularizer. It can be used to promote filtered reconstructions."""

    __reg_name__ = "l1med"

    def __init__(self, weight: Union[float, ArrayLike], filt_size: int = 3):
        BaseRegularizer_med.__init__(self, weight, filt_size=filt_size, norm=DataFidelity_l1())


class Regularizer_l2med(BaseRegularizer_med):
    """l2-norm median filter regularizer. It can be used to promote filtered reconstructions."""

    __reg_name__ = "l2med"

    def __init__(self, weight: Union[float, ArrayLike], filt_size: int = 3):
        BaseRegularizer_med.__init__(self, weight, filt_size=filt_size, norm=DataFidelity_l2())


class Regularizer_fft(BaseRegularizer):
    """Fourier regularizer. It can be used to promote sparse reconstructions in the Fourier domain."""

    __reg_name__ = "fft"

    def __init__(
        self,
        weight: Union[float, ArrayLike],
        ndims: int = 2,
        axes: Optional[Sequence[int]] = None,
        mask: str = "exp",
        norm: DataFidelityBase = DataFidelity_l12(),
    ):
        super().__init__(weight=weight, norm=norm)

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

        self.mask = mask

    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
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

        if not isinstance(self.norm, DataFidelity_l1):
            return self.weight
        else:
            return 1


# ---- Constraints ----


class Constraint_LowerLimit(BaseRegularizer):
    """Lower limit constraint. It can be used to promote reconstructions in certain regions of solution space."""

    __reg_name__ = "lowlim"

    def info(self) -> str:
        """
        Return the regularizer info.

        Returns
        -------
        str
            Regularizer info string.
        """
        return self.__reg_name__ + "(l:%g" % self.limit + ")"

    def __init__(self, limit, norm: DataFidelityBase = DataFidelity_l2()):
        super().__init__(weight=1, norm=norm)
        self.limit = limit

    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
        self.dtype = primal.dtype
        self.op = operators.TransformIdentity(primal.shape)

        self.norm.assign_data(self.limit, sigma=1)

        if not isinstance(self.norm, DataFidelity_l1):
            return self.weight
        else:
            return 1

    def update_dual(self, dual: ArrayLike, primal: ArrayLike) -> None:
        dual += primal

    def apply_proximal(self, dual: ArrayLike) -> None:
        dual[dual > self.limit] = self.limit
        self.norm.apply_proximal(dual)


class Constraint_UpperLimit(BaseRegularizer):
    """Upper limit constraint. It can be used to promote reconstructions in certain regions of solution space."""

    __reg_name__ = "uplim"

    def info(self) -> str:
        """
        Return the regularizer info.

        Returns
        -------
        str
            Regularizer info string.
        """
        return self.__reg_name__ + "(l:%g" % self.limit + ")"

    def __init__(self, limit, norm: DataFidelityBase = DataFidelity_l2()):
        super().__init__(weight=1, norm=norm)
        self.limit = limit

    def initialize_sigma_tau(self, primal: ArrayLike) -> Union[float, ArrayLike]:
        self.dtype = primal.dtype
        self.op = operators.TransformIdentity(primal.shape)

        self.norm.assign_data(self.limit, sigma=1)

        if not isinstance(self.norm, DataFidelity_l1):
            return self.weight
        else:
            return 1

    def update_dual(self, dual: ArrayLike, primal: ArrayLike) -> None:
        dual += primal

    def apply_proximal(self, dual: ArrayLike) -> None:
        dual[dual < self.limit] = self.limit
        self.norm.apply_proximal(dual)
