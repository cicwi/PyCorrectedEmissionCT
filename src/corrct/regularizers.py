#!/usr/bin/env python3
"""
Regularizers module.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
from collections.abc import Sequence

import numpy as np
import scipy.ndimage as spimg
from numpy.typing import DTypeLike, NDArray

from . import data_terms as dt
from . import operators

try:
    import pywt

    has_pywt = True
    use_swtn = pywt.version.version >= "1.0.2"  # type: ignore
except ImportError:
    has_pywt = False
    use_swtn = False
    print("WARNING - pywt was not found")


NDArrayInt = NDArray[np.integer]


# ---- Regularizers ----


class BaseRegularizer(ABC):
    """
    Initialize a base regularizer class, that defines the Regularizer object interface.

    Parameters
    ----------
    weight : Union[float, NDArray]
        The weight of the regularizer.
    norm : DataFidelityBase
        The norm of the regularizer minimization.
    """

    __reg_name__ = ""

    weight: NDArray
    dtype: DTypeLike
    op: Union[operators.BaseTransform, None]

    sigma: Union[float, NDArray]

    upd_mask: Optional[NDArray]

    def __init__(
        self,
        weight: Union[float, NDArray],
        norm: dt.DataFidelityBase,
        upd_mask: Optional[NDArray] = None,
        dtype: DTypeLike = np.float32,
    ):
        self.weight = np.array(weight)
        self.dtype = dtype
        self.op = None
        self.norm = norm
        self.upd_mask = upd_mask

    def info(self) -> str:
        """
        Return the regularizer info.

        Returns
        -------
        str
            Regularizer info string.
        """
        return self.__reg_name__ + f"(w:{self.weight.max():.3e})"

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
    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        """
        Initialize the internal state, operator, and sigma. It then returns the tau.

        Parameters
        ----------
        primal : NDArray
            The primal vector.

        Returns
        -------
        Union[float, NDArray]
            The tau to be used in the SIRT or PDHG algorithm.
        """

    def initialize_dual(self) -> NDArray:
        """
        Return the initialized dual.

        Returns
        -------
        NDArray
            Initialized (zero) dual.
        """
        if self.op is None:
            raise ValueError("Regularizer not initialized! Please use method: `initialize_sigma_tau`.")

        return np.zeros(self.op.adj_shape, dtype=self.dtype)

    def update_dual(self, dual: NDArray, primal: NDArray) -> None:
        """
        Update the dual in-place.

        Parameters
        ----------
        dual : NDArray
            Current stat of the dual.
        primal : NDArray
            Primal or over-relaxation of the primal.
        """
        if self.op is None:
            raise ValueError("Regularizer not initialized! Please use method: `initialize_sigma_tau`.")

        dual += self.sigma * self.op(primal)

    def apply_proximal(self, dual: NDArray) -> None:
        """
        Apply the proximal operator to the dual in-place.

        Parameters
        ----------
        dual : NDArray
            The dual to be applied the proximal on.
        """
        if isinstance(self.norm, dt.DataFidelity_l1):
            self.norm.apply_proximal(dual, self.weight)
        else:
            self.norm.apply_proximal(dual)

    def compute_update_primal(self, dual: NDArray) -> NDArray:
        """
        Compute the partial update of a primal term, from this regularizer.

        Parameters
        ----------
        dual : NDArray
            The dual associated to this regularizer.

        Returns
        -------
        upd : NDArray
            The update to the primal.
        """
        if self.op is None:
            raise ValueError("Regularizer not initialized! Please use method: `initialize_sigma_tau`.")

        upd = self.op.T(dual)
        if self.upd_mask is not None:
            upd *= self.upd_mask
        if not isinstance(self.norm, dt.DataFidelity_l1):
            upd *= self.weight
        return upd

    def _check_primal(self, primal: NDArray) -> None:
        if self.dtype != primal.dtype:
            print(f"WARNING: Regularizer dtype ({self.dtype}) and primal dtype ({primal.dtype}) are different!")
            self.dtype = primal.dtype


class Regularizer_Grad(BaseRegularizer):
    """Gradient regularizer.

    When used with l1-norms, it promotes piece-wise constant reconstructions.
    When used with l2-norm, it promotes smooth reconstructions.

    Parameters
    ----------
    weight : Union[float, NDArray]
        The weight of the regularizer.
    ndims : int, optional
        The number of dimensions. The default is 2.
    axes : Sequence, optional
        The axes over which it computes the gradient. If None, it uses the last 2. The default is None.
    pad_mode: str, optional
        The padding mode to use for the linear convolution. The default is "edge".
    norm : DataFidelityBase, optional
        The norm of the regularizer minimization. The default is DataFidelity_l12().
    """

    __reg_name__ = "grad"

    def __init__(
        self,
        weight: Union[float, NDArray],
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l12(),
    ):
        super().__init__(weight=weight, norm=norm, upd_mask=upd_mask)

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes
        self.pad_mode = pad_mode.lower()

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

        self.op = operators.TransformGradient(primal.shape, axes=self.axes, pad_mode=self.pad_mode)

        self.sigma = 0.5
        self.norm.assign_data(None, sigma=self.sigma)

        tau = 2 * self.ndims
        if not isinstance(self.norm, dt.DataFidelity_l1):
            tau *= self.weight
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau


class Regularizer_TV1D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 1D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "TV1D"

    def __init__(
        self,
        weight: Union[float, NDArray],
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l12(),
    ):
        super().__init__(weight=weight, ndims=1, axes=axes, pad_mode=pad_mode, norm=norm, upd_mask=upd_mask)


class Regularizer_TV2D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 2D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "TV2D"

    def __init__(
        self,
        weight: Union[float, NDArray],
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l12(),
    ):
        super().__init__(weight=weight, ndims=2, axes=axes, pad_mode=pad_mode, norm=norm, upd_mask=upd_mask)


class Regularizer_TV3D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 3D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "TV3D"

    def __init__(
        self,
        weight: Union[float, NDArray],
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l12(),
    ):
        super().__init__(weight=weight, ndims=3, axes=axes, pad_mode=pad_mode, norm=norm, upd_mask=upd_mask)


class Regularizer_HubTV2D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 2D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "HubTV2D"

    def __init__(
        self,
        weight: Union[float, NDArray],
        huber_size: float,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
    ):
        super().__init__(
            weight=weight,
            ndims=2,
            axes=axes,
            pad_mode=pad_mode,
            upd_mask=upd_mask,
            norm=dt.DataFidelity_Huber(huber_size, l2_axis=0),
        )


class Regularizer_HubTV3D(Regularizer_Grad):
    """Total Variation (TV) regularizer in 3D. It can be used to promote piece-wise constant reconstructions."""

    __reg_name__ = "HubTV3D"

    def __init__(
        self,
        weight: Union[float, NDArray],
        huber_size: float,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
    ):
        super().__init__(
            weight=weight,
            ndims=3,
            axes=axes,
            pad_mode=pad_mode,
            upd_mask=upd_mask,
            norm=dt.DataFidelity_Huber(huber_size, l2_axis=0),
        )


class Regularizer_smooth1D(Regularizer_Grad):
    """It can be used to promote smooth reconstructions."""

    __reg_name__ = "smooth1D"

    def __init__(
        self,
        weight: Union[float, NDArray],
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l2(),
    ):
        super().__init__(weight=weight, ndims=1, axes=axes, pad_mode=pad_mode, norm=norm, upd_mask=upd_mask)


class Regularizer_smooth2D(Regularizer_Grad):
    """It can be used to promote smooth reconstructions."""

    __reg_name__ = "smooth2D"

    def __init__(
        self,
        weight: Union[float, NDArray],
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l2(),
    ):
        super().__init__(weight=weight, ndims=2, axes=axes, pad_mode=pad_mode, norm=norm, upd_mask=upd_mask)


class Regularizer_smooth3D(Regularizer_Grad):
    """It can be used to promote smooth reconstructions."""

    __reg_name__ = "smooth3D"

    def __init__(
        self,
        weight: Union[float, NDArray],
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l2(),
    ):
        super().__init__(weight=weight, ndims=3, axes=axes, pad_mode=pad_mode, norm=norm, upd_mask=upd_mask)


class Regularizer_lap(BaseRegularizer):
    """Laplacian regularizer. It can be used to promote smooth reconstructions."""

    __reg_name__ = "lap"

    def __init__(
        self,
        weight: Union[float, NDArray],
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
    ):
        super().__init__(weight=weight, norm=dt.DataFidelity_l1(), upd_mask=upd_mask)

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes
        self.pad_mode = pad_mode.lower()

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

        self.op = operators.TransformLaplacian(primal.shape, axes=self.axes, pad_mode=self.pad_mode)

        self.sigma = 0.25
        self.norm.assign_data(None, sigma=self.sigma)

        tau = 4 * self.ndims
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau


class Regularizer_lap1D(Regularizer_lap):
    """Laplacian regularizer in 1D. It can be used to promote smooth reconstructions."""

    __reg_name__ = "lap1D"

    def __init__(
        self,
        weight,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
    ):
        Regularizer_lap.__init__(self, weight=weight, ndims=1, axes=axes, pad_mode=pad_mode, upd_mask=upd_mask)


class Regularizer_lap2D(Regularizer_lap):
    """Laplacian regularizer in 2D. It can be used to promote smooth reconstructions."""

    __reg_name__ = "lap2D"

    def __init__(
        self,
        weight,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
    ):
        Regularizer_lap.__init__(self, weight=weight, ndims=2, axes=axes, pad_mode=pad_mode, upd_mask=upd_mask)


class Regularizer_lap3D(Regularizer_lap):
    """Laplacian regularizer in 3D. It can be used to promote smooth reconstructions."""

    __reg_name__ = "lap3D"

    def __init__(
        self,
        weight,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
    ):
        Regularizer_lap.__init__(self, weight=weight, ndims=3, axes=axes, pad_mode=pad_mode, upd_mask=upd_mask)


class Regularizer_l1(BaseRegularizer):
    """l1-norm regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "l1"

    def __init__(
        self,
        weight: Union[float, NDArray],
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l1(),
    ):
        super().__init__(weight=weight, norm=norm, upd_mask=upd_mask)

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

        self.op = operators.TransformIdentity(primal.shape)

        self.norm.assign_data(None, sigma=1)

        tau = 1.0
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau

    def update_dual(self, dual: NDArray, primal: NDArray) -> None:
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
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
        normalized: bool = False,
        min_approx: bool = True,
        norm: dt.DataFidelityBase = dt.DataFidelity_l1(),
    ):
        if not has_pywt:
            raise ValueError("Cannot use wavelet regularizer because pywavelets is not installed.")
        if not use_swtn:
            raise ValueError("Cannot use stationary wavelet regularizer because pywavelets is too old (<1.0.2).")
        super().__init__(weight=weight, norm=norm, upd_mask=upd_mask)
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

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

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
        self.dec_func_mult = np.concatenate(([lo_dec_mult**self.level], self.dec_func_mult.flatten()))

        self.rec_func_mult = (
            self.op.wlet_rec_filter_mult[None, :] * (lo_rec_mult ** np.arange(self.level - 1, -1, -1))[:, None]
        )
        self.rec_func_mult = np.concatenate(([lo_rec_mult**self.level], self.rec_func_mult.flatten()))

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

            tau = np.ones_like(self.rec_func_mult) * ((2**self.ndims) - 1)
            tau[0] += 1
            tau = np.sum(tau / self.rec_func_mult)

        if not isinstance(self.norm, dt.DataFidelity_l1):
            tau *= self.weight
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau

    def update_dual(self, dual: NDArray, primal: NDArray) -> None:
        if self.op is None:
            raise ValueError("Regularizer not initialized! Please use method: `initialize_sigma_tau`.")

        upd = self.op(primal)
        if not self.normalized:
            upd *= self.sigma
        dual += upd
        if not self.min_approx:
            dual[0, ...] = 0

    def apply_proximal(self, dual: NDArray) -> None:
        if isinstance(self.norm, dt.DataFidelity_l12):
            tmp_dual = dual[1:]
            tmp_dual = tmp_dual.reshape([-1, self.level, *dual.shape[1:]])
            self.norm.apply_proximal(tmp_dual, self.weight)
            tmp_dual = dual[0:1:]
            self.norm.apply_proximal(tmp_dual, self.weight)
        else:
            super().apply_proximal(dual)


class Regularizer_l1swl(Regularizer_swl):
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions in the wavelet domain."""

    __reg_name__ = "l1swl"

    def __init__(
        self,
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
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
            upd_mask=upd_mask,
            normalized=normalized,
            min_approx=min_approx,
            norm=dt.DataFidelity_l1(),
        )


class Regularizer_l12swl(Regularizer_swl):
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions in the wavelet domain."""

    __reg_name__ = "l12swl"

    def __init__(
        self,
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
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
            upd_mask=upd_mask,
            normalized=normalized,
            min_approx=min_approx,
            norm=dt.DataFidelity_l12(),
        )


class Regularizer_Hub_swl(Regularizer_swl):
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions in the wavelet domain."""

    __reg_name__ = "Hubswl"

    def __init__(
        self,
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
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
            upd_mask=upd_mask,
            normalized=normalized,
            min_approx=min_approx,
            norm=dt.DataFidelity_Huber(huber_size),
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
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
        min_approx: bool = True,
        norm: dt.DataFidelityBase = dt.DataFidelity_l1(),
    ):
        if not has_pywt:
            raise ValueError("Cannot use wavelet regularizer because pywavelets is not installed.")
        super().__init__(weight=weight, norm=norm, upd_mask=upd_mask)
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

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

        self.op = operators.TransformDecimatedWavelet(
            primal.shape, wavelet=self.wavelet, level=self.level, axes=self.axes, pad_on_demand=self.pad_on_demand
        )

        filt_bank_l1norm = np.linalg.norm(self.op.w.filter_bank, ord=1, axis=-1)
        lo_dec_mult = filt_bank_l1norm[0] ** self.ndims
        lo_rec_mult = filt_bank_l1norm[2] ** self.ndims

        self.dec_func_mult = (
            self.op.wlet_dec_filter_mult[None, :] * (lo_dec_mult ** np.arange(self.level - 1, -1, -1))[:, None]
        )
        self.dec_func_mult = np.concatenate(([lo_dec_mult**self.level], self.dec_func_mult.flatten()))

        self.rec_func_mult = (
            self.op.wlet_rec_filter_mult[None, :] * (lo_rec_mult ** np.arange(self.level - 1, -1, -1))[:, None]
        )
        self.rec_func_mult = np.concatenate(([lo_rec_mult**self.level], self.rec_func_mult.flatten()))

        # self.dec_func_mult = 2 ** np.arange(self.level, 0, -1)
        # self.rec_func_mult = self.dec_func_mult

        tmp_sigma = [np.ones(self.op.sub_band_shapes[0], self.dtype) * self.dec_func_mult[0]]
        count = 0
        for ii_l in range(self.level):
            d = {}
            for label in self.op.sub_band_shapes[ii_l + 1].keys():
                # d[label] = np.ones(self.op.sub_band_shapes[ii_l + 1][label], self.dtype) * self.dec_func_mult[ii_l]
                d[label] = np.ones(self.op.sub_band_shapes[ii_l + 1][label], self.dtype) * self.dec_func_mult[count]
                count += 1
            tmp_sigma.append(d)
        self.sigma, _ = pywt.coeffs_to_array(tmp_sigma, axes=self.axes)
        self.norm.assign_data(None, sigma=self.sigma)

        tau = np.ones_like(self.rec_func_mult) * ((2**self.ndims) - 1)
        tau[0] += 1
        tau = np.sum(tau / self.rec_func_mult)

        if not isinstance(self.norm, dt.DataFidelity_l1):
            tau *= self.weight
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau

    def update_dual(self, dual: NDArray, primal: NDArray) -> None:
        super().update_dual(dual, primal)
        if not self.min_approx:
            if self.op is None:
                raise ValueError("Regularizer not initialized! Please use method: `initialize_sigma_tau`.")

            op_wl: operators.TransformDecimatedWavelet = self.op  # type: ignore
            slices = [slice(0, x) for x in op_wl.sub_band_shapes[0]]
            dual[tuple(slices)] = 0

    def apply_proximal(self, dual: NDArray) -> None:
        if isinstance(self.norm, dt.DataFidelity_l12):
            op_wl: operators.TransformDecimatedWavelet = self.op
            coeffs = pywt.array_to_coeffs(dual, op_wl.slicing_info)
            for ii_l in range(1, len(coeffs)):
                c_l = coeffs[ii_l]
                labels = []
                details = []
                for lab, det in c_l.items():
                    labels.append(lab)
                    details.append(det)
                c_ll = np.stack(details, axis=0)
                self.norm.apply_proximal(c_ll, self.weight)
                for ii, lab in enumerate(labels):
                    c_l[lab] = c_ll[ii]
                coeffs[ii_l] = c_l
            self.norm.apply_proximal(coeffs[0], self.weight)
            dual[:] = pywt.coeffs_to_array(coeffs)[0]
        else:
            super().apply_proximal(dual)


class Regularizer_l1dwl(Regularizer_dwl):
    """l1-norm decimated wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "l1dwl"

    def __init__(
        self,
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
    ):
        super().__init__(
            weight,
            wavelet,
            level,
            ndims=ndims,
            axes=axes,
            pad_on_demand=pad_on_demand,
            upd_mask=upd_mask,
            norm=dt.DataFidelity_l1(),
        )


class Regularizer_l12dwl(Regularizer_dwl):
    """l1-norm decimated wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "l12dwl"

    def __init__(
        self,
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
    ):
        super().__init__(
            weight,
            wavelet,
            level,
            ndims=ndims,
            axes=axes,
            pad_on_demand=pad_on_demand,
            upd_mask=upd_mask,
            norm=dt.DataFidelity_l12(),
        )


class Regularizer_Hub_dwl(Regularizer_dwl):
    """l1-norm decimated wavelet regularizer. It can be used to promote sparse reconstructions."""

    __reg_name__ = "Hubdwl"

    def __init__(
        self,
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
        huber_size: Optional[int] = None,
    ):
        super().__init__(
            weight,
            wavelet,
            level,
            ndims=ndims,
            axes=axes,
            pad_on_demand=pad_on_demand,
            upd_mask=upd_mask,
            norm=dt.DataFidelity_Huber(huber_size),
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

    def __init__(
        self,
        weight: Union[float, NDArray],
        filt_size: int = 3,
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l1(),
    ):
        super().__init__(weight=weight, norm=norm, upd_mask=upd_mask)
        self.filt_size = filt_size

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

        self.op = operators.TransformIdentity(primal.shape)
        self.norm.assign_data(None, sigma=1)

        if not isinstance(self.norm, dt.DataFidelity_l1):
            tau = self.weight
        else:
            tau = 1.0
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau

    def update_dual(self, dual: NDArray, primal: NDArray) -> None:
        dual += primal - spimg.median_filter(primal, self.filt_size)


class Regularizer_l1med(BaseRegularizer_med):
    """l1-norm median filter regularizer. It can be used to promote filtered reconstructions."""

    __reg_name__ = "l1med"

    def __init__(self, weight: Union[float, NDArray], filt_size: int = 3):
        BaseRegularizer_med.__init__(self, weight, filt_size=filt_size, norm=dt.DataFidelity_l1())


class Regularizer_l2med(BaseRegularizer_med):
    """l2-norm median filter regularizer. It can be used to promote filtered reconstructions."""

    __reg_name__ = "l2med"

    def __init__(self, weight: Union[float, NDArray], filt_size: int = 3):
        BaseRegularizer_med.__init__(self, weight, filt_size=filt_size, norm=dt.DataFidelity_l2())


class Regularizer_fft(BaseRegularizer):
    """Fourier regularizer. It can be used to promote sparse reconstructions in the Fourier domain."""

    __reg_name__ = "fft"

    def __init__(
        self,
        weight: Union[float, NDArray],
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        fft_filter: str = "exp",
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l12(),
    ):
        super().__init__(weight=weight, norm=norm, upd_mask=upd_mask)

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = axes

        self.fft_filter = fft_filter

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

        self.op = operators.TransformFourier(primal.shape, axes=self.axes)

        if isinstance(self.fft_filter, str):
            coords = [np.fft.fftfreq(s) for s in self.op.adj_shape[self.axes]]
            coords = np.array(np.meshgrid(*coords, indexing="ij"))

            if self.fft_filter.lower() == "delta":
                self.sigma = 1 - np.all(coords == 0, axis=0)
            elif self.fft_filter.lower() == "exp":
                self.sigma = 1 - np.exp(-np.sqrt(np.sum(coords**2, axis=0)) * 12)
            elif self.fft_filter.lower() == "exp2":
                self.sigma = 1 - np.exp(-np.sum(coords**2, axis=0) * 36)
            else:
                raise ValueError('Unknown FFT mask: %s. Options are: "delta", "exp". and "exp2".' % self.fft_filter)

            new_shape = np.ones_like(self.op.adj_shape)
            new_shape[self.axes] = self.op.adj_shape[self.axes]
            self.sigma = np.reshape(self.sigma, new_shape)
        else:
            self.sigma = 1

        self.norm.assign_data(None, sigma=self.sigma)

        if not isinstance(self.norm, dt.DataFidelity_l1):
            tau = self.weight
        else:
            tau = 1.0
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau


# Multi-channel regularizers


class Regularizer_TNV(Regularizer_Grad):
    """Total Nuclear Variation (TNV) regularizer.

    It can be used to promote piece-wise constant reconstructions, for multi-channel volumes.
    """

    __reg_name__ = "TNV"

    def __init__(
        self,
        weight: Union[float, NDArray],
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_mode: str = "edge",
        upd_mask: Optional[NDArray] = None,
        spectral_norm: dt.DataFidelityBase = dt.DataFidelity_l1(),
        x_ref: Optional[NDArray] = None,
    ):
        super().__init__(weight=weight, ndims=ndims, axes=axes, pad_mode=pad_mode, upd_mask=upd_mask)

        # Here we assume that the channels will be the rows and the derivatives the columns
        self.norm = dt.DataFidelity_ln(ln_axes=(1, 0), spectral_norm=spectral_norm)
        self.x_ref = x_ref

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        tau = super().initialize_sigma_tau(primal)

        if self.x_ref is not None:
            if self.op is None:
                raise ValueError("Regularizer should have been initialized... this is a bug!")

            self.q_ref = self.op(self.x_ref)
            self.q_ref = np.expand_dims(self.q_ref, axis=1)

        return tau


class Regularizer_VTV(Regularizer_Grad):
    """Vectorial Total Variation (VTV) regularizer.

    It can be used to promote piece-wise constant reconstructions, for multi-channel volumes.
    """

    __reg_name__ = "VTV"

    def __init__(
        self,
        weight: Union[float, NDArray],
        ndims: int = 2,
        pwise_der_norm: Union[int, float] = 2,
        pwise_chan_norm: Union[int, float] = np.inf,
        x_ref: Optional[NDArray] = None,
        upd_mask: Optional[NDArray] = None,
    ):
        super().__init__(weight=weight, ndims=ndims, upd_mask=upd_mask)

        self.pwise_der_norm = pwise_der_norm
        if self.pwise_der_norm not in [1, 2, np.inf]:
            self._raise_pwise_norm_error()

        self.pwise_chan_norm = pwise_chan_norm
        if self.pwise_chan_norm not in [1, 2, np.inf]:
            self._raise_pwise_norm_error()

        if x_ref is not None:
            self.initialize_sigma_tau(x_ref)
            q_ref = self.initialize_dual()
            self.update_dual(q_ref, x_ref)
            self.q_ref = np.expand_dims(q_ref, axis=1)
        else:
            self.q_ref = None

    def _raise_pwise_norm_error(self):
        raise ValueError(
            "The only supported point-wise norm exponents are: 1, 2, and Inf."
            + f" Provided the following instead: derivatives={self.pwise_der_norm}, channel={self.pwise_chan_norm}"
        )

    def apply_proximal(self, dual: NDArray) -> None:
        # Following assignments will detach the local array from the original one
        dual_tmp = dual.copy()

        dual_is_scalar = len(dual_tmp.shape) == (self.ndims + 1)
        if dual_is_scalar:
            dual_tmp = np.expand_dims(dual_tmp, axis=1)

        if self.q_ref is not None:
            dual_tmp = np.concatenate((dual_tmp, self.q_ref), axis=1)

        if self.pwise_der_norm == 1:
            grad_norm = np.abs(dual_tmp)
        elif self.pwise_der_norm == 2:
            grad_norm = np.linalg.norm(dual_tmp, axis=0, ord=2, keepdims=True)
        elif self.pwise_der_norm == np.inf:
            grad_norm = np.linalg.norm(dual_tmp, axis=0, ord=1, keepdims=True)
        else:
            self._raise_pwise_norm_error()

        if self.pwise_chan_norm == 1:
            dual_norm = grad_norm
        elif self.pwise_chan_norm == 2:
            dual_norm = np.linalg.norm(grad_norm, axis=1, ord=2, keepdims=True)
        elif self.pwise_chan_norm == np.inf:
            dual_norm = np.linalg.norm(grad_norm, axis=1, ord=1, keepdims=True)
        else:
            self._raise_pwise_norm_error()

        dual_tmp /= np.fmax(dual_norm, self.weight)
        dual_tmp *= self.weight

        if self.q_ref is not None:
            dual_tmp = dual_tmp[:, : dual_tmp.shape[1] - 1 :, ...]

        if dual_is_scalar:
            dual_tmp = np.squeeze(dual_tmp, axis=1)

        dual[:] = dual_tmp[:]  # Replacing values


class Regularizer_lnswl(Regularizer_l1swl):
    """Nuclear-norm Wavelet regularizer.

    It can be used to promote compressed multi-channel reconstructions.
    """

    __reg_name__ = "lnswl"

    def __init__(
        self,
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
        normalized: bool = False,
        min_approx: bool = True,
        spectral_norm: dt.DataFidelityBase = dt.DataFidelity_l1(),
        x_ref: Optional[NDArray] = None,
    ):
        super().__init__(
            weight,
            wavelet,
            level,
            ndims=ndims,
            axes=axes,
            pad_on_demand=pad_on_demand,
            upd_mask=upd_mask,
            normalized=normalized,
            min_approx=min_approx,
        )

        self.norm = dt.DataFidelity_ln(ln_axes=(1, 0), spectral_norm=spectral_norm)
        self.x_ref = x_ref

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        tau = super().initialize_sigma_tau(primal)

        if self.x_ref is not None:
            if self.op is None:
                raise ValueError("Regularizer should have been initialized... this is a bug!")

            self.q_ref = self.op(self.x_ref)
            self.q_ref = np.expand_dims(self.q_ref, axis=1)

        return tau


class Regularizer_vl1wl(Regularizer_l1swl):
    """l1-norm vectorial Wavelet regularizer. It can be used to promote compressed reconstructions."""

    __reg_name__ = "vl1wl"

    def __init__(
        self,
        weight: Union[float, NDArray],
        wavelet: str,
        level: int,
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        pad_on_demand: str = "constant",
        upd_mask: Optional[NDArray] = None,
        normalized: bool = False,
        min_approx: bool = True,
        pwise_lvl_norm: Union[int, float] = 1,
        pwise_chan_norm: Union[int, float] = np.inf,
        x_ref: Optional[NDArray] = None,
    ):
        super().__init__(
            weight,
            wavelet,
            level,
            ndims=ndims,
            axes=axes,
            pad_on_demand=pad_on_demand,
            upd_mask=upd_mask,
            normalized=normalized,
            min_approx=min_approx,
        )

        self.pwise_lvl_norm = pwise_lvl_norm
        if self.pwise_lvl_norm not in [1, 2, np.inf]:
            self._raise_pwise_norm_error()

        self.pwise_chan_norm = pwise_chan_norm
        if self.pwise_chan_norm not in [1, 2, np.inf]:
            self._raise_pwise_norm_error()

        self.x_ref = x_ref
        self.q_ref = None

    def _raise_pwise_norm_error(self):
        raise ValueError(
            "The only supported point-wise norm exponents are: 1, 2, and Inf."
            + f" Provided the following instead: level={self.pwise_lvl_norm}, channel={self.pwise_chan_norm}"
        )

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        tau = super().initialize_sigma_tau(primal)

        if self.x_ref is not None:
            if self.op is None:
                raise ValueError("Regularizer should have been initialized... this is a bug!")

            self.q_ref = self.op(self.x_ref)

        return tau

    def apply_proximal(self, dual: NDArray) -> None:
        dual_tmp = dual.copy()

        if self.q_ref is not None:
            dual_tmp = np.concatenate((dual_tmp, self.q_ref), axis=1)

        if self.pwise_lvl_norm == 1:
            lvl_norm = np.abs(dual_tmp)
        elif self.pwise_lvl_norm == 2:
            lvl_norm = np.linalg.norm(dual_tmp, axis=0, ord=2, keepdims=True)
        elif self.pwise_lvl_norm == np.inf:
            lvl_norm = np.linalg.norm(dual_tmp, axis=0, ord=1, keepdims=True)
        else:
            self._raise_pwise_norm_error()

        if self.pwise_chan_norm == 1:
            dual_norm = lvl_norm
        elif self.pwise_chan_norm == 2:
            dual_norm = np.linalg.norm(lvl_norm, axis=1, ord=2, keepdims=True)
        elif self.pwise_chan_norm == np.inf:
            dual_norm = np.linalg.norm(lvl_norm, axis=1, ord=1, keepdims=True)
        else:
            self._raise_pwise_norm_error()

        dual_tmp /= np.fmax(dual_norm, self.weight)
        dual_tmp *= self.weight

        if self.q_ref is not None:
            dual_tmp = dual_tmp[:, : dual_tmp.shape[0] - 1 :, ...]

        dual[:] = dual_tmp[:]


class Regularizer_vSVD(BaseRegularizer):
    """Regularizer based on the Singular Value Decomposition.

    It can be used to promote similar reconstructions across different channels.
    """

    __reg_name__ = "vsvd"

    def __init__(
        self,
        weight: Union[float, NDArray],
        ndims: int = 2,
        axes: Union[Sequence[int], NDArray, None] = None,
        axis_channels: Sequence[int] = (0,),
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l1(),
    ):
        super().__init__(weight=weight, norm=norm, upd_mask=upd_mask)

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=int)
        elif not ndims == len(axes):
            print("WARNING - Number of axes different from number of dimensions. Updating dimensions accordingly.")
            ndims = len(axes)
        self.ndims = ndims
        self.axes = np.array(axes)
        self.axis_channels = axis_channels

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

        self.op = operators.TransformSVD(primal.shape, axes_rows=self.axis_channels, axes_cols=self.axes, rescale=True)

        self.sigma = 1
        self.norm.assign_data(None, sigma=self.sigma)

        if not isinstance(self.norm, dt.DataFidelity_l1):
            tau = self.weight
        else:
            tau = 1.0
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau


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

    def __init__(
        self,
        limit: Union[float, NDArray],
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l2(),
    ):
        super().__init__(weight=1, norm=norm, upd_mask=upd_mask)
        self.limit = limit

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

        self.op = operators.TransformIdentity(primal.shape)

        self.norm.assign_data(self.limit, sigma=1)

        if not isinstance(self.norm, dt.DataFidelity_l1):
            tau = self.weight
        else:
            tau = 1.0
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau

    def update_dual(self, dual: NDArray, primal: NDArray) -> None:
        dual += primal

    def apply_proximal(self, dual: NDArray) -> None:
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

    def __init__(
        self,
        limit: Union[float, NDArray],
        upd_mask: Optional[NDArray] = None,
        norm: dt.DataFidelityBase = dt.DataFidelity_l2(),
    ):
        super().__init__(weight=1, norm=norm, upd_mask=upd_mask)
        self.limit = limit

    def initialize_sigma_tau(self, primal: NDArray) -> Union[float, NDArray]:
        self._check_primal(primal)

        self.op = operators.TransformIdentity(primal.shape)

        self.norm.assign_data(self.limit, sigma=1)

        if not isinstance(self.norm, dt.DataFidelity_l1):
            tau = self.weight
        else:
            tau = 1.0
        if self.upd_mask is not None:
            tau = tau * self.upd_mask
        return tau

    def update_dual(self, dual: NDArray, primal: NDArray) -> None:
        dual += primal

    def apply_proximal(self, dual: NDArray) -> None:
        dual[dual < self.limit] = self.limit
        self.norm.apply_proximal(dual)
