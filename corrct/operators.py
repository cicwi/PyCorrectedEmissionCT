#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operators module.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.signal as spsig

import copy as cp

from numpy.typing import ArrayLike, NDArray

from typing import Callable, Optional, Sequence, Tuple, Union

from abc import abstractmethod

try:
    import pywt

    has_pywt = True
    use_swtn = pywt.version.version >= "1.0.2"  # type: ignore
    if not use_swtn:
        print("WARNING - pywavelets version is too old (<1.0.2)")
except ImportError:
    has_pywt = False
    use_swtn = False
    print("WARNING - pywt was not found")


NDArrayInt = NDArray[np.integer]


class BaseTransform(LinearOperator):
    """Base operator class.

    It implements the linear operator behavior that can be used with the solvers in the `.solvers` module,
    and by the solvers in `scipy.sparse.linalg`.
    """

    dir_shape: NDArrayInt
    adj_shape: NDArrayInt

    def __init__(self):
        """Initialize the base operator class.

        It assumes that the fields `dir_shape` and `adj_shape` have been set during the initialization of the derived classes.
        """
        num_cols = np.prod(self.dir_shape)
        num_rows = np.prod(self.adj_shape)
        super().__init__(np.float32, [num_rows, num_cols])
        self.is_dir_operator = True

    def _matvec(self, x: NDArray) -> NDArray:
        """Implement the direct operator for column vectors from the right.

        :param x: Either row from the left or column from the right.
        :type x: numpy.array_like
        """
        if self.is_dir_operator:
            x = x.reshape(self.dir_shape)
            return self._op_direct(x).flatten()
        else:
            x = x.reshape(self.adj_shape)
            return self._op_adjoint(x).flatten()

    def rmatvec(self, x: NDArray) -> NDArray:
        """Implement the direct operator for row vectors from the left.

        :param x: Either row from the left or column from the right on transpose.
        :type x: numpy.array_like
        """
        if self.is_dir_operator:
            x = x.reshape(self.adj_shape)
            return self._op_adjoint(x).flatten()
        else:
            x = x.reshape(self.dir_shape)
            return self._op_direct(x).flatten()

    def _transpose(self):
        """Create the transpose operator.

        :returns: The transpose operator
        :rtype: BaseTransform
        """
        Op_t = cp.copy(self)
        Op_t.shape = [Op_t.shape[1], Op_t.shape[0]]
        Op_t.is_dir_operator = False
        return Op_t

    def _adjoint(self):
        return self._transpose()

    def absolute(self):
        """Return the projection operator using the absolute value of the projection coefficients.

        :returns: The absolute value operator
        :rtype: ProjectorOperator
        """
        return self

    def explicit(self) -> NDArray:
        """Return the explicit transformation matrix associated to the operator.

        :returns: The explicit transformation matrix
        :rtype: NDArray
        """
        He = np.empty(self.shape, dtype=self.dtype)
        if self.is_dir_operator:
            dim_size = np.prod(self.dir_shape)
        else:
            dim_size = np.prod(self.adj_shape)
        for ii in range(dim_size):
            xii = np.zeros((dim_size,))
            xii[ii] = 1
            He[:, ii] = self * xii
        return He

    def __call__(self, x: NDArray) -> NDArray:
        """Apply the operator to the input vector.

        :param x: Input vector.
        :type x: NDArray

        :returns: The result of the application of the operator on the input vector.
        :rtype: NDArray
        """
        if self.is_dir_operator:
            return self._op_direct(x)
        else:
            return self._op_adjoint(x)

    @abstractmethod
    def _op_direct(self, x: NDArray) -> NDArray:
        """Apply the operator to the data.

        Parameters
        ----------
        x : NDArray
            Data to process.

        Returns
        -------
        NDArray
            The processed data.
        """

    @abstractmethod
    def _op_adjoint(self, x: NDArray) -> NDArray:
        """Apply the adjoint operator to the data.

        Parameters
        ----------
        x : NDArray
            Data to process.

        Returns
        -------
        NDArray
            The processed data.
        """


class TransformFunctions(BaseTransform):
    """Transform class that uses callables."""

    def __init__(
        self,
        dir_shape: ArrayLike,
        adj_shape: ArrayLike,
        A: Callable[[NDArray], NDArray],
        At: Optional[Callable[[NDArray], NDArray]] = None,
    ) -> None:
        """Initialize the callable transform.

        If the adjoint of the function is not given, the function is considered symmetric.

        Parameters
        ----------
        dir_shape : ArrayLike
            Shape of the direct space.
        adj_shape : ArrayLike
            Shape of the adjoint space.
        A : Callable[[NDArray], NDArray]
            The transform function.
        At : Optional[Callable[[NDArray], NDArray]], optional
            The adjoint transform function, by default None
        """
        self.dir_shape = np.array(dir_shape, ndmin=1, dtype=int)
        self.adj_shape = np.array(adj_shape, ndmin=1, dtype=int)
        self.A = A
        self.At = At
        super().__init__()

    def _op_direct(self, x: NDArray) -> NDArray:
        """Apply the operator to the data.

        Parameters
        ----------
        x : NDArray
            Data to process.

        Returns
        -------
        NDArray
            The processed data.
        """
        return self.A(x)

    def _op_adjoint(self, x: NDArray) -> NDArray:
        """Apply the adjoint operator to the data.

        Parameters
        ----------
        x : NDArray
            Data to process.

        Returns
        -------
        NDArray
            The processed data.
        """
        if self.At is not None:
            return self.At(x)
        else:
            return self.A(x)

    def absolute(self):
        """Compute the absolute value of the operator. Raise an error, because not supported.

        Raises
        ------
        AttributeError
            Not supported operation.
        """
        raise AttributeError("Callable transform class does not support computing its absolute value.")


class ProjectorOperator(BaseTransform):
    """Base projector class that fixes the projection interface."""

    @property
    def vol_shape(self) -> NDArrayInt:
        """Expose the direct space shape as volume shape.

        Returns
        -------
        NDArray
            The volume shape.
        """
        return self.dir_shape

    @property
    def prj_shape(self) -> NDArrayInt:
        """Expose the adjoint space shape as projection shape.

        Returns
        -------
        NDArray
            The projection shape.
        """
        return self.adj_shape

    @vol_shape.setter
    def vol_shape(self, new_shape: Union[Sequence[int], NDArray]) -> None:
        self.dir_shape = np.array(new_shape, ndmin=1, dtype=int)

    @prj_shape.setter
    def prj_shape(self, new_shape: Union[Sequence[int], NDArray]) -> None:
        self.adj_shape = np.array(new_shape, ndmin=1, dtype=int)

    @abstractmethod
    def fp(self, x: NDArray) -> NDArray:
        """Define the interface for the forward-projection.

        :param x: Input volume.
        :type x: NDArray

        :returns: The projection data.
        :rtype: NDArray
        """

    @abstractmethod
    def bp(self, x: NDArray) -> NDArray:
        """Define the interface for the back-projection.

        :param x: Input projection data.
        :type x: NDArray

        :returns: The back-projected volume.
        :rtype: NDArray
        """

    def _op_direct(self, x: NDArray) -> NDArray:
        return self.fp(x)

    def _op_adjoint(self, x: NDArray) -> NDArray:
        return self.bp(x)


class TransformIdentity(BaseTransform):
    """Identity operator."""

    def __init__(self, x_shape: ArrayLike):
        """Identity operator.

        :param x_shape: Shape of the data.
        :type x_shape: ArrayLike
        """
        self.dir_shape = np.array(x_shape, ndmin=1, dtype=int)
        self.adj_shape = np.array(x_shape, ndmin=1, dtype=int)
        super().__init__()

    def _op_direct(self, x: NDArray) -> NDArray:
        return x

    def _op_adjoint(self, x: NDArray) -> NDArray:
        return x


class TransformDiagonalScaling(BaseTransform):
    """Diagonal scaling operator."""

    scale: NDArray

    def __init__(self, x_shape: ArrayLike, scale: Union[float, ArrayLike]):
        """Diagonal scaling operator.

        :param x_shape: Shape of the data.
        :type x_shape: ArrayLike
        :param scale: Operator diagonal.
        :type scale: float or ArrayLike
        """
        self.scale = np.array(scale)
        self.dir_shape = np.array(x_shape, ndmin=1, dtype=int)
        self.adj_shape = np.array(x_shape, ndmin=1, dtype=int)
        super().__init__()

    def absolute(self):
        """Return the projection operator using the absolute value of the projection coefficients.

        :returns: The absolute value operator
        :rtype: Diagonal operator of the absolute values
        """
        return TransformDiagonalScaling(self.dir_shape, np.abs(self.scale))

    def _op_direct(self, x: NDArray) -> NDArray:
        return self.scale * x

    def _op_adjoint(self, x: NDArray) -> NDArray:
        return self.scale * x


class TransformConvolution(BaseTransform):
    """
    Convolution operator.

    Parameters
    ----------
    x_shape : ArrayLike
        Shape of the direct space.
    kernel : ArrayLike
        The convolution kernel.
    pad_mode: str, optional
        The padding mode to use for the linear convolution. The default is "edge".
    is_symm : bool, optional
        Whether the operator is symmetric or not. The default is True.
    flip_adjoint : bool, optional
        Whether the adjoint kernel should be flipped. The default is False.
        This is useful when the kernel is not symmetric.
    """

    kernel: NDArray
    pad_mode: str
    is_symm: bool
    flip_adjoint: bool

    def __init__(
        self, x_shape: ArrayLike, kernel: ArrayLike, pad_mode: str = "edge", is_symm: bool = True, flip_adjoint: bool = False
    ):
        self.dir_shape = np.array(x_shape, ndmin=1, dtype=int)
        self.adj_shape = np.array(x_shape, ndmin=1, dtype=int)

        self.kernel = np.array(kernel, ndmin=len(self.dir_shape))

        self.pad_mode = pad_mode.lower()
        self.is_symm = is_symm
        self.flip_adjoint = flip_adjoint

        super().__init__()

    def absolute(self) -> "TransformConvolution":
        """
        Return the convolution operator using the absolute value of the kernel coefficients.

        Returns
        -------
        TransformConvolution
            The absolute value of the convolution operator.
        """
        return TransformConvolution(self.dir_shape, np.abs(self.kernel))

    def _pad_valid(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        pad_width = (np.array(self.kernel.shape) - 1) // 2
        return np.pad(x, pad_width=pad_width[:, None], mode=self.pad_mode), pad_width  # type: ignore

    def _crop_valid(self, x: NDArray, pad_width: NDArray) -> NDArray:
        slices = [slice(pw if pw else None, -pw if pw else None) for pw in pad_width]
        return x[tuple(slices)]

    def _op_direct(self, x: NDArray) -> NDArray:
        x, pw = self._pad_valid(x)
        x = spsig.convolve(x, self.kernel, mode="same")
        return self._crop_valid(x, pw)

    def _op_adjoint(self, x: NDArray) -> NDArray:
        if self.is_symm:
            x, pw = self._pad_valid(x)
            if self.flip_adjoint:
                adj_kernel = np.flip(self.kernel)
            else:
                adj_kernel = self.kernel
            x = spsig.convolve(x, adj_kernel, mode="same")
            return self._crop_valid(x, pw)
        else:
            return x


class BaseWaveletTransform(BaseTransform):
    """Base Wavelet transform."""

    axes: NDArrayInt
    wavelet: str

    def _initialize_filter_bank(self) -> None:
        num_axes = len(self.axes)
        self.labels = [bin(x)[2:].zfill(num_axes).replace("0", "a").replace("1", "d") for x in range(1, 2**num_axes)]

        self.w = pywt.Wavelet(self.wavelet)  # type: ignore
        filt_bank_l1norm = np.linalg.norm(self.w.filter_bank, ord=1, axis=-1)
        self.wlet_dec_filter_mult = np.array(
            [(filt_bank_l1norm[0] ** lab.count("a")) * (filt_bank_l1norm[1] ** lab.count("d")) for lab in self.labels]
        )
        self.wlet_rec_filter_mult = np.array(
            [(filt_bank_l1norm[2] ** lab.count("a")) * (filt_bank_l1norm[3] ** lab.count("d")) for lab in self.labels]
        )


class TransformDecimatedWavelet(BaseWaveletTransform):
    """Decimated wavelet Transform operator."""

    def __init__(
        self, x_shape: ArrayLike, wavelet: str, level: int, axes: Optional[ArrayLike] = None, pad_on_demand: str = "edge"
    ):
        """Decimated wavelet Transform operator.

        :param x_shape: Shape of the data to be wavelet transformed.
        :type x_shape: ArrayLike
        :param wavelet: Wavelet type
        :type wavelet: string
        :param level: Numer of wavelet decomposition levels
        :type level: int
        :param axes: Axes along which to do the transform, defaults to None
        :type axes: int or tuple of int, optional
        :param pad_on_demand: Padding type to fit the `2 ** level` shape requirements, defaults to 'edge'
        :type pad_on_demand: string, optional. Options are all the `numpy.pad` padding modes.

        :raises ValueError: In case the pywavelets package is not available or its version is not adequate.
        """
        x_shape = np.array(x_shape, ndmin=1, dtype=int)

        if not has_pywt:
            raise ValueError("Cannot use Wavelet transform because pywavelets is not installed.")

        self.wavelet = wavelet
        self.level = level

        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1, dtype=int)

        self.pad_on_demand = pad_on_demand

        self._initialize_filter_bank()

        num_axes = len(self.axes)

        self.dir_shape = x_shape

        self.sub_band_shapes = pywt.wavedecn_shapes(
            self.dir_shape, self.wavelet, mode=self.pad_on_demand, level=self.level, axes=self.axes
        )
        self.adj_shape = self.dir_shape.copy()
        for ax in self.axes:
            self.adj_shape[ax] = self.sub_band_shapes[0][ax] + np.sum(
                [self.sub_band_shapes[x]["d" * num_axes][ax] for x in range(1, self.level + 1)]
            )
        self.slicing_info = None

        super().__init__()

    def direct_dwt(self, x: NDArray) -> list:
        """Perform the direct wavelet transform.

        :param x: Data to transform.
        :type x: NDArray

        :return: Transformed data.
        :rtype: list
        """
        return pywt.wavedecn(x, wavelet=self.wavelet, axes=self.axes, mode=self.pad_on_demand, level=self.level)

    def inverse_dwt(self, y: list) -> NDArray:
        """Perform the inverse wavelet transform.

        :param x: Data to anti-transform.
        :type x: list

        :return: Anti-transformed data.
        :rtype: NDArray
        """
        rec = pywt.waverecn(y, wavelet=self.wavelet, axes=self.axes, mode=self.pad_on_demand)
        if not np.all(rec.shape == self.dir_shape):
            slices = [slice(0, s) for s in self.dir_shape]
            rec = rec[tuple(slices)]
        return rec

    def _op_direct(self, x: NDArray) -> NDArray:
        c = self.direct_dwt(x)
        y, self.slicing_info = pywt.coeffs_to_array(c, axes=self.axes)
        return y

    def _op_adjoint(self, y: NDArray) -> NDArray:
        if self.slicing_info is None:
            _ = self._op_direct(np.zeros(self.dir_shape))

        c = pywt.array_to_coeffs(y, self.slicing_info)
        return self.inverse_dwt(c)


class TransformStationaryWavelet(BaseWaveletTransform):
    """Stationary avelet Transform operator."""

    def __init__(
        self,
        x_shape: ArrayLike,
        wavelet: str,
        level: int,
        axes: Optional[ArrayLike] = None,
        pad_on_demand: str = "edge",
        normalized: bool = True,
    ):
        """Stationary wavelet Transform operator.

        :param x_shape: Shape of the data to be wavelet transformed.
        :type x_shape: ArrayLike
        :param wavelet: Wavelet type
        :type wavelet: string
        :param level: Numer of wavelet decomposition levels
        :type level: int
        :param axes: Axes along which to do the transform, defaults to None
        :type axes: int or tuple of int, optional
        :param pad_on_demand: Padding type to fit the `2 ** level` shape requirements, defaults to 'constant'
        :type pad_on_demand: string, optional. Options are all the `numpy.pad` padding modes.
        :param normalized: Whether to use a normalized transform. Defaults to True.
        :type normalized: boolean, optional.

        :raises ValueError: In case the pywavelets package is not available or its version is not adequate.
        """
        x_shape = np.array(x_shape, ndmin=1, dtype=int)

        if not has_pywt:
            raise ValueError("Cannot use Wavelet transform because pywavelets is not installed.")
        if not use_swtn:
            raise ValueError("Cannot use Wavelet transform because pywavelets is too old (<1.0.2).")

        self.wavelet = wavelet
        self.level = level
        self.normalized = normalized

        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1, dtype=int)

        self.pad_on_demand = pad_on_demand

        self._initialize_filter_bank()

        self.dir_shape = x_shape
        if self.pad_on_demand is not None:
            alignment = 2**self.level
            x_axes = np.array(self.dir_shape)[np.array(self.axes)]
            self.pad_axes = (alignment - x_axes % alignment) % alignment

            adj_x_shape = cp.deepcopy(self.dir_shape)
            adj_x_shape[np.array(self.axes)] += self.pad_axes
        else:
            adj_x_shape = self.dir_shape
        self.adj_shape = np.array((self.level * (2 ** len(self.axes) - 1) + 1, *adj_x_shape))

        super().__init__()

    def direct_swt(self, x: NDArray) -> list:
        """Perform the direct wavelet transform.

        :param x: Data to transform.
        :type x: NDArray

        :return: Transformed data.
        :rtype: list
        """
        if self.pad_on_demand is not None and np.any(self.pad_axes):
            for ax in np.nonzero(self.pad_axes)[0]:
                pad_l = np.ceil(self.pad_axes[ax] / 2).astype(int)
                pad_h = np.floor(self.pad_axes[ax] / 2).astype(int)
                pad_width = [(0, 0)] * len(x.shape)
                pad_width[self.axes[ax]] = (pad_l, pad_h)
                x = np.pad(x, pad_width, mode=self.pad_on_demand)
        return pywt.swtn(x, wavelet=self.wavelet, axes=self.axes, norm=self.normalized, level=self.level, trim_approx=True)

    def inverse_swt(self, y: list) -> NDArray:
        """Perform the inverse wavelet transform.

        :param x: Data to anti-transform.
        :type x: list

        :return: Anti-transformed data.
        :rtype: NDArray
        """
        x = pywt.iswtn(y, wavelet=self.wavelet, axes=self.axes, norm=self.normalized)
        if self.pad_on_demand is not None and np.any(self.pad_axes):
            for ax in np.nonzero(self.pad_axes)[0]:
                pad_l = np.ceil(self.pad_axes[ax] / 2).astype(int)
                pad_h = np.floor(self.pad_axes[ax] / 2).astype(int)
                slices = [slice(None)] * len(x.shape)
                slices[self.axes[ax]] = slice(pad_l, x.shape[self.axes[ax]] - pad_h, 1)
                x = x[tuple(slices)]
        return x

    def _op_direct(self, x: NDArray) -> NDArray:
        y = self.direct_swt(x)
        y = [y[0]] + [y[lvl][x] for lvl in range(1, self.level + 1) for x in self.labels]
        return np.array(y)

    def _op_adjoint(self, y: NDArray) -> NDArray:
        def get_lvl_pos(lvl):
            return (lvl - 1) * (2 ** len(self.axes) - 1) + 1

        x = [y[0]] + [
            dict(((k, y[ii_lbl + get_lvl_pos(lvl), ...]) for ii_lbl, k in enumerate(self.labels)))
            for lvl in range(1, self.level + 1)
        ]
        return self.inverse_swt(x)


class TransformGradient(BaseTransform):
    """
    Gradient operator.

    Parameters
    ----------
    x_shape : ArrayLike
        Shape of the data to be transformed.
    axes : Optional[ArrayLike], optional
        Axes along which to do the gradient. The default is None.
    pad_mode : str, optional
        Padding mode of the gradient. The default is "edge".
    """

    def __init__(self, x_shape: ArrayLike, axes: Optional[ArrayLike] = None, pad_mode: str = "edge"):
        x_shape = np.array(x_shape, ndmin=1, dtype=int)

        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1, dtype=int)
        self.ndims = len(x_shape)

        self.pad_mode = pad_mode.lower()

        self.dir_shape = x_shape
        self.adj_shape = np.array((len(self.axes), *self.dir_shape), ndmin=1, dtype=int)

        super().__init__()

    def gradient(self, x: NDArray) -> NDArray:
        """Compute the gradient.

        :param x: Input data.
        :type x: NDArray

        :return: Gradient of data.
        :rtype: NDArray
        """
        d = [np.array([])] * len(self.axes)
        for ii, ax in enumerate(self.axes):
            padding = np.zeros((self.ndims, 2), dtype=int)
            padding[ax, 1] = 1
            temp_x = np.pad(x, padding, mode=self.pad_mode)  # type: ignore
            d[ii] = np.diff(temp_x, n=1, axis=ax)
        return np.stack(d, axis=0)

    def divergence(self, x: NDArray) -> NDArray:
        """Compute the divergence - transpose of gradient.

        :param x: Input data.
        :type x: NDArray

        :return: Divergence of data.
        :rtype: NDArray
        """
        d = [np.array([])] * len(self.axes)
        for ii, ax in enumerate(self.axes):
            padding = np.zeros((self.ndims, 2), dtype=int)
            padding[ax, 0] = 1
            temp_x = np.pad(x[ii, ...], padding, mode=self.pad_mode)  # type: ignore
            d[ii] = np.diff(temp_x, n=1, axis=ax)
        return np.sum(np.stack(d, axis=0), axis=0)

    def _op_direct(self, x: NDArray) -> NDArray:
        return self.gradient(x)

    def _op_adjoint(self, y: NDArray) -> NDArray:
        return -self.divergence(y)


class TransformFourier(BaseTransform):
    """Fourier transform operator."""

    def __init__(self, x_shape: ArrayLike, axes: Optional[ArrayLike] = None):
        """Fourier transform.

        :param x_shape: Shape of the data to be wavelet transformed.
        :type x_shape: ArrayLike
        :param axes: Axes along which to do the gradient, defaults to None
        :type axes: int or tuple of int, optional
        """
        x_shape = np.array(x_shape, ndmin=1, dtype=int)

        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1, dtype=int)
        self.ndims = len(x_shape)

        self.dir_shape = x_shape
        self.adj_shape = np.array((2, *self.dir_shape), ndmin=1, dtype=int)

        super().__init__()

    def fft(self, x: NDArray) -> NDArray:
        """Compute the fft.

        :param x: Input data.
        :type x: NDArray

        :return: FFT of data.
        :rtype: NDArray
        """
        d = np.empty(self.adj_shape, dtype=x.dtype)
        x_f = np.fft.fftn(x, axes=tuple(self.axes), norm="ortho")
        d[0, ...] = x_f.real
        d[1, ...] = x_f.imag
        return d

    def ifft(self, x: NDArray) -> NDArray:
        """Compute the inverse of the fft.

        :param x: Input data.
        :type x: NDArray

        :return: iFFT of data.
        :rtype: NDArray
        """
        d = x[0, ...] + 1j * x[1, ...]
        return np.fft.ifftn(d, axes=tuple(self.axes), norm="ortho").real

    def _op_direct(self, x: NDArray) -> NDArray:
        return self.fft(x)

    def _op_adjoint(self, y: NDArray) -> NDArray:
        return self.ifft(y)


class TransformLaplacian(BaseTransform):
    """
    Laplacian transform operator.

    Parameters
    ----------
    x_shape : ArrayLike
        Shape of the data to be transformed.
    axes : ArrayLike, optional
        Axes along which to do the Laplacian. The default is None.
    pad_mode : str, optional
        Padding mode of the Laplacian. The default is "edge".
    """

    def __init__(self, x_shape: ArrayLike, axes: Optional[ArrayLike] = None, pad_mode: str = "edge"):
        x_shape = np.array(x_shape, ndmin=1, dtype=int)

        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1, dtype=int)
        self.ndims = len(x_shape)

        self.pad_mode = pad_mode.lower()

        self.dir_shape = x_shape
        self.adj_shape = x_shape

        super().__init__()

    def laplacian(self, x: NDArray) -> NDArray:
        """Compute the laplacian.

        :param x: Input data.
        :type x: NDArray

        :return: Gradient of data.
        :rtype: NDArray
        """
        d = [np.array([])] * len(self.axes)
        for ii, ax in enumerate(self.axes):
            padding = np.zeros((self.ndims, 2), dtype=int)
            padding[ax, :] = 1
            temp_x = np.pad(x, padding, mode=self.pad_mode)  # type: ignore
            d[ii] = np.diff(temp_x, n=2, axis=ax)
        return np.sum(d, axis=0)

    def _op_direct(self, x: NDArray) -> NDArray:
        return self.laplacian(x)

    def _op_adjoint(self, y: NDArray) -> NDArray:
        return self.laplacian(y)


class TransformSVD(BaseTransform):
    """Singular value decomposition based decomposition operator."""

    U: Optional[NDArray]
    Vt: Optional[NDArray]

    def __init__(self, x_shape, axes_rows=(0,), axes_cols=(-1,), rescale: bool = False):
        """Singular value decomposition operator.

        The SVD decomposition will be done over the flattened rows vs flatted cols.
        This means that the channels should always be the rows (expected to be only
        one dimension, usually), while the volume dimensions should always be the
        columns (expected to be the last two or three dimensions).

        :param x_shape: Shape of the data to be wavelet transformed.
        :type x_shape: `numpy.array_like`
        :param axes_rows: Axes expanded in rows of the SVD, defaults to (0, )
        :type axes_rows: tuple of int, optional
        :param axes_cols: Axes expanded in cols of the SVD, defaults to (-1, )
        :type axes_cols: tuple of int, optional

        :raises IndexError: In case the the axes are outside the range.
        """
        self.dir_shape = np.array(x_shape, ndmin=1, dtype=int)

        self.axes_rows = np.atleast_1d(axes_rows) % len(self.dir_shape)
        self.axes_cols = np.atleast_1d(axes_cols) % len(self.dir_shape)

        # Dimensions to decompose
        self.append_dims = np.concatenate((self.axes_rows, self.axes_cols)).astype(int)
        # Dimensions to NOT decompose
        temp_dims = np.arange(len(self.dir_shape), dtype=int)
        self.invariant_dims = np.delete(temp_dims, self.append_dims)
        self.invariant_dims_shape = self.dir_shape[self.invariant_dims]

        # Transpose operation to prepare data for decomposition
        self.fwd_transpose = np.concatenate((self.invariant_dims, self.append_dims))
        # Transpose operation to recover data after re-composition
        self.bwd_transpose = np.argsort(self.fwd_transpose)

        self.axes_rows_shape = self.dir_shape[self.axes_rows]
        self.axes_cols_shape = self.dir_shape[self.axes_cols]
        self.axes_rows_size = (np.prod(self.axes_rows_shape),)
        self.axes_cols_size = (np.prod(self.axes_cols_shape),)

        # Reshape operation to prepare data for decomposition
        self.fwd_shape = np.concatenate((self.invariant_dims_shape, self.axes_rows_size, self.axes_cols_size))
        # Reshape operation to recover data after re-composition
        self.bwd_shape = np.concatenate((self.invariant_dims_shape, self.axes_rows_shape, self.axes_cols_shape))

        self.adj_shape = np.concatenate((self.invariant_dims_shape, np.fmin(self.axes_rows_size, self.axes_cols_size)))

        self.U = None
        self.Vt = None

        self.rescale = rescale

        super().__init__()

    def direct_svd(self, x):
        """Performs the SVD decomposition.

        :param x: Data to transform.
        :type x: `numpy.array_like`

        :return: Transformed data.
        :rtype: tuple(U, s, Vt)
        """
        return np.linalg.svd(x, full_matrices=False)

    def inverse_svd(self, U, s, Vt):
        """Performs the inverse SVD decomposition.

        :param U: Rows of the SVD dcomposition.
        :type U: `numpy.array_like`
        :param s: Singular values.
        :type s: `numpy.array_like`
        :param Vt: Columns of the SVD dcomposition.
        :type Vt: `numpy.array_like`

        :return: Anti-transformed data.
        :rtype: `numpy.array_like`
        """
        return np.matmul(U, s[..., None] * Vt)

    def _op_direct(self, x):
        x = np.transpose(x, self.fwd_transpose)
        x = np.reshape(x, self.fwd_shape)

        (self.U, s, self.Vt) = self.direct_svd(x)

        if self.rescale:
            s /= np.sqrt(self.Vt.shape[-1] * self.U.shape[-2])
            self.Vt *= np.sqrt(self.Vt.shape[-1])
            self.U *= np.sqrt(self.U.shape[-2])

        return s

    def _op_adjoint(self, y):
        if self.U is None or self.Vt is None:
            raise ValueError("Operator not initialized!")

        x = self.inverse_svd(self.U, y, self.Vt)

        x = np.reshape(x, self.bwd_shape)
        return np.transpose(x, self.bwd_transpose)


if __name__ == "__main__":
    test_vol = np.zeros((10, 10), dtype=np.float32)

    H = TransformStationaryWavelet(test_vol.shape, "db1", 2)
    Htw = H.T.explicit()
    Hw = H.explicit()

    D = TransformGradient(test_vol.shape)
    Dg = D.explicit()
