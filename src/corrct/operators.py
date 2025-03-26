#!/usr/bin/env python3
"""
Operators module.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import copy as cp
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Callable, Optional, Union
import numpy as np
import scipy.signal as spsig
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator


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


class BaseTransform(LinearOperator, ABC):
    """Base operator class.

    It implements the linear operator behavior that can be used with the solvers in the `.solvers` module,
    and by the solvers in `scipy.sparse.linalg`.

    Parameters
    ----------
    dir_shape : NDArrayInt
        Shape of the direct space.
    adj_shape : NDArrayInt
        Shape of the adjoint space

    Attributes
    ----------
    is_dir_operator : bool
        Flag indicating if the operator is a direct operator.

    Notes
    -----
        It assumes that the fields `dir_shape` and `adj_shape` have been set during the initialization of the derived classes.
    """

    dir_shape: NDArrayInt
    adj_shape: NDArrayInt

    def __init__(self) -> None:
        """Initialize the base operator class."""
        num_cols = np.prod(self.dir_shape)
        num_rows = np.prod(self.adj_shape)
        super().__init__(np.float32, [num_rows, num_cols])
        self.is_dir_operator = True

    def _matvec(self, x: NDArray) -> NDArray:
        """Implement the direct operator for column vectors from the right.

        Parameters
        ----------
        x : NDArray
            Either row from the left or column from the right.

        Returns
        -------
        NDArray
            Result of applying the direct operator.
        """
        if self.is_dir_operator:
            x = x.reshape(self.dir_shape)
            return self._op_direct(x).flatten()
        else:
            x = x.reshape(self.adj_shape)
            return self._op_adjoint(x).flatten()

    def rmatvec(self, x: NDArray) -> NDArray:
        """Implement the direct operator for row vectors from the left.

        Parameters
        ----------
        x : NDArray
            Either row from the left or column from the right on transpose.

        Returns
        -------
        NDArray
            Result of applying the direct operator for row vectors.
        """
        if self.is_dir_operator:
            x = x.reshape(self.adj_shape)
            return self._op_adjoint(x).flatten()
        else:
            x = x.reshape(self.dir_shape)
            return self._op_direct(x).flatten()

    def _transpose(self) -> "BaseTransform":
        """Create the transpose operator.

        Returns
        -------
        BaseTransform
            The transpose operator.
        """
        Op_t = cp.copy(self)
        Op_t.shape = [Op_t.shape[1], Op_t.shape[0]]
        Op_t.is_dir_operator = False
        return Op_t

    def _adjoint(self) -> "BaseTransform":
        """Create the adjoint operator.

        Returns
        -------
        BaseTransform
            The adjoint operator.
        """
        return self._transpose()

    def absolute(self) -> "BaseTransform":
        """Return the absolute value of the operator.

        Returns
        -------
        BaseTransform
            The absolute value operator.
        """
        return self

    def explicit(self) -> NDArray:
        """Return the explicit transformation matrix associated with the operator.

        Returns
        -------
        NDArray
            The explicit transformation matrix, as a NumPy array.
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
        """Apply the operator to the input vector `x`.

        Parameters
        ----------
        x : NDArray
            Input vector.

        Returns
        -------
        result : NDArray
            The result of applying the operator to `x`.
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
        dir_shape: Union[ArrayLike, NDArray],
        adj_shape: Union[ArrayLike, NDArray],
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

    def absolute(self) -> "TransformFunctions":
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
        """
        Define the interface of the forward-projection.

        Parameters
        ----------
        x : NDArray
            Input volume.

        Returns
        -------
        NDArray
            The projection data.
        """

    @abstractmethod
    def bp(self, x: NDArray) -> NDArray:
        """
        Define the interface for the back-projection.

        Parameters
        ----------
        x : NDArray
            Input projection data.

        Returns
        -------
        NDArray
            The back-projected volume.
        """

    def _op_direct(self, x: NDArray) -> NDArray:
        return self.fp(x)

    def _op_adjoint(self, x: NDArray) -> NDArray:
        return self.bp(x)

    def get_pre_weights(self) -> Union[NDArray, None]:
        """Compute the pre-weights of the projector geometry (notably for cone-beam geometries).

        Returns
        -------
        Union[NDArray, None]
            The computed detector weights
        """
        return None


class TransformIdentity(BaseTransform):
    """Identity operator."""

    def __init__(self, x_shape: Union[ArrayLike, NDArray]):
        """Identity operator.

        Parameters
        ----------
        x_shape : ArrayLike | NDArray
            Shape of the data.
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

    def __init__(self, x_shape: Union[ArrayLike, NDArray], scale: Union[ArrayLike, NDArray]):
        """Diagonal scaling operator.

        Parameters
        ----------
        x_shape : ArrayLike
            Shape of the data.
        scale : float or ArrayLike
            Operator diagonal.
        """
        self.scale = np.array(scale)
        self.dir_shape = np.array(x_shape, ndmin=1, dtype=int)
        self.adj_shape = np.array(x_shape, ndmin=1, dtype=int)
        super().__init__()

    def absolute(self) -> "TransformDiagonalScaling":
        """Return the projection operator using the absolute value of the projection coefficients.

        Returns
        -------
        TransformDiagonalScaling
            The absolute value operator
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

    def _pad_valid(self, x: NDArray) -> tuple[NDArray, NDArray]:
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


class BaseWaveletTransform(BaseTransform, ABC):
    """Base Wavelet transform."""

    axes: NDArrayInt
    wavelet: str
    labels: list[str]

    wlet_dec_filter_mult: NDArray
    wlet_rec_filter_mult: NDArray

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
        self,
        x_shape: Union[ArrayLike, NDArray],
        wavelet: str,
        level: int,
        axes: Optional[ArrayLike] = None,
        pad_on_demand: str = "edge",
    ):
        """
        Decimated wavelet Transform operator.

        Parameters
        ----------
        x_shape : ArrayLike
            Shape of the data to be wavelet transformed.
        wavelet : str
            Wavelet type.
        level : int
            Number of wavelet decomposition levels.
        axes : int or tuple of int, optional
            Axes along which to do the transform. Defaults to None.
        pad_on_demand : str, optional
            Padding type to fit the `2 ** level` shape requirements. Defaults to 'edge'.
            Options are all the `numpy.pad` padding modes.

        Raises
        ------
        ValueError
            If the pywavelets package is not available or its version is not adequate.
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
        """
        Perform the direct wavelet transform.

        Parameters
        ----------
        x : NDArray
            Data to transform.

        Returns
        -------
        list
            Transformed data.
        """
        return pywt.wavedecn(x, wavelet=self.wavelet, axes=self.axes, mode=self.pad_on_demand, level=self.level)

    def inverse_dwt(self, y: list) -> NDArray:
        """
        Perform the inverse wavelet transform.

        Parameters
        ----------
        y : list
            Data to anti-transform.

        Returns
        -------
        NDArray
            Anti-transformed data.
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

    def _op_adjoint(self, x: NDArray) -> NDArray:
        if self.slicing_info is None:
            _ = self._op_direct(np.zeros(self.dir_shape))

        c = pywt.array_to_coeffs(x, self.slicing_info)
        return self.inverse_dwt(c)


class TransformStationaryWavelet(BaseWaveletTransform):
    """Stationary wavelet Transform operator."""

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

        Parameters
        ----------
        x_shape : ArrayLike
            The shape of the data to be wavelet transformed.
        wavelet : str
            The type of wavelet to use.
        level : int
            Number of wavelet decomposition levels.
        axes : int or tuple of int, optional
            Axes along which to perform the transform. Default is None.
        pad_on_demand : str, optional
            The padding type to fit the `2 ** level` shape requirements.
            Default is 'constant'. Options are all the `numpy.pad` padding modes.
        normalized : bool, optional
            Whether to use a normalized transform. Default is True.

        Raises
        ------
        ValueError
            If the pywavelets package is not available or its version is not adequate.
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
        """
        Perform the direct wavelet transform.

        Parameters
        ----------
        x : NDArray
            Data to transform.

        Returns
        -------
        list
            Transformed data.
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
        """
        Perform the inverse wavelet transform.

        Parameters
        ----------
        y : list
            Data to anti-transform.

        Returns
        -------
        NDArray
            Anti-transformed data.
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

    def _op_adjoint(self, x: NDArray) -> NDArray:
        def get_lvl_pos(lvl):
            return (lvl - 1) * (2 ** len(self.axes) - 1) + 1

        y = [x[0]] + [
            {k: x[ii_lbl + get_lvl_pos(lvl), ...] for ii_lbl, k in enumerate(self.labels)} for lvl in range(1, self.level + 1)
        ]
        return self.inverse_swt(y)


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
        """
        Compute the gradient.

        Parameters
        ----------
        x : NDArray
            Input data.

        Returns
        -------
        NDArray
            Gradient of data.
        """
        d = [np.array([])] * len(self.axes)
        for ii, ax in enumerate(self.axes):
            padding = np.zeros((self.ndims, 2), dtype=int)
            padding[ax, 1] = 1
            temp_x = np.pad(x, padding, mode=self.pad_mode)  # type: ignore
            d[ii] = np.diff(temp_x, n=1, axis=ax)
        return np.stack(d, axis=0)

    def divergence(self, x: NDArray) -> NDArray:
        """
        Compute the divergence - transpose of gradient.

        Parameters
        ----------
        x : NDArray
            Input data.

        Returns
        -------
        NDArray
            Divergence of data.
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

    def _op_adjoint(self, x: NDArray) -> NDArray:
        return -self.divergence(x)


class TransformFourier(BaseTransform):
    """Fourier transform operator."""

    def __init__(self, x_shape: ArrayLike, axes: Optional[ArrayLike] = None):
        """
        Fourier transform.

        Parameters
        ----------
        x_shape : ArrayLike
            Shape of the data to be Fourier transformed.
        axes : Optional[ArrayLike], optional
            Axes along which to do the Fourier transform.

        Returns
        -------
        None
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
        """
        Compute the fft.

        Parameters
        ----------
        x : NDArray
            Input data.

        Returns
        -------
        NDArray
            FFT of data.
        """
        d = np.empty(self.adj_shape, dtype=x.dtype)
        x_f = np.fft.fftn(x, axes=tuple(self.axes), norm="ortho")
        d[0, ...] = x_f.real
        d[1, ...] = x_f.imag
        return d

    def ifft(self, x: NDArray) -> NDArray:
        """Compute the inverse of the fft.

        Parameters
        ----------
        x : NDArray
            Input data.

        Returns
        -------
        NDArray
            iFFT of data.
        """
        d = x[0, ...] + 1j * x[1, ...]
        return np.fft.ifftn(d, axes=tuple(self.axes), norm="ortho").real

    def _op_direct(self, x: NDArray) -> NDArray:
        return self.fft(x)

    def _op_adjoint(self, x: NDArray) -> NDArray:
        return self.ifft(x)


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

        Parameters
        ----------
        x : NDArray
            Input data.

        Returns
        -------
        NDArray
            Laplacian of the input data.
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

    def _op_adjoint(self, x: NDArray) -> NDArray:
        return self.laplacian(x)


class TransformSVD(BaseTransform):
    """Singular value decomposition based decomposition operator."""

    U: Optional[NDArray]
    Vt: Optional[NDArray]

    rescale: bool

    def __init__(
        self,
        x_shape: ArrayLike,
        axes_rows: Union[Sequence[int], NDArray] = (0,),
        axes_cols: Union[Sequence[int], NDArray] = (-1,),
        rescale: bool = False,
    ):
        """
        Singular value decomposition operator.

        The SVD decomposition will be done over the flattened rows vs flatted cols.
        This means that the channels should always be the rows (expected to be only
        one dimension, usually), while the volume dimensions should always be the
        columns (expected to be the last two or three dimensions).

        Parameters
        ----------
        x_shape : numpy.array_like
            Shape of the data to be wavelet transformed.
        axes_rows : tuple of int, optional
            Axes expanded in rows of the SVD. Defaults to (0, ).
        axes_cols : tuple of int, optional
            Axes expanded in cols of the SVD. Defaults to (-1, ).

        Raises
        ------
        IndexError
            In case the axes are outside the range.
        """
        self.dir_shape = np.array(x_shape, ndmin=1, dtype=int)

        self.axes_rows = np.array(axes_rows, ndmin=1) % len(self.dir_shape)
        self.axes_cols = np.array(axes_cols, ndmin=1) % len(self.dir_shape)

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

    def direct_svd(self, x: NDArray):
        """
        Performs the SVD decomposition.

        Parameters
        ----------
        x : NDArray
            Data to transform.

        Returns
        -------
        tuple(U, s, Vt)
            Transformed data.
        """
        return np.linalg.svd(x, full_matrices=False, compute_uv=True)

    def inverse_svd(self, U: NDArray, s: NDArray, Vt: NDArray) -> NDArray:
        """
        Performs the inverse SVD decomposition.

        Parameters
        ----------
        U : NDArray
            Rows of the SVD decomposition.
        s : NDArray
            Singular values.
        Vt : NDArray
            Columns of the SVD decomposition.

        Returns
        -------
        NDArray
            Anti-transformed data.
        """
        return np.matmul(U, s[..., None] * Vt)

    def _op_direct(self, x: NDArray) -> NDArray:
        x = np.transpose(x, self.fwd_transpose)
        x = np.reshape(x, self.fwd_shape)

        (self.U, s, self.Vt) = self.direct_svd(x)

        if self.rescale:
            s /= np.sqrt(self.Vt.shape[-1] * self.U.shape[-2])
            self.Vt *= np.sqrt(self.Vt.shape[-1])
            self.U *= np.sqrt(self.U.shape[-2])

        return s

    def _op_adjoint(self, x: NDArray) -> NDArray:
        if self.U is None or self.Vt is None:
            raise ValueError("Operator not initialized!")

        y = self.inverse_svd(self.U, x, self.Vt)

        y = np.reshape(y, self.bwd_shape)
        return np.transpose(y, self.bwd_transpose)


if __name__ == "__main__":
    test_vol = np.zeros((10, 10), dtype=np.float32)

    H = TransformStationaryWavelet(test_vol.shape, "db1", 2)
    Htw = H.T.explicit()
    Hw = H.explicit()

    D = TransformGradient(test_vol.shape)
    Dg = D.explicit()
