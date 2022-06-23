#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operators module.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import scipy.sparse.linalg as spsla
import scipy.signal as spsig

import copy as cp

from numpy.typing import ArrayLike

from typing import Optional, Union

try:
    import pywt

    has_pywt = True
    use_swtn = pywt.version.version >= "1.0.2"
    if not use_swtn:
        print("WARNING - pywavelets version is too old (<1.0.2)")
except ImportError:
    has_pywt = False
    use_swtn = False
    print("WARNING - pywt was not found")


class BaseTransform(spsla.LinearOperator):
    """Base operator class.

    It implements the linear operator behavior that can be used with the solvers in the `.solvers` module,
    and by the solvers in `scipy.sparse.linalg`.
    """

    def __init__(self):
        """Initialize the base operator class.

        It assumes that the fields `dir_shape` and `adj_shape` have been set during the initialization of the derived classes.
        """
        num_cols = np.prod(self.dir_shape)
        num_rows = np.prod(self.adj_shape)
        super().__init__(np.float32, [num_rows, num_cols])
        self.is_dir_operator = True

    def _matvec(self, x: ArrayLike) -> ArrayLike:
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

    def rmatvec(self, x: ArrayLike) -> ArrayLike:
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

    def explicit(self):
        """Return the explicit transformation matrix associated to the operator.

        :returns: The explicit transformation matrix
        :rtype: ArrayLike
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

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply the operator to the input vector.

        :param x: Input vector.
        :type x: ArrayLike

        :returns: The result of the application of the operator on the input vector.
        :rtype: ArrayLike
        """
        if self.is_dir_operator:
            return self._op_direct(x)
        else:
            return self._op_adjoint(x)

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError()

    def _op_adjoint(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError()


class ProjectorOperator(BaseTransform):
    """Base projector class that fixes the projection interface."""

    def __init__(self):
        """Initialize the projector operator class.

        It sets the fields `dir_shape` and `adj_shape`, from the fields `vol_shape` and `prj_shape` respectively.
        These two other fields need to have been defined in a derived class.
        """
        self.dir_shape = self.vol_shape
        self.adj_shape = self.prj_shape
        super().__init__()

    def fp(self, x: ArrayLike) -> ArrayLike:
        """Define the interface for the forward-projection.

        :param x: Input volume.
        :type x: ArrayLike

        :returns: The projection data.
        :rtype: ArrayLike
        """
        raise NotImplementedError()

    def bp(self, x: ArrayLike) -> ArrayLike:
        """Define the interface for the back-projection.

        :param x: Input projection data.
        :type x: ArrayLike

        :returns: The back-projected volume.
        :rtype: ArrayLike
        """
        raise NotImplementedError()

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        return self.fp(x)

    def _op_adjoint(self, x: ArrayLike) -> ArrayLike:
        return self.bp(x)


class TransformIdentity(BaseTransform):
    """Identity operator."""

    def __init__(self, x_shape: ArrayLike):
        """Identity operator.

        :param x_shape: Shape of the data.
        :type x_shape: ArrayLike
        """
        self.dir_shape = np.array(x_shape)
        self.adj_shape = np.array(x_shape)
        super().__init__()

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        return x

    def _op_adjoint(self, x: ArrayLike) -> ArrayLike:
        return x


class TransformDiagonalScaling(BaseTransform):
    """Diagonal scaling operator."""

    def __init__(self, x_shape: ArrayLike, scale: Union[float, ArrayLike]):
        """Diagonal scaling operator.

        :param x_shape: Shape of the data.
        :type x_shape: ArrayLike
        :param scale: Operator diagonal.
        :type scale: float or ArrayLike
        """
        self.scale = scale
        self.dir_shape = np.array(x_shape)
        self.adj_shape = np.array(x_shape)
        super().__init__()

    def absolute(self):
        """Return the projection operator using the absolute value of the projection coefficients.

        :returns: The absolute value operator
        :rtype: Diagonal operator of the absolute values
        """
        return TransformDiagonalScaling(self.dir_shape, np.abs(self.scale))

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        return self.scale * x

    def _op_adjoint(self, x: ArrayLike) -> ArrayLike:
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

    def __init__(
        self, x_shape: ArrayLike, kernel: ArrayLike, pad_mode: str = "edge", is_symm: bool = True, flip_adjoint: bool = False
    ):
        self.dir_shape = np.array(x_shape)
        self.adj_shape = np.array(x_shape)

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

    def _pad_valid(self, x: ArrayLike) -> ArrayLike:
        pad_width = (np.array(self.kernel.shape) - 1) // 2
        return np.pad(x, pad_width=pad_width[:, None], mode=self.pad_mode), pad_width

    def _crop_valid(self, x: ArrayLike, pad_width: ArrayLike) -> ArrayLike:
        slices = [slice(pw if pw else None, -pw if pw else None) for pw in pad_width]
        return x[tuple(slices)]

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        x, pw = self._pad_valid(x)
        x = spsig.convolve(x, self.kernel, mode="same")
        return self._crop_valid(x, pw)

    def _op_adjoint(self, x: ArrayLike) -> ArrayLike:
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

    def _initialize_filter_bank(self) -> None:
        num_axes = len(self.axes)
        self.labels = [bin(x)[2:].zfill(num_axes).replace("0", "a").replace("1", "d") for x in range(1, 2**num_axes)]

        self.w = pywt.Wavelet(self.wavelet)
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
        if not has_pywt:
            raise ValueError("Cannot use Wavelet transform because pywavelets is not installed.")

        self.wavelet = wavelet
        self.level = level

        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1)

        self.pad_on_demand = pad_on_demand

        self._initialize_filter_bank()

        num_axes = len(self.axes)

        self.dir_shape = np.array(x_shape)

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

    def direct_dwt(self, x: ArrayLike) -> ArrayLike:
        """Perform the direct wavelet transform.

        :param x: Data to transform.
        :type x: ArrayLike

        :return: Transformed data.
        :rtype: list
        """
        return pywt.wavedecn(x, wavelet=self.wavelet, axes=self.axes, mode=self.pad_on_demand, level=self.level)

    def inverse_dwt(self, y: ArrayLike) -> ArrayLike:
        """Perform the inverse wavelet transform.

        :param x: Data to anti-transform.
        :type x: list

        :return: Anti-transformed data.
        :rtype: ArrayLike
        """
        rec = pywt.waverecn(y, wavelet=self.wavelet, axes=self.axes, mode=self.pad_on_demand)
        if not np.all(rec.shape == self.dir_shape):
            slices = [slice(0, s) for s in self.dir_shape]
            rec = rec[tuple(slices)]
        return rec

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        c = self.direct_dwt(x)
        y, self.slicing_info = pywt.coeffs_to_array(c, axes=self.axes)
        return y

    def _op_adjoint(self, y: ArrayLike) -> ArrayLike:
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
        if not has_pywt:
            raise ValueError("Cannot use Wavelet transform because pywavelets is not installed.")
        if not use_swtn:
            raise ValueError("Cannot use Wavelet transform because pywavelets is too old (<1.0.2).")

        self.wavelet = wavelet
        self.level = level
        self.normalized = normalized

        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1)

        self.pad_on_demand = pad_on_demand

        self._initialize_filter_bank()

        self.dir_shape = np.array(x_shape)
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

    def direct_swt(self, x: ArrayLike) -> ArrayLike:
        """Perform the direct wavelet transform.

        :param x: Data to transform.
        :type x: ArrayLike

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

    def inverse_swt(self, y: ArrayLike) -> ArrayLike:
        """Perform the inverse wavelet transform.

        :param x: Data to anti-transform.
        :type x: list

        :return: Anti-transformed data.
        :rtype: ArrayLike
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

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        y = self.direct_swt(x)
        y = [y[0]] + [y[lvl][x] for lvl in range(1, self.level + 1) for x in self.labels]
        return np.array(y)

    def _op_adjoint(self, y: ArrayLike) -> ArrayLike:
        def get_lvl_pos(lvl):
            return (lvl - 1) * (2 ** len(self.axes) - 1) + 1

        y = [y[0]] + [
            dict(((k, y[ii_lbl + get_lvl_pos(lvl), ...]) for ii_lbl, k in enumerate(self.labels)))
            for lvl in range(1, self.level + 1)
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
        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1)
        self.ndims = len(x_shape)

        self.pad_mode = pad_mode

        self.dir_shape = np.array(x_shape)
        self.adj_shape = np.array((len(self.axes), *self.dir_shape))

        super().__init__()

    def gradient(self, x: ArrayLike) -> ArrayLike:
        """Compute the gradient.

        :param x: Input data.
        :type x: ArrayLike

        :return: Gradient of data.
        :rtype: ArrayLike
        """
        d = [None] * len(self.axes)
        for ii in range(len(self.axes)):
            ind = -(ii + 1)
            padding = [(0, 0)] * self.ndims
            padding[ind] = (0, 1)
            temp_x = np.pad(x, padding, mode=self.pad_mode)
            d[ind] = np.diff(temp_x, n=1, axis=ind)
        return np.stack(d, axis=0)

    def divergence(self, x: ArrayLike) -> ArrayLike:
        """Compute the divergence - transpose of gradient.

        :param x: Input data.
        :type x: ArrayLike

        :return: Divergence of data.
        :rtype: ArrayLike
        """
        d = [None] * len(self.axes)
        for ii in range(len(self.axes)):
            ind = -(ii + 1)
            padding = [(0, 0)] * self.ndims
            padding[ind] = (1, 0)
            temp_x = np.pad(x[ind, ...], padding, mode=self.pad_mode)
            d[ind] = np.diff(temp_x, n=1, axis=ind)
        return np.sum(np.stack(d, axis=0), axis=0)

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        return self.gradient(x)

    def _op_adjoint(self, y: ArrayLike) -> ArrayLike:
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
        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1)
        self.ndims = len(x_shape)

        self.dir_shape = np.array(x_shape)
        self.adj_shape = np.array((2, *self.dir_shape))

        super().__init__()

    def fft(self, x: ArrayLike) -> ArrayLike:
        """Compute the fft.

        :param x: Input data.
        :type x: ArrayLike

        :return: FFT of data.
        :rtype: ArrayLike
        """
        d = np.empty(self.adj_shape, dtype=x.dtype)
        x_f = np.fft.fftn(x, axes=self.axes, norm="ortho")
        d[0, ...] = x_f.real
        d[1, ...] = x_f.imag
        return d

    def ifft(self, x: ArrayLike) -> ArrayLike:
        """Compute the inverse of the fft.

        :param x: Input data.
        :type x: ArrayLike

        :return: iFFT of data.
        :rtype: ArrayLike
        """
        d = x[0, ...] + 1j * x[1, ...]
        return np.fft.ifftn(d, axes=self.axes, norm="ortho").real

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        return self.fft(x)

    def _op_adjoint(self, y: ArrayLike) -> ArrayLike:
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
        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=int)
        self.axes = np.array(axes, ndmin=1)
        self.ndims = len(x_shape)

        self.pad_mode = pad_mode

        self.dir_shape = np.array(x_shape)
        self.adj_shape = np.array(x_shape)

        super().__init__()

    def laplacian(self, x: ArrayLike) -> ArrayLike:
        """Compute the laplacian.

        :param x: Input data.
        :type x: ArrayLike

        :return: Gradient of data.
        :rtype: ArrayLike
        """
        d = [None] * len(self.axes)
        for ii in range(len(self.axes)):
            ind = -(ii + 1)
            padding = [(0, 0)] * self.ndims
            padding[ind] = (1, 1)
            temp_x = np.pad(x, padding, mode=self.pad_mode)
            d[ind] = np.diff(temp_x, n=2, axis=ind)
        return np.sum(d, axis=0)

    def _op_direct(self, x: ArrayLike) -> ArrayLike:
        return self.laplacian(x)

    def _op_adjoint(self, y: ArrayLike) -> ArrayLike:
        return self.laplacian(y)


if __name__ == "__main__":
    test_vol = np.zeros((10, 10), dtype=np.float32)

    H = TransformStationaryWavelet(test_vol.shape, "db1", 2)
    Htw = H.T.explicit()
    Hw = H.explicit()

    D = TransformGradient(test_vol.shape)
    Dg = D.explicit()
