# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:36:39 2020

@author: VIGANO
"""

import numpy as np
import scipy.sparse.linalg as spsla

import copy as cp

try:
    import pywt
    has_pywt = True
    use_swtn = pywt.version.version >= '1.0.2'
    if not use_swtn:
        print('WARNING - pywavelets version is too old (<1.0.2)')
except ImportError:
    has_pywt = False
    use_swtn = False
    print('WARNING - pywt was not found')


class BaseTransform(spsla.LinearOperator):
    """Base operator class, that implements the linear operator behavior that
    can be used with the solvers in the `.solvers` module and the solvers in
    `scipy.sparse.linalg`.
    """

    def __init__(self):
        num_cols = np.prod(self.dir_shape)
        num_rows = np.prod(self.adj_shape)
        super().__init__(np.float32, [num_rows, num_cols])
        self.is_dir_operator = True

    def _matvec(self, x):
        """Implement the direct operator for column vectors from the right.

        :param x: Either row from the left or column from the right.
        :type x: numpy.array_like
        """
        if self.is_dir_operator:
            x = np.reshape(x, self.dir_shape)
            return self._op_direct(x).flatten()
        else:
            x = np.reshape(x, self.adj_shape)
            return self._op_adjoint(x).flatten()

    def rmatvec(self, x):
        """Implement the direct operator for row vectors from the left.

        :param x: Either row from the left or column from the right on transpose.
        :type x: numpy.array_like
        """
        if self.is_dir_operator:
            x = np.reshape(x, self.adj_shape)
            return self._op_adjoint(x).flatten()
        else:
            x = np.reshape(x, self.dir_shape)
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
        """Returns the projection operator using the absolute value of the
        projection coefficients.

        :returns: The absolute value operator
        :rtype: ProjectorOperator
        """
        return self

    def explicit(self):
        """Returns the explicit transformation matrix associated to the operator.

        :returns: The explicit transformation matrix
        :rtype: `numpy.array_like`
        """
        He = np.empty(self.shape, dtype=self.dtype)
        if self.is_dir_operator:
            dim_size = np.prod(self.dir_shape)
        else:
            dim_size = np.prod(self.adj_shape)
        for ii in range(dim_size):
            xii = np.zeros((dim_size, ))
            xii[ii] = 1
            He[:, ii] = self * xii
        return He

    def __call__(self, x):
        if self.is_dir_operator:
            return self._op_direct(x)
        else:
            return self._op_adjoint(x)

    def _op_direct(self, x):
        raise NotImplementedError()

    def _op_adjoint(self, x):
        raise NotImplementedError()


class ProjectorOperator(BaseTransform):
    """Base projector class that fixes the projection interface.
    """

    def __init__(self):
        self.dir_shape = self.vol_shape
        self.adj_shape = self.proj_shape
        super().__init__()

    def fp(self, x):
        raise NotImplementedError()

    def bp(self, x):
        raise NotImplementedError()

    def _op_direct(self, x):
        return self.fp(x)

    def _op_adjoint(self, x):
        return self.bp(x)


class TranformIdentity(BaseTransform):
    """Identity operator.
    """

    def __init__(self, x_shape):
        """Identity operator.

        :param x_shape: Shape of the data.
        :type x_shape: `numpy.array_like`
        """
        self.dir_shape = np.array(x_shape)
        self.adj_shape = np.array(x_shape)
        super().__init__()

    def _op_direct(self, x):
        return x

    def _op_adjoint(self, x):
        return x


class TranformDiagonalScaling(BaseTransform):
    """Diagonal scaling operator.
    """

    def __init__(self, x_shape, scale):
        """Diagonal scaling operator.

        :param x_shape: Shape of the data.
        :type x_shape: `numpy.array_like`
        :param scale: Operator diagonal.
        :type scale: float or `numpy.array_like`
        """
        self.scale = scale
        self.dir_shape = np.array(x_shape)
        self.adj_shape = np.array(x_shape)
        super().__init__()

    def absolute(self):
        """Returns the projection operator using the absolute value of the
        projection coefficients.

        :returns: The absolute value operator
        :rtype: Diagonal operator of the absolute values
        """
        return TranformDiagonalScaling(self.dir_shape, np.abs(self.scale))

    def _op_direct(self, x):
        return self.scale * x

    def _op_adjoint(self, x):
        return self.scale * x


class TranformWavelet(BaseTransform):
    """Wavelet tranform operator.
    """

    def __init__(self, x_shape, wavelet, level, axes=None, pad_on_demand='constant'):
        """Wavelet tranform operator.

        :param x_shape: Shape of the data to be wavelet transformed.
        :type x_shape: `numpy.array_like`
        :param wavelet: Wavelet type
        :type wavelet: string
        :param level: Numer of wavelet decomposition levels
        :type level: int
        :param axes: Axes along which to do the transform, defaults to None
        :type axes: int or tuple of int, optional
        :param pad_on_demand: Padding type to fit the `2 ** level` shape requirements, defaults to 'constant'
        :type pad_on_demand: string, optional. Options are all the `numpy.pad` padding modes.

        :raises ValueError: In case the pywavelets package is not available or its version is not adequate.
        """
        if not has_pywt:
            raise ValueError('Cannot use Wavelet transform because pywavelets is not installed.')
        if not use_swtn:
            raise ValueError('Cannot use Wavelet transform because pywavelets is too old (<1.0.2).')

        self.wavelet = wavelet
        self.level = level

        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=np.int)
        self.axes = axes

        self.pad_on_demand = pad_on_demand

        num_axes = len(self.axes)
        self.labels = [
            bin(x)[2:].zfill(num_axes).replace('0', 'a').replace('1', 'd') for x in range(1, 2 ** num_axes)]

        self.dir_shape = np.array(x_shape)
        if self.pad_on_demand is not None:
            alignment = 2 ** self.level
            x_axes = np.array(self.dir_shape)[np.array(self.axes)]
            self.pad_axes = (alignment - x_axes % alignment) % alignment

            adj_x_shape = cp.deepcopy(self.dir_shape)
            adj_x_shape[self.axes] += self.pad_axes
        else:
            adj_x_shape = self.dir_shape
        self.adj_shape = np.array((self.level * (2 ** len(self.axes) - 1) + 1, *adj_x_shape))

        super().__init__()

    def direct_swt(self, x):
        """Performs the direct wavelet transform.

        :param x: Data to transform.
        :type x: `numpy.array_like`

        :return: Transformed data.
        :rtype: list
        """
        if self.pad_on_demand is not None and np.any(self.pad_axes):
            for ax in np.nonzero(self.pad_axes)[0]:
                pad_l = np.ceil(self.pad_axes[ax] / 2).astype(np.int)
                pad_h = np.floor(self.pad_axes[ax] / 2).astype(np.int)
                pad_width = [(0, 0)] * len(x.shape)
                pad_width[self.axes[ax]] = (pad_l, pad_h)
                x = np.pad(x, pad_width, mode=self.pad_on_demand)
        return pywt.swtn(x, wavelet=self.wavelet, axes=self.axes, norm=True, level=self.level, trim_approx=True)

    def inverse_swt(self, y):
        """Performs the inverse wavelet transform.

        :param x: Data to anti-transform.
        :type x: list

        :return: Anti-transformed data.
        :rtype: `numpy.array_like`
        """
        x = pywt.iswtn(y, wavelet=self.wavelet, axes=self.axes, norm=True)
        if self.pad_on_demand is not None and np.any(self.pad_axes):
            for ax in np.nonzero(self.pad_axes)[0]:
                pad_l = np.ceil(self.pad_axes[ax] / 2).astype(np.int)
                pad_h = np.floor(self.pad_axes[ax] / 2).astype(np.int)
                slices = [slice(None)] * len(x.shape)
                slices[self.axes[ax]] = slice(pad_l, x.shape[self.axes[ax]]-pad_h, 1)
                x = x[tuple(slices)]
        return x

    def _op_direct(self, x):
        y = self.direct_swt(x)
        y = [y[0]] + [y[l][x] for l in range(1, self.level + 1) for x in self.labels]
        return np.array(y)

    def _op_adjoint(self, y):
        def get_lvl_pos(l):
            return (l - 1) * (2 ** len(self.axes) - 1) + 1
        y = [y[0]] + [
            dict(((k, y[ii + get_lvl_pos(l), ...]) for ii, k in enumerate(self.labels))) for l in range(1, self.level + 1)]
        return self.inverse_swt(y)


class TransformGradient(BaseTransform):

    def __init__(self, x_shape, axes=None):
        """Gradient transform.

        :param x_shape: Shape of the data to be wavelet transformed.
        :type x_shape: `numpy.array_like`
        :param axes: Axes along which to do the gradient, defaults to None
        :type axes: int or tuple of int, optional
        """
        if axes is None:
            axes = np.arange(-len(x_shape), 0, dtype=np.int)
        self.axes = axes
        self.ndims = len(x_shape)

        self.dir_shape = np.array(x_shape)
        self.adj_shape = np.array((len(self.axes), *self.dir_shape))

        super().__init__()

    def gradient(self, x):
        """Computes the gradient.

        :param x: Input data.
        :type x: `numpy.array_like`

        :return: Gradient of data.
        :rtype: `numpy.array_like`
        """
        d = [None] * len(self.axes)
        for ii in range(len(self.axes)):
            ind = -(ii + 1)
            padding = [(0, 0)] * self.ndims
            padding[ind] = (0, 1)
            temp_x = np.pad(x, padding, mode='constant')
            d[ind] = np.diff(temp_x, n=1, axis=ind)
        return np.stack(d, axis=0)

    def divergence(self, x):
        """Computes the divergence (transpose of gradient).

        :param x: Input data.
        :type x: `numpy.array_like`

        :return: Divergence of data.
        :rtype: `numpy.array_like`
        """
        d = [None] * len(self.axes)
        for ii in range(len(self.axes)):
            ind = -(ii + 1)
            padding = [(0, 0)] * self.ndims
            padding[ind] = (1, 0)
            temp_x = np.pad(x[ind, ...], padding, mode='constant')
            d[ind] = np.diff(temp_x, n=1, axis=ind)
        return np.sum(np.stack(d, axis=0), axis=0)

    def _op_direct(self, x):
        return self.gradient(x)

    def _op_adjoint(self, y):
        return - self.divergence(y)


if __name__ == '__main__':
    test_vol = np.zeros((10, 10), dtype=np.float32)

    H = TranformWavelet(test_vol.shape, 'db1', 2)
    Htw = H.T.explicit()
    Hw = H.explicit()

    D = TransformGradient(test_vol.shape)
    Dg = D.explicit()
