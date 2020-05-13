#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solvers for the tomographic reconstruction problem.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import scipy.sparse as sps

from numpy import random as rnd

import time as tm

try:
    import pywt
    has_pywt = True
    use_swtn = pywt.version.version >= '1.0.2'
except ImportError:
    has_pywt = False
    use_swtn = False
    print('WARNING - pywt was not found')


class BaseRegularizer(object):
    """Base regularizer class that defines the Regularizer object interface.
    """

    __reg_name__ = ''

    def __init__(self, weight):
        self.weight = weight

    def upper(self):
        return self.__reg_name__.upper()

    def lower(self):
        return self.__reg_name__.lower()

    def initialize_sigma_tau(self):
        raise NotImplementedError()

    def initialize_dual(self, primal):
        raise NotImplementedError()

    def update_dual(self, dual, primal):
        raise NotImplementedError()

    def apply_proximal(self, dual):
        raise NotImplementedError()

    def compute_update_primal(self, dual):
        raise NotImplementedError()


class Regularizer_TV(BaseRegularizer):
    """Total Variation (TV) regularizer. It can be used to promote piece-wise
    constant reconstructions.
    """

    __reg_name__ = 'TV'

    def __init__(self, weight, ndims=2):
        BaseRegularizer.__init__(self, weight=weight)
        self.ndims = ndims

    def initialize_sigma_tau(self):
        self.sigma = 0.5
        return self.weight * 2 * self.ndims

    def initialize_dual(self, primal):
        self.primal_shape = primal.shape
        return np.zeros([self.ndims, *self.primal_shape], dtype=primal.dtype)

    def update_dual(self, dual, primal):
        dual += self.sigma * self.gradient(primal)

    def apply_proximal(self, dual):
        dual_dir_norm_l2 = np.linalg.norm(dual, ord=2, axis=0, keepdims=True)
        dual /= np.fmax(1, dual_dir_norm_l2)

    def compute_update_primal(self, dual):
        return self.weight * self.divergence(dual)

    def gradient(self, x):
        d = [None] * self.ndims
        for ii in range(self.ndims):
            ind = -(ii + 1)
            padding = [(0, 0)] * len(self.primal_shape)
            padding[ind] = (0, 1)
            temp_x = np.pad(x, padding, mode='constant')
            d[ind] = np.diff(temp_x, n=1, axis=ind)
        return np.stack(d, axis=0)

    def divergence(self, x):
        d = [None] * self.ndims
        for ii in range(self.ndims):
            ind = -(ii + 1)
            padding = [(0, 0)] * len(self.primal_shape)
            padding[ind] = (1, 0)
            temp_x = np.pad(x[ind, ...], padding, mode='constant')
            d[ind] = np.diff(temp_x, n=1, axis=ind)
        return - np.sum(np.stack(d, axis=0), axis=0)


class Regularizer_TV2D(Regularizer_TV):
    """Total Variation (TV) regularizer in 2D. It can be used to promote
    piece-wise constant reconstructions.
    """

    __reg_name__ = 'TV2D'

    def __init__(self, weight):
        Regularizer_TV.__init__(self, weight=weight, ndims=2)


class Regularizer_TV3D(Regularizer_TV):
    """Total Variation (TV) regularizer in 3D. It can be used to promote
    piece-wise constant reconstructions.
    """

    __reg_name__ = 'TV3D'

    def __init__(self, weight):
        Regularizer_TV.__init__(self, weight=weight, ndims=3)


class Regularizer_lap(BaseRegularizer):
    """Laplacian regularizer. It can be used to promote smooth reconstructions.
    """

    __reg_name__ = 'lap'

    def __init__(self, weight, ndims=2):
        BaseRegularizer.__init__(self, weight=weight)
        self.ndims = ndims

    def initialize_sigma_tau(self):
        self.sigma = 0.25
        return self.weight * 4 * self.ndims

    def initialize_dual(self, primal):
        return np.zeros(primal.shape, dtype=primal.dtype)

    def update_dual(self, dual, primal):
        dual += self.sigma * self.laplacian(primal)

    def apply_proximal(self, dual):
        dual_dir_norm_l2 = np.linalg.norm(dual, ord=2, axis=0, keepdims=True)
        dual /= np.fmax(1, dual_dir_norm_l2)

    def compute_update_primal(self, dual):
        return self.weight * self.laplacian(dual)

    def laplacian(self, x):
        d = [None] * self.ndims
        for ii in range(self.ndims):
            ind = -(ii + 1)
            padding = [(0, 0)] * self.ndims
            padding[ind] = (1, 1)
            temp_x = np.pad(x, padding, mode='edge')
            d[ind] = np.diff(temp_x, n=2, axis=ind)
        return np.sum(d, axis=0)


class Regularizer_lap2D(Regularizer_lap):
    """Laplacian regularizer in 2D. It can be used to promote smooth
    reconstructions.
    """

    __reg_name__ = 'lap2D'

    def __init__(self, weight):
        Regularizer_lap.__init__(self, weight=weight, ndims=2)


class Regularizer_lap3D(Regularizer_lap):
    """Laplacian regularizer in 3D. It can be used to promote smooth
    reconstructions.
    """

    __reg_name__ = 'lap3D'

    def __init__(self, weight):
        Regularizer_lap.__init__(self, weight=weight, ndims=3)


class Regularizer_l1(BaseRegularizer):
    """l1-norm regularizer. It can be used to promote sparse reconstructions.
    """

    __reg_name__ = 'l1'

    def initialize_sigma_tau(self):
        return self.weight

    def initialize_dual(self, primal):
        return np.zeros(primal.shape, dtype=primal.dtype)

    def update_dual(self, dual, primal):
        dual += primal

    def apply_proximal(self, dual):
        dual /= np.fmax(1, np.abs(dual))

    def compute_update_primal(self, dual):
        return self.weight * dual


class Regularizer_l1wl(BaseRegularizer):
    """l1-norm Wavelet regularizer. It can be used to promote sparse reconstructions.
    """

    __reg_name__ = 'l1wl'

    def __init__(self, weight, wavelet, level, ndims=2, axes=None, pad_on_demand=False):
        if not has_pywt:
            raise ValueError('Cannot use l1wl regularizer because pywavelets is not installed.')
        if not use_swtn:
            raise ValueError('Cannot use l1wl regularizer because pywavelets is too old (<1.0.2).')
        BaseRegularizer.__init__(self, weight=weight)
        self.wavelet = wavelet
        self.level = level
        self.ndims = ndims

        if axes is None:
            axes = np.arange(-ndims, 0, dtype=np.int)
        self.axes = axes

        self.pad_on_demand = pad_on_demand

    def _op_direct(self, x):
        if self.pad_on_demand:
            alignment = 2 ** self.level
            x_axes = np.array(x.shape)[np.array(self.axes)]
            self.pad_axes = (alignment - x_axes % alignment) % alignment
            for ax in np.nonzero(self.pad_axes)[0]:
                pad_l = np.ceil(self.pad_axes[ax] / 2).astype(np.int)
                pad_h = np.floor(self.pad_axes[ax] / 2).astype(np.int)
                pad_width = [(0, 0)] * len(x.shape)
                pad_width[self.axes[ax]] = (pad_l, pad_h)
                x = np.pad(x, pad_width, mode='edge')
        return pywt.swtn(x, wavelet=self.wavelet, axes=self.axes, level=self.level)

    def _op_adjoint(self, y):
        x = pywt.iswtn(y, wavelet=self.wavelet, axes=self.axes)
        if self.pad_on_demand and np.any(self.pad_axes):
            for ax in np.nonzero(self.pad_axes)[0]:
                pad_l = np.ceil(self.pad_axes[ax] / 2).astype(np.int)
                pad_h = np.floor(self.pad_axes[ax] / 2).astype(np.int)
                slices = [slice(None)] * len(x.shape)
                slices[self.axes[ax]] = slice(pad_l, -pad_h, 1)
                x = x[tuple(slices)]
        return x

    def initialize_sigma_tau(self):
        self.sigma = 1 / (2 ** np.arange(self.level, 0, -1))
        return self.weight

    def initialize_dual(self, primal):
        self.primal_shape = primal.shape
        return self._op_direct(np.zeros(primal.shape, dtype=primal.dtype))

    def update_dual(self, dual, primal):
        d = self._op_direct(primal)
        for ii_l in range(self.level):
            for k in dual[ii_l].keys():
                dual[ii_l][k] += d[ii_l][k] * self.sigma[ii_l]

    def apply_proximal(self, dual):
        for ii_l in range(self.level):
            for k in dual[ii_l].keys():
                dual[ii_l][k] /= np.fmax(1, np.abs(dual[ii_l][k]))

    def compute_update_primal(self, dual):
        return self.weight * self._op_adjoint(dual)


class Solver(object):
    """Base solver class.
    """

    def __init__(self, verbose=False, relaxation=1, tolerance=None):
        self.verbose = verbose
        self.relaxation = relaxation
        self.tolerance = tolerance

    def upper(self):
        return type(self).__name__.upper()

    def lower(self):
        return type(self).__name__.lower()

    def _initialize_data_operators(self, A, At):
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


class Sart(Solver):
    """Solver class implementing the Simultaneous Algebraic Reconstruction
    Technique (SART) algorithm.
    """

    def __call__(  # noqa: C901
            self, A, b, iterations, A_num_rows, x0=None, At=None,
            lower_limit=None, upper_limit=None, x_mask=None, b_mask=None):
        """
        """
        data_type = b.dtype

        c_in = tm.time()

        # Back-projection diagonal re-scaling
        b_ones = np.ones_like(b)
        if b_mask is not None:
            b_ones *= b_mask
        tau = [At(b_ones[ii, :], ii) for ii in range(A_num_rows)]
        tau = np.abs(np.stack(tau, axis=0))
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = self.relaxation / tau

        # Forward-projection diagonal re-scaling
        x_ones = np.ones(tau.shape[1:], dtype=data_type)
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
            res_norm_rel = np.ones((iterations, )) * self.tolerance
        else:
            res_norm_rel = None

        c_init = tm.time()

        rows_sequence = rnd.permutation(A_num_rows)

        if self.verbose:
            print("- Performing %s iterations (init: %g seconds): " % (
                    self.upper(), c_init - c_in), end='', flush=True)
        for ii in range(iterations):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (
                        ii, iterations, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

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

            if self.verbose:
                print(('\b') * len(prnt_str), end='', flush=True)
                print((' ') * len(prnt_str), end='', flush=True)
                print(('\b') * len(prnt_str), end='', flush=True)

            if self.tolerance is not None:
                res = np.empty_like(b)
                for ii_a in rows_sequence:
                    res[..., ii_a, :] = A(x, ii_a) - b[..., ii_a, :]
                if b_mask is not None:
                    res *= b_mask
                res_norm_rel[ii] = np.linalg.norm(res) / res_norm_0

                if self.tolerance > res_norm_rel[ii]:
                    break

        if self.verbose:
            print("Done %d in %g seconds." % (iterations, tm.time() - c_in))

        return (x, res_norm_rel)


class Sirt(Solver):
    """Solver class implementing the Simultaneous Iterative Reconstruction
    Technique (SIRT) algorithm.
    """

    def __init__(
            self, verbose=False, tolerance=None, relaxation=1.95,
            regularizer=None):
        Solver.__init__(
            self, verbose=verbose, tolerance=tolerance, relaxation=relaxation)
        self.regularizer = regularizer

    def __call__(  # noqa: C901
            self, A, b, iterations, x0=None, At=None, lower_limit=None,
            upper_limit=None, x_mask=None, b_mask=None):
        """
        """
        (A, At) = self._initialize_data_operators(A, At)

        data_type = b.dtype

        c_in = tm.time()

        # Back-projection diagonal re-scaling
        tau = np.ones(b.shape, data_type)
        if b_mask is not None:
            tau *= b_mask
        tau = np.abs(At(tau))
        if self.regularizer is not None:
            tau += self.regularizer.initialize_sigma_tau()
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = self.relaxation / tau

        # Forward-projection diagonal re-scaling
        sigma = np.ones(tau.shape, dtype=data_type)
        if x_mask is not None:
            sigma *= x_mask
        sigma = np.abs(A(sigma))
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        if x0 is None:
            x0 = At(b * sigma) * tau
        x = x0

        if self.regularizer is not None:
            q = self.regularizer.initialize_dual(x)

        if self.tolerance is not None:
            res_norm_0 = np.linalg.norm(b.flatten())
            res_norm_rel = np.ones((iterations, )) * self.tolerance
        else:
            res_norm_rel = None

        c_init = tm.time()

        if self.verbose:
            print("- Performing %s iterations (init: %g seconds): " % (
                    self.upper(), c_init - c_in), end='', flush=True)
        for ii in range(iterations):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (
                        ii, iterations, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            res = b - A(x)
            if b_mask is not None:
                res *= b_mask

            if self.tolerance is not None:
                res_norm_rel[ii] = np.linalg.norm(res.flatten()) / res_norm_0
                if self.tolerance > res_norm_rel[ii]:
                    break

            if self.regularizer is not None:
                self.regularizer.update_dual(q, x)
                self.regularizer.apply_proximal(q)

            upd = At(res * sigma)
            if self.regularizer is not None:
                upd -= self.regularizer.compute_update_primal(q)
            x += upd * tau

            if lower_limit is not None:
                x = np.fmax(x, lower_limit)
            if upper_limit is not None:
                x = np.fmin(x, upper_limit)
            if x_mask is not None:
                x *= x_mask

            if self.verbose:
                print(('\b') * len(prnt_str), end='', flush=True)
                print((' ') * len(prnt_str), end='', flush=True)
                print(('\b') * len(prnt_str), end='', flush=True)

        if self.verbose:
            print("Done %d in %g seconds." % (iterations, tm.time() - c_in))

        return (x, res_norm_rel)


class CP(Solver):
    """Solver class implementing the primal-dual algorithm from Chambolle and
    Pock.
    It allows to specify two types of data fidelity terms: l2-norm and
    Kulback-Leibler. The first assumes Gaussian noise and the second Poisson
    noise as dominant sources of noise in the data.
    This solver also allows to specify a chosen regularizer, from the ones
    based on the BaseRegularizer interface.
    """

    def __init__(
            self, verbose=False, tolerance=None, relaxation=0.95,
            data_term='l2', regularizer=None):
        Solver.__init__(
            self, verbose=verbose, tolerance=tolerance, relaxation=relaxation)
        self.data_term = data_term
        self.regularizer = regularizer

    def power_method(self, A, At, b, iterations=5):
        data_type = b.dtype

        x = np.random.rand(*b.shape).astype(data_type)
        x /= np.linalg.norm(x)
        x = At(x)

        x_norm = np.linalg.norm(x)
        L = x_norm

        for ii in range(iterations):
            x /= x_norm
            x = At(A(x))

            x_norm = np.linalg.norm(x)
            L = np.sqrt(x_norm)

        return (L, x.shape)

    def __call__(  # noqa: C901
            self, A, b, iterations, x0=None, At=None, upper_limit=None,
            lower_limit=None, x_mask=None, b_mask=None, precondition=False):
        """
        """
        (A, At) = self._initialize_data_operators(A, At)
        if precondition:
            try:
                At_abs = At.absolute()
                A_abs = A.absolute()
            except AttributeError:
                print(A, At)
                print('WARNING: Turning off preconditioning because system matrix does not support absolute')
                precondition = False

        data_type = b.dtype

        c_in = tm.time()

        if precondition:
            tau = np.ones(b.shape, data_type)
            if b_mask is not None:
                tau *= b_mask
            tau = np.abs(At_abs(tau))
            if self.regularizer is not None:
                tau += self.regularizer.initialize_sigma_tau()
            tau[(tau / np.max(tau)) < 1e-5] = 1
            tau = self.relaxation / tau

            x_shape = tau.shape

            sigma = np.ones(tau.shape, dtype=data_type)
            if x_mask is not None:
                sigma *= x_mask
            sigma = np.abs(A_abs(sigma))
            sigma[(sigma / np.max(sigma)) < 1e-5] = 1
            sigma = 0.95 / sigma
        else:
            (L, x_shape) = self.power_method(A, At, b)
            tau = L
            if self.regularizer is not None:
                tau += self.regularizer.initialize_sigma_tau()
            tau = self.relaxation / tau
            sigma = 0.95 / L

        sigma1 = 1 / (1 + sigma)

        if x0 is None:
            x0 = np.zeros(x_shape, data_type)
        x = x0
        x_relax = x

        p = np.zeros(b.shape, dtype=data_type)

        if self.data_term.lower() == 'kl':
            b_prox = 4 * sigma * b
        else:
            b_prox = sigma * b

        if self.regularizer is not None:
            q = self.regularizer.initialize_dual(x)

        if self.tolerance is not None:
            res_norm_0 = np.linalg.norm(b.flatten())
            res_norm_rel = np.ones((iterations, )) * self.tolerance
        else:
            res_norm_rel = None

        c_init = tm.time()

        if self.verbose:
            reg_info = ''
            if self.regularizer is not None:
                reg_info = '-' + self.regularizer.upper()
            print("- Performing CP-%s%s iterations (init: %g seconds): " % (
                    self.data_term, reg_info, c_init - c_in), end='', flush=True)
        for ii in range(iterations):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (ii, iterations, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            p += A(x_relax) * sigma

            if self.data_term.lower() == 'kl':
                p = (1 + p - np.sqrt((p - 1) ** 2 + b_prox)) / 2
            else:
                p -= b_prox
                if self.data_term.lower() == 'l1':
                    p /= np.fmax(1, np.abs(p))
                elif self.data_term.lower() == 'l2':
                    p *= sigma1
                else:
                    raise ValueError("Unknown data term: %s" % self.data_term)

            if b_mask is not None:
                p *= b_mask

            if self.regularizer is not None:
                self.regularizer.update_dual(q, x)
                self.regularizer.apply_proximal(q)

            upd = At(p)
            if self.regularizer is not None:
                upd += self.regularizer.compute_update_primal(q)
            x_new = x - upd * tau

            if lower_limit is not None:
                x_new = np.fmax(x_new, lower_limit)
            if upper_limit is not None:
                x_new = np.fmin(x_new, upper_limit)
            if x_mask is not None:
                x_new *= x_mask

            x_relax = x_new + (x_new - x)
            x = x_new

            if self.verbose:
                print(('\b') * len(prnt_str), end='', flush=True)
                print((' ') * len(prnt_str), end='', flush=True)
                print(('\b') * len(prnt_str), end='', flush=True)

            if self.tolerance is not None:
                Ax = A(x)
                res = b - Ax
                res_norm_rel[ii] = np.linalg.norm(res.flatten()) / res_norm_0
                if self.tolerance > res_norm_rel[ii]:
                    break

        if self.verbose:
            print("Done %d in %g seconds." % (iterations, tm.time() - c_in))

        return (x, res_norm_rel)
