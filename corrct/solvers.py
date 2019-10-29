#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solvers for the tomographic reconstruction problem.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands
"""

import numpy as np

from numpy import random as rnd

import time as tm


class BaseRegularizer(object):
    """Base regularizer class that defines the Regularizer object interface.
    """

    def __init__(self, weight):
        self.weight = weight

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


class Regularizer_TV2D(BaseRegularizer):
    """Total Variation (TV) regularizer. It can be used to promote piece-wise
    constant reconstructions.
    """

    def initialize_sigma_tau(self):
        self.sigma = 0.5
        return self.weight * 2 * 2

    def initialize_dual(self, primal):
        return np.zeros([2, *primal.shape], dtype=primal.dtype)

    def update_dual(self, dual, primal):
        dual += self.sigma * self.gradient(primal)

    def apply_proximal(self, dual):
        dual_dir_norm_l2 = np.linalg.norm(dual, ord=2, axis=0, keepdims=True)
        dual /= np.fmax(1, dual_dir_norm_l2)

    def compute_update_primal(self, dual):
        return self.weight * self.divergence(dual)

    def gradient(self, x):
        d0 = np.pad(x, ((0, 1), (0, 0)), mode='constant')
        d0 = np.diff(d0, n=1, axis=0)
        d1 = np.pad(x, ((0, 0), (0, 1)), mode='constant')
        d1 = np.diff(d1, n=1, axis=1)
        return np.stack((d0, d1), axis=0)

    def divergence(self, x):
        x0, x1 = x[0, ...], x[1, ...]
        d0 = np.pad(x0, ((1, 0), (0, 0)), mode='constant')
        d1 = np.pad(x1, ((0, 0), (1, 0)), mode='constant')
        return np.diff(d0, n=1, axis=0) + np.diff(d1, n=1, axis=1)


class Regularizer_l1(BaseRegularizer):
    """l1-norm regularizer. It can be used to promote sparse reconstructions.
    """

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


class Solver(object):
    """Base solver class.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def upper(self):
        return type(self).__name__.upper()

    def lower(self):
        return type(self).__name__.lower()


class Sart(Solver):
    """Solver class implementing the Simultaneous Algebraic Reconstruction
    Technique (SART) algorithm.
    """

    def __call__(
            self, A, b, iterations, A_num_rows, x0=None, At=None,
            lower_limit=None, upper_limit=None, x_mask=None, b_mask=None):
        """
        """
        data_type = b.dtype

        c_in = tm.time()

        # Back-projection diagonal re-scaling
        b_ones = np.ones_like(b)
        tau = [At(b_ones, ii) for ii in range(A_num_rows)]
        tau = np.abs(np.stack(tau))
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = 1 / tau

        # Forward-projection diagonal re-scaling
        x_ones = np.ones(tau.shape[1:], dtype=data_type)
        sigma = np.empty_like(b)
        for ii in range(A_num_rows):
            sigma[ii, ...] = A(x_ones, ii)
        sigma = np.abs(sigma)
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        if x0 is None:
            x0 = np.zeros_like(x_ones)
        x = x0

        c_init = tm.time()

        rows_sequence = rnd.permutation(A_num_rows)

        if self.verbose:
            print("- Performing SIRT iterations (init: %g seconds): " % (
                    c_init - c_in), end='', flush=True)
        for ii in range(iterations):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (
                        ii, iterations, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            for ii_a in rows_sequence:

                res = A(x, ii_a) - b[ii_a, :]
                if b_mask is not None:
                    res *= b_mask[ii_a, :]

                x -= At(res * sigma[ii_a, :], ii_a) * tau[ii_a, ...]

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
            print("Done in %g seconds." % (tm.time() - c_in))

        return (x, None)


class Sirt(Solver):
    """Solver class implementing the Simultaneous Iterative Reconstruction
    Technique (SIRT) algorithm.
    """

    def __call__(
            self, A, b, iterations, x0=None, At=None, lower_limit=None,
            upper_limit=None, x_mask=None, b_mask=None):
        """
        """
        data_type = b.dtype

        c_in = tm.time()

        # Back-projection diagonal re-scaling
        tau = np.abs(At(np.ones(b.shape, data_type)))
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = 1 / tau

        # Forward-projection diagonal re-scaling
        sigma = np.abs(A(np.ones(tau.shape, dtype=data_type)))
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        if x0 is None:
            x0 = At(b * sigma) * tau
        x = x0

        res_norm_0 = np.linalg.norm(b.flatten())
        res_norm_rel = np.empty((iterations, ))

        c_init = tm.time()

        if self.verbose:
            print("- Performing SIRT iterations (init: %g seconds): " % (
                    c_init - c_in), end='', flush=True)
        for ii in range(iterations):
            if self.verbose:
                prnt_str = "%03d/%03d (avg: %g seconds)" % (
                        ii, iterations, (tm.time() - c_init) / np.fmax(ii, 1))
                print(prnt_str, end='', flush=True)

            res = b - A(x)
            if b_mask is not None:
                res *= b_mask
            res_norm_rel[ii] = np.linalg.norm(res.flatten()) / res_norm_0

            x += At(res * sigma) * tau

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
            print("Done in %g seconds." % (tm.time() - c_in))

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

    def __init__(self, verbose=False, data_term='l2', regularizer=None):
        Solver.__init__(self, verbose=verbose)
        self.data_term = data_term
        self.regularizer = regularizer

    def __call__(
            self, A, b, iterations, x0=None, At=None, upper_limit=None,
            lower_limit=None, x_mask=None, b_mask=None):
        """
        """
        data_type = b.dtype

        c_in = tm.time()

        tau = np.abs(At(np.ones(b.shape, dtype=data_type)))
        if self.regularizer is not None:
            tau += self.regularizer.initialize_sigma_tau()
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = 1 / tau

        sigma = np.abs(A(np.ones(tau.shape, dtype=data_type)))
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        sigma1 = 1 / (1 + sigma)

        if x0 is None:
            x0 = np.zeros_like(tau)
        x = x0
        x_relax = x

        p = np.zeros(b.shape, dtype=data_type)

        if self.data_term.lower() == 'kl':
            b_prox = 4 * sigma * b
        else:
            b_prox = sigma * b

        if self.regularizer is not None:
            q = self.regularizer.initialize_dual(x)

        c_init = tm.time()

        if self.verbose:
            print("- Performing CP-%s iterations (init: %g seconds): " % (self.data_term, c_init - c_in), end='', flush=True)
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

        if self.verbose:
            print("Done in %g seconds." % (tm.time() - c_in))

        return (x, None)


