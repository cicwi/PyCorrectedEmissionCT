#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `corrct.solvers` package."""

import numpy as np
import copy as cp

import unittest

from corrct import solvers
from corrct import utils_proc


eps = np.finfo(np.float32).eps


class TestRegularizers(unittest.TestCase):
    """Tests for the regularizers in `corrct.solvers` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        test_vols_shape = (100, 100, 10)
        self.vol_rand_2d = np.fmin(np.random.rand(*test_vols_shape[:2]) + eps, 1)
        self.vol_rand_3d = np.fmin(np.random.rand(*test_vols_shape) + eps, 1)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def _test_Regularizer_l1(self, vol):
        weight = 0.5
        reg = solvers.Regularizer_l1(weight)

        tau = reg.initialize_sigma_tau()
        assert tau == weight

        dual = reg.initialize_dual(vol)
        assert np.all(dual.shape == vol.shape)
        assert np.all(dual == 0)

        copy_dual = cp.deepcopy(dual)
        reg.update_dual(dual, vol)
        assert not np.all(dual == copy_dual)

        assert np.all(dual == vol)

        dual += 0.5
        copy_dual = cp.deepcopy(dual)
        reg.apply_proximal(dual)
        assert np.all(np.isclose(dual, np.fmin(1, copy_dual)))

        dual = copy_dual - 2
        copy_dual = cp.deepcopy(dual)
        reg.apply_proximal(dual)
        assert np.all(np.isclose(dual, np.fmax(-1, copy_dual)))

        upd = reg.compute_update_primal(dual)
        assert np.all(dual * weight == upd)

    def test_000_Regularizer_l1_2D(self):
        """Test l1-min regularizer in 2D."""
        self._test_Regularizer_l1(self.vol_rand_2d)

    def test_001_Regularizer_l1_3D(self):
        """Test l1-min regularizer in 3D."""
        self._test_Regularizer_l1(self.vol_rand_3d)


class TestSolvers(unittest.TestCase):
    """Tests for the solvers in `corrct.solvers` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        test_vols_shape = (10, 10, 5)
        self.vol_rand_2d = np.fmin(np.random.rand(*test_vols_shape[:2]) + eps, 1)
        self.proj_matrix_rand_2d = np.random.rand(5 * self.vol_rand_2d.size, self.vol_rand_2d.size)
        self.data_rand_2d = self.fwd_op(self.proj_matrix_rand_2d, self.vol_rand_2d)

        self.vol_flat_2d = utils_proc.get_circular_mask(test_vols_shape[:2], -2)
        self.proj_matrix_flat_2d = np.random.rand(2 * self.vol_flat_2d.size, self.vol_flat_2d.size)
        self.data_flat_2d = self.fwd_op(self.proj_matrix_flat_2d, self.vol_flat_2d)
        self.data_flat_2d += (2 * np.random.rand(*self.data_flat_2d.shape) - 1) * 1e-3

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def fwd_op(self, M, x):
        return np.dot(M, x.flatten())
    def bwd_op(self, M, y, x_shape):
        return np.reshape(np.dot(y.flatten(), M), x_shape)

    def test_000_SIRT(self):
        """Test SIRT algorithm in 2D."""

        def A(x):
            return self.fwd_op(self.proj_matrix_rand_2d, x)
        def At(y):
            return self.bwd_op(self.proj_matrix_rand_2d, y, self.vol_rand_2d.shape)

        algo = solvers.Sirt()
        (sol, residual) = algo(A, self.data_rand_2d, 2500, At=At)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_rand_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_rand_2d, atol=1e-3))

    def test_001_CPLS(self):
        """Test Chambolle-Pock least-squares algorithm in 2D."""

        def A(x):
            return self.fwd_op(self.proj_matrix_rand_2d, x)
        def At(y):
            return self.bwd_op(self.proj_matrix_rand_2d, y, self.vol_rand_2d.shape)

        algo = solvers.CP()
        (sol, residual) = algo(A, self.data_rand_2d, 2000, At=At)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_rand_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_rand_2d, atol=1e-3))

    def test_002_SIRTTV(self):
        """Test SIRT TV-min algorithm in 2D."""

        def A(x):
            return self.fwd_op(self.proj_matrix_flat_2d, x)
        def At(y):
            return self.bwd_op(self.proj_matrix_flat_2d, y, self.vol_flat_2d.shape)

        reg = solvers.Regularizer_TV2D(1e-3)
        algo = solvers.CP(regularizer=reg)
        (sol, residual) = algo(A, self.data_flat_2d, 2500, At=At)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_flat_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_flat_2d, atol=1e-3))

    def test_003_CPLSTV(self):
        """Test Chambolle-Pock least-squares TV-min algorithm in 2D."""

        def A(x):
            return self.fwd_op(self.proj_matrix_flat_2d, x)
        def At(y):
            return self.bwd_op(self.proj_matrix_flat_2d, y, self.vol_flat_2d.shape)

        reg = solvers.Regularizer_TV2D(1e-3)
        algo = solvers.CP(regularizer=reg)
        (sol, residual) = algo(A, self.data_flat_2d, 2500, At=At)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_flat_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_flat_2d, atol=1e-3))


