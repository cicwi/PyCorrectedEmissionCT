#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `corrct.solvers` package."""

import numpy as np
import copy as cp

import unittest

from corrct import solvers
from corrct import utils_proc
from corrct import utils_test


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

        tau = reg.initialize_sigma_tau(vol)
        assert tau == weight

        dual = reg.initialize_dual()
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
        assert np.all(np.isclose(dual * weight, upd))

    def _test_Regularizer_l1swl(self, vol):
        weight = 0.5
        ndims = len(vol.shape)
        level = 2
        reg = solvers.Regularizer_l1swl(weight, 'db1', level, ndims=ndims, min_approx=True)

        reg.initialize_sigma_tau(vol)

        dual = reg.initialize_dual()
        assert np.all(dual.shape[1:] == utils_test.roundup_to_pow2(vol.shape, level))
        assert dual.shape[0] == ((2 ** ndims - 1) * level + 1)
        assert np.all(dual == 0)

        copy_dual = cp.deepcopy(dual)
        reg.update_dual(dual, vol)
        sigma_shape = [-1] + [1] * (len(reg.op.adj_shape)-1)
        assert np.all(np.isclose(dual, copy_dual + reg.op(vol) * np.reshape(reg.sigma, sigma_shape)))

        upd = reg.compute_update_primal(dual)
        assert np.all(np.isclose(upd, weight * reg.op.T(dual)))

        copy_dual = cp.deepcopy(dual)
        reg.apply_proximal(dual)
        assert np.all(np.isclose(dual, np.fmax(np.fmin(1, copy_dual), -1)))

    def _test_Regularizer_TV(self, vol):
        weight = 0.5
        ndims = len(vol.shape)
        reg = solvers.Regularizer_Grad(weight, ndims=ndims)

        tau = reg.initialize_sigma_tau(vol)
        assert tau == (weight * 2 * ndims)

        dual = reg.initialize_dual()
        assert np.all(dual.shape[1:] == vol.shape)
        assert dual.shape[0] == ndims
        assert np.all(dual == 0)

        copy_dual = cp.deepcopy(dual)
        reg.update_dual(dual, vol)
        assert np.all(np.isclose(dual, copy_dual + reg.op(vol) / 2))

        dual = np.random.random(dual.shape)
        upd = reg.compute_update_primal(dual)
        assert np.all(np.isclose(reg.op.T(dual) * weight, upd))

    def _test_Regularizer_lap(self, vol):
        weight = 0.25
        ndims = len(vol.shape)
        reg = solvers.Regularizer_lap(weight, ndims=ndims)

        tau = reg.initialize_sigma_tau(vol)
        assert tau == (weight * 4 * ndims)

        dual = reg.initialize_dual()
        assert np.all(dual.shape == vol.shape)
        assert np.all(dual == 0)

        copy_dual = cp.deepcopy(dual)
        reg.update_dual(dual, vol)
        assert np.all(np.isclose(dual, copy_dual + reg.op(vol) / 4))

        dual = np.random.random(dual.shape)
        upd = reg.compute_update_primal(dual)
        assert np.all(np.isclose(reg.op.T(dual) * weight, upd))

    def test_000_Regularizer_l1_2D(self):
        """Test l1-min regularizer in 2D."""
        self._test_Regularizer_l1(self.vol_rand_2d)

    def test_001_Regularizer_l1_3D(self):
        """Test l1-min regularizer in 3D."""
        self._test_Regularizer_l1(self.vol_rand_3d)

    def test_002_Regularizer_l1swl_2D(self):
        """Test l1-min wavelet regularizer in 2D."""
        self._test_Regularizer_l1swl(self.vol_rand_2d)

    def test_003_Regularizer_l1swl_3D(self):
        """Test l1-min wavelet regularizer in 3D."""
        self._test_Regularizer_l1swl(self.vol_rand_3d)

    def test_004_Regularizer_TV_2D(self):
        """Test TV regularizer in 2D."""
        self._test_Regularizer_TV(self.vol_rand_2d)

    def test_005_Regularizer_TV_3D(self):
        """Test TV regularizer in 3D."""
        self._test_Regularizer_TV(self.vol_rand_3d)

    def test_004_Regularizer_lap_2D(self):
        """Test l1-min laplacian regularizer in 2D."""
        self._test_Regularizer_lap(self.vol_rand_2d)

    def test_005_Regularizer_lap_3D(self):
        """Test l1-min laplacian regularizer in 3D."""
        self._test_Regularizer_lap(self.vol_rand_3d)


class TestSolvers(unittest.TestCase):
    """Tests for the solvers in `corrct.solvers` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.test_vols_shape = (10, 10, 5)

        self.vol_rand_2d = np.fmin(np.random.rand(*self.test_vols_shape[:2]) + eps, 1)
        self.vol_flat_2d = utils_proc.get_circular_mask(self.test_vols_shape[:2], -2)

        self.proj_matrix_2d = (np.random.rand(5 * self.vol_rand_2d.size, self.vol_rand_2d.size) > 0.5).astype(np.float32)

        self.data_rand_2d = self.fwd_op(self.proj_matrix_2d, self.vol_rand_2d)

        self.data_flat_2d = self.fwd_op(self.proj_matrix_2d, self.vol_flat_2d)
        self.data_flat_2d += np.random.randn(*self.data_flat_2d.shape) * 1e-3

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def fwd_op(self, M, x):
        return np.dot(M, x.flatten())

    def bwd_op(self, M, y, x_shape):
        return np.reshape(np.dot(y.flatten(), M), x_shape)

    def get_A_At(self, vol_dims):
        if vol_dims.lower() == '2d':
            def A(x):
                return self.fwd_op(self.proj_matrix_2d, x)

            def At(y):
                return self.bwd_op(self.proj_matrix_2d, y, self.test_vols_shape[:2])

        else:
            raise ValueError('Only 2D implemented.')
        return (A, At)

    def test_000_SIRT(self):
        """Test SIRT algorithm in 2D."""

        (A, At) = self.get_A_At('2d')
        algo = solvers.Sirt()
        (sol, residual) = algo(A, self.data_rand_2d, 2500, At=At)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_rand_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_rand_2d, atol=1e-3))

    def test_001_CPLS(self):
        """Test Chambolle-Pock least-squares algorithm in 2D."""

        (A, At) = self.get_A_At('2d')
        algo = solvers.CP()
        (sol, residual) = algo(A, self.data_rand_2d, 2000, At=At)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_rand_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_rand_2d, atol=1e-3))

    def test_002_SIRTTV(self):
        """Test SIRT TV-min algorithm in 2D."""

        (A, At) = self.get_A_At('2d')
        reg = solvers.Regularizer_TV2D(1e-4)
        algo = solvers.Sirt(regularizer=reg)
        (sol, residual) = algo(A, self.data_flat_2d, 2500, At=At)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_flat_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_flat_2d, atol=1e-3))

    def test_003_CPLSTV(self):
        """Test Chambolle-Pock unconstrained least-squares TV-min algorithm in 2D."""

        (A, At) = self.get_A_At('2d')
        reg = solvers.Regularizer_TV2D(1e-4)
        algo = solvers.CP(regularizer=reg)
        (sol, residual) = algo(A, self.data_flat_2d, 2500, At=At)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_flat_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_flat_2d, atol=1e-3))

    def test_004_CPLSTV_unconstrained(self):
        """Test Chambolle-Pock unconstrained least-squares TV-min algorithm in 2D."""

        (A, At) = self.get_A_At('2d')
        reg = solvers.Regularizer_TV2D(1e-4)
        algo = solvers.CP(regularizer=reg)
        (sol, residual) = algo(A, self.data_flat_2d, 2500, At=At)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_flat_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_flat_2d, atol=1e-3))

    def test_005_CPLSTV_constrained01(self):
        """Test Chambolle-Pock constrained [0, 1] least-squares TV-min algorithm in 2D."""

        (A, At) = self.get_A_At('2d')
        reg = solvers.Regularizer_TV2D(1e-4)
        algo = solvers.CP(regularizer=reg)
        (sol, residual) = algo(A, self.data_flat_2d, 2500, At=At, lower_limit=0, upper_limit=1)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_flat_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_flat_2d, atol=1e-3))

    def test_006_CPLSTV_unconstrained_precond(self):
        """Test Chambolle-Pock preconditioned, unconstrained least-squares TV-min algorithm in 2D."""

        (A, At) = self.get_A_At('2d')
        reg = solvers.Regularizer_TV2D(1e-4)
        algo = solvers.CP(regularizer=reg)
        (sol, residual) = algo(A, self.data_flat_2d, 2500, At=At, precondition=True)

        print('Max absolute deviation is: {}. '.format(np.max(np.abs(self.vol_flat_2d - sol))),
              end='', flush=True)
        assert np.all(np.isclose(sol, self.vol_flat_2d, atol=1e-3))
