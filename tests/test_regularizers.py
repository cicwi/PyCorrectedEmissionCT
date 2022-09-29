#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test `corrct.regularizers` package.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

import pytest

from corrct import regularizers
from corrct import testing


eps = np.finfo(np.float32).eps


@pytest.fixture(scope="class")
def bootstrap_base(request):
    test_vols_shape = (100, 100, 10)
    cls = request.cls
    cls.vol_rand_2d = np.fmin(np.random.rand(*test_vols_shape[:2]) + eps, 1)
    cls.vol_rand_3d = np.fmin(np.random.rand(*test_vols_shape) + eps, 1)


@pytest.mark.usefixtures("bootstrap_base")
class TestRegularizers:
    """Tests for the regularizers in `corrct.regularizers` package."""

    def _test_Regularizer_l1(self, vol):
        weight = 0.5
        reg = regularizers.Regularizer_l1(weight)

        tau = reg.initialize_sigma_tau(vol)
        assert tau == 1

        dual = reg.initialize_dual()
        assert np.all(dual.shape == vol.shape)
        assert np.all(dual == 0)

        copy_dual = dual.copy()
        reg.update_dual(dual, vol)
        assert not np.all(dual == copy_dual)

        assert np.all(dual == vol)

        dual += weight / 2
        copy_dual = dual.copy()
        reg.apply_proximal(dual)
        assert np.all(np.isclose(dual, np.fmin(weight, copy_dual)))

        dual = copy_dual - weight * 2
        copy_dual = dual.copy()
        reg.apply_proximal(dual)
        assert np.all(np.isclose(dual, np.fmax(-weight, copy_dual)))

        upd = reg.compute_update_primal(dual)
        assert np.all(np.isclose(dual, upd))

    def _test_Regularizer_l1swl(self, vol):
        weight = 0.5
        ndims = len(vol.shape)
        level = 2
        reg = regularizers.Regularizer_l1swl(weight, "db1", level, ndims=ndims, min_approx=True)

        reg.initialize_sigma_tau(vol)

        dual = reg.initialize_dual()
        assert np.all(dual.shape[1:] == testing.roundup_to_pow2(vol.shape, level))
        assert dual.shape[0] == ((2**ndims - 1) * level + 1)
        assert np.all(dual == 0)

        copy_dual = dual.copy()
        reg.update_dual(dual, vol)
        sigma_shape = [-1] + [1] * (len(reg.op.adj_shape) - 1)
        assert np.all(np.isclose(dual, copy_dual + reg.op(vol) * np.reshape(reg.sigma, sigma_shape)))

        upd = reg.compute_update_primal(dual)
        assert np.all(np.isclose(upd, reg.op.T(dual)))

        copy_dual = dual.copy()
        reg.apply_proximal(dual)
        assert np.all(np.isclose(dual, np.fmax(np.fmin(weight, copy_dual), -weight)))

    def _test_Regularizer_TV(self, vol):
        weight = 0.5
        ndims = len(vol.shape)
        reg = regularizers.Regularizer_Grad(weight, ndims=ndims)

        tau = reg.initialize_sigma_tau(vol)
        assert tau == (2 * ndims)

        dual = reg.initialize_dual()
        assert np.all(dual.shape[1:] == vol.shape)
        assert dual.shape[0] == ndims
        assert np.all(dual == 0)

        copy_dual = dual.copy()
        reg.update_dual(dual, vol)
        assert np.all(np.isclose(dual, copy_dual + reg.op(vol) / 2))

        dual = np.random.random(dual.shape)
        upd = reg.compute_update_primal(dual)
        assert np.all(np.isclose(reg.op.T(dual), upd))

    def _test_Regularizer_lap(self, vol):
        weight = 0.25
        ndims = len(vol.shape)
        reg = regularizers.Regularizer_lap(weight, ndims=ndims)

        tau = reg.initialize_sigma_tau(vol)
        assert tau == (4 * ndims)

        dual = reg.initialize_dual()
        assert np.all(dual.shape == vol.shape)
        assert np.all(dual == 0)

        copy_dual = dual.copy()
        reg.update_dual(dual, vol)
        assert np.all(np.isclose(dual, copy_dual + reg.op(vol) / 4))

        dual = np.random.random(dual.shape)
        upd = reg.compute_update_primal(dual)
        assert np.all(np.isclose(reg.op.T(dual), upd))

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
