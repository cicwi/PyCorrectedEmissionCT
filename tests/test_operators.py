#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `corrct.operators` package."""

import numpy as np

import unittest

from corrct import operators
from corrct import utils_test


eps = np.finfo(np.float32).eps


class TestOperators(unittest.TestCase):
    """Base test class for operators in `corrct.operators` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        test_vols_shape = (29, 29)
        self.vol_ones_2d = np.ones(test_vols_shape)

    def tearDown(self):
        """Tear down test fixtures, if any."""


class TestTransformGradient(TestOperators):
    """Test for TransformGradient class in `corrct.operators` package."""

    def test_000_gradient(self):
        """Test gradient operator 2D."""
        D = operators.TransformGradient(self.vol_ones_2d.shape, pad_mode="constant")

        g = D(self.vol_ones_2d)
        assert np.all(g.shape[1:] == self.vol_ones_2d.shape)
        assert g.shape[0] == 2

        assert np.all(np.isclose(g[0, -1, :], -1, atol=eps))
        assert np.all(np.isclose(g[1, :, -1], -1, atol=eps))

    def test_001_minus_divergence(self):
        """Test divergence operator 2D."""
        D = operators.TransformGradient(self.vol_ones_2d.shape, pad_mode="constant")

        g = np.ones(D.adj_shape)
        d = D.T(g)
        assert np.all(d.shape == self.vol_ones_2d.shape)

        assert np.all(np.isclose(g[0, 0, :], 1, atol=eps))
        assert np.all(np.isclose(g[1, :, 0], 1, atol=eps))

    def test_002_explicit_gradient(self):
        """Test explicit gradient operator 2D."""
        D = operators.TransformGradient(self.vol_ones_2d.shape, pad_mode="constant")
        g0 = D(self.vol_ones_2d)

        De = D.explicit()
        ge = De.dot(self.vol_ones_2d.flatten())
        assert g0.size == ge.size

        ge = np.reshape(ge, D.adj_shape)
        assert np.all(np.isclose(g0, ge, atol=eps))

    def test_003_explicit_minus_divergence(self):
        """Test explicit transposed gradient operator 2D."""
        D = operators.TransformGradient(self.vol_ones_2d.shape, pad_mode="constant")
        g = np.ones(D.adj_shape)
        d0 = D.T(g)

        Dte = D.T.explicit()
        de = Dte.dot(np.ones(D.adj_shape).flatten())
        assert d0.size == de.size

        de = np.reshape(de, D.dir_shape)
        assert np.all(np.isclose(d0, de, atol=eps))


class TestTransformLaplacian(TestOperators):
    """Test for TransformLaplacian class in `corrct.operators` package."""

    def test_000_laplcian(self):
        """Test laplacian operator 2D."""
        L = operators.TransformLaplacian(self.vol_ones_2d.shape)

        g = L(self.vol_ones_2d)
        assert np.all(g.shape == self.vol_ones_2d.shape)
        assert np.all(np.isclose(g, 0, atol=eps))

        test_line = np.ones((3,))
        test_line[1] = 0
        L = operators.TransformLaplacian(test_line.shape)
        g = L(test_line)
        assert np.all(np.isclose(g, [-1, 2, -1], atol=eps))

    def test_001_explicit_gradient(self):
        """Test explicit laplacian operator 2D."""
        L = operators.TransformLaplacian(self.vol_ones_2d.shape)
        g0 = L(self.vol_ones_2d)

        Le = L.explicit()
        ge = Le.dot(self.vol_ones_2d.flatten())
        assert g0.size == ge.size

        ge = np.reshape(ge, L.adj_shape)
        assert np.all(np.isclose(g0, ge, atol=eps))


class TestTransformStationaryWavelet(TestOperators):
    """Tests for the TransformStationaryWavelet class in `corrct.operators` package."""

    def test_000_transform(self):
        """Test Haar wavelet transform in 2D."""
        wl_dec_level = 3
        H = operators.TransformStationaryWavelet(self.vol_ones_2d.shape, "db1", wl_dec_level)

        w = H(self.vol_ones_2d)
        assert np.all(w.shape[1:] == utils_test.roundup_to_pow2(self.vol_ones_2d.shape, wl_dec_level))
        assert w.shape[0] == 10

        wtw = H.T(w)
        assert np.all(wtw.shape == self.vol_ones_2d.shape)

        print("Max absolute deviation is: {}. ".format(np.max(np.abs(self.vol_ones_2d - wtw))), end="", flush=True)
        assert np.all(np.isclose(wtw, self.vol_ones_2d, atol=eps * 1e2))

    def test_001_explicit_transform(self):
        """Test explicit Haar transform in 2D."""
        H = operators.TransformStationaryWavelet(self.vol_ones_2d.shape, "db1", 3)
        w0 = H(self.vol_ones_2d)

        He = H.explicit()
        we = He.dot(self.vol_ones_2d.flatten())
        assert w0.size == we.size

        we = np.reshape(we, H.adj_shape)
        assert np.all(np.isclose(w0, we, atol=eps))

    def test_002_explicit_inverse_transform(self):
        """Test explicit inverse Haar transform in 2D."""
        H = operators.TransformStationaryWavelet(self.vol_ones_2d.shape, "db1", 2)
        wt0 = H.T(np.ones((H.adj_shape)))

        Hte = H.T.explicit()
        wte = Hte.dot(np.ones(H.adj_shape).flatten())
        assert wt0.size == wte.size

        wte = np.reshape(wte, H.dir_shape)
        assert np.all(np.isclose(wt0, wte, atol=eps))
