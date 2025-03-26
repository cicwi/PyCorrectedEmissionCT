#!/usr/bin/env python

"""Tests for `corrct.operators` package."""

import numpy as np
import pytest
import scipy.signal as spsig

from corrct import operators, testing

eps = float(np.finfo(np.float32).eps)


@pytest.fixture(scope="class")
def bootstrap_base(request):
    """Set up the testing classes."""
    cls = request.cls
    test_vols_shape = (29, 29)
    cls.vol_ones_2d = np.ones(test_vols_shape)


@pytest.mark.usefixtures("bootstrap_base")
class TestTransformGradient:
    """Test for TransformGradient class in `corrct.operators` package."""

    def test_000_gradient(self):
        """Test gradient operator 2D."""
        D = operators.TransformGradient(self.vol_ones_2d.shape, pad_mode="constant")

        g = D(self.vol_ones_2d)
        assert np.allclose(g.shape[1:], self.vol_ones_2d.shape)
        assert g.shape[0] == 2

        assert np.allclose(g[0, -1, :], -1, atol=eps)
        assert np.allclose(g[1, :, -1], -1, atol=eps)

    def test_001_minus_divergence(self):
        """Test divergence operator 2D."""
        D = operators.TransformGradient(self.vol_ones_2d.shape, pad_mode="constant")

        g = np.ones(D.adj_shape)
        d = D.T(g)
        assert np.allclose(d.shape, self.vol_ones_2d.shape)

        assert np.allclose(g[0, 0, :], 1, atol=eps)
        assert np.allclose(g[1, :, 0], 1, atol=eps)

    def test_002_explicit_gradient(self):
        """Test explicit gradient operator 2D."""
        D = operators.TransformGradient(self.vol_ones_2d.shape, pad_mode="constant")
        g0 = D(self.vol_ones_2d)

        De = D.explicit()
        ge = De.dot(self.vol_ones_2d.flatten())
        assert g0.size == ge.size

        ge = np.reshape(ge, D.adj_shape)
        assert np.allclose(g0, ge, atol=eps)

    def test_003_explicit_minus_divergence(self):
        """Test explicit transposed gradient operator 2D."""
        D = operators.TransformGradient(self.vol_ones_2d.shape, pad_mode="constant")
        g = np.ones(D.adj_shape)
        d0 = D.T(g)

        Dte = D.T.explicit()
        de = Dte.dot(np.ones(D.adj_shape).flatten())
        assert d0.size == de.size

        de = np.reshape(de, D.dir_shape)
        assert np.allclose(d0, de, atol=eps)


@pytest.mark.usefixtures("bootstrap_base")
class TestTransformLaplacian:
    """Test for TransformLaplacian class in `corrct.operators` package."""

    def test_000_laplacian(self):
        """Test laplacian operator 2D."""
        L = operators.TransformLaplacian(self.vol_ones_2d.shape)

        g = L(self.vol_ones_2d)
        assert np.allclose(g.shape, self.vol_ones_2d.shape)
        assert np.allclose(g, 0, atol=eps)

        test_line = np.ones((3,))
        test_line[1] = 0
        L = operators.TransformLaplacian(test_line.shape)
        g = L(test_line)
        assert np.allclose(g, [-1, 2, -1], atol=eps)

    def test_001_explicit_gradient(self):
        """Test explicit laplacian operator 2D."""
        L = operators.TransformLaplacian(self.vol_ones_2d.shape)
        g0 = L(self.vol_ones_2d)

        Le = L.explicit()
        ge = Le.dot(self.vol_ones_2d.flatten())
        assert g0.size == ge.size

        ge = np.reshape(ge, L.adj_shape)
        assert np.allclose(g0, ge, atol=eps)


@pytest.mark.usefixtures("bootstrap_base")
class TestTransformStationaryWavelet:
    """Tests for the TransformStationaryWavelet class in `corrct.operators` package."""

    def test_000_transform(self):
        """Test Haar wavelet transform in 2D."""
        wl_dec_level = 3
        H = operators.TransformStationaryWavelet(self.vol_ones_2d.shape, "db1", wl_dec_level)

        w = H(self.vol_ones_2d)
        assert np.allclose(w.shape[1:], testing.roundup_to_pow2(self.vol_ones_2d.shape, wl_dec_level))
        assert w.shape[0] == 10

        wtw = H.T(w)
        assert np.allclose(wtw.shape, self.vol_ones_2d.shape)

        print(f"Max absolute deviation is: {np.max(np.abs(self.vol_ones_2d - wtw))}. ", end="", flush=True)
        assert np.allclose(wtw, self.vol_ones_2d, atol=eps * 1e2)

    def test_001_explicit_transform(self):
        """Test explicit Haar transform in 2D."""
        H = operators.TransformStationaryWavelet(self.vol_ones_2d.shape, "db1", 3)
        w0 = H(self.vol_ones_2d)

        He = H.explicit()
        we = He.dot(self.vol_ones_2d.flatten())
        assert w0.size == we.size

        we = np.reshape(we, H.adj_shape)
        assert np.allclose(w0, we, atol=eps)

    def test_002_explicit_inverse_transform(self):
        """Test explicit inverse Haar transform in 2D."""
        H = operators.TransformStationaryWavelet(self.vol_ones_2d.shape, "db1", 2)
        wt0 = H.T(np.ones(H.adj_shape))

        Hte = H.T.explicit()
        wte = Hte.dot(np.ones(H.adj_shape).flatten())
        assert wt0.size == wte.size

        wte = np.reshape(wte, H.dir_shape)
        assert np.allclose(wt0, wte, atol=eps)


@pytest.fixture(scope="class")
def bootstrap_convolution(request):
    """Set up the convolution test class."""
    cls = request.cls

    cls.prj_shape_v1u = (32, 1, 32)
    cls.prj_shape_vu = (32, 32)
    cls.prj_shape_vwu = (32, 8, 32)

    cls.prj_shape_1u = cls.prj_shape_vwu[-2::]
    cls.prj_shape_u = cls.prj_shape_vwu[-1::]
    cls.prj_shape_wu = cls.prj_shape_vwu[-2::]

    cls.prj_vu = np.random.rand(*cls.prj_shape_vu)
    cls.prj_vwu = np.random.rand(*cls.prj_shape_vwu)

    cls.kernel_1u = np.ones((1, 3))
    cls.kernel_u = np.ones((3,))
    cls.kernel_v1u = np.ones((3, 1, 3))
    cls.kernel_vu = np.ones((3, 3))


@pytest.mark.usefixtures("bootstrap_convolution")
class TestTransformConvolution:
    """Tests for the TransformConvolution class in `corrct.operators` package."""

    def test_000_initialize(self):
        """Test the initialization of the convolution operator."""
        C = operators.TransformConvolution(self.prj_shape_vu, kernel=self.kernel_u)
        assert len(C.kernel.shape) == len(C.dir_shape), "Incorrect kernel dimensions initialization (img vu, ker u)"

        C = operators.TransformConvolution(self.prj_shape_vu, kernel=self.kernel_vu)
        assert len(C.kernel.shape) == len(C.dir_shape), "Incorrect kernel dimensions initialization (img vu, ker vu)"

        C = operators.TransformConvolution(self.prj_shape_vwu, kernel=self.kernel_u)
        assert len(C.kernel.shape) == len(C.dir_shape), "Incorrect kernel dimensions initialization (img vwu, ker u)"

        C = operators.TransformConvolution(self.prj_shape_vwu, kernel=self.kernel_vu)
        assert len(C.kernel.shape) == len(C.dir_shape), "Incorrect kernel dimensions initialization (img vwu, ker u)"

    def test_001_direct_shapes(self):
        """Test the output dimensions of the convolution."""
        C = operators.TransformConvolution(self.prj_shape_vu, kernel=self.kernel_u)
        conv_prj = C(self.prj_vu)
        assert len(conv_prj.shape) == len(self.prj_vu.shape), "Non-matching output dimensions (img vu, ker u)"

        C = operators.TransformConvolution(self.prj_shape_vu, kernel=self.kernel_vu)
        conv_prj = C(self.prj_vu)
        assert len(conv_prj.shape) == len(self.prj_vu.shape), "Non-matching output dimensions (img vu, ker vu)"

        C = operators.TransformConvolution(self.prj_shape_vwu, kernel=self.kernel_u)
        conv_prj = C(self.prj_vwu)
        assert len(conv_prj.shape) == len(self.prj_vwu.shape), "Non-matching output dimensions (img vwu, ker u)"

        C = operators.TransformConvolution(self.prj_shape_vwu, kernel=self.kernel_vu)
        conv_prj = C(self.prj_vwu)
        assert len(conv_prj.shape) == len(self.prj_vwu.shape), "Non-matching output dimensions (img vwu, ker vu)"

    def test_002_direct_results_vu_u(self):
        """Test the output correctness of the convolution for the case  (img vu, ker u)."""
        C = operators.TransformConvolution(self.prj_shape_vu, kernel=self.kernel_u, pad_mode="constant")
        conv_prj = C(self.prj_vu)
        conv_prj_ref = spsig.convolve(self.prj_vu, self.kernel_u[None, :], mode="same")
        assert np.allclose(conv_prj, conv_prj_ref, rtol=1e-7), "Non-matching output (img vu, ker u)"

    def test_002_direct_results_vu_vu(self):
        """Test the output correctness of the convolution for the case  (img vu, ker vu)."""
        C = operators.TransformConvolution(self.prj_shape_vu, kernel=self.kernel_vu, pad_mode="constant")
        conv_prj = C(self.prj_vu)
        conv_prj_ref = spsig.convolve(self.prj_vu, self.kernel_vu, mode="same")
        assert np.allclose(conv_prj, conv_prj_ref, rtol=1e-7), "Non-matching output (img vu, ker vu)"

    def test_002_direct_results_vwu_u(self):
        """Test the output correctness of the convolution for the case  (img vwu, ker u)."""
        C = operators.TransformConvolution(self.prj_shape_vwu, kernel=self.kernel_u, pad_mode="constant")
        conv_prj = C(self.prj_vwu)
        conv_prj_ref = spsig.convolve(self.prj_vwu, self.kernel_u[None, None, :], mode="same")
        assert np.allclose(conv_prj, conv_prj_ref, rtol=1e-7), "Non-matching output (img vwu, ker u)"

    def test_002_direct_results_vwu_vu(self):
        """Test the output correctness of the convolution for the case  (img vwu, ker vu)."""
        C = operators.TransformConvolution(self.prj_shape_vwu, kernel=self.kernel_vu[..., None, :], pad_mode="constant")
        conv_prj = C(self.prj_vwu)
        conv_prj_ref = spsig.convolve(self.prj_vwu, self.kernel_vu[..., None, :], mode="same")
        assert np.allclose(conv_prj, conv_prj_ref, rtol=1e-7), "Non-matching output (img vwu, ker vu)"
