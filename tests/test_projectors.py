#!/usr/bin/env python3
"""
Test `corrct.projectors` package.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.ndimage as spndi
import skimage.data as skd
import skimage.transform as skt
from numpy.typing import NDArray

import corrct as cct
from corrct import _projector_backends as backends

eps = np.finfo(np.float32).eps


def _radon_rot_sk_w(x: NDArray, angles_rad: NDArray, shift: Optional[float] = None) -> NDArray:
    """
    Compute the Radon transform using skimage's warp function.

    Parameters
    ----------
    x : NDArray
        Input image.
    angles_rad : NDArray
        Angles in radians.
    shift : float, optional
        Shift value. Default is None.

    Returns
    -------
    NDArray
        Radon transform of the input image.
    """
    prj = np.empty((len(angles_rad), x.shape[-1]), dtype=np.float32)
    c = (np.array(x.shape) - 1) / 2

    I = np.eye(3, dtype=np.float32)
    C = np.array([[0, 0, c[1]], [0, 0, c[0]], [0, 0, 0]], dtype=np.float32)
    T1 = I - C
    T2 = I + C
    if shift is not None:
        S = np.array([[0, 0, shift], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    else:
        S = np.zeros_like(I)

    for ii, angle in enumerate(angles_rad):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, sin_a, 0], [-sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)

        # Transformation matrix, that takes the center into account
        T2_R_T1 = T2.dot(R.dot(T1))
        rotated_x = skt.warp(x, T2_R_T1, clip=False)
        if shift is not None:
            rotated_x = skt.warp(rotated_x, (I + S), clip=False)

        prj[ii, ...] = rotated_x.sum(axis=0)
    return prj


def _radon_rot_sp(x: NDArray, angles_rad: NDArray) -> NDArray:
    """
    Compute the Radon transform using scipy's rotate function.

    Parameters
    ----------
    x : NDArray
        Input image.
    angles_rad : NDArray
        Angles in radians.

    Returns
    -------
    NDArray
        Radon transform of the input image.
    """
    prj = np.empty((len(angles_rad), x.shape[-1]), dtype=np.float32)
    for ii, a in enumerate(np.rad2deg(angles_rad)):
        prj[ii, ...] = spndi.rotate(x, -a, order=1, reshape=False).sum(axis=0)
    return prj


@pytest.fixture(scope="function")
def base_test_data() -> tuple[NDArray, NDArray, float]:
    """
    Fixture to provide base test data.

    Returns
    -------
    tuple[NDArray, NDArray, float]
        Phantom image, angles in radians, and shift value in pixels.
    """
    ph = skd.shepp_logan_phantom()
    angles_deg = np.arange(0, 180)
    angles_rad = angles_deg / 180 * np.pi
    shift = 20
    return ph, angles_rad, shift


def _test_centered_sinogram(ph: NDArray, angles_rad: NDArray, ref_function: Callable) -> None:
    """
    Test the centered sinogram.

    Parameters
    ----------
    ph : NDArray
        Phantom image.
    angles_rad : NDArray
        Angles in radians.
    ref_function : callable
        Reference function to compute the Radon transform.
    """
    debug = False

    ph_added = ph + 0.1 * cct.processing.circular_mask(ph.shape)

    prj_ref = ref_function(ph_added, angles_rad)

    with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, rot_axis_shift_pix=None) as prj:
        prj_cmp = prj(ph_added)

        assert prj.projector_backend.__class__.__name__ in [
            "ProjectorBackendASTRA",
            "ProjectorBackendDirectASTRA",
        ], "The projector backend is not Astra's or Direct Astra's. This function will not work as intended"

    rel_diff = (prj_ref - prj_cmp) / prj_ref.max()

    if debug:
        print(np.max(np.abs(prj_ref - prj_cmp)))
        print(np.max(np.abs(rel_diff)))
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(prj_ref)
        axs[1].imshow(prj_cmp)
        axs[2].imshow(prj_ref - prj_cmp)
        fig.tight_layout()
        plt.show()

    assert np.all(np.isclose(rel_diff, 0, atol=0.015)), "Reference radon transform and astra-toolbox do not match"


@pytest.mark.skipif(not cct.projectors.astra_available, reason="astra-toolbox not available")
def test_centered_sinogram_sk(base_test_data: tuple[NDArray, NDArray, float]) -> None:
    """
    Test the centered sinogram using skimage's warp function.

    Parameters
    ----------
    base_test_data : tuple[NDArray, NDArray, float]
        Base test data containing:
        - Phantom image (NDArray)
        - Angles in radians (NDArray)
        - Shift value in pixels (float)
    """
    ph, angles_rad, _ = base_test_data
    _test_centered_sinogram(ph, angles_rad, _radon_rot_sk_w)


@pytest.mark.skipif(not cct.projectors.astra_available, reason="astra-toolbox not available")
def test_centered_sinogram_sp(base_test_data: tuple[NDArray, NDArray, float]) -> None:
    """
    Test the centered sinogram using scipy's rotate function.

    Parameters
    ----------
    base_test_data : tuple[NDArray, NDArray, float]
        Base test data containing:
        - Phantom image (NDArray)
        - Angles in radians (NDArray)
        - Shift value in pixels (float)
    """
    ph, angles_rad, _ = base_test_data
    _test_centered_sinogram(ph, angles_rad, _radon_rot_sp)


@pytest.mark.skipif(not cct.projectors.astra_available, reason="astra-toolbox not available")
def test_shifted_sinogram(base_test_data: tuple[NDArray, NDArray, float]) -> None:
    """
    Test the shifted sinogram.

    Parameters
    ----------
    base_test_data : tuple[NDArray, NDArray, float]
        Base test data containing:
        - Phantom image (NDArray)
        - Angles in radians (NDArray)
        - Shift value in pixels (float)
    """
    ph, angles_rad, shift = base_test_data
    debug = False

    ph_added = ph + 0.1 * cct.processing.circular_mask(ph.shape)

    prj_ref = _radon_rot_sk_w(ph_added, angles_rad, shift)

    with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, rot_axis_shift_pix=shift) as A:
        prj_cmp = A(ph_added)

    rel_diff = (prj_ref - prj_cmp) / prj_ref.max()

    if debug:
        print(np.max(np.abs(prj_ref - prj_cmp)))
        print(np.max(np.abs(rel_diff)))
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(prj_ref)
        axs[1].imshow(prj_cmp)
        axs[2].imshow(prj_ref - prj_cmp)
        fig.tight_layout()
        plt.show()

    assert np.all(np.isclose(rel_diff, 0, atol=0.015)), "Reference radon transform and astra-toolbox do not match"


@pytest.mark.skipif(not cct.projectors.astra_available, reason="astra-toolbox not available")
def test_astra_backends(base_test_data: tuple[NDArray, NDArray, float]) -> None:
    """
    Test the astra backends.

    Parameters
    ----------
    base_test_data : tuple[NDArray, NDArray, float]
        Base test data containing:
        - Phantom image (NDArray)
        - Angles in radians (NDArray)
        - Shift value in pixels (float)
    """
    ph, angles_rad, shift = base_test_data
    debug = False

    projector_legacy = backends.ProjectorBackendASTRA()
    projector_direct = backends.ProjectorBackendDirectASTRA()

    solver = cct.solvers.SIRT()
    num_iterations = 25

    prj_ref = _radon_rot_sk_w(ph, angles_rad, shift)

    with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, rot_axis_shift_pix=shift, backend=projector_legacy) as A:
        rec_vol_leg, _ = solver(A, prj_ref, iterations=num_iterations)

    with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, rot_axis_shift_pix=shift, backend=projector_direct) as A:
        rec_vol_dir, _ = solver(A, prj_ref, iterations=num_iterations)

    diff_rec = rec_vol_leg - rec_vol_dir
    rel_diff = diff_rec / rec_vol_leg.max()

    if debug:
        print(np.max(np.abs(diff_rec)))
        print(np.max(np.abs(rel_diff)))
        fig, ax = plt.subplots(1, 3, figsize=[9, 3])
        ax[0].imshow(rec_vol_leg)
        ax[1].imshow(rec_vol_dir)
        ax[2].imshow(diff_rec)
        fig.tight_layout()
        plt.show()

    assert np.allclose(rel_diff, 0.0, atol=0.0002), "Legacy and direct astra-toolbox projectors do not match"
