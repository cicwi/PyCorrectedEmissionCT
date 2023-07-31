#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test `corrct.projectors` package.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import scipy.ndimage as spndi

import skimage.data as skd
import skimage.transform as skt

import matplotlib.pyplot as plt

import pytest

import corrct as cct
from corrct import _projector_backends as backends


eps = np.finfo(np.float32).eps


def _radon_rot_sk_w(x, angles_rad, shift=None):
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


def _radon_rot_sp(x, angles_rad):
    prj = np.empty((len(angles_rad), x.shape[-1]), dtype=np.float32)
    for ii, a in enumerate(np.rad2deg(angles_rad)):
        prj[ii, ...] = spndi.rotate(x, -a, order=1, reshape=False).sum(axis=0)
    return prj


@pytest.fixture(scope="class")
def bootstrap_base(request):
    cls = request.cls
    cls.ph = skd.shepp_logan_phantom()

    cls.angles_deg = np.arange(0, 180)
    cls.angles_rad = cls.angles_deg / 180 * np.pi

    cls.shift = 20


@pytest.mark.skipif(not cct.projectors.astra_available, reason="astra-toolbox not available")
@pytest.mark.usefixtures("bootstrap_base")
class TestProjectors:
    """Tests for the projectors in `corrct.projectors` package."""

    def _test_centered_sinogram(self, ref_function):
        debug = False

        ph_added = self.ph + 0.1 * cct.processing.circular_mask(self.ph.shape)

        prj_ref = ref_function(ph_added, self.angles_rad)

        with cct.projectors.ProjectorUncorrected(self.ph.shape, self.angles_rad, rot_axis_shift_pix=None) as A:
            prj_cmp = A(ph_added)

        rel_diff = (prj_ref - prj_cmp) / prj_ref.max()

        if debug:
            print(np.max(np.abs(prj_ref - prj_cmp)))
            print(np.max(np.abs(rel_diff)))
            f, ax = plt.subplots(3, 1)
            ax[0].imshow(prj_ref)
            ax[1].imshow(prj_cmp)
            ax[2].imshow(prj_ref - prj_cmp)
            plt.show()

        assert np.all(np.isclose(rel_diff, 0, atol=0.015)), "Reference radon transform and astra-toolbox do not match"

    def test_centered_sinogram_sk(self):
        self._test_centered_sinogram(_radon_rot_sk_w)

    def test_centered_sinogram_sp(self):
        self._test_centered_sinogram(_radon_rot_sp)

    def test_shifted_sinogram(self):
        debug = False

        ph_added = self.ph + 0.1 * cct.processing.circular_mask(self.ph.shape)

        prj_ref = _radon_rot_sk_w(ph_added, self.angles_rad, self.shift)

        with cct.projectors.ProjectorUncorrected(self.ph.shape, self.angles_rad, rot_axis_shift_pix=self.shift) as A:
            prj_cmp = A(ph_added)

        rel_diff = (prj_ref - prj_cmp) / prj_ref.max()

        if debug:
            print(np.max(np.abs(prj_ref - prj_cmp)))
            print(np.max(np.abs(rel_diff)))
            f, ax = plt.subplots(3, 1)
            ax[0].imshow(prj_ref)
            ax[1].imshow(prj_cmp)
            ax[2].imshow(prj_ref - prj_cmp)
            plt.show()

        assert np.all(np.isclose(rel_diff, 0, atol=0.015)), "Reference radon transform and astra-toolbox do not match"

    def test_astra_backends(self):
        debug = False

        projector_legacy = backends.ProjectorBackendASTRA()
        projector_direct = backends.ProjectorBackendDirectASTRA()

        solver = cct.solvers.SIRT()
        num_iterations = 25

        prj_ref = _radon_rot_sk_w(self.ph, self.angles_rad, self.shift)

        with cct.projectors.ProjectorUncorrected(
            self.ph.shape, self.angles_rad, rot_axis_shift_pix=self.shift, backend=projector_legacy
        ) as A:
            rec_vol_leg, _ = solver(A, prj_ref, iterations=num_iterations)

        with cct.projectors.ProjectorUncorrected(
            self.ph.shape, self.angles_rad, rot_axis_shift_pix=self.shift, backend=projector_direct
        ) as A:
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

        assert np.all(np.isclose(rel_diff, 0, atol=0.0002)), "Legacy and direct astra-toolbox projectors do not match"
