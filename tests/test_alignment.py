#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test `corrct.projectors` package.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

import pytest

import skimage.data as skd

import matplotlib.pyplot as plt

import corrct as cct


@pytest.fixture(scope="class")
def bootstrap_base(request):
    cls = request.cls
    cls.ph = skd.shepp_logan_phantom()

    cls.data_type = np.float32

    # Basic geometry parameters
    angles_start = 0
    angles_range = 180
    angles_num = 41
    cls.angles_deg = np.linspace(angles_start, angles_start + angles_range, angles_num, endpoint=True)
    cls.angles_rad = np.deg2rad(cls.angles_deg)

    cls.theo_rot_axis = -1.25

    # Randomized shift errors
    sigma_error = 0.25
    linear_error = -0.05
    exponential_error = 7.5

    random_drifts = sigma_error * np.random.randn(angles_num)
    linear_drifts = linear_error * np.linspace(-(angles_num - 1) / 2, (angles_num - 1) / 2, angles_num)
    exponential_drifts = exponential_error * np.exp(-np.linspace(0, 5, angles_num))

    cls.theo_shifts = random_drifts + linear_drifts + exponential_drifts + cls.theo_rot_axis
    cls.theo_shifts = np.around(cls.theo_shifts, decimals=2)

    cls.com_ph_yx = cct.processing.post.com(cls.ph)
    prj_geom = cct.models.ProjectionGeometry.get_default_parallel(geom_type="2d")
    cls.recenter = cct.alignment.RecenterVolume(prj_geom, cls.angles_rad)

    cls.shift = 20


@pytest.mark.skipif(not cct.projectors.astra_available, reason="astra-toolbox not available")
@pytest.mark.usefixtures("bootstrap_base")
class TestAlignment:
    """Tests for the alignment methods in `corrct.alignment` package."""

    @pytest.mark.parametrize("add_noise", [(False,), (True,)])
    def test_pre(self, add_noise: bool):
        debug = False

        vol_geom = cct.models.get_vol_geom_from_volume(self.ph)

        with cct.projectors.ProjectorUncorrected(vol_geom, self.angles_rad, self.theo_shifts) as A:
            data_theo = A(self.ph)

        if add_noise:
            # Adding noise
            NUM_PHOTONS = 1e1
            BACKGROUND_AVG = 2e0
            ADD_POISSON = True
            data_noise, data_theo, background = cct.testing.add_noise(
                data_theo, num_photons=NUM_PHOTONS, add_poisson=ADD_POISSON, background_avg=BACKGROUND_AVG
            )

            data_test = data_noise - background
        else:
            data_test = data_theo

        # Setting up the pre-alignment routine
        align_pre = cct.alignment.DetectorShiftsPRE(data_test, self.angles_rad, verbose=debug)

        # Runnning pre-alignment
        diffs_u_pre, cor = align_pre.fit_u()
        shifts_u_pre = cor + diffs_u_pre

        solver_opts = dict(lower_limit=0.0)
        ITERATIONS = 100

        solver = cct.solvers.SIRT()
        if debug:
            with cct.projectors.ProjectorUncorrected(vol_geom, self.angles_rad, self.theo_rot_axis) as A:
                rec_noise_theocor, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)
            with cct.projectors.ProjectorUncorrected(vol_geom, self.angles_rad, cor) as A:
                rec_noise_precor, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)
        with cct.projectors.ProjectorUncorrected(vol_geom, self.angles_rad, shifts_u_pre) as A:
            rec_noise_pre, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)

        # Recentering the reconstructions on the phantom's center-of-mass -> moving the shifts accordingly
        if debug:
            theo_rot_axis = self.recenter.recenter_to(self.theo_rot_axis, rec_noise_theocor, self.com_ph_yx)
            cor = self.recenter.recenter_to(cor, rec_noise_precor, self.com_ph_yx)
        shifts_u_pre = self.recenter.recenter_to(shifts_u_pre, rec_noise_pre, self.com_ph_yx)

        if debug:
            print(f"{self.theo_shifts = }")
            print(f"{shifts_u_pre = }")
            print(f"{self.theo_shifts - shifts_u_pre = }")
            print(f"{np.max(np.abs(self.theo_shifts - shifts_u_pre)) = }")

            fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=[5, 2.5])
            axs.plot(self.theo_shifts, label="Ground truth")
            axs.plot(theo_rot_axis, label="Center-of-rotation (theoretical)")
            axs.plot(cor, label="Center-of-rotation (computed)")
            axs.plot(shifts_u_pre, label="Pre-alignment shifts")
            axs.grid()
            axs.legend()
            fig.tight_layout()
            plt.show()

        tolerance = 0.8 if add_noise else 0.3
        assert np.all(
            np.isclose(shifts_u_pre, self.theo_shifts, atol=tolerance)
        ), "Theoretical and computed shifts do not match"
