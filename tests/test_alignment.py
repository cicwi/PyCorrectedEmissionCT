#!/usr/bin/env python3
"""
Test `corrct.projectors` package.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import skimage.data as skd
from numpy.typing import NDArray
import corrct as cct


phantom = skd.shepp_logan_phantom()

ITERATIONS = 100

NUM_PHOTONS = 1e1
BACKGROUND_AVG = 2e0
ADD_POISSON = True


def _get_angles(angles_num: int = 41) -> NDArray:
    angles_start = 0
    angles_range = 180

    angles_deg = np.linspace(angles_start, angles_start + angles_range, angles_num, endpoint=True)
    return np.deg2rad(angles_deg)


def _get_shifts(angles_num: int, theo_rot_axis: float = -1.25) -> NDArray:
    sigma_error = 0.25
    linear_error = -0.05
    exponential_error = 7.5

    random_drifts = sigma_error * np.random.randn(angles_num)
    linear_drifts = linear_error * np.linspace(-(angles_num - 1) / 2, (angles_num - 1) / 2, angles_num)
    exponential_drifts = exponential_error * np.exp(-np.linspace(0, 5, angles_num))

    theo_shifts = random_drifts + linear_drifts + exponential_drifts + theo_rot_axis
    return np.around(theo_shifts, decimals=2)


def _generate_noise(data_theo: NDArray) -> NDArray:
    data_noise, data_theo, background = cct.testing.add_noise(
        data_theo, num_photons=NUM_PHOTONS, add_poisson=ADD_POISSON, background_avg=BACKGROUND_AVG
    )

    return data_noise - background


def test_api_det_shifts_vu():
    """Test the API of the detector VU shifts manipulation in `models`."""
    num_angles: int = 41
    angles_deg = np.linspace(0, 180, num_angles)
    shifts_u = np.random.randn(num_angles)
    shifts_v = np.random.randn(num_angles)

    shifts_vu = cct.models.combine_shifts_vu(shifts_v, shifts_u)
    assert np.all(shifts_vu[-1] == shifts_u), "Wrong composition of shifts from `cct.models.combine_shifts_vu`"
    assert np.all(shifts_vu[-2] == shifts_v), "Wrong composition of shifts from `cct.models.combine_shifts_vu`"

    prj_geom = cct.models.get_prj_geom_parallel(geom_type="2d")
    prj_geom.set_detector_shifts_vu(shifts_u)

    msg = f"Wrong dimensions of `det_pos_xyz` in ProjectionGeometry. Should be 2-dimensional, of shape: {[num_angles, 3]}"
    assert prj_geom.det_pos_xyz.ndim == 2, msg
    assert np.all(np.array(prj_geom.det_pos_xyz.shape) == [num_angles, 3]), msg
    assert np.all(prj_geom.det_pos_xyz[:, -3] == shifts_u), "The shifts along U are wrong"

    rot_prj_geom = prj_geom.rotate(np.deg2rad(10))
    assert rot_prj_geom.det_pos_xyz.ndim == 2, msg
    assert np.all(np.array(rot_prj_geom.det_pos_xyz.shape) == [num_angles, 3]), msg

    rot_prj_geom = prj_geom.rotate(np.deg2rad(angles_deg))
    assert rot_prj_geom.det_pos_xyz.ndim == 2, msg
    assert np.all(np.array(rot_prj_geom.det_pos_xyz.shape) == [num_angles, 3]), msg

    prj_geom = cct.models.get_prj_geom_parallel(geom_type="3d")
    prj_geom.set_detector_shifts_vu(shifts_vu)

    msg = f"Detector position in ProjectionGeometry should be 2-dimensional, of shape: {[num_angles, 3]}"
    assert prj_geom.det_pos_xyz.ndim == 2, msg
    assert np.all(np.array(prj_geom.det_pos_xyz.shape) == [num_angles, 3]), msg
    assert np.all(prj_geom.det_pos_xyz[:, -3] == shifts_u), "The shifts along U are wrong"
    assert np.all(prj_geom.det_pos_xyz[:, -1] == shifts_v), "The shifts along v are wrong"

    rot_prj_geom = prj_geom.rotate(np.deg2rad(10))
    assert rot_prj_geom.det_pos_xyz.ndim == 2, msg
    assert np.all(np.array(rot_prj_geom.det_pos_xyz.shape) == [num_angles, 3]), msg

    rot_prj_geom = prj_geom.rotate(np.deg2rad(angles_deg))
    assert rot_prj_geom.det_pos_xyz.ndim == 2, msg
    assert np.all(np.array(rot_prj_geom.det_pos_xyz.shape) == [num_angles, 3]), msg

    prj_geom = cct.models.get_prj_geom_parallel(geom_type="3d")
    prj_geom.set_detector_shifts_vu(shifts_vu)

    prj_geom.set_detector_tilt(np.pi / 2)

    assert np.all(np.isclose(prj_geom.det_pos_xyz[:, -1], -shifts_u)), "The tilted shifts along U are wrong"
    assert np.all(np.isclose(prj_geom.det_pos_xyz[:, -3], shifts_v)), "The tilted shifts along v are wrong"

    prj_geom.set_detector_tilt(np.pi / 2)

    assert np.all(np.isclose(prj_geom.det_pos_xyz[:, -3], -shifts_u)), "The tilted shifts along U are wrong"
    assert np.all(np.isclose(prj_geom.det_pos_xyz[:, -1], -shifts_v)), "The tilted shifts along v are wrong"


@pytest.mark.skipif(not cct.projectors.astra_available, reason="astra-toolbox not available")
@pytest.mark.parametrize("add_noise", [(False,), (True,)])
def test_pre_alignment(add_noise: bool, theo_rot_axis: float = -1.25):
    """Test pre-alignment routines.

    Parameters
    ----------
    add_noise : bool
        Whether to add noise
    theo_rot_axis : float, optional
        The theoretical rotation axis position, by default -1.25
    """
    debug = False

    vol_geom = cct.models.get_vol_geom_from_volume(phantom)
    angles_rad = _get_angles()
    theo_shifts = _get_shifts(len(angles_rad), theo_rot_axis=theo_rot_axis)

    with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, theo_shifts) as prj:
        prj_data = prj(phantom)

    if add_noise:
        prj_data = _generate_noise(prj_data)

    # Setting up the pre-alignment routine
    align_pre = cct.alignment.DetectorShiftsPRE(prj_data, angles_rad, verbose=debug)

    # Running pre-alignment
    diffs_u_pre, cor = align_pre.fit_u()
    shifts_u_pre = cor + diffs_u_pre

    solver_opts = dict(lower_limit=0.0)

    solver = cct.solvers.SIRT()
    if debug:
        with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, theo_rot_axis) as prj:
            rec_noise_theocor, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)
        with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, cor) as prj:
            rec_noise_precor, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)
    with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, shifts_u_pre) as prj:
        rec_noise_pre, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)

    com_ph_yx = cct.processing.post.com(phantom)
    prj_geom = cct.models.ProjectionGeometry.get_default_parallel(geom_type="2d")
    recenter = cct.alignment.RecenterVolume(prj_geom, angles_rad)

    # Re-centering the reconstructions on the phantom's center-of-mass -> moving the shifts accordingly
    if debug:
        theo_rot_axis_per_angle = np.ones_like(angles_rad) * theo_rot_axis
        theo_rot_axis_per_angle = recenter.recenter_to(theo_rot_axis_per_angle, rec_noise_theocor, com_ph_yx)
        cor = recenter.recenter_to(cor, rec_noise_precor, com_ph_yx)
    shifts_u_pre = recenter.recenter_to(shifts_u_pre, rec_noise_pre, com_ph_yx)

    if debug:
        print(f"{theo_shifts = }")
        print(f"{shifts_u_pre = }")
        print(f"{theo_shifts - shifts_u_pre = }")
        print(f"{np.max(np.abs(theo_shifts - shifts_u_pre)) = }")

        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=[5, 2.5])
        axs.plot(theo_shifts, label="Ground truth")
        axs.plot(theo_rot_axis_per_angle, label="Center-of-rotation (theoretical)")
        axs.plot(cor, label="Center-of-rotation (computed)")
        axs.plot(shifts_u_pre, label="Pre-alignment shifts")
        axs.grid()
        axs.legend()
        fig.tight_layout()
        plt.show()

    tolerance = 0.8 if add_noise else 0.3
    assert np.all(np.isclose(shifts_u_pre, theo_shifts, atol=tolerance)), "Theoretical and computed shifts do not match"
