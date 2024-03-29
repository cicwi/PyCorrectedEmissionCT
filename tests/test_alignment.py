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


phantom_sl2d = skd.shepp_logan_phantom()
phantom_nuc3d, background_nuc3d, _ = cct.testing.create_phantom_nuclei3d(FoV_size=100)

ITERATIONS = 100

NUM_PHOTONS = 1e1
BACKGROUND_AVG = 2e0
ADD_POISSON = True


def _get_angles(angles_num: int = 41, angles_start: float = 0.0, angles_range: float = 180.0) -> NDArray:
    angles_deg = np.linspace(angles_start, angles_start + angles_range, angles_num, endpoint=True)
    return np.deg2rad(angles_deg)


def _get_shifts(angles_num: int, theo_rot_axis: float = 0.0) -> NDArray:
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

    vol_geom = cct.models.get_vol_geom_from_volume(phantom_sl2d)
    angles_rad = _get_angles()
    theo_shifts = _get_shifts(len(angles_rad), theo_rot_axis=theo_rot_axis)

    with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, theo_shifts) as prj:
        prj_data = prj(phantom_sl2d)

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

    com_ph_yx = cct.processing.post.com(phantom_sl2d)
    prj_geom = cct.models.ProjectionGeometry.get_default_parallel(geom_type="2d")
    recenter = cct.alignment.RecenterVolume(prj_geom, angles_rad)

    # Re-centering the reconstructions on the phantom's center-of-mass -> moving the shifts accordingly
    shifts_u_pre = recenter.to_com(shifts_u_pre, rec_noise_pre, com_ph_yx)

    if debug:
        theo_rot_axis_per_angle = np.ones_like(angles_rad) * theo_rot_axis
        theo_rot_axis_per_angle = recenter.to_com(theo_rot_axis_per_angle, rec_noise_theocor, com_ph_yx)
        cor = recenter.to_com(cor, rec_noise_precor, com_ph_yx)

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


@pytest.mark.skipif(not cct.projectors.astra_available, reason="astra-toolbox not available")
@pytest.mark.parametrize("add_noise", [(False,), (True,)])
def test_pre_alignment_3d(add_noise: bool, theo_rot_axis: float = -1.25):
    """Test pre-alignment routines.

    Parameters
    ----------
    add_noise : bool
        Whether to add noise
    theo_rot_axis : float, optional
        The theoretical rotation axis position, by default -1.25
    """
    debug = False

    phantom = phantom_nuc3d - background_nuc3d
    phantom /= phantom.max()

    vol_geom = cct.models.get_vol_geom_from_volume(phantom)
    angles_rad = _get_angles(angles_range=360)
    theo_shifts_u = _get_shifts(len(angles_rad), theo_rot_axis=theo_rot_axis) / 5
    theo_shifts_v = _get_shifts(len(angles_rad)) / 20
    theo_shifts_vu = cct.models.combine_shifts_vu(theo_shifts_v, theo_shifts_u)
    prj_geom = cct.models.get_prj_geom_parallel()
    prj_geom.set_detector_shifts_vu(theo_shifts_vu)

    with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, prj_geom=prj_geom) as prj:
        prj_data = prj(phantom)

    if debug:
        fig, axs = plt.subplots(1, 2, figsize=[8, 2.5])
        axs[0].imshow(phantom[phantom.shape[0] // 2])
        axs[0].set_title("Phantom")
        axs[1].imshow(prj_data[phantom.shape[0] // 2])
        axs[1].set_title("Sinogram")
        fig.tight_layout()

    if add_noise:
        prj_data = _generate_noise(prj_data)

    # Setting up the pre-alignment routine
    align_pre = cct.alignment.DetectorShiftsPRE(prj_data, angles_rad, verbose=debug)

    # Running pre-alignment
    shifts_v_pre = align_pre.fit_v()
    shifts_u_pre, cor_pre = align_pre.fit_u()
    shifts_vu_pre = cct.models.combine_shifts_vu(shifts_v_pre, shifts_u_pre)
    prj_geom = cct.models.get_prj_geom_parallel()
    prj_geom.set_detector_shifts_vu(shifts_vu_pre, cor_pos_u=cor_pre)

    solver_opts = dict(lower_limit=0.0)

    solver = cct.solvers.SIRT()
    if debug:
        with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, theo_rot_axis) as prj:
            rec_noise_theocor, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)
        with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, cor_pre) as prj:
            rec_noise_precor, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)
    with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, prj_geom=prj_geom) as prj:
        rec_noise_xc, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)

    com_ph_yx = cct.processing.post.com(phantom)
    prj_geom = cct.models.get_prj_geom_parallel()
    recenter = cct.alignment.RecenterVolume(prj_geom, angles_rad)

    # Re-centering the reconstructions on the phantom's center-of-mass -> moving the shifts accordingly
    shifts_vu_pre = recenter.to_com(shifts_vu_pre, rec_noise_xc, com_ph_yx)
    if debug:
        theo_rot_axis_per_angle = cct.models.combine_shifts_vu(
            np.zeros_like(angles_rad), np.ones_like(angles_rad) * theo_rot_axis
        )
        theo_rot_axis_per_angle = recenter.to_com(theo_rot_axis_per_angle, rec_noise_theocor, com_ph_yx)
        cor_pre = cct.models.combine_shifts_vu(np.zeros_like(angles_rad), np.ones_like(angles_rad) * cor_pre)
        cor_pre = recenter.to_com(cor_pre, rec_noise_precor, com_ph_yx)

        print(f"{theo_shifts_vu = }")
        print(f"{shifts_vu_pre = }")
        print(f"{theo_shifts_vu - shifts_vu_pre = }")
        print(f"{np.max(np.abs(theo_shifts_vu - shifts_vu_pre)) = }")

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[8, 2.5])
        axs[0].plot(theo_shifts_vu[0], label="Ground truth")
        axs[0].plot(shifts_vu_pre[0], label="Pre-alignment shifts")
        axs[0].grid()
        axs[0].legend()
        axs[0].set_title("Vertical shifts")
        axs[1].plot(theo_shifts_vu[1], label="Ground truth")
        axs[1].plot(theo_rot_axis_per_angle[1], label="Center-of-rotation (theoretical)")
        axs[1].plot(cor_pre[1], label="Center-of-rotation (computed)")
        axs[1].plot(shifts_vu_pre[1], label="Pre-alignment shifts")
        axs[1].grid()
        axs[1].legend()
        axs[1].set_title("Horizontal shifts")
        fig.tight_layout()
        plt.show()

    tolerance = 0.8 if add_noise else 0.3
    assert np.all(np.isclose(shifts_vu_pre, theo_shifts_vu, atol=tolerance)), "Theoretical and computed shifts do not match"
