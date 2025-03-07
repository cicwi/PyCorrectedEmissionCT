#!/usr/bin/env python3
"""
Test `corrct.projectors` package.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import pytest
import skimage.data as skd
import skimage.transform as skt
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


def test_set_detector_tilt():
    """
    Test ProjectionGeometry.set_detector_tilt method with various scenarios of angles, axes and source tilting.

    Checks if detector u and v vectors are correctly tilted after calling the function. Also checks if the source is
    correctly tilted when specified.
    """
    # Setup: Create a default geometry
    prj_geom_orig = cct.models.get_prj_geom_parallel(geom_type="3d")

    prj_geom_test = prj_geom_orig.copy()
    # Test 1: Single tilt along the default axis (0, 1, 0)
    angles_t_rad = np.deg2rad(30)
    prj_geom_test.set_detector_tilt(angles_t_rad)

    rotation = Rotation.from_rotvec(angles_t_rad * np.array([0, 1, 0]))
    expected_u = rotation.apply(prj_geom_orig.det_u_xyz)
    assert np.allclose(prj_geom_test.det_u_xyz, expected_u), "Failed: Single tilt along default axis"

    prj_geom_test = prj_geom_orig.copy()
    # Test 2: Single tilt along a custom axis
    angles_t_rad = np.deg2rad(-45)
    tilt_axis = (1, 0, 0)
    prj_geom_test.set_detector_tilt(angles_t_rad, tilt_axis=tilt_axis)

    rotation = Rotation.from_rotvec(angles_t_rad * np.array(tilt_axis))
    expected_v = rotation.apply(prj_geom_orig.det_v_xyz)
    assert np.allclose(prj_geom_test.det_v_xyz, expected_v), "Failed: Single tilt along custom axis"

    prj_geom_test = prj_geom_orig.copy()
    # Test 3: Multiple tilts along different axes
    angles_t_rad = np.deg2rad([45, -60])
    tilt_axis = [(1, 0, 0), (0, 1, 0)]
    prj_geom_test.set_detector_tilt(angles_t_rad, tilt_axis=tilt_axis)

    expected_u = prj_geom_orig.det_u_xyz
    for angle, axis in zip(angles_t_rad, tilt_axis):
        rotations = Rotation.from_rotvec(angle * np.array(axis))  # type: ignore
        expected_u = rotations.apply(expected_u)

    assert np.allclose(prj_geom_test.det_u_xyz, expected_u), "Failed: Multiple tilts along different axes"

    prj_geom_test = prj_geom_orig.copy()
    # Test 4: Tilt the source as well
    angles_t_rad = np.deg2rad(-90)
    tilt_source = True
    prj_geom_test.set_detector_tilt(angles_t_rad, tilt_source=tilt_source)

    rotation = Rotation.from_rotvec(angles_t_rad * np.array([0, 1, 0]))
    expected_src = rotation.apply(prj_geom_orig.src_pos_xyz)
    assert np.allclose(prj_geom_test.src_pos_xyz, expected_src), "Failed: Tilting the source"


def test_fit_image_rotation_and_scale():
    """
    Test the fit_image_rotation_and_scale function from the cct.alignment.fitting module.

    This function should return a tuple (angle, scale) that represents the rotation angle
    and scaling factor required to align an input image with a reference image. The
    function is tested using various combinations of identical images, simple rotations,
    scaling, and both rotation and scaling. It is also tested with images of different
    shapes, in which case it should raise a ValueError.
    """
    # Load the camera image from scikit-image's data package
    img: NDArray = skd.camera() / 255

    # Test with two identical images should return 0 degrees and 1 scale
    result = cct.alignment.fitting.fit_image_rotation_and_scale(img, img)
    assert np.allclose(result, (0.0, 1.0))

    # Test with a simple rotation of an image by 10 degrees
    img_10 = skt.rotate(img, angle=10.0)
    result = cct.alignment.fitting.fit_image_rotation_and_scale(img, img_10)
    assert np.allclose(result, (10.0, 1.0), rtol=0.01)

    # Test with a simple rotation of an image by 45 degrees
    img_45 = skt.rotate(img, angle=45.0)
    result = cct.alignment.fitting.fit_image_rotation_and_scale(img, img_45)
    assert np.allclose(result, (45.0, 1.0), rtol=0.01)

    # Test with a rotation of an image by 360 degrees
    img_360 = np.copy(img)
    result = cct.alignment.fitting.fit_image_rotation_and_scale(img, img_360)
    assert np.allclose(result, (0.0, 1.0), rtol=0.01)

    # Test with a scaling of an image by 10% using scikit-image's rescale function
    img_scaled: NDArray = skt.rescale(img, scale=1.1)
    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
    img_scaled_cropped = img_scaled[
        center_y - img.shape[0] // 2 : center_y + img.shape[0] // 2,
        center_x - img.shape[1] // 2 : center_x + img.shape[1] // 2,
    ]
    result = cct.alignment.fitting.fit_image_rotation_and_scale(img, img_scaled_cropped)
    assert np.allclose(result, (0.0, 1.1), rtol=0.01, atol=0.05)

    # Test with a rotation and scaling of an image
    img_scaled = skt.rescale(img, scale=1.1)
    img_scaled_cropped = img_scaled[
        center_y - img.shape[0] // 2 : center_y + img.shape[0] // 2,
        center_x - img.shape[1] // 2 : center_x + img.shape[1] // 2,
    ]
    img_10_scaled = skt.rotate(img_scaled_cropped, angle=10.0)
    result = cct.alignment.fitting.fit_image_rotation_and_scale(img, img_10_scaled)
    assert np.allclose(result, (10.0, 1.1), rtol=0.01)

    # Test with images of different shapes should raise a ValueError
    try:
        cct.alignment.fitting.fit_image_rotation_and_scale(img, np.random.rand(10, 20))
        assert False, "Expected a ValueError"
    except ValueError:
        pass


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
    assert np.allclose(shifts_u_pre, theo_shifts, atol=tolerance), "Theoretical and computed shifts do not match"


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

    phantom = np.fmax(phantom_nuc3d - background_nuc3d, 0.0)
    phantom /= phantom.max()

    vol_geom = cct.models.get_vol_geom_from_volume(phantom)
    angles_rad = _get_angles(angles_range=360)
    theo_shifts_u = _get_shifts(len(angles_rad), theo_rot_axis=theo_rot_axis) / 5.0
    theo_rot_axis = theo_rot_axis / 5.0
    theo_shifts_v = _get_shifts(len(angles_rad)) / 20.0
    theo_shifts_vu = cct.models.combine_shifts_vu(theo_shifts_v, theo_shifts_u)
    prj_geom = cct.models.get_prj_geom_parallel()
    prj_geom.set_detector_shifts_vu(theo_shifts_vu)

    with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, prj_geom=prj_geom) as prj:
        prj_data = prj(phantom)

    if debug:
        fig, axs = plt.subplots(1, 2, figsize=[8, 2.5])
        axs[0].imshow(phantom_nuc3d[phantom.shape[0] // 2])
        axs[0].set_title("Phantom + background")
        axs[1].imshow(background_nuc3d[phantom.shape[0] // 2])
        axs[1].set_title("Background")
        fig.tight_layout()

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
    shifts_v_pre = align_pre.fit_v(use_derivative=False)
    shifts_u_pre, cor_pre = align_pre.fit_u(background=5.0)
    shifts_vu_pre = cct.models.combine_shifts_vu(shifts_v_pre, shifts_u_pre)
    prj_geom = cct.models.get_prj_geom_parallel()
    prj_geom.set_detector_shifts_vu(shifts_vu_pre, cor_pos_u=cor_pre)

    solver_opts = dict(lower_limit=0.0)

    solver = cct.solvers.SIRT()
    with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, theo_shifts_vu) as prj:
        rec_noise_theo, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)
    with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, prj_geom=prj_geom) as prj:
        rec_noise_pre, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)
    if debug:
        with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, theo_rot_axis) as prj:
            rec_noise_theocor, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)
        with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, cor_pre) as prj:
            rec_noise_precor, _ = solver(prj, prj_data, iterations=ITERATIONS, **solver_opts)

    com_ph_yx = cct.processing.post.com(phantom)
    prj_geom = cct.models.get_prj_geom_parallel()
    recenter = cct.alignment.RecenterVolume(prj_geom, angles_rad)

    theo_shifts_vu = recenter.to_com(theo_shifts_vu, rec_noise_theo, com_ph_yx)
    shifts_vu_pre = recenter.to_com(shifts_vu_pre, rec_noise_pre, com_ph_yx)

    # Re-centering the reconstructions on the phantom's center-of-mass -> moving the shifts accordingly
    # shifts_vu_pre = recenter.to_com(shifts_vu_pre, rec_noise_xc, com_ph_yx)
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
        print(f"{np.max(np.abs(theo_shifts_vu[0] - shifts_vu_pre[0])) = }")
        print(f"{np.max(np.abs(theo_shifts_vu[1] - shifts_vu_pre[1])) = }")

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
        axs[1].plot(shifts_vu_pre[1] - theo_shifts_vu[1], label="Diff")
        axs[1].grid()
        axs[1].legend()
        axs[1].set_title("Horizontal shifts")
        fig.tight_layout()
        plt.show()

    tolerance = 0.8 if add_noise else 0.25
    assert np.allclose(
        shifts_vu_pre[0], theo_shifts_vu[0], atol=tolerance
    ), "Theoretical and computed vertical shifts do not match"
    tolerance = 0.8 if add_noise else 0.5
    assert np.allclose(
        shifts_vu_pre[1], theo_shifts_vu[1], atol=tolerance
    ), "Theoretical and computed horizontal shifts do not match"
