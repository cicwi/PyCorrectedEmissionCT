# test_models.py

import numpy as np
import pytest
from src.corrct.models import (
    ProjectionGeometry,
    VolumeGeometry,
    get_prj_geom_parallel,
    get_prj_geom_cone,
    get_vol_geom_from_data,
    get_vol_geom_from_volume,
    combine_shifts_vu,
    get_rot_axis_dir,
)


def test_projection_geometry_initialization():
    """
    Test the initialization of the ProjectionGeometry class.
    """
    geom = ProjectionGeometry(
        geom_type="parallel3d",
        src_pos_xyz=np.array([0.0, -1.0, 0.0]),
        det_pos_xyz=np.zeros(3),
        det_u_xyz=np.array([1.0, 0.0, 0.0]),
        det_v_xyz=np.array([0.0, 0.0, 1.0]),
        rot_dir_xyz=np.array([0.0, 0.0, -1.0]),
    )
    assert geom.geom_type == "parallel3d"
    assert np.array_equal(geom.src_pos_xyz, np.array([[0.0, -1.0, 0.0]]))
    assert np.array_equal(geom.det_pos_xyz, np.array([[0.0, 0.0, 0.0]]))
    assert np.array_equal(geom.det_u_xyz, np.array([[1.0, 0.0, 0.0]]))
    assert np.array_equal(geom.det_v_xyz, np.array([[0.0, 0.0, 1.0]]))
    assert np.array_equal(geom.rot_dir_xyz, np.array([[0.0, 0.0, -1.0]]))


def test_volume_geometry_initialization():
    """
    Test the initialization of the VolumeGeometry class.
    """
    vol_geom = VolumeGeometry(_vol_shape_xyz=np.array([100, 100, 50]), vox_size=1.0)
    assert np.array_equal(vol_geom.shape_xyz, np.array([100, 100, 50]))
    assert vol_geom.vox_size == 1.0


def test_get_prj_geom_parallel():
    """
    Test the generation of a parallel beam geometry.
    """
    geom = get_prj_geom_parallel(geom_type="3d", rot_axis_shift_pix=None, rot_axis_dir="clockwise")
    assert geom.geom_type == "parallel3d"
    assert np.array_equal(geom.src_pos_xyz, np.array([[0.0, -1.0, 0.0]]))
    assert np.array_equal(geom.det_pos_xyz, np.array([[0.0, 0.0, 0.0]]))
    assert np.array_equal(geom.det_u_xyz, np.array([[1.0, 0.0, 0.0]]))
    assert np.array_equal(geom.det_v_xyz, np.array([[0.0, 0.0, 1.0]]))
    assert np.array_equal(geom.rot_dir_xyz, np.array([[0.0, 0.0, -1.0]]))


def test_get_prj_geom_cone():
    """
    Test the generation of a cone beam geometry.
    """
    geom = get_prj_geom_cone(src_to_sam_dist=100.0, rot_axis_shift_pix=None, rot_axis_dir="clockwise")
    assert geom.geom_type == "cone"
    assert np.array_equal(geom.src_pos_xyz, np.array([[0.0, -100.0, 0.0]]))
    assert np.array_equal(geom.det_pos_xyz, np.array([[0.0, 0.0, 0.0]]))
    assert np.array_equal(geom.det_u_xyz, np.array([[1.0, 0.0, 0.0]]))
    assert np.array_equal(geom.det_v_xyz, np.array([[0.0, 0.0, 1.0]]))
    assert np.array_equal(geom.rot_dir_xyz, np.array([[0.0, 0.0, -1.0]]))


def test_get_vol_geom_from_data():
    """
    Test the generation of a volume geometry from data.
    """
    data = np.random.rand(10, 20, 30)
    vol_geom = get_vol_geom_from_data(data, data_format="dvwu")
    assert np.array_equal(vol_geom.shape_xyz, np.array([30, 30, 10]))


def test_get_vol_geom_from_volume():
    """
    Test the generation of a volume geometry from a volume.
    """
    volume = np.random.rand(10, 20, 30)
    vol_geom = get_vol_geom_from_volume(volume)
    # XXX: There might be a mess up here between X and Y!!
    assert np.array_equal(vol_geom.shape_xyz, np.array([20, 30, 10]))


def test_combine_shifts_vu():
    """
    Test the combination of vertical and horizontal shifts.
    """
    shifts_v = np.array([1, 2, 3])
    shifts_u = np.array([4, 5, 6])
    combined_shifts = combine_shifts_vu(shifts_v, shifts_u)
    assert np.array_equal(combined_shifts, np.array([[1, 2, 3], [4, 5, 6]]))

    # Test with invalid shapes
    with pytest.raises(ValueError, match="Expected 1-dimensional array for vertical shifts"):
        combine_shifts_vu(np.array([[1, 2], [3, 4]]), shifts_u)

    with pytest.raises(ValueError, match="Expected 1-dimensional array for horizontal shifts"):
        combine_shifts_vu(shifts_v, np.array([[4, 5], [6, 7]]))

    with pytest.raises(ValueError, match="Number of vertical shifts"):
        combine_shifts_vu(np.array([1, 2]), shifts_u)


def test_get_rot_axis_dir():
    """
    Test the processing of the rotation axis direction.
    """
    rot_dir = get_rot_axis_dir("clockwise")
    assert np.array_equal(rot_dir, np.array([0.0, 0.0, -1.0]))

    rot_dir = get_rot_axis_dir("counter-clockwise")
    assert np.array_equal(rot_dir, np.array([0.0, 0.0, 1.0]))

    rot_dir = get_rot_axis_dir(np.array([1.0, 0.0, 0.0]))
    assert np.array_equal(rot_dir, np.array([1.0, 0.0, 0.0]))

    # Test with invalid direction
    with pytest.raises(ValueError, match="Rotation axis direction invalid_dir not allowed"):
        get_rot_axis_dir("invalid_dir")
