import numpy as np
import pytest
from src.corrct.struct_illum import (
    reorder_masks,
    decompose_qr_masks,
    estimate_resolution,
    MaskCollection,
    MaskGeneratorPoint,
    MaskGeneratorBernoulli,
    MaskGeneratorHalfGaussian,
    MaskGeneratorMURA,
    ProjectorGhostImaging,
)


def test_reorder_masks():
    """
    Test the reorder_masks function.

    This function tests the reordering of masks and buckets based on a given shift.
    """
    masks = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    buckets = np.array([1, 2])
    shift = 1
    reordered_masks, reordered_buckets = reorder_masks(masks, buckets, shift)
    assert np.array_equal(reordered_masks, np.array([[[5, 6], [7, 8]], [[1, 2], [3, 4]]]))
    assert np.array_equal(reordered_buckets, np.array([2, 1]))


def test_decompose_qr_masks():
    """
    Test the decompose_qr_masks function.

    This function tests the decomposition of masks into Q and R1t matrices.
    """
    masks = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    Q, R1t = decompose_qr_masks(masks)
    assert Q.shape == masks.shape
    assert R1t.shape == (masks.shape[0], masks.shape[0])


def test_mask_collection():
    """
    Test the MaskCollection class.

    This function tests the creation and properties of the MaskCollection class.
    """
    masks_enc = np.random.rand(2, 2, 2)
    masks_dec = np.random.rand(2, 2, 2)
    mc = MaskCollection(masks_enc, masks_dec)
    assert mc.shape_fov == (2, 2)
    assert mc.shape_shifts == (2,)
    assert mc.num_buckets == 2
    assert mc.num_pixels == 4


def test_mask_generator_point():
    """
    Test the MaskGeneratorPoint class.

    This function tests the creation and properties of the MaskGeneratorPoint class.
    """
    mg = MaskGeneratorPoint(fov_size_mm=10, req_res_mm=1)
    assert mg.shape_fov[0] == 10
    assert mg.shape_mask[0] == 1
    assert mg.shape_shifts[0] == 10


def test_mask_generator_bernoulli():
    """
    Test the MaskGeneratorBernoulli class.

    This function tests the creation and properties of the MaskGeneratorBernoulli class.
    """
    mg = MaskGeneratorBernoulli(fov_size_mm=10, req_res_mm=1)
    assert mg.shape_fov[0] == 10
    assert mg.shape_mask[0] == 10
    assert mg.shape_shifts[0] == 12


def test_mask_generator_half_gaussian():
    """
    Test the MaskGeneratorHalfGaussian class.

    This function tests the creation and properties of the MaskGeneratorHalfGaussian class.
    """
    mg = MaskGeneratorHalfGaussian(fov_size_mm=10, req_res_mm=1)
    assert mg.shape_fov[0] == 10
    assert mg.shape_mask[0] == 10
    assert mg.shape_shifts[0] == 12


def test_mask_generator_mura():
    """
    Test the MaskGeneratorMURA class.

    This function tests the creation and properties of the MaskGeneratorMURA class.
    """
    mg = MaskGeneratorMURA(fov_size_mm=10, req_res_mm=1)
    assert mg.shape_fov[0] == 13
    assert mg.shape_mask[0] == 13
    assert mg.shape_shifts[0] == 13


def test_projector_ghost_imaging():
    """
    Test the ProjectorGhostImaging class.

    This function tests the creation and properties of the ProjectorGhostImaging class,
    as well as its forward projection and least-squares reconstruction methods.
    """
    masks = np.random.rand(2, 2, 2)
    projector = ProjectorGhostImaging(masks)
    assert projector.mc.shape_fov == (2, 2)
    assert projector.mc.shape_shifts == (2,)
    assert projector.mc.num_buckets == 2
    assert projector.mc.num_pixels == 4

    image = np.random.rand(2, 2)
    bucket_vals = projector.fp(image)
    assert bucket_vals.shape == (2,)

    reconstructed_image = projector.fbp(bucket_vals)
    assert reconstructed_image.shape == (2, 2)


if __name__ == "__main__":
    pytest.main()
