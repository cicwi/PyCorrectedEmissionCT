# -*- coding: utf-8 -*-

"""Package for data pre-processing and post-processing."""

__author__ = """Nicola VIGANÒ"""
__email__ = "N.R.Vigano@cwi.nl"

from . import pre  # noqa: F401, F402
from . import post  # noqa: F401, F402
from . import misc  # noqa: F401, F402

import numpy as np

from typing import Sequence, Optional, Union
from numpy.typing import DTypeLike, NDArray


eps = np.finfo(np.float32).eps

NDArrayInt = NDArray[np.signedinteger]


def circular_mask(
    vol_shape_zxy: Union[Sequence[int], NDArrayInt],
    radius_offset: float = 0,
    coords_ball: Union[Sequence[int], NDArrayInt, None] = None,
    vol_origin_zxy: Optional[Sequence[float]] = None,
    mask_drop_off: str = "const",
    super_sampling: int = 1,
    dtype: DTypeLike = np.float32,
) -> NDArray:
    """
    Compute a circular mask for the reconstruction volume.

    Parameters
    ----------
    vol_shape_zxy : Sequence[int] | NDArrayInt
        The size of the volume.
    radius_offset : float, optional
        The offset with respect to the volume edge. The default is 0.
    coords_ball : Sequence[int] | NDArrayInt | None, optional
        The coordinates to consider for the non-masked region. The default is None.
    vol_origin_zxy : Optional[Sequence[float]], optional
        The origin of the coordinates in voxels. The default is None.
    mask_drop_off : str, optional
        The mask data type. Allowed types: "const" | "sinc". The default is "const".
    super_sampling : int, optional
        The pixel super sampling to be used for the mask. The default is 1.
    dtype : DTypeLike, optional
        The type of mask. The default is np.float32.

    Raises
    ------
    ValueError
        In case of unknown mask_drop_off value, or mismatching volume origin and shape.

    Returns
    -------
    NDArray
        The circular mask.
    """
    vol_shape_zxy_s = np.array(vol_shape_zxy, dtype=int) * super_sampling

    coords = [
        np.linspace(-(s - 1) / (2 * super_sampling), (s - 1) / (2 * super_sampling), s, dtype=dtype) for s in vol_shape_zxy_s
    ]
    if vol_origin_zxy:
        if len(coords) != len(vol_origin_zxy):
            raise ValueError(f"The volume shape ({len(coords)}), and the origin shape ({len(vol_origin_zxy)}) should match")
        coords = [c + vol_origin_zxy[ii] for ii, c in enumerate(coords)]
    coords = np.meshgrid(*coords, indexing="ij")

    if coords_ball is None:
        coords_ball = np.arange(-np.fmin(2, len(vol_shape_zxy_s)), 0, dtype=int)
    else:
        coords_ball = np.array(coords_ball, dtype=int)

    radius = np.min(vol_shape_zxy_s[coords_ball]) / (2 * super_sampling) + radius_offset

    coords = np.stack(coords, axis=0)
    if coords_ball.size == 1:
        dists = np.abs(coords[coords_ball, ...])
    else:
        dists = np.sqrt(np.sum(coords[coords_ball, ...] ** 2, axis=0))

    if mask_drop_off.lower() == "const":
        mask = (dists <= radius).astype(dtype)
    elif mask_drop_off.lower() == "sinc":
        cut_off = np.min(vol_shape_zxy_s[coords_ball]) / np.sqrt(2) - radius
        outter_region = 1.0 - (dists <= radius)
        outter_vals = 1.0 - np.sinc((dists - radius) / cut_off)
        mask = np.fmax(1 - outter_region * outter_vals, 0.0).astype(dtype)
    else:
        raise ValueError("Unknown drop-off function: %s" % mask_drop_off)

    if super_sampling > 1:
        new_shape = np.stack([np.array(vol_shape_zxy), np.ones_like(vol_shape_zxy) * super_sampling], axis=1).flatten()
        mask = mask.reshape(new_shape)
        mask = np.mean(mask, axis=tuple(np.arange(1, len(vol_shape_zxy) * 2, 2, dtype=int)))

    return mask


def compute_variance_poisson(
    Is: NDArray, I0: Optional[NDArray] = None, var_I0: Optional[NDArray] = None, normalized: bool = True
) -> NDArray:
    """
    Compute the variance of a signal subject to Poisson noise, against a reference intensity.

    The reference intensity can also be subject to Poisson and Gaussian noise.
    If the variance of the reference intensity is not passed, it will be assumed to be Poisson.

    Parameters
    ----------
    Is : NDArray
        The signal intensity.
    I0 : Optional[NDArray], optional
        The reference intensity. The default is None.
    var_I0 : Optional[NDArray], optional
        The variance of the reference intensity. The default is None.
        If not given, it will be assumed to be equal to I0.
    normalized : bool, optional
        Whether to renormalize by the mean of the reference intensity.

    Returns
    -------
    NDArray
        The computed variance.
    """
    var_Is = np.abs(Is)
    Is = np.fmax(Is, eps)

    if I0 is not None:
        if var_I0 is None:
            var_I0 = np.abs(I0)
        I0 = np.fmax(I0, eps)

        Is2 = Is**2
        I02 = I0**2
        variance = (Is2 / I02) * (var_Is / Is2 + var_I0 / I02)
        if normalized:
            variance *= np.mean(I0)
        return variance
    else:
        return var_Is


def compute_variance_transmission(
    Is: NDArray, I0: NDArray, var_I0: Optional[NDArray] = None, normalized: bool = True
) -> NDArray:
    """
    Compute the variance of a linearized attenuation (transmission) signal, against a reference intensity.

    Parameters
    ----------
    Is : NDArray
        The transmitted signal.
    I0 : NDArray
        The reference intensity.
    var_I0 : Optional[NDArray], optional
        The variance of the reference intensity. The default is None.
        If not given, it will be assumed to be equal to I0.
    normalized : bool, optional
        Whether to renormalize by the mean of the reference intensity.

    Returns
    -------
    NDArray
        The computed variance.
    """
    var_Is = np.abs(Is)
    Is = np.fmax(Is, eps)

    if var_I0 is None:
        var_I0 = np.abs(I0)
    I0 = np.fmax(I0, eps)

    Is2 = Is**2
    I02 = I0**2
    variance = (Is / I0) * (var_Is / Is2 + var_I0 / I02)
    if normalized:
        variance *= np.mean(I0)
    return variance


def compute_variance_weight(
    variance: NDArray, *, percentile: float = 0.001, normalized: bool = False, use_std: bool = False, semilog: bool = False
) -> NDArray:
    """
    Compute the weight associated to the given variance, in a weighted least-squares context.

    Parameters
    ----------
    variance : NDArray
        The variance of the signal.
    percentile : float
        Minimum percentile to discard. The default is 0.001 (0.1%)
    normalized : bool, optional
        Scale the largest weight to 1. The default is False.
    use_std : bool, optional
        Use the standard deviation instead of the variance.
    semilog : bool, optional
        Scale the variance over a logarithmic curve. It can be beneficial with
        high dynamic range data. The default is False.

    Returns
    -------
    NDArray
        The computed weights.
    """
    variance = np.abs(variance)

    sorted_variances = np.sort(variance.flatten())
    percentiles_variances = np.cumsum(sorted_variances) / np.sum(sorted_variances)
    ind_threshold = np.fmin(np.fmax(0, np.sum(percentiles_variances < percentile)), percentiles_variances.size)
    min_val = np.fmax(sorted_variances[ind_threshold], eps)

    min_nonzero_variance = np.fmax(np.min(variance[variance > 0]), min_val)
    weight = np.fmax(variance, min_nonzero_variance)

    if normalized:
        weight /= min_nonzero_variance
    if use_std:
        weight = np.sqrt(weight)
    if semilog:
        weight = np.log(weight + float(normalized is False)) + 1
    return 1 / weight
