# -*- coding: utf-8 -*-
"""
Pre-processing routines.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

import skimage.transform as skt

from typing import Sequence, Optional, Union
from numpy.typing import DTypeLike, NDArray


eps = np.finfo(np.float32).eps


def pad_sinogram(
    sinogram: NDArray, width: Union[int, Sequence[int], NDArray], pad_axis: int = -1, mode: str = "edge", **kwds
) -> NDArray:
    """
    Pad the sinogram.

    Parameters
    ----------
    sinogram : NDArray
        The sinogram to pad.
    width : Union[int, Sequence[int]]
        The width of the padding. Normally, it should either be an int or a tuple(int, int).
    pad_axis : int, optional
        The axis to pad. The default is -1.
    mode : str, optional
        The padding type (from numpy.pad). The default is "edge".
    **kwds :
        The numpy.pad arguments.

    Returns
    -------
    NDArray
        The padded sinogram.
    """
    width = np.array(width, ndmin=1)

    pad_size = np.zeros((len(sinogram.shape), len(width)), dtype=int)
    pad_size[pad_axis, :] = width

    return np.pad(sinogram, pad_size, mode=mode.lower(), **kwds)  # type: ignore


def apply_flat_field(
    projs: NDArray,
    flats: NDArray,
    darks: Optional[NDArray] = None,
    crop: Optional[Sequence[int]] = None,
    dtype: DTypeLike = np.float32,
) -> NDArray:
    """
    Apply flat field.

    Parameters
    ----------
    projs : NDArray
        Projections.
    flats : NDArray
        Flat fields.
    darks : Optional[NDArray], optional
        Dark noise. The default is None.
    crop : Optional[Sequence[int]], optional
        Crop region. The default is None.
    dtype : DTypeLike, optional
        Data type of the processed data. The default is np.float32.

    Returns
    -------
    NDArray
        Falt-field corrected and linearized projections.
    """
    projs = np.ascontiguousarray(projs, dtype=dtype)
    flats = np.ascontiguousarray(flats, dtype=dtype)

    if crop is not None:
        projs = projs[..., crop[0] : crop[2], crop[1] : crop[3]]
        flats = flats[..., crop[0] : crop[2], crop[1] : crop[3]]
        if darks is not None:
            darks = darks[..., crop[0] : crop[2], crop[1] : crop[3]]

    if darks is not None:
        darks = np.ascontiguousarray(darks, dtype=dtype)
        projs = projs - darks
        flats = flats - darks

    flats = np.mean(flats, axis=0)

    return projs / flats


def apply_minus_log(projs: NDArray) -> NDArray:
    """
    Apply -log.

    Parameters
    ----------
    projs : NDArray
        Projections.

    Returns
    -------
    NDArray
        Linearized projections.
    """
    return np.fmax(-np.log(projs), 0.0)


def rotate_proj_stack(data_vwu: NDArray, rot_angle_deg: float) -> NDArray:
    """
    Rotate the projection stack.

    Parameters
    ----------
    data_vwu : NDArray
        The projection stack, with dimensions [v, w, u] (vertical, omega / sample rotation, horizontal).
    rot_angle_deg : float
        The rotation angle in degrees.

    Returns
    -------
    NDArray
        The rotated projection stack.
    """
    data_vwu_r = np.empty_like(data_vwu)
    for ii in range(data_vwu.shape[-2]):
        data_vwu_r[:, ii, :] = skt.rotate(data_vwu[:, ii, :], -rot_angle_deg, clip=False)
    return data_vwu_r
