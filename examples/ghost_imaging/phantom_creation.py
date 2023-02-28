# -*- coding: utf-8 -*-
"""
Ghost-imaging phantom creation.

@author: Nicola VIGANÒ, The European Synchrotron, Grenoble, France
"""

import numpy as np

import corrct as cct

import skimage.data as skd
import skimage.transform as skt

from numpy.typing import NDArray, DTypeLike
from typing import Tuple


def create_phantom_shepplogan_2D(FoV_size: int, dtype: DTypeLike = np.float32) -> NDArray:
    phantom = skd.shepp_logan_phantom()
    phantom = skt.rescale(phantom, scale=FoV_size / phantom.shape[0], mode="reflect", multichannel=False)
    phantom = np.array(phantom, dtype=dtype)
    phantom[np.isclose(phantom, 0.0)] += 0.0001
    return phantom[None, ...]


def create_phantom_dots_2D(FoV_size: int, dtype: DTypeLike = np.float32) -> NDArray:
    phantom = np.zeros((1, FoV_size, FoV_size), dtype=dtype)
    if FoV_size == 101:
        phantom[0, ...] = cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-45)
        phantom = np.roll(phantom, 20, axis=1)
        phantom = np.roll(phantom, 10, axis=2)
        phantom[0, ...] += cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-45)
        phantom = np.roll(phantom, -20, axis=2)
        phantom = np.roll(phantom, 10, axis=1)
        phantom[0, ...] += cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-45)
        phantom = np.roll(phantom, -15, axis=1)
    elif FoV_size == 37:
        phantom[0, ...] = cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-12)
        phantom = np.roll(phantom, 14, axis=1)
        phantom = np.roll(phantom, 8, axis=2)
        phantom[0, ...] += cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-12)
        phantom = np.roll(phantom, -14, axis=2)
        phantom = np.roll(phantom, 8, axis=1)
        phantom[0, ...] += cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-12)
        phantom = np.roll(phantom, 8, axis=2)
        phantom = np.roll(phantom, -10, axis=1)
    else:
        phantom[0, ...] = cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-12)
        phantom = np.roll(phantom, 8, axis=1)
        phantom = np.roll(phantom, 4, axis=2)
        phantom[0, ...] += cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-12)
        phantom = np.roll(phantom, -8, axis=2)
        phantom = np.roll(phantom, 4, axis=1)
        phantom[0, ...] += cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-12)
        phantom = np.roll(phantom, -6, axis=1)
    phantom[phantom == 0] += 0.1
    return phantom.astype(np.float32)


def create_phantom_dots_overlap_2D(FoV_size: int, dtype: DTypeLike = np.float32) -> NDArray:
    phantom = np.zeros((1, FoV_size, FoV_size), dtype=dtype)
    phantom[0, ...] = cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-6)
    phantom = np.roll(phantom, 6, axis=1)
    phantom = np.roll(phantom, 4, axis=2)
    phantom[0, ...] += cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-6)
    phantom = np.roll(phantom, -8, axis=2)
    phantom = np.roll(phantom, 4, axis=1)
    phantom[0, ...] += cct.processing.circular_mask((FoV_size, FoV_size), radius_offset=-6)
    phantom = np.roll(phantom, 4, axis=2)
    phantom = np.roll(phantom, -5, axis=1)
    return phantom.astype(np.float32)


def create_phantom_cells_3D(FoV_size: int, apply_mask: bool = True, dtype: DTypeLike = np.float32) -> Tuple[NDArray, NDArray]:
    phantom = np.array(skd.cells3d(), dtype=dtype).swapaxes(0, 1) / 5e6
    int_profile = phantom[0].mean(axis=-1, keepdims=True)
    phantom[0] /= int_profile / int_profile.mean()
    scale_factor = FoV_size / phantom.shape[-1]
    phantom = skt.rescale(phantom, scale=FoV_size / phantom.shape[-1], mode="reflect", channel_axis=0)
    resolution_um = np.array([0.29, 0.26, 0.26], dtype=dtype) / scale_factor
    water_mask = cct.processing.circular_mask([FoV_size, FoV_size], dtype=dtype)
    phantom = np.pad(phantom, pad_width=([0, 1], [0, 0], [0, 0], [0, 0]), mode="constant", constant_values=1.0)
    if apply_mask:
        phantom = phantom * water_mask
    return phantom, resolution_um
