#!/usr/bin/env python3
"""
Fiducial marker tracking routines.

@author: Nicola VIGANÒ, ESRF - The European Synchrotron, Grenoble, France,
and CEA-IRIG, Grenoble, France
"""

from typing import Union

import numpy as np
import scipy.ndimage as spimg
from numpy.typing import ArrayLike, NDArray

from . import fitting


def track_marker(prj_data: NDArray, marker_vu: NDArray, stack_axis: int = -2) -> NDArray:
    """Track marker position in a stack of images.

    Parameters
    ----------
    prj_data_vwu : NDArray
        The projection data.
    marker_vu : NDArray
        The fiducial marker to track in VU.
    stack_axis : int, optional
        The axis along which the images are stacked. The default is -2.

    Returns
    -------
    NDArray
        List of positions for each image.
    """
    marker_v1u = np.expand_dims(marker_vu, stack_axis).astype(np.float32)
    marker_pos = fitting.fit_shifts_vu_xc(prj_data, marker_v1u, stack_axis=stack_axis, normalize_fourier=False)
    marker_pos = marker_pos.swapaxes(-2, -1) + np.array(marker_vu.shape) / 2
    return marker_pos


def create_marker_disk(
    data_shape_vu: Union[ArrayLike, NDArray], radius: float, super_sampling: int = 5, conv: bool = True
) -> NDArray:
    """
    Create a Disk probe object, that will be used for tracking a calibration object's movement.

    Parameters
    ----------
    data_shape_vu : ArrayLike
        Shape of the images (vertical, horizontal).
    radius : float
        Radius of the probe.
    super_sampling : int, optional
        Super sampling of the coordinates used for creation. The default is 5.
    conv : bool, optional
        Whether to convolve the initial probe with itself. The default is True.

    Returns
    -------
    NDArray
        An image of the same size as the projections, that contains the marker in the center.
    """
    data_shape_vu = np.array(data_shape_vu, dtype=int) * super_sampling

    # coords = [np.linspace(-(s - 1) / 2, (s - 1) / 2, s, dtype=np.float32) for s in data_shape_vu]
    coords = [np.fft.fftfreq(d, 1 / d) for d in data_shape_vu]
    coords = np.stack(np.meshgrid(*coords, indexing="ij"), axis=0)
    pix_rr = np.sqrt(np.sum(coords**2, axis=0))

    probe = pix_rr < radius * super_sampling
    probe = np.roll(probe, super_sampling // 2, axis=tuple(np.arange(len(data_shape_vu))))
    new_shape = np.stack([data_shape_vu // super_sampling, np.ones_like(data_shape_vu) * super_sampling], axis=1).flatten()
    probe = probe.reshape(new_shape)
    probe = np.mean(probe, axis=tuple(np.arange(1, len(data_shape_vu) * 2, 2, dtype=int)))

    probe = np.fft.fftshift(probe)

    if conv:
        probe = spimg.convolve(probe, probe)

    return probe
