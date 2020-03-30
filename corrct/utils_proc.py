# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:25:14 2020

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np


def get_circular_mask(vol_shape_yx, radius_offset=0, data_type=np.float32):
    """Computes a circular mask for the reconstruction volume.

    :param vol_shape_yx: The size of the volume (numpy.array_like)
    :param radius_offset: The offset with respect to the volume edge. Optinal, default: 0 (float)
    :param data_type: The mask data type. Optional, default: np.float32 (numpy.dtype)

    :returns: The circular mask.
    :rtype: (numpy.array_like)
    """
    yy = np.linspace(-(vol_shape_yx[0]-1)/2, (vol_shape_yx[0]-1)/2, vol_shape_yx[0], dtype=data_type)
    xx = np.linspace(-(vol_shape_yx[1]-1)/2, (vol_shape_yx[1]-1)/2, vol_shape_yx[1], dtype=data_type)
    (yy, xx) = np.meshgrid(yy, xx, indexing='ij')

    return np.sqrt(xx ** 2 + yy ** 2) <= ((vol_shape_yx[0]) / 2 + radius_offset)


def pad_sinogram(sinogram, width, pad_axis=-1, mode='edge', **kwds):
    """Pads the sinogram.

    :param sinogram: The sinogram to pad. (numpy.array_like)
    :param width: The width of the padding. Could be a scalar or a tuple of two integers. (int or tuple(int, int) )
    :param pad_axis: The axis to pad. Optional, default: -1. (int)
    :param mode: The padding type (from numpy.pad). Optional, default: 'edge'. (string)
    :param **kwds: The numpy.pad arguments.

    :returns: The padded sinogram.
    :rtype: (numpy.array_like)
    """
    pad_size = [(0, 0)] * len(sinogram.shape)
    if len(width) == 1:
        width = (width, width)
    pad_size[pad_axis] = width

    return np.pad(sinogram, pad_size, mode=mode, **kwds)
