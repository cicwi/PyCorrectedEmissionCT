# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:25:14 2020

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np


def get_circular_mask(vol_shape_yx, radius_offset=0, data_type=np.float32):
    """
    Computes a circular mask for the reconstruction volume.

    Parameters
    ----------
    :param vol_shape_yx: numpy.array_like
        The size of the volume.
    :param radius_offset: float, optional
        The offset with respect to the volume edge. The default is 0.
    :param data_type: numpy.dtype, optional
        The mask data type. The default is np.float32.

    Returns
    -------
    :returns: The circular mask.
    :rtype: (numpy.array_like)
    """
    yy = np.linspace(-(vol_shape_yx[0]-1)/2, (vol_shape_yx[0]-1)/2, vol_shape_yx[0], dtype=data_type)
    xx = np.linspace(-(vol_shape_yx[1]-1)/2, (vol_shape_yx[1]-1)/2, vol_shape_yx[1], dtype=data_type)
    (yy, xx) = np.meshgrid(yy, xx, indexing='ij')

    return np.sqrt(xx ** 2 + yy ** 2) <= ((vol_shape_yx[0]) / 2 + radius_offset)


def pad_sinogram(sinogram, width, pad_axis=-1, mode='edge', **kwds):
    """
    Pads the sinogram.

    Parameters
    ----------
    :param sinogram: numpy.array_like
        The sinogram to pad.
    :param width: int or tuple(int, int)
        The width of the padding. Could be a scalar or a tuple of two integers.
    :param pad_axis: int, optional
        The axis to pad. The default is -1.
    :param mode: string, optional
        The padding type (from numpy.pad). The default is 'edge'.
    :param **kwds: The numpy.pad arguments.

    Returns
    -------
    :returns: The padded sinogram.
    :rtype: (numpy.array_like)
    """
    pad_size = [(0, 0)] * len(sinogram.shape)
    if len(width) == 1:
        width = (width, width)
    pad_size[pad_axis] = width

    return np.pad(sinogram, pad_size, mode=mode, **kwds)
