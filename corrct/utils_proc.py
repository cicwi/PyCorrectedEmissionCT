# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:25:14 2020

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from . import operators
from . import solvers


def get_circular_mask(vol_shape, radius_offset=0, coords_ball=None, mask_drop_off="const", data_type=np.float32):
    """Computes a circular mask for the reconstruction volume.

    :param vol_shape: The size of the volume.
    :type vol_shape: numpy.array_like
    :param radius_offset: The offset with respect to the volume edge.
    :type radius_offset: float. Optional, default: 0
    :param coords_ball: The coordinates to consider for the non-masked region.
    :type coords_ball: list of dimensions. Optional, default: None
    :param data_type: The mask data type.
    :type data_type: numpy.dtype. Optional, default: np.float32

    :returns: The circular mask.
    :rtype: (numpy.array_like)
    """
    vol_shape = np.array(vol_shape, dtype=np.int)

    coords = [np.linspace(-(s - 1) / 2, (s - 1) / 2, s, dtype=data_type) for s in vol_shape]
    coords = np.meshgrid(*coords, indexing="ij")

    if coords_ball is None:
        coords_ball = np.arange(-np.fmin(2, len(vol_shape)), 0, dtype=np.int)
    else:
        coords_ball = np.array(coords_ball, dtype=np.int)

    radius = np.min(vol_shape[coords_ball]) / 2 + radius_offset

    coords = np.stack(coords, axis=0)
    if coords_ball.size == 1:
        dists = np.abs(coords[coords_ball, ...])
    else:
        dists = np.sqrt(np.sum(coords[coords_ball, ...] ** 2, axis=0))

    if mask_drop_off.lower() == "const":
        return dists <= radius
    elif mask_drop_off.lower() == "sinc":
        cut_off = np.min(vol_shape[coords_ball]) / np.sqrt(2) - radius
        outter_region = 1 - (dists <= radius)
        outter_vals = 1 - np.sinc((dists - radius) / cut_off)
        return np.fmax(1 - outter_region * outter_vals, 0)
    else:
        raise ValueError("Unknown drop-off function: %s" % mask_drop_off)


def pad_sinogram(sinogram, width, pad_axis=-1, mode="edge", **kwds):
    """Pads the sinogram.

    :param sinogram: The sinogram to pad.
    :type sinogram: numpy.array_like
    :param width: The width of the padding.
    :type width: either an int or tuple(int, int)
    :param pad_axis: The axis to pad.
    :type pad_axis: int. Optional, default: -1
    :param mode: The padding type (from numpy.pad).
    :type mode: string. Optional, default: 'edge'.
    :param kwds: The numpy.pad arguments.

    :returns: The padded sinogram.
    :rtype: (numpy.array_like)
    """
    pad_size = [(0, 0)] * len(sinogram.shape)
    if len(width) == 1:
        width = (width, width)
    pad_size[pad_axis] = width

    return np.pad(sinogram, pad_size, mode=mode, **kwds)


def apply_flat_field(projs, flats, darks=None, crop=None, data_type=np.float32):
    """Apply flat field.

    :param projs: Projections
    :type projs: numpy.array_like
    :param flats: Flat fields
    :type flats: numpy.array_like
    :param darks: Dark noise, defaults to None
    :type darks: numpy.array_like, optional
    :param crop: Crop region, defaults to None
    :type crop: numpy.array_like, optional
    :param data_type: numpy.dtype, defaults to np.float32
    :type data_type: Data type of the processed data, optional

    :return: Falt-field corrected and linearized projections
    :rtype: numpy.array_like
    """
    if crop is not None:
        projs = projs[..., crop[0] : crop[2], crop[1] : crop[3]]
        flats = flats[..., crop[0] : crop[2], crop[1] : crop[3]]
        if darks is not None:
            darks = darks[..., crop[0] : crop[2], crop[1] : crop[3]]

    if darks is not None:
        projs -= darks
        flats -= darks

    flats = np.mean(flats.astype(data_type), axis=0)

    return projs.astype(data_type) / flats


def apply_minus_log(projs):
    """Apply -log.

    :param projs: Projections
    :type projs: numpy.array_like

    :return: Falt-field corrected and linearized projections
    :rtype: numpy.array_like
    """
    return np.fmax(-np.log(projs), 0.0)


def denoise_image(
    img, reg_weight=1e-2, stddev=None, error_norm="l2b", iterations=250, axes=(-2, -1), lower_limit=None, verbose=False
):
    """Image denoiser based on (simple, weighted or dead-zone) least-squares and wavelets.
    The weighted least-squares requires the local pixel-wise standard deviations.
    It can be used to denoise sinograms and projections.

    :param img: The image or sinogram to denoise.
    :type img: `numpy.array_like`
    :param reg_weight: Weight of the regularization term, defaults to 1e-2
    :type reg_weight: float, optional
    :param stddev: The local standard deviations. If None, it performs a standard least-squares.
    :type stddev: `numpy.array_like`, optional
    :param error_norm: The error weighting mechanism. When using std_dev, options are: {'l2b'} | 'l1b' | 'hub' | 'wl2' \
    (corresponding to: 'l2 dead-zone', 'l1 dead-zone', 'Huber', 'weighted least-squares').
    :type error_norm: str, optional
    :param iterations: Number of iterations, defaults to 250
    :type iterations: int, optional
    :param axes: Axes along which the regularization should be done, defaults to (-2, -1)
    :type iterations: int or tuple, optional
    :param lower_limit: Lower clipping limit of the image, defaults to None
    :type iterations: float, optional
    :param verbose: Turn verbosity on, defaults to False
    :type verbose: boolean, optional

    :return: Denoised image or sinogram.
    :rtype: `numpy.array_like`
    """

    def compute_wls_weights(stddev, At, reg_weights):
        stddev_zeros = stddev == 0
        stddev_valid = np.invert(stddev_zeros)
        min_valid_stddev = np.min(stddev[stddev_valid])

        reg_weights = reg_weights * (At(stddev_zeros) == 0) * min_valid_stddev
        img_weights = min_valid_stddev / np.fmax(stddev, min_valid_stddev)
        return (img_weights, reg_weights)

    def compute_lsb_weights(stddev):
        stddev_zeros = stddev == 0
        stddev_valid = np.invert(stddev_zeros)
        min_valid_stddev = np.min(stddev[stddev_valid])
        return np.fmax(stddev, min_valid_stddev)

    OpI = operators.TransformIdentity(img.shape)

    if stddev is not None:
        if error_norm.lower() == "l2b":
            img_weight = compute_lsb_weights(stddev)
            data_term = solvers.DataFidelity_l2b(img_weight)
        elif error_norm.lower() == "l1b":
            img_weight = compute_lsb_weights(stddev)
            data_term = solvers.DataFidelity_l1b(img_weight)
        elif error_norm.lower() == "hub":
            img_weight = compute_lsb_weights(stddev)
            data_term = solvers.DataFidelity_Huber(img_weight)
        elif error_norm.lower() == "wl2":
            (img_weight, reg_weight) = compute_wls_weights(stddev, OpI.T, reg_weight)
            data_term = solvers.DataFidelity_wl2(img_weight)
        else:
            raise ValueError('Unknown error method: "%s". Options are: {"l2b"} | "l1b" | "hub" | "wl2"' % error_norm)
    else:
        data_term = error_norm

    if isinstance(axes, int):
        axes = (axes,)

    reg_wl = solvers.Regularizer_l1swl(reg_weight, "bior4.4", 2, axes=axes, normalized=False)
    sol_wls_wl = solvers.CP(verbose=verbose, regularizer=reg_wl, data_term=data_term)

    (denoised_img, _) = sol_wls_wl(OpI, img, iterations, x0=img, lower_limit=lower_limit)
    return denoised_img
