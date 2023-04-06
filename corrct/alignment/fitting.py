#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitting routines.

Created on Tue May 17 12:11:58 2022

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import numpy.polynomial

import scipy.optimize as spopt
import scipy.ndimage as spimg

from typing import Sequence, Tuple, Optional, Union
from numpy.typing import ArrayLike, NDArray


NDArrayFloat = NDArray[np.floating]


eps = np.finfo(np.float32).eps


def fit_shifts_u_sad(data_wu: NDArrayFloat, proj_wu: NDArrayFloat, error_norm: int = 1, decimals: int = 1) -> NDArrayFloat:
    """
    Find the U shifts between two sets of lines, by means of the sum-of-absolute-difference (SAD).

    Parameters
    ----------
    data_wu : NDArrayFloat
        The reference data.
    proj_wu : NDArrayFloat
        The other data.
    error_norm : int, optional
        The error norm to use, by default 1
    decimals : int, optional
        The precision of the result, by default 1

    Returns
    -------
    NDArrayFloat
        A list of one shift for each row.
    """
    padding = np.zeros((len(data_wu.shape), 2), dtype=int)
    padding[-1, :] = (int(np.ceil(data_wu.shape[-1] / 2)), int(np.floor(data_wu.shape[-1] / 2)))
    pad_data_wu = np.pad(data_wu, pad_width=padding, mode="edge")
    pad_proj_wu = np.pad(proj_wu, pad_width=padding, mode="edge")

    fft_proj_wu = np.fft.fft2(pad_proj_wu)

    delta = 1 / (10 ** (-decimals))
    num_shifts = np.ceil(data_wu.shape[-1] * np.sqrt(1 / delta)).astype(int)
    shifts = np.fft.fftfreq(num_shifts, 1 / (data_wu.shape[-1] * delta))

    vals = np.empty((len(shifts), data_wu.shape[-2]))
    for ii, s in enumerate(shifts):
        shifted_proj_wu = np.fft.ifft2(spimg.fourier_shift(fft_proj_wu, (0, s))).real
        vals[ii, :] = np.linalg.norm(pad_data_wu - shifted_proj_wu, axis=-1, ord=error_norm)

    iis_min = np.argmin(vals, axis=0)
    return shifts[list(iis_min)]


def fit_shifts_vu_xc(
    data_vwu: NDArrayFloat,
    proj_vwu: NDArrayFloat,
    pad_u: bool = False,
    normalize_fourier: bool = True,
    use_rfft: bool = True,
    decimals: int = 2,
) -> NDArrayFloat:
    """
    Find the VU shifts of the projected data, through cross-correlation.

    Parameters
    ----------
    data_vwu : NDArrayFloat
        The collected projection data.
    proj_vwu : NDArrayFloat
        The forward-projected images from the reconstruction.
    pad_u : bool, optional
        Pad the u coordinate. The default is False.
    normalize_fourier : bool, optional
        Whether to normalize the Fourier representation of the cross-correlation. The default is True.
    use_rfft : bool, optional
        Whether to use the `rfft` transform in place of the complex `fft` transform. The default is True.
    decimals : int, optional
        Decimals for the truncation of the sub-pixel  The default is 2.

    Returns
    -------
    NDArrayFloat
        The VU shifts.
    """
    num_angles = data_vwu.shape[-2]

    if use_rfft:
        local_fftn = np.fft.rfftn
        local_ifftn = np.fft.irfftn
    else:
        local_fftn = np.fft.fftn
        local_ifftn = np.fft.ifftn

    fft_dims = np.delete(np.arange(-len(data_vwu.shape), 0), -2)
    old_fft_shapes = np.array(np.array(data_vwu.shape)[fft_dims], ndmin=1, dtype=int)
    new_fft_shapes = old_fft_shapes.copy()
    if pad_u:
        new_fft_shapes[-1] *= 2
    cc_coords = [np.fft.fftfreq(s, 1 / s) for s in new_fft_shapes]

    if len(fft_dims) == 2:
        shifts_vu = np.empty((len(data_vwu.shape) - 1, num_angles))
        for ii in range(num_angles):
            # For performance reasons, it is better to do the fft on each image
            data_vwu_f = local_fftn(data_vwu[..., ii, :], s=list(new_fft_shapes))
            proj_vwu_f = local_fftn(proj_vwu[..., ii, :], s=list(new_fft_shapes))

            cc_f = data_vwu_f * proj_vwu_f.conj()
            if normalize_fourier:
                cc_f /= np.fmax(np.abs(cc_f), eps)
            cc: NDArrayFloat = local_ifftn(cc_f).real

            f_vals, f_coords = extract_peak_region_nd(cc, cc_coords=cc_coords)
            shifts_vu[..., ii] = np.array([f_coords[0][1], f_coords[1][1]])

            if decimals > 0:
                f_vals_v = f_vals[:, 1]
                f_vals_u = f_vals[1, :]

                sub_pixel_v = refine_max_position_1d(f_vals_v, decimals=decimals)
                sub_pixel_u = refine_max_position_1d(f_vals_u, decimals=decimals)

                shifts_vu[..., ii] += [sub_pixel_v, sub_pixel_u]
    else:
        data_vwu_f = local_fftn(data_vwu, s=list(new_fft_shapes), axes=list(fft_dims))
        proj_vwu_f = local_fftn(proj_vwu, s=list(new_fft_shapes), axes=list(fft_dims))

        ccs_f = data_vwu_f * proj_vwu_f.conj()
        if normalize_fourier:
            ccs_f /= np.fmax(np.abs(ccs_f).max(axis=-1, keepdims=True), eps)
        ccs = local_ifftn(ccs_f, axes=fft_dims).real

        f_vals, fh = extract_peak_regions_1d(ccs, axis=-1, cc_coords=cc_coords[-1])
        shifts_vu = fh[1, :]

        if decimals > 0:
            shifts_vu += refine_max_position_1d(f_vals, decimals=decimals)

    # upsample_factor = int(1 / 10 ** (-decimals))
    # shifts_vu = np.empty((len(data_vwu.shape) - 1, num_angles))
    # for ii in range(num_angles):
    #     shifts_vu[..., ii] = skr.phase_cross_correlation(
    #         data_vwu[..., ii, :], proj_vwu[..., ii, :], upsample_factor=upsample_factor, return_error=False
    #     )

    return shifts_vu


def sinusoid(
    x: Union[NDArrayFloat, float], a: Union[NDArrayFloat, float], p: Union[NDArrayFloat, float], b: Union[NDArrayFloat, float]
) -> NDArrayFloat:
    """Compute the values of a sine function.

    Parameters
    ----------
    x : NDArrayFloat | float
        The independent variable.
    a : NDArrayFloat | float
        The amplitude of the sine.
    p : NDArrayFloat | float
        The phase of the sine.
    b : NDArrayFloat | float
        The bias of the sine.

    Returns
    -------
    NDArrayFloat
        The computed values.
    """
    return a * np.sin(x + p) + b


def fit_sinusoid(angles: NDArrayFloat, values: NDArrayFloat, fit_l1: bool = False) -> Tuple[float, float, float]:
    """Fits a sinusoid to the given values.

    Parameters
    ----------
    angles : NDArrayFloat
        Angles where to evaluate the sinusoid.
    values : NDArrayFloat
        Values of the sinusoid.
    fit_l1 : bool, optional
        Whether to use l1 fit instead of the l2 fit, by default False

    Returns
    -------
    Tuple[float, float, float]
        The amplitude, phase and bias of the sinusoid.
    """
    a0 = (values.max() - values.min()) / 2
    b0 = (values.max() + values.min()) / 2

    (a, p, b), _ = spopt.curve_fit(sinusoid, angles, values, p0=[a0, 0, b0])

    if fit_l1:

        def f(apb: NDArrayFloat) -> float:
            a, p, b = apb[0], apb[1], apb[2]
            pred_sinusoid = sinusoid(angles, a, p, b)
            l1_diff = np.linalg.norm(pred_sinusoid - values, ord=1)
            return float(l1_diff)

        apb = spopt.minimize(f, np.array([a, p, b]))
        (a, p, b) = apb.x

    return a, p, b


def extract_peak_regions_1d(
    cc: NDArrayFloat, axis: int = -1, peak_radius: int = 1, cc_coords: Union[ArrayLike, NDArray, None] = None
) -> Tuple[NDArrayFloat, Optional[NDArray]]:
    """
    Extract a region around the maximum value.

    Parameters
    ----------
    cc: NDArrayFloat
        Correlation image.
    axis: int, optional
        Find the max values along the specified direction. The default is -1.
    peak_radius: int, optional
        The l_inf radius of the area to extract around the peak. The default is 1.
    cc_coords: ArrayLike, optional
        The coordinates of `cc` along the selected axis. The default is None.

    Returns
    -------
    f_vals: NDArrayFloat
        The extracted function values.
    fc_ax: NDArrayFloat
        The coordinates of the extracted values, along the selected axis.
    """
    if len(cc.shape) == 1:
        cc = cc[None, ...]
    img_shape = np.array(cc.shape)
    if not (len(img_shape) == 2):
        raise ValueError(
            "The input image should be either a 1 or 2-dimensional array. Array of shape: [%s] was given."
            % (" ".join(("%d" % s for s in cc.shape)))
        )
    other_axis = (axis + 1) % 2
    # get pixel having the maximum value of the correlation array
    pix_max = np.argmax(cc, axis=axis)

    # select a n neighborhood for the many 1D sub-pixel fittings (with wrapping)
    p_ax_range = np.arange(-peak_radius, peak_radius + 1)
    p_ax = (pix_max[None, :] + p_ax_range[:, None]) % img_shape[axis]

    p_ln = np.tile(np.arange(0, img_shape[other_axis])[None, :], [2 * peak_radius + 1, 1])

    # extract the pixel coordinates along the axis
    if cc_coords is None:
        fc_ax = None
    else:
        cc_coords = np.array(cc_coords, ndmin=1)
        fc_ax = cc_coords[p_ax.flatten()].reshape(p_ax.shape)

    # extract the correlation values
    if other_axis == 0:
        f_vals = cc[p_ln, p_ax]
    else:
        f_vals = cc[p_ax, p_ln]

    return (f_vals, fc_ax)


def refine_max_position_1d(
    f_vals: NDArrayFloat, fx: Union[ArrayLike, NDArray, None] = None, return_vertex_val: bool = False, decimals: int = 2
) -> Union[NDArrayFloat, Tuple[NDArrayFloat, NDArrayFloat]]:
    """Compute the sub-pixel max position of the given function sampling.

    Parameters
    ----------
    f_vals: NDArrayFloat
        Function values of the sampled points
    fx: ArrayLike, optional
        Coordinates of the sampled points
    return_vertex_val: boolean, option
        Enables returning the vertex values. Defaults to False.

    Raises
    ------
    ValueError
        In case position and values do not have the same size, or in case
        the fitted maximum is outside the fitting region.

    Returns
    -------
    float
        Estimated function max, according to the coordinates in fx.
    """
    if not len(f_vals.shape) in (1, 2):
        raise ValueError(
            "The fitted values should be either one or a collection of 1-dimensional arrays. Array of shape: [%s] was given."
            % (" ".join(("%d" % s for s in f_vals.shape)))
        )
    num_vals = f_vals.shape[0]

    if fx is None:
        fx_half_size = (num_vals - 1) / 2
        fx = np.linspace(-fx_half_size, fx_half_size, num_vals)
    else:
        fx = np.squeeze(fx)
        if not (len(fx.shape) == 1 and np.all(fx.size == num_vals)):
            raise ValueError(
                "Base coordinates should have the same length as values array. Sizes of fx: %d, f_vals: %d"
                % (fx.size, num_vals)
            )

    if len(f_vals.shape) == 1:
        # using Polynomial.fit, because supposed to be more numerically
        # stable than previous solutions (according to numpy).
        poly = np.polynomial.Polynomial.fit(fx, f_vals, deg=2)
        coeffs = poly.convert().coef
    else:
        coords = np.array([np.ones(num_vals), fx, fx**2])
        coeffs = np.linalg.lstsq(coords.T, f_vals, rcond=None)[0]

    # For a 1D parabola `f(x) = c + bx + ax^2`, the vertex position is:
    # x_v = -b / 2a.
    vertex_x = -coeffs[1, ...] / (2 * coeffs[2, ...])
    vertex_x = np.around(vertex_x, decimals=decimals)

    vertex_min_x = np.min(fx)
    vertex_max_x = np.max(fx)
    lower_bound_ok = vertex_min_x < vertex_x
    upper_bound_ok = vertex_x < vertex_max_x
    if not np.all(lower_bound_ok * upper_bound_ok):
        if len(f_vals.shape) == 1:
            message = (
                f"Fitted position {vertex_x} is outide the input margins [{vertex_min_x}, {vertex_max_x}]."
                + f" Input values: {f_vals}"
            )
        else:
            message = "Fitted positions outide the input margins [{}, {}]: {} below and {} above".format(
                vertex_min_x,
                vertex_max_x,
                np.sum(1 - lower_bound_ok),
                np.sum(1 - upper_bound_ok),
            )
        raise ValueError(message)

    if return_vertex_val:
        vertex_val = coeffs[0, ...] + vertex_x * coeffs[1, ...] / 2
        vertex_val = np.around(vertex_val, decimals=decimals)
        return vertex_x, vertex_val
    else:
        return vertex_x


def extract_peak_region_nd(
    cc: NDArrayFloat, peak_radius: int = 1, cc_coords: Union[Tuple[Union[Sequence, NDArray]], None] = None
) -> Tuple[NDArray, Optional[Sequence[NDArray]]]:
    """
    Extract a region around the maximum value.

    Parameters
    ----------
    cc: NDArrayFloat
        Correlation image.
    peak_radius: int, optional
        The l_inf radius of the area to extract around the peak. The default is 1.
    cc_coords: ArrayLike, optional
        The coordinates of `cc`. The default is None.

    Returns
    -------
    f_vals: NDArrayFloat
        The extracted function values.
    f_coords: Tuple[NDArrayFloat]
        The coordinates of the extracted values.
    """
    img_shape = np.array(cc.shape)
    # get pixel having the maximum value of the correlation array
    pix_max_corr = np.argmax(cc)
    peak_pos = np.unravel_index(pix_max_corr, img_shape)

    # select a n x n neighborhood for the sub-pixel fitting (with wrapping)
    peak_ranges = [np.arange(p - peak_radius, p + peak_radius + 1) % img_shape[ii] for ii, p in enumerate(peak_pos)]

    # extract the (v, h) pixel coordinates
    if cc_coords is None:
        f_coords = None
    else:
        f_coords = [coords[pr] for coords, pr in zip(cc_coords, peak_ranges)]

    # extract the correlation values
    peak_ranges = np.meshgrid(*peak_ranges, indexing="ij")
    f_vals = cc[tuple(peak_ranges)]

    return f_vals, f_coords


def refine_max_position_2d(
    f_vals: NDArrayFloat, fy: Union[ArrayLike, NDArray, None] = None, fx: Union[ArrayLike, NDArray, None] = None
) -> NDArray:
    """Compute the sub-pixel max position of the given function sampling.

    Parameters
    ----------
    f_vals: NDArrayFloat
        Function values of the sampled points
    fy: ArrayLike, optional
        Vertical coordinates of the sampled points
    fx: ArrayLike, optional
        Horizontal coordinates of the sampled points

    Raises
    ------
    ValueError
        In case position and values do not have the same size, or in case
        the fitted maximum is outside the fitting region.

    Returns
    -------
    tuple(float, float)
        Estimated (vertical, horizontal) function max, according to the
        coordinates in fy and fx.
    """
    if not (len(f_vals.shape) == 2):
        raise ValueError(
            "The fitted values should form a 2-dimensional array. Array of shape: [%s] was given."
            % (" ".join(("%d" % s for s in f_vals.shape)))
        )
    if fy is None:
        fy = np.linspace(-1, 1, f_vals.shape[0])
        y_scaling = (f_vals.shape[0] - 1) / 2
    else:
        fy = np.array(fy, ndmin=1)
        y_scaling = 1.0
        if not (len(fy.shape) == 1 and np.all(fy.size == f_vals.shape[0])):
            raise ValueError(
                "Vertical coordinates should have the same length as values matrix. Sizes of fy: %d, f_vals: [%s]"
                % (fy.size, " ".join(("%d" % s for s in f_vals.shape)))
            )

    if fx is None:
        fx = np.linspace(-1, 1, f_vals.shape[1])
        x_scaling = (f_vals.shape[1] - 1) / 2
    else:
        fx = np.array(fx, ndmin=1)
        x_scaling = 1.0
    if not (len(fx.shape) == 1 and np.all(fx.size == f_vals.shape[1])):
        raise ValueError(
            "Horizontal coordinates should have the same length as values matrix. Sizes of fx: %d, f_vals: [%s]"
            % (fx.size, " ".join(("%d" % s for s in f_vals.shape)))
        )

    fy, fx = np.meshgrid(fy, fx, indexing="ij")
    fy = fy.flatten()
    fx = fx.flatten()

    coords = np.array([np.ones(f_vals.size), fy, fx, fy * fx, fy**2, fx**2])
    coeffs = np.linalg.lstsq(coords.T, f_vals.flatten(), rcond=None)[0]
    coeffs *= [1, y_scaling, x_scaling, y_scaling * x_scaling, y_scaling**2, x_scaling**2]

    # For a 1D parabola `f(x) = ax^2 + bx + c`, the vertex position is:
    # x_v = -b / 2a. For a 2D parabola, the vertex position is:
    # (y, x)_v = - b / A, where:
    A = [[2 * coeffs[4], coeffs[3]], [coeffs[3], 2 * coeffs[5]]]
    b = coeffs[1:3]
    vertex_yx = np.linalg.lstsq(A, -b, rcond=None)[0]

    vertex_min_yx = [np.min(fy), np.min(fx)]
    vertex_max_yx = [np.max(fy), np.max(fx)]
    if np.any(vertex_yx < vertex_min_yx) or np.any(vertex_yx > vertex_max_yx):
        print(f_vals)
        raise ValueError(
            f"Fitted (yx: {vertex_yx}) positions are outside the input margins"
            + f" y: [{vertex_min_yx[0]}, {vertex_max_yx[0]}], and x: [{vertex_min_yx[1]}, {vertex_max_yx[1]}]."
            + f" Input values: {f_vals}"
        )
    return vertex_yx
