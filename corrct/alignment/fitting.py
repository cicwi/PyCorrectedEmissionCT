#!/usr/bin/env python3
"""
Fitting routines.

Created on Tue May 17 12:11:58 2022

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from typing import Literal, Optional, Union, Sequence

import numpy as np
from numpy.polynomial import Polynomial
import scipy.ndimage as spimg
import scipy.optimize as spopt
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
    stack_axis: int = -2,
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
    stack_axis : int, optional
        The axis along which the VU images are stacked. The default is -2.
    decimals : int, optional
        Decimals for the truncation of the sub-pixel  The default is 2.

    Returns
    -------
    NDArrayFloat
        The VU shifts.
    """
    num_angles = data_vwu.shape[stack_axis]

    if use_rfft:
        local_fftn = np.fft.rfftn
        local_ifftn = np.fft.irfftn
    else:
        local_fftn = np.fft.fftn
        local_ifftn = np.fft.ifftn

    fft_dims = np.delete(np.arange(-len(data_vwu.shape), 0), stack_axis)
    u_axis = fft_dims[-1]

    old_fft_shapes = np.array(np.array(data_vwu.shape)[fft_dims], ndmin=1, dtype=int)
    new_fft_shapes = old_fft_shapes.copy()
    if pad_u:
        new_fft_shapes[u_axis] *= 2
    cc_coords = [np.fft.fftfreq(s, 1 / s) for s in new_fft_shapes]

    if len(fft_dims) == 2:
        shifts_vu = np.empty((len(data_vwu.shape) - 1, num_angles))
        slices = [slice(None)] * len(data_vwu.shape)
        for ii_a in range(num_angles):
            # For performance reasons, it is better to do the fft on each image
            slices[stack_axis] = slice(ii_a, ii_a + 1)
            data_vu = data_vwu[tuple(slices)].squeeze(axis=stack_axis)
            if proj_vwu.shape[stack_axis] == 1:
                proj_vu = proj_vwu.squeeze(axis=stack_axis)
            else:
                proj_vu = proj_vwu[tuple(slices)].squeeze(axis=stack_axis)
            data_vwu_f = local_fftn(data_vu, s=list(new_fft_shapes))
            proj_vwu_f = local_fftn(proj_vu, s=list(new_fft_shapes))

            cc_f = data_vwu_f * proj_vwu_f.conj()
            if normalize_fourier:
                cc_f /= np.fmax(np.abs(cc_f), eps)
            cc_r: NDArrayFloat = local_ifftn(cc_f).real

            f_vals, f_coords = extract_peak_region_nd(cc_r, cc_coords=cc_coords)
            shifts_vu[..., ii_a] = np.array([f_coords[0][1], f_coords[1][1]])

            if decimals > 0:
                f_vals_v = f_vals[:, 1]
                f_vals_u = f_vals[1, :]

                sub_pixel_v = refine_max_position_1d(f_vals_v, decimals=decimals)
                sub_pixel_u = refine_max_position_1d(f_vals_u, decimals=decimals)

                shifts_vu[..., ii_a] += [sub_pixel_v, sub_pixel_u]
    else:
        data_vwu_f = local_fftn(data_vwu, s=list(new_fft_shapes), axes=list(fft_dims))
        proj_vwu_f = local_fftn(proj_vwu, s=list(new_fft_shapes), axes=list(fft_dims))

        ccs_f = data_vwu_f * proj_vwu_f.conj()
        if normalize_fourier:
            ccs_f /= np.fmax(np.abs(ccs_f).max(axis=u_axis, keepdims=True), eps)
        ccs = local_ifftn(ccs_f, axes=fft_dims).real

        f_vals, f_h = extract_peak_regions_1d(ccs, axis=u_axis, cc_coords=cc_coords[u_axis])
        shifts_vu = f_h[1, :]

        if decimals > 0:
            shifts_vu += refine_max_position_1d(f_vals, decimals=decimals)

    # import skimage.registration as skr
    # upsample_factor = int(1 / 10 ** (-decimals))
    # shifts_vu = np.empty((len(data_vwu.shape) - 1, num_angles))
    # for ii in range(num_angles):
    #     shifts_vu[..., ii] = skr.phase_cross_correlation(
    #         data_vwu[..., ii, :], proj_vwu[..., ii, :], upsample_factor=upsample_factor, return_error=False
    #     )

    return shifts_vu


def fit_shifts_zyx_xc(
    ref_vol_zyx: NDArrayFloat,
    rec_vol_zyx: NDArrayFloat,
    pad_zyx: bool = False,
    normalize_fourier: bool = True,
    use_rfft: bool = True,
    decimals: int = 2,
) -> NDArrayFloat:
    """
    Find the ZYX shifts of the volume, through cross-correlation.

    Parameters
    ----------
    ref_vol_zyx : NDArrayFloat
        The reference volume.
    rec_vol_zyx : NDArrayFloat
        The reconstructed volume to register.
    pad_zyx : bool, optional
        Pad the ZYX coordinates. The default is False.
    normalize_fourier : bool, optional
        Whether to normalize the Fourier representation of the cross-correlation. The default is True.
    use_rfft : bool, optional
        Whether to use the `rfft` transform in place of the complex `fft` transform. The default is True.
    decimals : int, optional
        Decimals for the truncation of the sub-pixel  The default is 2.

    Returns
    -------
    NDArrayFloat
        The ZYX shifts.
    """
    if use_rfft:
        local_fftn = np.fft.rfftn
        local_ifftn = np.fft.irfftn
    else:
        local_fftn = np.fft.fftn
        local_ifftn = np.fft.ifftn

    fft_dims = np.arange(-np.fmin(ref_vol_zyx.ndim, 3), 0)
    old_fft_shapes = np.array(np.array(ref_vol_zyx.shape)[fft_dims], ndmin=1, dtype=int)
    new_fft_shapes = old_fft_shapes.copy()
    if pad_zyx:
        new_fft_shapes *= 2
    cc_coords = [np.fft.fftfreq(s, 1 / s) for s in new_fft_shapes]

    ref_vol_zyx_f = local_fftn(ref_vol_zyx, s=list(new_fft_shapes), axes=fft_dims)
    rec_vol_zyx_f = local_fftn(rec_vol_zyx, s=list(new_fft_shapes), axes=fft_dims)

    cc_f = ref_vol_zyx_f * rec_vol_zyx_f.conj()
    if normalize_fourier:
        cc_f /= np.fmax(np.abs(cc_f), eps)
    cc: NDArrayFloat = local_ifftn(cc_f).real

    f_vals, f_coords = extract_peak_region_nd(cc, cc_coords=cc_coords)
    shifts_zyx = np.array([coords[1] for coords in f_coords])

    if decimals > 0:
        for ii, dim in enumerate(fft_dims):
            slices = [slice(1, 2)] * ref_vol_zyx.ndim
            slices[dim] = slice(None)
            f_vals_slice = f_vals[tuple(slices)].flatten()

            sub_pixel_pos = refine_max_position_1d(f_vals_slice, decimals=decimals)

            shifts_zyx[ii] += sub_pixel_pos

    return shifts_zyx


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


def fit_sinusoid(angles: NDArrayFloat, values: NDArrayFloat, fit_l1: bool = False) -> tuple[float, float, float]:
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
) -> tuple[NDArrayFloat, Optional[NDArray]]:
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
            % (" ".join("%d" % s for s in cc.shape))
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
    f_vals: NDArrayFloat, f_x: Union[ArrayLike, NDArray, None] = None, return_vertex_val: bool = False, decimals: int = 2
) -> Union[NDArrayFloat, tuple[NDArrayFloat, NDArrayFloat]]:
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
            % (" ".join("%d" % s for s in f_vals.shape))
        )
    num_vals = f_vals.shape[0]

    if f_x is None:
        f_x_half_size = (num_vals - 1) / 2
        f_x = np.linspace(-f_x_half_size, f_x_half_size, num_vals)
    else:
        f_x = np.squeeze(f_x)
        if not (len(f_x.shape) == 1 and np.all(f_x.size == num_vals)):
            raise ValueError(
                "Base coordinates should have the same length as values array. Sizes of fx: %d, f_vals: %d"
                % (f_x.size, num_vals)
            )

    if len(f_vals.shape) == 1:
        # using Polynomial.fit, because supposed to be more numerically
        # stable than previous solutions (according to numpy).
        poly = Polynomial.fit(f_x, f_vals, deg=2)
        coeffs = poly.convert().coef
    else:
        coords = np.array([np.ones(num_vals), f_x, f_x**2])
        coeffs = np.linalg.lstsq(coords.T, f_vals, rcond=None)[0]

    # For a 1D parabola `f(x) = c + bx + ax^2`, the vertex position is:
    # x_v = -b / 2a.
    vertex_x = -coeffs[1, ...] / (2 * coeffs[2, ...])
    vertex_x = np.around(vertex_x, decimals=decimals)

    vertex_min_x = np.min(f_x)
    vertex_max_x = np.max(f_x)
    lower_bound_ok = vertex_min_x < vertex_x
    upper_bound_ok = vertex_x < vertex_max_x
    if not np.all(lower_bound_ok * upper_bound_ok):
        if len(f_vals.shape) == 1:
            message = (
                f"Fitted position {vertex_x} is outide the input margins [{vertex_min_x}, {vertex_max_x}]."
                f" Input values: {f_vals}"
            )
        else:
            message = (
                f"Fitted positions outide the input margins [{vertex_min_x}, {vertex_max_x}]:"
                f" {np.sum(1 - lower_bound_ok)} below and {np.sum(1 - upper_bound_ok)} above"
            )
        raise ValueError(message)

    if return_vertex_val:
        vertex_val = coeffs[0, ...] + vertex_x * coeffs[1, ...] / 2
        vertex_val = np.around(vertex_val, decimals=decimals)
        return vertex_x, vertex_val
    else:
        return vertex_x


def extract_peak_region_nd(
    cc: NDArrayFloat, peak_radius: int = 1, cc_coords: Union[tuple[Union[Sequence, NDArray]], None] = None
) -> tuple[NDArray, Optional[Sequence[NDArray]]]:
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
            % (" ".join("%d" % s for s in f_vals.shape))
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
                % (fy.size, " ".join("%d" % s for s in f_vals.shape))
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
            % (fx.size, " ".join("%d" % s for s in f_vals.shape))
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


def fit_parabola_min(
    fun_x: Union[ArrayLike, NDArray],
    fun_vals: Union[ArrayLike, NDArray],
    scale: Literal["linear", "log"] = "linear",
    decimals: int = 2,
) -> tuple[float, float, Optional[tuple[NDArray, NDArray]]]:
    """Parabolic fit local function stationary point.

    Parameters
    ----------
    fx : ArrayLike
        Parameter values.
    f_vals : ArrayLike
        Objective function costs of each parameter value.
    scale : str, optional
        Scale of the fit. Options are: "log" | "linear". The default is "log".

    Returns
    -------
    min_fx : float
        Expected parameter value of the fitted minimum.
    min_f_val : float
        Expected objective function cost of the fitted minimum.
    """
    fun_x = np.array(fun_x)
    fun_vals = np.array(fun_vals)

    if len(fun_x) < 3 or len(fun_vals) < 3 or len(fun_x) != len(fun_vals):
        raise ValueError(
            "Lengths of the parameter values and function values should be identical and >= 3."
            + f"Given: fx={len(fun_x)}, f_vals={len(fun_vals)}"
        )

    if scale.lower() == "log":
        to_fit = np.log10

        def from_fit(vals):
            return 10**vals

    elif scale.lower() == "linear":

        def to_fit(vals):
            return vals

        from_fit = to_fit
    else:
        raise ValueError(f"Parameter 'scale' should be either 'log' or 'linear', given '{scale}' instead")

    min_pos = np.argmin(fun_vals)
    if min_pos == 0:
        print("WARNING: minimum value at the beginning of the lambda range.")
        fx_fit = to_fit(fun_x[:3])
        f_vals_fit = fun_vals[:3]
    elif min_pos == (len(fun_vals) - 1):
        print("WARNING: minimum value at the end of the lambda range.")
        fx_fit = to_fit(fun_x[-3:])
        f_vals_fit = fun_vals[-3:]
    else:
        fx_fit = to_fit(fun_x[min_pos - 1 : min_pos + 2])
        f_vals_fit = fun_vals[min_pos - 1 : min_pos + 2]

    # using Polynomial.fit, because it is supposed to be more numerically
    # stable than previous solutions (according to numpy).
    poly = Polynomial.fit(fx_fit, f_vals_fit, deg=2)
    coeffs = poly.convert().coef
    if coeffs[2] <= 0:
        print("WARNING: fitted curve is concave. Returning minimum measured point.")
        return fun_x[min_pos], fun_vals[min_pos], None

    # For a 1D parabola `f(x) = c + bx + ax^2`, the vertex position is:
    # x_v = -b / 2a.
    vertex_pos = -coeffs[1] / (2 * coeffs[2])
    vertex_val = coeffs[0] + vertex_pos * coeffs[1] / 2

    vertex_pos = np.around(vertex_pos, decimals=decimals)
    vertex_val = np.around(vertex_val, decimals=decimals)

    min_fx, min_f_val = from_fit(vertex_pos), vertex_val
    if min_fx < fun_x[0] or min_fx > fun_x[-1]:
        print(
            f"WARNING: fitted stationary point {min_fx} is outside input range [{fun_x[0]}, {fun_x[-1]}]."
            + " Returning minimum measured point."
        )
        return fun_x[min_pos], fun_vals[min_pos], None

    return min_fx, min_f_val, (coeffs, fx_fit)


class Ellipse:
    """
    Initialize ellipse class, used for fitting acquisition geometry parameters.

    Parameters
    ----------
    prj_points_vu : ArrayLike
        List ofprojected positions over the detector of a test object.
    prj_center_vu : Optional[ArrayLike], optional
        Projected position of the rotation center. The default is None.
    """

    def __init__(self, prj_points_vu: Union[ArrayLike, NDArray], prj_center_vu: Union[ArrayLike, NDArray, None] = None):
        self.prj_points_vu = np.array(prj_points_vu)
        if prj_center_vu is not None:
            prj_center_vu = np.array(prj_center_vu)
        self.prj_center_vu = prj_center_vu

    def fit_prj_center(self, rescale: bool = True, least_squares: bool = True) -> NDArray:
        """
        Fit the projected circle position.

        Parameters
        ----------
        rescale : bool, optional
            Whether to rescale the data within the interval [-1, 1]. The default is True.
        least_squares : bool, optional
            Whether to use the least-squares (l2-norm) fit or l1-norm. The default is True.

        Returns
        -------
        ArrayLike
            The fitted center position.
        """
        c_vu = np.mean(self.prj_points_vu, axis=0)
        pos_vu = self.prj_points_vu - c_vu

        if rescale:
            scale_vu = np.max(pos_vu, axis=0) - np.min(pos_vu, axis=0)
            pos_vu /= scale_vu
        else:
            scale_vu = 1.0

        num_lines = pos_vu.shape[-2] // 2
        pos1_vu = pos_vu[:num_lines, :]
        pos2_vu = pos_vu[num_lines : num_lines * 2, :]

        diffs_vu = pos2_vu - pos1_vu
        b = np.cross(pos1_vu, pos2_vu)
        A = np.stack([diffs_vu[:, -1], -diffs_vu[:, -2]], axis=-1)

        p_vu = np.linalg.lstsq(A, b, rcond=None)[0]
        if not least_squares:

            def _func(params: NDArrayFloat) -> float:
                predicted_b = A.dot(params)
                l1_diff = np.linalg.norm(predicted_b - b, ord=1)
                return float(l1_diff)

            opt_p_vu = spopt.minimize(_func, p_vu)
            p_vu = opt_p_vu.x

        return p_vu * scale_vu + c_vu

    def fit_parameters(self, rescale: bool = True, least_squares: bool = True) -> NDArray:
        """
        Fit the ellipse parameters.

        Parameters
        ----------
        rescale : bool, optional
            Whether to rescale the data within the interval [-1, 1]. The default is True.
        least_squares : bool, optional
            Whether to use the least-squares (l2-norm) fit or l1-norm. The default is True.

        Returns
        -------
        ArrayLike
            The fitted ellipse parameters.
        """
        # First we fit 5 intermediate variables
        p_u: NDArray = self.prj_points_vu[:, -1]
        p_v: NDArray = self.prj_points_vu[:, -2]

        if rescale:
            c_h = np.mean(p_u)
            c_v = np.mean(p_v)
            p_u = p_u - c_h
            p_v = p_v - c_v

            p_u_scaling = np.abs(p_u).max()
            p_v_scaling = np.abs(p_v).max()
            p_u /= p_u_scaling
            p_v /= p_v_scaling
        else:
            c_h = 0.0
            c_v = 0.0
            p_u_scaling = 1.0
            p_v_scaling = 1.0

        A = np.stack([p_u**2, -2 * p_u, -2 * p_v, 2 * p_u * p_v, np.ones_like(p_u)], axis=-1)
        b = -(p_v**2)

        params = np.linalg.lstsq(A, b, rcond=None)[0]
        if not least_squares:

            def _func(pars: NDArrayFloat) -> float:
                predicted_b = A.dot(pars)
                l1_diff = np.linalg.norm(predicted_b - b, ord=1)
                return float(l1_diff)

            opt_params = spopt.minimize(_func, params)
            params = opt_params.x

        if rescale:
            params[0] *= (p_v_scaling**2) / (p_u_scaling**2)
            params[1] *= (p_v_scaling**2) / p_u_scaling
            params[2] *= p_v_scaling
            params[3] *= p_v_scaling / p_u_scaling
            params[4] *= p_v_scaling**2

        u = (params[1] - params[2] * params[3]) / (params[0] - params[3] ** 2)
        v = (params[0] * params[2] - params[1] * params[3]) / (params[0] - params[2] * params[3])

        a = params[0] / (params[0] * u**2 + v**2 + 2 * params[3] * u * v - params[4])
        b = a / params[0]
        c = params[3] * b

        return np.array([b, a, c, v + c_v, u + c_h])

    def fit_ellipse_centroid(self, rescale: bool = True, least_squares: bool = True) -> NDArray:
        """
        Fit the ellipse parameters, when assuming the center of mass of the points as center of the ellipse.

        Parameters
        ----------
        rescale : bool, optional
            Whether to rescale the data within the interval [-1, 1]. The default is True.
        least_squares : bool, optional
            Whether to use the least-squares (l2-norm) fit or l1-norm. The default is True.

        Returns
        -------
        ArrayLike
            The fitted ellipse parameters.
        """
        # First we fit 3 intermediate variables
        num_to_keep = (self.prj_points_vu.shape[0] // 2) * 2
        p_u = self.prj_points_vu[:num_to_keep, -1]
        p_v = self.prj_points_vu[:num_to_keep, -2]

        u = np.mean(p_u)
        v = np.mean(p_v)

        p_u = p_u - u
        p_v = p_v - v

        if rescale:
            p_u_scaling = np.abs(p_u).max()
            p_v_scaling = np.abs(p_v).max()
            p_u /= p_u_scaling
            p_v /= p_v_scaling
        else:
            p_u_scaling = 1.0
            p_v_scaling = 1.0

        A = np.stack([p_u**2, p_u * p_v, np.ones_like(p_u)], axis=-1)
        b = -(p_v**2)

        params = np.linalg.lstsq(A, b, rcond=None)[0]
        if not least_squares:

            def _func(pars: NDArrayFloat) -> float:
                predicted_b = A.dot(pars)
                l1_diff = np.linalg.norm(predicted_b - b, ord=1)
                return float(l1_diff)

            opt_params = spopt.minimize(_func, params)
            params = opt_params.x

        if rescale:
            params[0] *= (p_v_scaling**2) / (p_u_scaling**2)
            params[1] *= p_v_scaling / p_u_scaling
            params[2] *= p_v_scaling**2

        a = -params[0] / params[2]
        b = -1 / params[2]
        c = -params[1] / params[2]

        return np.array([b, a, c, v, u])

    @staticmethod
    def predict_v(ell_params: Union[ArrayLike, NDArray], uus: Union[ArrayLike, NDArray]) -> tuple[NDArray, NDArray]:
        """Predict V coordinates of ellipse from its parameters, and U coordinates.

        Parameters
        ----------
        ell_params : Union[ArrayLike, NDArray]
            The ellipse parameters
        uus : Union[ArrayLike, NDArray]
            The U coordinates

        Returns
        -------
        tuple[NDArray, NDArray]
            The corresponding top and bottom V coordinates
        """
        b, a, c, v, u = np.array(ell_params)
        uus = np.array(uus)

        a_tilde = b
        b_tilde = 2 * (-b * v + c * uus - c * u)
        c_tilde = -(1 - a * (uus - u) ** 2 - b * v**2 + 2 * c * v * (uus - u))
        delta_tilde = np.sqrt(b_tilde**2 - 4 * a_tilde * c_tilde)
        v_1 = (-b_tilde + delta_tilde) / (2 * a_tilde)
        v_2 = (-b_tilde - delta_tilde) / (2 * a_tilde)

        return v_1, v_2
