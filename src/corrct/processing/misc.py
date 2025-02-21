"""
Miscellaneous processing routines.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import scipy as sp

from typing import Optional, Union, Callable
from collections.abc import Sequence
from numpy.typing import NDArray, ArrayLike, DTypeLike

import matplotlib.pyplot as plt


eps = np.finfo(np.float32).eps

NDArrayInt = NDArray[np.signedinteger]


def circular_mask(
    vol_shape_zxy: Union[Sequence[int], NDArrayInt],
    radius_offset: float = 0,
    coords_ball: Union[Sequence[int], NDArrayInt, None] = None,
    ball_norm: float = 2,
    vol_origin_zxy: Union[Sequence[float], NDArray, None] = None,
    taper_func: Optional[str] = None,
    taper_target: Union[str, float] = "edge",
    super_sampling: int = 1,
    squeeze: bool = True,
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
    ball_norm : float, optional
        The norm of the ball. The default is 2.
    vol_origin_zxy : Optional[Sequence[float]], optional
        The origin of the coordinates in voxels. The default is None.
    taper_func : str, optional
        The mask data type. Allowed types: "const" | "cos". The default is "const".
    taper_target : str | float, optional
        The size target. Allowed values: "edge" | "diagonal". The default is "edge".
    super_sampling : int, optional
        The pixel super sampling to be used for the mask. The default is 1.
    squeeze : bool, optional
        Whether to squeeze the mask. The default is True.
    dtype : DTypeLike, optional
        The type of mask. The default is np.float32.

    Raises
    ------
    ValueError
        In case of unknown taper_func value, or mismatching volume origin and shape.

    Returns
    -------
    NDArray
        The circular mask.
    """
    vol_shape_zxy_s = np.array(vol_shape_zxy, dtype=int) * super_sampling

    coords = [
        np.linspace(-(s - 1) / (2 * super_sampling), (s - 1) / (2 * super_sampling), s, dtype=dtype) for s in vol_shape_zxy_s
    ]
    if vol_origin_zxy is not None:
        if len(coords) != len(vol_origin_zxy):
            raise ValueError(f"The volume shape ({len(coords)}), and the origin shape ({len(vol_origin_zxy)}) should match")
        coords = [c + vol_origin_zxy[ii] for ii, c in enumerate(coords)]
    coords = np.meshgrid(*coords, indexing="ij")

    if coords_ball is None:
        coords_ball = np.arange(-np.fmin(2, len(vol_shape_zxy_s)), 0, dtype=int)
    else:
        coords_ball = np.array(coords_ball, dtype=int)

    max_radius = np.min(vol_shape_zxy_s[coords_ball]) / (2 * super_sampling) + radius_offset

    coords = np.stack(coords, axis=0)
    if coords_ball.size == 1:
        dists = np.abs(coords[coords_ball, ...])
    else:
        dists = np.linalg.norm(coords[coords_ball, ...], axis=0, ord=ball_norm)

    if taper_func is None:
        mask = (dists <= max_radius).astype(dtype)
    elif isinstance(taper_func, str):
        if isinstance(taper_target, str):
            if taper_target.lower() == "edge":
                cut_off_denom = 2
                cut_off_offset = 0
            elif taper_target.lower() == "diagonal":
                cut_off_denom = np.sqrt(2)
                cut_off_offset = 0
            else:
                raise ValueError(
                    f"When `taper_target` is str, it should be one of: 'edge' | 'diagonal', but {taper_target} passed instead."
                )
        else:
            cut_off_denom = 2
            if taper_target < radius_offset:
                print(f"WARNING: parameter `taper_target`={taper_target} is smaller than `radius_offset`={radius_offset}.")
            cut_off_offset = np.fmax(taper_target, radius_offset)

        if taper_func.lower() == "cos":
            cut_off_radius = np.min(vol_shape_zxy_s[coords_ball]) / (cut_off_denom * super_sampling) + cut_off_offset
            cut_off_size = cut_off_radius - max_radius
            outter_vals = np.cos(np.fmax(dists - max_radius, 0) / cut_off_size * np.pi) / 2 + 0.5
            mask = (outter_vals * (dists < cut_off_radius)).astype(dtype)
        else:
            raise ValueError(f"Unknown taper function: {taper_func}")
    else:
        raise ValueError(f"Parameter `taper_func` should either be a string or None.")

    if super_sampling > 1:
        new_shape = np.stack([np.array(vol_shape_zxy), np.ones_like(vol_shape_zxy) * super_sampling], axis=1).flatten()
        mask = mask.reshape(new_shape)
        mask = np.mean(mask, axis=tuple(np.arange(1, len(vol_shape_zxy) * 2, 2, dtype=int)))

    if squeeze:
        mask = np.squeeze(mask)

    return mask


def ball(
    data_shape_vu: ArrayLike,
    radius: Union[int, float],
    super_sampling: int = 5,
    dtype: DTypeLike = np.float32,
    func: Optional[Callable] = None,
) -> ArrayLike:
    """
    Compute a ball with specified radius.

    Parameters
    ----------
    data_shape_vu : ArrayLike
        Shape of the output array.
    radius : int | float
        Radius of the ball.
    super_sampling : int, optional
        Super-sampling for having smoother ball edges. The default is 5.
    dtype : DTypeLike, optional
        Type of the output. The default is np.float32.
    func : Optional[Callable], optional
        Point-wise function for the local values. The default is None.

    Returns
    -------
    ArrayLike
        The ball.
    """
    data_shape_vu = np.array(data_shape_vu, dtype=int) * super_sampling

    # coords = [np.linspace(-(s - 1) / 2, (s - 1) / 2, s, dtype=np.float32) for s in data_shape_vu]
    coords = [np.fft.fftfreq(d, 1 / d) for d in data_shape_vu]
    coords = np.stack(np.meshgrid(*coords, indexing="ij"), axis=0)

    r = np.sqrt(np.sum(coords**2, axis=0)) / super_sampling

    probe = (r < radius).astype(dtype)
    if func is not None:
        probe *= func(r)

    probe = np.roll(probe, super_sampling // 2, axis=tuple(np.arange(len(data_shape_vu))))
    new_shape = np.stack([data_shape_vu // super_sampling, np.ones_like(data_shape_vu) * super_sampling], axis=1).flatten()
    probe = probe.reshape(new_shape)
    probe = np.mean(probe, axis=tuple(np.arange(1, len(data_shape_vu) * 2, 2, dtype=int)))

    return np.fft.fftshift(probe)


def azimuthal_integration(img: NDArray, axes: Sequence[int] = (-2, -1), domain: str = "direct") -> NDArray:
    """
    Compute the azimuthal integration of a n-dimensional image or a stack of them.

    Parameters
    ----------
    img : NDArray
        The image or stack of images.
    axes : tuple(int, int), optional
        Axes of that need to be azimuthally integrated. The default is (-2, -1).
    domain : string, optional
        Domain of the integration. Options are: "direct" | "fourier". Default is "direct".

    Raises
    ------
    ValueError
        Error returned when not passing images or wrong axes.

    Returns
    -------
    NDArray
        The azimuthally integrated profile.
    """
    num_dims_int = len(axes)
    num_dims_img = len(img.shape)

    if num_dims_img < num_dims_int:
        raise ValueError(
            "Input image ({num_dims_img}D) should be at least the same dimensionality"
            " of the axes for the integration (#{num_dims_int})."
        )
    if len(axes) == 0:
        raise ValueError("Input axes should be at least 1.")

    # Compute the coordinates of the pixels along the chosen axes
    img_axes_dims = np.array(np.array(img.shape)[list(axes)], ndmin=1)
    if domain.lower() == "direct":
        half_dims = (img_axes_dims - 1) / 2
        coords = [np.linspace(-h, h, d) for h, d in zip(half_dims, img_axes_dims)]
    else:
        coords = [np.fft.fftfreq(d, 1 / d) for d in img_axes_dims]
    coords = np.stack(np.meshgrid(*coords, indexing="ij"))
    r = np.sqrt(np.sum(coords**2, axis=0))

    # Reshape the volume to have the axes to be integrates as right-most axes
    img_tr_op = np.array([*range(len(img.shape))])
    img_tr_op = np.concatenate((np.delete(img_tr_op, obj=axes), img_tr_op[list(axes)]))
    img = np.transpose(img, img_tr_op)

    if num_dims_img > num_dims_int:
        img_old_shape = img.shape[:-num_dims_int]
        img = np.reshape(img, [-1, *img_axes_dims])

    # Compute the linear interpolation coefficients
    r_l = np.floor(r)
    r_u = r_l + 1
    w_l = (r_u - r) * img
    w_u = (r - r_l) * img

    # Do the azimuthal integration as a histogram operation
    r_all = np.concatenate((r_l.flatten(), r_u.flatten())).astype(int)
    if num_dims_img > num_dims_int:
        num_imgs = img.shape[0]
        az_img = []
        for ii in range(num_imgs):
            w_all = np.concatenate((w_l[ii, ...].flatten(), w_u[ii, ...].flatten()))
            az_img.append(np.bincount(r_all, weights=w_all))
        az_img = np.array(az_img)
        return np.reshape(az_img, (*img_old_shape, az_img.shape[-1]))  # type: ignore
    else:
        w_all = np.concatenate((w_l.flatten(), w_u.flatten()))
        return np.bincount(r_all, weights=w_all)


def lines_intersection(
    line_1: NDArray,
    line_2: Union[float, NDArray],
    position: str = "first",
    x_lims: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> Optional[tuple[float, float]]:
    """
    Compute the intersection point between two lines.

    Parameters
    ----------
    line_1 : NDArray
        The first line.
    line_2 : float | NDArray
        The second line. It can be a scalar representing a horizontal line.
    position : str, optional
        The position of the point to select. Either "first" or "last".
        The default is "first".

    Raises
    ------
    ValueError
        If position is neither "first" nor "last".

    Returns
    -------
    Tuple[float, float] | None
        It returns either the requested crossing point, or None in case the
        point was not found.
    """
    line_1 = np.array(np.squeeze(line_1), ndmin=1)
    line_2 = np.array(np.squeeze(line_2), ndmin=1)
    # Find the transition points, by first finding where line_2 is above line_1
    crossing_points = np.where(line_2 > line_1, 0, 1)
    crossing_points = np.abs(np.diff(crossing_points))

    if x_lims is not None:
        if x_lims[0] is None:
            if x_lims[1] is None:
                raise ValueError("When passing `x_lims`, at least one of the values should not be None.")
            else:
                bias = 0
                crossing_points = crossing_points[: x_lims[1]]
        else:
            bias = x_lims[0]
            if x_lims[1] is None:
                crossing_points = crossing_points[x_lims[0] :]
            else:
                crossing_points = crossing_points[x_lims[0] : x_lims[1]]
    else:
        bias = 0

    crossing_points = np.where(crossing_points)[0]

    if crossing_points.size == 0:
        print("No crossing found!")
        return None

    if position.lower() == "first":
        point_l = crossing_points[0] + bias
    elif position.lower() == "last":
        point_l = crossing_points[-1] + bias
    else:
        raise ValueError(f"Crossing position: {position} unknown. Please choose either 'first' or 'last'.")

    x1 = 0.0
    x2 = 1.0
    y1 = line_1[point_l]
    y2 = line_1[point_l + 1]
    x3 = 0.0
    x4 = 1.0
    if line_2.size == 1:
        y3 = line_2
        y4 = line_2
    else:
        y3 = line_2[point_l]
        y4 = line_2[point_l + 1]

    # From wikipedia: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    p_den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    p_x_num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    p_y_num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    p_x = p_x_num / p_den + point_l
    p_y = p_y_num / p_den

    return float(p_x), float(p_y)


def norm_cross_corr(
    img1: NDArray,
    img2: Optional[NDArray] = None,
    axes: Sequence[int] = (-2, -1),
    t_match: bool = False,
    mode_full: bool = True,
    compute_profile: bool = True,
    plot: bool = True,
) -> Union[NDArray, tuple[NDArray, NDArray]]:
    """
    Compute the normalized cross-correlation between two images.

    Parameters
    ----------
    img1 : NDArray
        The first image.
    img2 : NDArray, optional
        The second images. If None, it computes the auto-correlation. The default is None.
    axes : Sequence[int], optional
        Axes along which to compute the cross-correlation. The default is (-2, -1).
    t_match : bool, optional
        Whether to perform the cross-correlation for template matching. The default is False.
    mode_full : bool, optional
        Whether to return the "full" or "same" convolution size. The default is True.
    compute_profile : bool, optional
        Whether to compute the azimuthal integration of the cross-correlation or not. The default is True.
    plot : bool, optional
        Whether to plot the profile of the cross-correlation curve. The default is True.

    Returns
    -------
    NDArray
        The one-dimensional cross-correlation curve.
    """

    def local_sum(x: NDArray, axes: Sequence[int]) -> NDArray:
        padding = np.zeros(len(x.shape), dtype=int)
        for a in axes:
            padding[a] = x.shape[a]
        y: NDArray[np.floating] = np.pad(x, padding)
        for a in axes:
            y = np.cumsum(y, axis=a)
            slicing1 = [slice(None)] * len(x.shape)
            slicing1[a] = slice(x.shape[a], -1)
            slicing2 = [slice(None)] * len(x.shape)
            slicing2[a] = slice(0, -x.shape[a] - 1)
            y = y[tuple(slicing1)] - y[tuple(slicing2)]
        return y

    if img2 is None:
        is_autocorrelation = True
        img2 = img1
    else:
        is_autocorrelation = False

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    for a in axes:
        img2 = np.flip(img2, axis=a)
    cc = sp.signal.fftconvolve(img1, img2, mode="full", axes=axes)

    if not mode_full:
        slices = [slice(None)] * len(cc.shape)
        for a in axes:
            start_ind = (cc.shape[a] - img1.shape[a]) // 2
            end_ind = start_ind + img1.shape[a]
            slices[a] = slice(start_ind, end_ind)
        slices = tuple(slices)
        cc = cc[slices]

    if t_match:
        local_sums_img2 = local_sum(img2, axes=axes)
        local_sums_img2_2 = local_sum(img2**2, axes=axes)
        if not mode_full:
            local_sums_img2 = local_sums_img2[slices]
            local_sums_img2_2 = local_sums_img2_2[slices]

        cc_n = cc - local_sums_img2 * np.mean(img1)

        cc_n /= np.std(img1) * np.sqrt(np.prod(np.array(img1.shape)[list(axes)]))

        diff_local_sums = local_sums_img2_2 - (local_sums_img2**2) / np.prod(np.array(img2.shape)[list(axes)])
        cc_n /= np.sqrt(diff_local_sums.clip(0, None))
    else:
        cc_n = cc / (np.linalg.norm(img1) * np.linalg.norm(img2))

    cc_n = np.fft.ifftshift(cc_n)

    if compute_profile:
        cc_l = azimuthal_integration(cc_n, axes=axes, domain="fourier")
        cc_o = azimuthal_integration(np.ones_like(cc_n), axes=axes, domain="fourier")

        cc_l /= cc_o
        cc_l = cc_l[: np.min(np.array(img1.shape)[list(axes)])]

        if plot:
            p_xy = lines_intersection(cc_l, 0.5, position="first")

            fig, axs = plt.subplots(1, 1, figsize=(7, 3.5))
            axs.plot(cc_l, label="Auto-correlation" if is_autocorrelation else "Cross-correlation")
            axs.plot(np.ones_like(cc_l) * 0.5, label="Half-maximum")
            if p_xy is not None:
                axs.scatter(p_xy[0], p_xy[1])
                axs.plot([p_xy[0], p_xy[0]], [0, 1], label=f"Resolution: {p_xy[0]:.3} pix")
            axs.grid()
            axs.legend(fontsize=13)
            axs.tick_params(labelsize=16)
            axs.set_title("Cross-correlation")
            fig.tight_layout()
            plt.show(block=False)

        return cc_n, cc_l
    else:
        return cc_n


def inspect_fourier_img(img: NDArray, remove_zero: bool = False) -> None:
    """Display Fourier representation of the input image.

    Parameters
    ----------
    img : NDArray
        Input image.
    remove_zero : bool, optional
        Remove the zero frequency value. The default is False.
    """
    img_f = np.squeeze(np.fft.fft2(img))
    if remove_zero is True:
        img_f[0, 0] = 0
    img_f_sh = np.fft.fftshift(img_f)

    f, axs = plt.subplots(2, 3)
    f.suptitle("Fourier representation")

    axs[0, 0].imshow(np.real(img_f_sh))
    axs[0, 0].set_title("Real")
    axs[0, 1].imshow(np.imag(img_f_sh))
    axs[0, 1].set_title("Imag")
    axs[0, 2].imshow(np.abs(img_f_sh))
    axs[0, 2].set_title("Abs")

    vert_img_f = np.fft.fftshift(img_f[:, 0])
    axs[1, 0].plot(np.stack((np.real(vert_img_f), np.imag(vert_img_f), np.abs(vert_img_f)), axis=1))
    axs[1, 0].set_title("Vertical profiles")

    horz_img_f = np.fft.fftshift(img_f[0, :])
    axs[1, 1].plot(np.stack((np.real(horz_img_f), np.imag(horz_img_f), np.abs(horz_img_f)), axis=1))
    axs[1, 1].set_title("Horizontal profiles")

    diag_img_f = np.fft.fftshift(np.diag(img_f))
    axs[1, 2].plot(np.stack((np.real(diag_img_f), np.imag(diag_img_f), np.abs(diag_img_f)), axis=1))
    axs[1, 2].set_title("Diagonal profiles")

    plt.show(block=False)
