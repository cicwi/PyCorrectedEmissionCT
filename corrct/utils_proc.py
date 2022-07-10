# -*- coding: utf-8 -*-
"""
Pre-processing and post-processing routines.

Created on Tue Mar 24 15:25:14 2020

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import scipy as sp

import skimage.transform as skt

from typing import Sequence, Optional, Tuple, Union, Callable

from numpy.typing import ArrayLike, DTypeLike, NDArray

import matplotlib.pyplot as plt


eps = np.finfo(np.float32).eps


def get_ball(
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


def get_circular_mask(
    vol_shape: Sequence[int],
    radius_offset: float = 0,
    coords_ball: Optional[NDArray[np.signedinteger]] = None,
    vol_origin: Optional[Sequence[float]] = None,
    mask_drop_off: str = "const",
    super_sampling: int = 1,
    dtype: DTypeLike = np.float32,
) -> NDArray:
    """
    Compute a circular mask for the reconstruction volume.

    Parameters
    ----------
    vol_shape : Sequence[int]
        The size of the volume.
    radius_offset : float, optional
        The offset with respect to the volume edge. The default is 0.
    coords_ball : Optional[Sequence[int]], optional
        The coordinates to consider for the non-masked region. The default is None.
    vol_origin : Optional[Sequence[float]], optional
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
    vol_shape_s = np.array(vol_shape, dtype=int) * super_sampling

    coords = [
        np.linspace(-(s - 1) / (2 * super_sampling), (s - 1) / (2 * super_sampling), s, dtype=dtype) for s in vol_shape_s
    ]
    if vol_origin:
        if len(coords) != len(vol_origin):
            raise ValueError(f"The volume shape ({len(coords)}), and the origin shape ({len(vol_origin)}) should match")
        coords = [c + vol_origin[ii] for ii, c in enumerate(coords)]
    coords = np.meshgrid(*coords, indexing="ij")

    if coords_ball is None:
        coords_ball = np.arange(-np.fmin(2, len(vol_shape_s)), 0, dtype=int)
    else:
        coords_ball = np.array(coords_ball, dtype=int)

    radius = np.min(vol_shape_s[coords_ball]) / (2 * super_sampling) + radius_offset

    coords = np.stack(coords, axis=0)
    if coords_ball.size == 1:
        dists = np.abs(coords[coords_ball, ...])
    else:
        dists = np.sqrt(np.sum(coords[coords_ball, ...] ** 2, axis=0))

    if mask_drop_off.lower() == "const":
        mask = (dists <= radius).astype(dtype)
    elif mask_drop_off.lower() == "sinc":
        cut_off = np.min(vol_shape_s[coords_ball]) / np.sqrt(2) - radius
        outter_region = 1.0 - (dists <= radius)
        outter_vals = 1.0 - np.sinc((dists - radius) / cut_off)
        mask = np.fmax(1 - outter_region * outter_vals, 0.0).astype(dtype)
    else:
        raise ValueError("Unknown drop-off function: %s" % mask_drop_off)

    if super_sampling > 1:
        new_shape = np.stack([np.array(vol_shape), np.ones_like(vol_shape) * super_sampling], axis=1).flatten()
        mask = mask.reshape(new_shape)
        mask = np.mean(mask, axis=tuple(np.arange(1, len(vol_shape) * 2, 2, dtype=int)))

    return mask


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


def compute_variance_weigth(
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


def get_beam_profile(
    voxel_size_um: float, beam_fwhm_um: float, profile_size: int = 1, beam_shape: str = "gaussian", verbose: bool = False
) -> NDArray:
    """
    Compute the pencil beam integration point spread function.

    Parameters
    ----------
    voxel_size_um : float
        The integration length.
    beam_fwhm_um : float
        The beam FWHM.
    profile_size : int, optional
        The number of pixels of the PSF. The default is 1.
    verbose : bool, optional
        Whether to print verbose information. The default is False.

    Returns
    -------
    NDArray
        The beam PSF.
    """
    y_points = profile_size * 2 + 1
    extent_um = y_points * voxel_size_um
    num_points = (np.ceil(extent_um * 10) // 2).astype(int) * 2 + 1
    half_voxel_size_um = voxel_size_um / 2

    eps = np.finfo(np.float32).eps

    xc = np.linspace(-extent_um / 2, extent_um / 2, num_points)

    # Box beam shape
    yb = np.abs(xc) < half_voxel_size_um
    yb[np.abs(np.abs(xc) - half_voxel_size_um) <= eps] = 0.5

    # Gaussian beam shape
    if beam_shape.lower() == "gaussian":
        yg = np.exp(-4 * np.log(2) * (xc**2) / (beam_fwhm_um**2))
    elif beam_shape.lower() == "lorentzian":
        beam_hwhm_um = beam_fwhm_um / 2
        yg = (beam_hwhm_um**2) / (xc**2 + beam_hwhm_um**2)
    elif beam_shape.lower() == "sech**2":
        # doi: 10.1364/ol.20.001160
        tau = beam_fwhm_um / (2 * np.arccosh(np.sqrt(2)))
        yg = 1 / np.cosh(xc / tau) ** 2
    else:
        raise ValueError(f"Unknown beam shape: {beam_shape.lower()}")

    yc = np.convolve(yb, yg, mode="same")
    yc = yc / np.max(yc)

    y = np.zeros((y_points,))
    for ii_p in range(y_points):
        # Finding the region that overlaps with the given adjacent voxel
        voxel_center_um = (ii_p - profile_size) * voxel_size_um
        yp = np.abs(xc - voxel_center_um) < half_voxel_size_um
        yp[(np.abs(xc - voxel_center_um) - half_voxel_size_um) < eps] = 0.5

        y[ii_p] = np.sum(yc * yp)

    # Renormalization
    y /= y.sum()

    if verbose:
        f, ax = plt.subplots(1, 2, figsize=[10, 5])
        ax[0].plot(xc, yb, label="Integration length")  # type: ignore
        ax[0].plot(xc, yg, label=f"{beam_shape.capitalize()} beam shape")  # type: ignore
        ax[0].plot(xc, yc, label="Resulting beam shape")  # type: ignore
        ax[0].legend()  # type: ignore
        ax[0].grid()  # type: ignore

        pixels_pos = np.linspace(-(y_points - 1) / 2, (y_points - 1) / 2, y_points)
        ax[1].bar(pixels_pos, y, label="PSF values", color="C1", width=1, linewidth=1.5, edgecolor="k", alpha=0.75)  # type: ignore
        ax[1].plot(xc / extent_um * profile_size * 2, yc, label="Resulting beam shape")  # type: ignore
        ax[1].legend()  # type: ignore
        ax[1].grid()  # type: ignore
        plt.tight_layout()

    return y


def compute_com(vol: NDArray, axes: Optional[ArrayLike] = None) -> NDArray:
    """
    Compute center-of-mass for given volume.

    Parameters
    ----------
    vol : NDArray
        The input volume.
    axes : ArrayLike, optional
        Axes on which to compute center-of-mass. The default is None.

    Returns
    -------
    NDArray
        The center-of-mass position.
    """
    if axes is None:
        axes = np.arange(len(vol.shape))
    else:
        axes = np.array(axes, ndmin=1)

    coords = [np.linspace(-(s - 1) / 2, (s - 1) / 2, s) for s in np.array(vol.shape)[list(axes)]]

    num_dims = len(vol.shape)
    com = np.empty((len(axes),))
    for ii, a in enumerate(axes):
        sum_axes = np.array(np.delete(np.arange(num_dims), a), ndmin=1, dtype=int)
        line = np.abs(vol).sum(axis=tuple(sum_axes))
        com[ii] = line.dot(coords[ii]) / line.sum()

    return com


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


def compute_frc(
    img1: NDArray,
    img2: NDArray,
    snrt: float = 0.2071,
    axes: Optional[Sequence[int]] = None,
    smooth: Optional[int] = 5,
    supersampling: int = 1,
) -> Tuple[NDArray, NDArray]:
    """
    Compute the FRC/FSC (Fourier ring/shell correlation) between two images / volumes.

    Please refer to the following article for more information:
        M. van Heel and M. Schatz, “Fourier shell correlation threshold criteria,”
        J. Struct. Biol., vol. 151, no. 3, pp. 250–262, Sep. 2005.

    Parameters
    ----------
    img1 : NDArray
        First image / volume.
    img2 : NDArray
        Second image / volume.
    snrt : float, optional
        SNR to be used for generating the threshold curve for resolution definition.
        The SNR value of 0.4142 corresponds to the hald-bit curve for a full dataset.
        When splitting datasets in two sub-datasets, that value needs to be halved.
        The default is 0.2071, which corresponds to the half-bit threashold for half dataset.
    axes : Sequence[int], optional
        The axes over which we want to compute the FRC/FSC.
        If None, all axes will be used The default is None.
    smooth : Optional[int], optional
        Size of the Hann smoothing window. The default is 5.
    supersampling : int, optional
        Supersampling factor of the images.
        Larger values increase the high-frequency range of the FRC/FSC function.
        The default is 1, which corresponds to the Nyquist frequency.

    Raises
    ------
    ValueError
        Error returned when not passing images of the same shape.

    Returns
    -------
    NDArray
        The computed FRC/FSC.
    NDArray
        The threshold curve corresponding to the given threshod SNR.
    """
    img1_shape = np.array(img1.shape)

    if axes is None:
        axes = list(np.arange(-len(img1_shape), 0))

    if img2 is None:
        if np.any(img1_shape[axes] % 2 == 1):
            raise ValueError(f"Image shape {img1_shape} along the chosen axes {axes} needs to be even.")
        raise NotImplementedError("Self FRC not implemented, yet.")
    else:
        img2_shape = np.array(img2.shape)
        if len(img1_shape) != len(img2_shape) or np.any(img1_shape != img2_shape):
            raise ValueError(
                f"Image #1 size {img1_shape} and image #2 size {img2_shape} are different, while they should be equal."
            )

    if supersampling > 1:
        base_grid = [np.linspace(-(d - 1) / 2, (d - 1) / 2, d) for d in img1_shape]

        interp_grid = [np.linspace(-(d - 1) / 2, (d - 1) / 2, d) for d in img1_shape]
        for a in axes:
            d = img1_shape[a] * 2
            interp_grid[a] = np.linspace(-(d - 1) / 4, (d - 1) / 4, d)
        interp_grid = np.meshgrid(*interp_grid, indexing="ij")
        interp_grid = np.transpose(interp_grid, [*range(1, len(img1_shape) + 1), 0])

        img1 = sp.interpolate.interpn(base_grid, img1, interp_grid, bounds_error=False, fill_value=None)
        img2 = sp.interpolate.interpn(base_grid, img2, interp_grid, bounds_error=False, fill_value=None)

        img1_shape = np.array(img1.shape)

    axes_shape = img1_shape[list(axes)]
    cut_off = np.min(axes_shape) // 2

    img1_f = np.fft.fftn(img1, axes=axes)
    img2_f = np.fft.fftn(img2, axes=axes)

    fc = img1_f * np.conj(img2_f)
    f1 = np.abs(img1_f) ** 2
    f2 = np.abs(img2_f) ** 2

    fc_r_int = azimuthal_integration(fc.real, axes=axes, domain="fourier")
    fc_i_int = azimuthal_integration(fc.imag, axes=axes, domain="fourier")
    fc_int = np.sqrt((fc_r_int**2) + (fc_i_int**2))
    f1_int = azimuthal_integration(f1, axes=axes, domain="fourier")
    f2_int = azimuthal_integration(f2, axes=axes, domain="fourier")

    f1s_f2s = f1_int * f2_int
    f1s_f2s = f1s_f2s + (f1s_f2s == 0)
    frc = fc_int / np.sqrt(f1s_f2s)

    rings_size = azimuthal_integration(np.ones_like(img1), axes=axes, domain="fourier")
    # Alternatively:
    # # The number of pixels in a ring is given by the surface.
    # # We compute the n-dimensional hyper-sphere surface, where n is given by the number of axes.
    # n = len(axes)
    # num_surf = 2 * np.pi ** (n / 2)
    # den_surf = sp.special.gamma(n / 2)
    # rings_size = np.concatenate(((1.0, ), num_surf / den_surf * np.arange(1, len(frc)) ** (n - 1)))

    Tnum = snrt + (2 * np.sqrt(snrt) + 1) / np.sqrt(rings_size)
    Tden = snrt + 1 + 2 * np.sqrt(snrt) / np.sqrt(rings_size)
    Thb = Tnum / Tden

    if smooth is not None and smooth > 1:
        win = sp.signal.windows.hann(smooth)
        win /= np.sum(win)
        frc = sp.ndimage.convolve(frc, win, mode="nearest")

    return frc[:cut_off], Thb[:cut_off]


def norm_cross_corr(
    img1: NDArray,
    img2: Optional[NDArray] = None,
    axes: Sequence[int] = (-2, -1),
    t_match: bool = False,
    mode_full: bool = True,
    compute_profile: bool = True,
    plot: bool = True,
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
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
        img2 = img1

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
            f, ax = plt.subplots()
            ax.plot(cc_l)
            ax.plot(np.ones_like(cc_l) * 0.5)
            ax.grid()
            f.tight_layout()

        return cc_n, cc_l
    else:
        return cc_n
