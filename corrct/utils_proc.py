# -*- coding: utf-8 -*-
"""
Pre-processing and post-processing routines.

Created on Tue Mar 24 15:25:14 2020

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import scipy as sp
import scipy.signal

from . import operators
from . import solvers
from . import utils_reg

from typing import Sequence, Optional, Union, Tuple

from numpy.typing import ArrayLike, DTypeLike


eps = np.finfo(np.float32).eps


def get_circular_mask(
    vol_shape: ArrayLike,
    radius_offset: float = 0,
    coords_ball: Optional[Sequence[int]] = None,
    vol_origin: Optional[Sequence[float]] = None,
    mask_drop_off: str = "const",
    super_sampling: int = 1,
    dtype: DTypeLike = np.float32,
) -> ArrayLike:
    """
    Compute a circular mask for the reconstruction volume.

    Parameters
    ----------
    vol_shape : ArrayLike
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
    ArrayLike
        The circular mask.
    """
    vol_shape = np.array(vol_shape, dtype=int) * super_sampling

    coords = [np.linspace(-(s - 1) / (2 * super_sampling), (s - 1) / (2 * super_sampling), s, dtype=dtype) for s in vol_shape]
    if vol_origin:
        if len(coords) != len(vol_origin):
            raise ValueError(f"The volume shape ({len(coords)}), and the origin shape ({len(vol_origin)}) should match")
        coords = [c + vol_origin[ii] for ii, c in enumerate(coords)]
    coords = np.meshgrid(*coords, indexing="ij")

    if coords_ball is None:
        coords_ball = np.arange(-np.fmin(2, len(vol_shape)), 0, dtype=int)
    else:
        coords_ball = np.array(coords_ball, dtype=int)

    radius = np.min(vol_shape[coords_ball]) / (2 * super_sampling) + radius_offset

    coords = np.stack(coords, axis=0)
    if coords_ball.size == 1:
        dists = np.abs(coords[coords_ball, ...])
    else:
        dists = np.sqrt(np.sum(coords[coords_ball, ...] ** 2, axis=0))

    if mask_drop_off.lower() == "const":
        mask = (dists <= radius).astype(dtype)
    elif mask_drop_off.lower() == "sinc":
        cut_off = np.min(vol_shape[coords_ball]) / np.sqrt(2) - radius
        outter_region = 1.0 - (dists <= radius)
        outter_vals = 1.0 - np.sinc((dists - radius) / cut_off)
        mask = np.fmax(1 - outter_region * outter_vals, 0.0).astype(dtype)
    else:
        raise ValueError("Unknown drop-off function: %s" % mask_drop_off)

    if super_sampling > 1:
        new_shape = np.stack([vol_shape // super_sampling, np.ones_like(vol_shape) * super_sampling], axis=1).flatten()
        mask = mask.reshape(new_shape)
        mask = np.mean(mask, axis=tuple(np.arange(1, len(vol_shape) * 2, 2, dtype=int)))

    return mask


def pad_sinogram(
    sinogram: ArrayLike, width: Union[int, Sequence[int]], pad_axis: int = -1, mode: str = "edge", **kwds
) -> ArrayLike:
    """
    Pad the sinogram.

    Parameters
    ----------
    sinogram : ArrayLike
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
    ArrayLike
        The padded sinogram.
    """
    pad_size = [(0, 0)] * len(sinogram.shape)
    if len(width) == 1:
        width = (width, width)
    pad_size[pad_axis] = width

    return np.pad(sinogram, pad_size, mode=mode, **kwds)


def apply_flat_field(
    projs: ArrayLike,
    flats: ArrayLike,
    darks: Optional[ArrayLike] = None,
    crop: Optional[Sequence[int]] = None,
    dtype: DTypeLike = np.float32,
) -> ArrayLike:
    """
    Apply flat field.

    Parameters
    ----------
    projs : ArrayLike
        Projections.
    flats : ArrayLike
        Flat fields.
    darks : Optional[ArrayLike], optional
        Dark noise. The default is None.
    crop : Optional[Sequence[int]], optional
        Crop region. The default is None.
    dtype : DTypeLike, optional
        Data type of the processed data. The default is np.float32.

    Returns
    -------
    ArrayLike
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


def apply_minus_log(projs: ArrayLike) -> ArrayLike:
    """
    Apply -log.

    Parameters
    ----------
    projs : ArrayLike
        Projections.

    Returns
    -------
    ArrayLike
        Linearized projections.
    """
    return np.fmax(-np.log(projs), 0.0)


def compute_variance_poisson(
    Is: ArrayLike, I0: Optional[ArrayLike] = None, var_I0: Optional[ArrayLike] = None, normalized: bool = True
) -> ArrayLike:
    """
    Compute the variance of a signal subject to Poisson noise, against a reference intensity.

    The reference intensity can also be subject to Poisson and Gaussian noise.
    If the variance of the reference intensity is not passed, it will be assumed to be Poisson.

    Parameters
    ----------
    Is : ArrayLike
        The signal intensity.
    I0 : Optional[ArrayLike], optional
        The reference intensity. The default is None.
    var_I0 : Optional[ArrayLike], optional
        The variance of the reference intensity. The default is None.
        If not given, it will be assumed to be equal to I0.
    normalized : bool, optional
        Whether to renormalize by the mean of the reference intensity.

    Returns
    -------
    ArrayLike
        The computed variance.
    """
    var_Is = np.abs(Is)
    Is = np.fmax(Is, eps)

    if I0 is not None:
        if var_I0 is None:
            var_I0 = np.abs(I0)
        I0 = np.fmax(I0, eps)

        Is2 = Is ** 2
        I02 = I0 ** 2
        variance = (Is2 / I02) * (var_Is / Is2 + var_I0 / I02)
        if normalized:
            variance *= np.mean(I0)
        return variance
    else:
        return var_Is


def compute_variance_transmission(
    Is: ArrayLike, I0: ArrayLike, var_I0: Optional[ArrayLike] = None, normalized: bool = True
) -> ArrayLike:
    """
    Compute the variance of a linearized attenuation (transmission) signal, against a reference intensity.

    Parameters
    ----------
    Is : ArrayLike
        The transmitted signal.
    I0 : ArrayLike
        The reference intensity.
    var_I0 : Optional[ArrayLike], optional
        The variance of the reference intensity. The default is None.
        If not given, it will be assumed to be equal to I0.
    normalized : bool, optional
        Whether to renormalize by the mean of the reference intensity.

    Returns
    -------
    ArrayLike
        The computed variance.
    """
    var_Is = np.abs(Is)
    Is = np.fmax(Is, eps)

    if var_I0 is None:
        var_I0 = np.abs(I0)
    I0 = np.fmax(I0, eps)

    Is2 = Is ** 2
    I02 = I0 ** 2
    variance = (Is / I0) * (var_Is / Is2 + var_I0 / I02)
    if normalized:
        variance *= np.mean(I0)
    return variance


def compute_variance_weigth(
    variance: ArrayLike, percentile: float = 0.001, normalized: bool = False, semilog: bool = False
) -> ArrayLike:
    """
    Compute the weight associated to the given variance, in a weighted least-squares context.

    Parameters
    ----------
    variance : ArrayLike
        The variance of the signal.
    percentile : float
        Minimum percentile to discard. The default is 0.001 (0.1%)
    normalized : bool, optional
        Scale the largest weight to 1. The default is False.
    semilog : bool, optional
        Scale the variance over a logarithmic curve. It can be beneficial with
        high dynamic range data. The default is False.

    Returns
    -------
    ArrayLike
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
    if semilog:
        weight = np.log(weight + float(normalized is False)) + 1
    return 1 / weight


def denoise_image(
    img: ArrayLike,
    reg_weight: Union[float, ArrayLike] = 1e-2,
    psf: Optional[ArrayLike] = None,
    variances: Optional[ArrayLike] = None,
    iterations: int = 250,
    axes: Sequence[int] = (-2, -1),
    lower_limit: Optional[float] = None,
    verbose: bool = False,
) -> ArrayLike:
    """
    Denoise an image.

    Image denoiser based on (flat or weighted) least-squares, with wavelet minimization regularization.
    The weighted least-squares requires the local pixel-wise variances.
    It can be used to denoise sinograms and projections.

    Parameters
    ----------
    img : ArrayLike
        The image or sinogram to denoise.
    reg_weight : Union[float, Sequence[float], ArrayLike], optional
        Weight of the regularization term. The default is 1e-2.
        If a sequence / array is passed, all the different values will be tested.
        The one minimizing the error over the cross-validation set will be chosen and returned.
    variances : Optional[ArrayLike], optional
        The local variance of the pixels, for a weighted least-squares minimization.
        If None, a standard least-squares minimization is performed.
        The default is None.
    iterations : int, optional
        Number of iterations. The default is 250.
    axes : Sequence[int], optional
        Axes along which the regularization should be done. The default is (-2, -1).
    lower_limit : Optional[float], optional
        Lower clipping limit of the image. The default is None.
    verbose : bool, optional
        Turn verbosity on. The default is False.

    Returns
    -------
    ArrayLike
        Denoised image or sinogram.
    """
    if psf is None:
        op = operators.TransformIdentity(img.shape)
        padding = None
    else:
        padding = [(0, )] * len(img.shape)
        for ii, p in enumerate(psf.shape):
            padding[axes[ii]] = (p, )
        new_shape = np.ones_like(img.shape)
        new_shape[list(axes)] = psf.shape

        img = np.pad(img, padding, mode="edge")
        if variances is not None:
            variances = np.pad(variances, padding, mode="edge")
        op = operators.TransformConvolution(img.shape, psf.reshape(new_shape))

    if variances is not None:
        variances = np.abs(variances)
        img_weights = compute_variance_weigth(variances, normalized=True, semilog=True)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(img_weights[:, 0, :])
        # plt.show()
        data_term = solvers.DataFidelity_wl2(img_weights)
    else:
        data_term = solvers.DataFidelity_l2()

    if isinstance(axes, int):
        axes = (axes,)

    def solver_spawn(lam_reg):
        # Using the PDHG solver from Chambolle and Pock
        # reg = solvers.Regularizer_Grad(lam_reg, axes=axes)
        reg = solvers.Regularizer_l1dwl(lam_reg, "bior4.4", 3, axes=axes)
        return solvers.PDHG(verbose=verbose, data_term=data_term, regularizer=reg, data_term_test=data_term)

    def solver_call(solver, b_test_mask=None):
        if b_test_mask is not None:
            med_img = sp.signal.medfilt2d(img, kernel_size=11)
            masked_pixels = b_test_mask > 0.5
            x0 = img.copy()
            x0[masked_pixels] = med_img[masked_pixels]
        else:
            x0 = img.copy()

        return solver(op, img, iterations, x0=x0, lower_limit=lower_limit, b_test_mask=b_test_mask)

    reg_weight = np.array(reg_weight)
    if reg_weight.size > 1:
        reg_help_cv = utils_reg.CrossValidation(img.shape, verbose=True, num_averages=5, plot_result=True)
        reg_help_cv.solver_spawning_function = solver_spawn
        reg_help_cv.solver_calling_function = solver_call

        f_avgs, f_stds, _ = reg_help_cv.compute_loss_values(reg_weight)

        reg_weight, _ = reg_help_cv.fit_loss_min(reg_weight, f_avgs)

    solver = solver_spawn(reg_weight)
    (denoised_img, _) = solver_call(solver, None)

    if padding is not None:
        slicing = [slice(p[0], -p[0]) if p[0] else slice(None) for p in padding]
        denoised_img = denoised_img[tuple(slicing)]
    return denoised_img


def azimuthal_integration(img: ArrayLike, axes: Sequence[int] = (-2, -1), domain: str = "direct") -> ArrayLike:
    """
    Compute the azimuthal integration of a n-dimensional image or a stack of them.

    Parameters
    ----------
    img : ArrayLike
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
    ArrayLike
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
    r = np.sqrt(np.sum(coords ** 2, axis=0))

    # Reshape the volume to have the axes to be integrates as right-most axes
    img_tr_op = [*range(len(img.shape))]
    for a in axes:
        img_tr_op.append(img_tr_op.pop(a))
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
        az_img = [None] * num_imgs
        for ii in range(num_imgs):
            w_all = np.concatenate((w_l[ii, ...].flatten(), w_u[ii, ...].flatten()))
            az_img[ii] = np.bincount(r_all, weights=w_all)
        az_img = np.array(az_img)
        return np.reshape(az_img, (*img_old_shape, az_img.shape[-1]))
    else:
        w_all = np.concatenate((w_l.flatten(), w_u.flatten()))
        return np.bincount(r_all, weights=w_all)


def compute_frc(
    img1: ArrayLike,
    img2: ArrayLike,
    snrt: float = 0.2071,
    axes: Optional[Sequence[int]] = None,
    smooth: Optional[int] = 5,
    supersampling: int = 1,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the FRC/FSC (Fourier ring/shell correlation) between two images / volumes.

    Please refer to the following article for more information:
        M. van Heel and M. Schatz, “Fourier shell correlation threshold criteria,”
        J. Struct. Biol., vol. 151, no. 3, pp. 250–262, Sep. 2005.

    Parameters
    ----------
    img1 : ArrayLike
        First image / volume.
    img2 : ArrayLike
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
    ArrayLike
        The computed FRC/FSC.
    ArrayLike
        The threshold curve corresponding to the given threshod SNR.
    """
    img1_shape = np.array(img1.shape)

    if axes is None:
        axes = np.arange(-len(img1_shape), 0)

    if img2 is None:
        if np.any(img1_shape[list(axes)] % 2 == 1):
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
    fc_int = np.sqrt((fc_r_int ** 2) + (fc_i_int ** 2))
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
