# -*- coding: utf-8 -*-
"""
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


def get_circular_mask(
    vol_shape: ArrayLike,
    radius_offset: float = 0,
    coords_ball: Optional[Sequence[int]] = None,
    mask_drop_off: str = "const",
    data_type: DTypeLike = np.float32
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
    mask_drop_off : str, optional
        The mask data type. Allowed types: "const" | "sinc". The default is "const".
    data_type : DTypeLike, optional
        The type of mask. The default is np.float32.

    Raises
    ------
    ValueError
        In case of unknown mask_drop_off value.

    Returns
    -------
    ArrayLike
        The circular mask.
    """
    vol_shape = np.array(vol_shape, dtype=np.intp)

    coords = [np.linspace(-(s - 1) / 2, (s - 1) / 2, s, dtype=data_type) for s in vol_shape]
    coords = np.meshgrid(*coords, indexing="ij")

    if coords_ball is None:
        coords_ball = np.arange(-np.fmin(2, len(vol_shape)), 0, dtype=np.intp)
    else:
        coords_ball = np.array(coords_ball, dtype=np.intp)

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
    data_type: DTypeLike = np.float32
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
    data_type : DTypeLike, optional
        Data type of the processed data. The default is np.float32.

    Returns
    -------
    ArrayLike
        Falt-field corrected and linearized projections.
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


def denoise_image(
    img: ArrayLike,
    reg_weight: Union[float, Sequence[float], ArrayLike] = 1e-2,
    variances: Optional[ArrayLike] = None,
    iterations: int = 250,
    axes: Sequence[int] = (-2, -1),
    lower_limit: Optional[float] = None,
    verbose: bool = False
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
    OpI = operators.TransformIdentity(img.shape)

    if variances is not None:
        variances = np.abs(variances)
        min_nonzero_vars = np.min(variances[variances > 0])
        img_weights = 1 / np.fmax(variances, min_nonzero_vars)

        data_term = solvers.DataFidelity_wl2(img_weights)
    else:
        data_term = solvers.DataFidelity_l2()

    if isinstance(axes, int):
        axes = (axes,)

    def solver_spawn(lam_reg):
        # Using the PDHG solver from Chambolle and Pock
        # reg = solvers.Regularizer_Grad(lam_reg, axes=axes)
        reg = solvers.Regularizer_l1swl(lam_reg, "bior4.4", 3, axes=axes, normalized=False)
        return solvers.CP(verbose=verbose, data_term=data_term, regularizer=reg, data_term_test=data_term)

    def solver_call(solver, b_test_mask=None):
        if b_test_mask is not None:
            med_img = sp.signal.medfilt2d(img, kernel_size=11)
            masked_pixels = b_test_mask > 0.5
            x0 = img.copy()
            x0[masked_pixels] = med_img[masked_pixels]
        else:
            x0 = img.copy()

        return solver(
            OpI, img, iterations, x0=x0, lower_limit=lower_limit, b_test_mask=b_test_mask
        )

    reg_weight = np.array(reg_weight)
    if reg_weight.size > 1:
        reg_help_cv = utils_reg.CrossValidation(img.shape, verbose=True, num_averages=5, plot_result=True)
        reg_help_cv.solver_spawning_function = solver_spawn
        reg_help_cv.solver_calling_function = solver_call

        f_avgs, f_stds, _ = reg_help_cv.compute_loss_values(reg_weight)

        reg_weight, _ = reg_help_cv.fit_loss_min(reg_weight, f_avgs)

    solver = solver_spawn(reg_weight)
    (denoised_img, _) = solver_call(solver, None)
    return denoised_img
