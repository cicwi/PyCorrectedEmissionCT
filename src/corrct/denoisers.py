"""
Advanced denoising methods.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from collections.abc import Callable, Sequence
from typing import overload

import numpy as np
import scipy.signal as spsig
from numpy.typing import NDArray

from . import data_terms, operators, param_tuning, regularizers, solvers

eps = np.finfo(np.float32).eps


def _default_regularizer_l1dwl(r_w: float | NDArray) -> regularizers.BaseRegularizer:
    return regularizers.Regularizer_l1dwl(r_w, "bior4.4", 3)


@overload
def denoise_image(
    img: NDArray,
    reg_weight: Sequence[float] | NDArray,
    psf: NDArray | None = None,
    pix_weights: NDArray | None = None,
    iterations: int = 250,
    regularizer: Callable = _default_regularizer_l1dwl,
    lower_limit: float | None = None,
    verbose: bool = True,
) -> tuple[NDArray, float]: ...


@overload
def denoise_image(
    img: NDArray,
    reg_weight: float,
    psf: NDArray | None = None,
    pix_weights: NDArray | None = None,
    iterations: int = 250,
    regularizer: Callable = _default_regularizer_l1dwl,
    lower_limit: float | None = None,
    verbose: bool = True,
) -> NDArray: ...


def denoise_image(
    img: NDArray,
    reg_weight: float | Sequence[float] | NDArray = 1e-2,
    psf: NDArray | None = None,
    pix_weights: NDArray | None = None,
    iterations: int = 250,
    regularizer: Callable = _default_regularizer_l1dwl,
    lower_limit: float | None = None,
    verbose: bool = True,
) -> NDArray | tuple[NDArray, float]:
    """
    Denoise an image.

    Image denoiser based on (flat or weighted) least-squares, with wavelet minimization regularization.
    The weighted least-squares requires the local pixel-wise weights.
    It can be used to denoise sinograms and projections.

    Parameters
    ----------
    img : NDArray
        The image to denoise.
    reg_weight : float | ArrayLike | NDArray, optional
        Weight of the regularization term. The default is 1e-2.
        If a sequence / array is passed, all the different values will be tested.
        The one minimizing the error over the cross-validation set will be chosen and returned.
    pix_weights : ArrayLike | NDArray | None, optional
        The local weights of the pixels, for a weighted least-squares minimization.
        If None, a standard least-squares minimization is performed. The default is None.
    iterations : int, optional
        Number of iterations. The default is 250.
    regularizer : Callable, optional
        The one-argument constructor of a regularizer. The default is the DWL regularizer.
    lower_limit : float | None, optional
        Lower clipping limit of the image. The default is None.
    verbose : bool, optional
        Turn verbosity on. The default is True.

    Returns
    -------
    NDArray
        Denoised image.
    """
    if psf is None:
        op = operators.TransformIdentity(img.shape)
    else:
        op = operators.TransformConvolution(img.shape, psf)

    if pix_weights is None:
        data_term = data_terms.DataFidelity_l2()
    else:
        data_term = data_terms.DataFidelity_wl2(pix_weights)

    def solver_run(lam_reg, b_val_mask: NDArray | None = None) -> tuple[NDArray, solvers.SolutionInfo]:
        # Using the PDHG solver from Chambolle and Pock
        reg = regularizer(lam_reg)
        solver = solvers.PDHG(
            verbose=verbose,
            data_term=data_term,
            regularizer=reg,
            data_term_val=data_term,
            leave_progress=False,
            criterion="loss_val",
        )

        x0 = img.copy()
        if b_val_mask is not None:
            med_img = spsig.medfilt2d(img, kernel_size=11)
            masked_pixels = b_val_mask > 0.5

            x0[masked_pixels] = med_img[masked_pixels]

        return solver(op, img, iterations, x0=x0, lower_limit=lower_limit, b_val_mask=b_val_mask)

    reg_weight = np.array(reg_weight)
    if reg_weight.size > 1:
        reg_help_cv = param_tuning.CrossValidation(img.shape, verbose=verbose, num_averages=3, plot_result=verbose)
        reg_help_cv.task_exec_function = solver_run

        f_avgs, _, _ = reg_help_cv.compute_loss_values(reg_weight)

        min_reg_weight, _ = reg_help_cv.fit_loss_min(reg_weight, f_avgs)
    else:
        min_reg_weight = reg_weight

    pix_mask = param_tuning.create_random_test_mask(img.shape)
    denoised_img, _ = solver_run(min_reg_weight, pix_mask)

    if reg_weight.size == 1:
        return denoised_img
    else:
        return denoised_img, float(min_reg_weight)
