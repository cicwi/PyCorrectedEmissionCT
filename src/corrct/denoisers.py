"""
Advanced denoising methods.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from collections.abc import Sequence
from typing import Callable, Optional, Union, overload

import numpy as np
import scipy.signal as spsig
from numpy.typing import NDArray

from . import data_terms, operators, param_tuning, regularizers, solvers

eps = np.finfo(np.float32).eps


def _default_regularizer_l1dwl(r_w: Union[float, NDArray]) -> regularizers.BaseRegularizer:
    return regularizers.Regularizer_l1dwl(r_w, "bior4.4", 3)


@overload
def denoise_image(
    img: NDArray,
    reg_weight: Union[Sequence[float], NDArray],
    psf: Optional[NDArray] = None,
    pix_weights: Optional[NDArray] = None,
    iterations: int = 250,
    regularizer: Callable = _default_regularizer_l1dwl,
    lower_limit: Optional[float] = None,
    verbose: bool = True,
) -> tuple[NDArray, float]: ...


@overload
def denoise_image(
    img: NDArray,
    reg_weight: float,
    psf: Optional[NDArray] = None,
    pix_weights: Optional[NDArray] = None,
    iterations: int = 250,
    regularizer: Callable = _default_regularizer_l1dwl,
    lower_limit: Optional[float] = None,
    verbose: bool = True,
) -> NDArray: ...


def denoise_image(
    img: NDArray,
    reg_weight: Union[float, Sequence[float], NDArray] = 1e-2,
    psf: Optional[NDArray] = None,
    pix_weights: Optional[NDArray] = None,
    iterations: int = 250,
    regularizer: Callable = _default_regularizer_l1dwl,
    lower_limit: Optional[float] = None,
    verbose: bool = True,
) -> Union[NDArray, tuple[NDArray, float]]:
    """
    Denoise an image.

    Image denoiser based on (flat or weighted) least-squares, with wavelet minimization regularization.
    The weighted least-squares requires the local pixel-wise weights.
    It can be used to denoise sinograms and projections.

    Parameters
    ----------
    img : NDArray
        The image to denoise.
    reg_weight : Union[float, ArrayLike, NDArray], optional
        Weight of the regularization term. The default is 1e-2.
        If a sequence / array is passed, all the different values will be tested.
        The one minimizing the error over the cross-validation set will be chosen and returned.
    pix_weights : Union[ArrayLike, NDArray, None], optional
        The local weights of the pixels, for a weighted least-squares minimization.
        If None, a standard least-squares minimization is performed. The default is None.
    iterations : int, optional
        Number of iterations. The default is 250.
    regularizer : Callable, optional
        The one-argument constructor of a regularizer. The default is the DWL regularizer.
    lower_limit : Optional[float], optional
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

    def solver_spawn(lam_reg):
        # Using the PDHG solver from Chambolle and Pock
        reg = regularizer(lam_reg)
        return solvers.PDHG(
            verbose=verbose, data_term=data_term, regularizer=reg, data_term_test=data_term, leave_progress=False
        )

    def solver_call(solver: solvers.Solver, b_test_mask: Optional[NDArray] = None) -> tuple[NDArray, solvers.SolutionInfo]:
        x0 = img.copy()
        if b_test_mask is not None:
            med_img = spsig.medfilt2d(img, kernel_size=11)
            masked_pixels = b_test_mask > 0.5

            x0[masked_pixels] = med_img[masked_pixels]

        return solver(op, img, iterations, x0=x0, lower_limit=lower_limit, b_test_mask=b_test_mask)

    reg_weight = np.array(reg_weight)
    if reg_weight.size > 1:
        reg_help_cv = param_tuning.CrossValidation(img.shape, verbose=verbose, num_averages=3, plot_result=verbose)
        reg_help_cv.solver_spawning_function = solver_spawn
        reg_help_cv.solver_calling_function = solver_call

        f_avgs, _, _ = reg_help_cv.compute_loss_values(reg_weight)

        min_reg_weight, _ = reg_help_cv.fit_loss_min(reg_weight, f_avgs)
    else:
        min_reg_weight = reg_weight

    solver = solver_spawn(min_reg_weight)
    denoised_img, _ = solver_call(solver, None)

    if reg_weight.size == 1:
        return denoised_img
    else:
        return denoised_img, min_reg_weight
