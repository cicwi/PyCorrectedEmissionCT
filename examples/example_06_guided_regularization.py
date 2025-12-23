# -*- coding: utf-8 -*-
"""
Example demonstrating the use of the cross-validation routines for finding regularization weights.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import os

import numpy as np
from numpy.typing import ArrayLike, NDArray

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as ax_g
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import corrct as cct
from corrct.solvers import SolutionInfo

try:
    import phantom
except ImportError:
    import sys

    sys.path.insert(0, '')
    cct.testing.download_phantom()
    import phantom


def cm2inch(x: ArrayLike) -> tuple[float, float]:
    """Convert cm to inch.

    Parameters
    ----------
    x : ArrayLike
        Sizes in cm.

    Returns
    -------
    Tuple[float, float]
        Sizes in inch.
    """
    return tuple(np.array(x) / 2.54)


def plot_loss_figure(
    lams_reg: NDArray,
    f_avgs: NDArray,
    f_stds: NDArray | None,
    label: str,
    err_l1: NDArray,
    err_l2: NDArray,
    show_zoom: bool = False,
) -> Figure:
    fig = plt.figure(figsize=cm2inch([32, 16]))
    gs = fig.add_gridspec(2, 2)
    ax_test = fig.add_subplot(gs[:, 0])
    ax_test.set_xscale("log", nonpositive="clip")
    if f_stds is not None:
        ax_test.errorbar(lams_reg, f_avgs, yerr=f_stds, ecolor=(0.5, 0.5, 0.5), elinewidth=1, capsize=2, label=label)
    else:
        ax_test.plot(lams_reg, f_avgs, label=label)

    if show_zoom:
        ax_ins_test = inset_axes(ax_test, width="30%", height="20%", loc=3, borderpad=0.5)
        ax_ins_test.yaxis.tick_right()
        ax_ins_test.xaxis.tick_top()
        ax_ins_test.set_xscale("log", nonpositive="clip")
        ax_ins_test.errorbar(lams_reg, f_avgs, yerr=f_stds, ecolor=[0.5, 0.5, 0.5], elinewidth=1, capsize=2)
        x_lims = [0.8, 12]
        ax_ins_test.set_xlim(x_lims[0], x_lims[1])
        useful_vals = np.logical_and(lams_reg > x_lims[0], lams_reg < x_lims[1])
        if f_stds is not None:
            y_lims = [np.min(f_avgs[useful_vals] - f_stds[useful_vals]), np.max(f_avgs[useful_vals] + f_stds[useful_vals])]
        else:
            y_lims = [np.min(f_avgs[useful_vals]), np.max(f_avgs[useful_vals])]
        y_range_5 = 0.05 * (y_lims[1] - y_lims[0])
        y_lims = [y_lims[0] - y_range_5, y_lims[1] + y_range_5]
        ax_ins_test.set_ylim(y_lims[0], y_lims[1])
        # ax_ins_test.set_ylim(0.000225, 0.000350)
        ax_ins_test.set_xticks([1, 10])
        ax_ins_test.grid()
        ax_g.inset_locator.mark_inset(ax_test, ax_ins_test, loc1=1, loc2=4, linestyle="--", alpha=0.5)
    ax_test.grid()
    for tl in ax_test.get_xticklabels():
        tl.set_fontsize(13)
    for tl in ax_test.get_yticklabels():
        tl.set_fontsize(13)
    ax_test.set_xlabel(r"$\lambda$ values", fontsize=16)
    ax_test.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
    ax_test.legend(fontsize=13)

    ax_err1 = fig.add_subplot(gs[0, 1], sharex=ax_test)
    ax_err1.set_xscale("log", nonpositive="clip")
    ax_err1.plot(lams_reg, err_l1, label="Error - l1-norm")
    for tl in ax_err1.get_xticklabels():
        tl.set_fontsize(13)
    for tl in ax_err1.get_yticklabels():
        tl.set_fontsize(13)
    ax_err1.set_xlabel(r"$\lambda$ values", fontsize=16)
    ax_err1.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
    ax_err1.grid()
    ax_err1.legend(fontsize=13)

    ax_err2 = fig.add_subplot(gs[1, 1], sharex=ax_test)
    ax_err2.set_xscale("log", nonpositive="clip")
    ax_err2.plot(lams_reg, err_l2, label="Error - l2-norm ^ 2")
    for tl in ax_err2.get_xticklabels():
        tl.set_fontsize(13)
    for tl in ax_err2.get_yticklabels():
        tl.set_fontsize(13)
    ax_err2.set_xlabel(r"$\lambda$ values", fontsize=16)
    ax_err2.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
    ax_err2.grid()
    ax_err2.legend(fontsize=13)
    fig.tight_layout()

    return fig


def iso_tv_seminorm(x: NDArray) -> float:
    """Compute the isotropic TV semi-norm.

    Used in the L-curve.

    Parameters
    ----------
    x : NDArray
        Input array.

    Returns
    -------
    float
        The isotropic TV semi-norm of the input array.
    """
    op = cct.operators.TransformGradient(x.shape)
    d = op(x)
    d = np.linalg.norm(d, axis=0, ord=2)
    return float(np.linalg.norm(d.flatten(), ord=1))


if __name__ == "__main__":
    # Create the phantom shape
    ph_or = np.squeeze(phantom.modified_shepp_logan([256, 256, 3]).astype(np.float32))
    ph_or = ph_or[:, :, 1]

    # Simulate the XRF-CT acquisition data (sinogram)
    ph, _, _ = cct.testing.phantom_assign_concentration(ph_or)
    sinogram, angles, expected_ph, background_avg = cct.testing.create_sino(
        ph, 30, add_poisson=True, dwell_time_s=2e-2, background_avg=1e-2, background_std=1e-4
    )

    iterations = 200
    lower_limit = 0.0
    vol_mask = cct.processing.circular_mask(ph_or.shape)

    # Subtract background from sinogram
    sino_substr = sinogram - background_avg

    # The pixel weights are computed from their variances
    sino_variance = cct.processing.noise.compute_variance_poisson(sinogram)
    sino_weights = cct.processing.noise.compute_variance_weight(sino_variance)

    # Data fitting term: weighted least-squares, based on the standard deviation of the noise.
    data_term_lsw = cct.data_terms.DataFidelity_wl2(sino_weights)

    reg = cct.regularizers.Regularizer_TV2D
    # reg = lambda l: cct.regularizers.Regularizer_l1swl(l, "haar", 4)

    # Instantiates the solver object, that is later used for computing the reconstruction
    def solver_init(lam_reg: float):
        # Using the PDHG solver from Chambolle and Pock
        return cct.solvers.PDHG(
            verbose=True, data_term=data_term_lsw, regularizer=reg(lam_reg), data_term_test=data_term_lsw, leave_progress=False
        )

    # Computes the reconstruction for a given solver and a given cross-validation data mask
    def solver_exec(solver, b_test_mask: NDArray | None = None) -> tuple[NDArray, SolutionInfo]:
        with cct.projectors.ProjectorUncorrected(ph.shape, angles) as prj:
            return solver(prj, sino_substr, iterations, x_mask=vol_mask, lower_limit=lower_limit, b_test_mask=b_test_mask)

    def solve_reg(lam_reg: float, b_test_mask: NDArray | None = None) -> tuple[NDArray, SolutionInfo]:
        solver = solver_init(lam_reg)
        return solver_exec(solver, b_test_mask)

    print("Reconstructing:")
    # Create the regularization weight finding helper object (using cross-validation)
    hpt_cv = cct.param_tuning.CrossValidation(sinogram.shape, verbose=True, num_averages=3)
    hpt_cv.task_exec_function = solve_reg
    # # Alternatively, one can use the two functions API, where the initialization is taken care by a dedicated function.
    # # This could be useful when desiring more precise execution and initialization times.
    # hpt_cv.task_init_function = solver_init
    # hpt_cv.task_exec_function = solver_exec

    # Define the regularization weight range
    lams_reg = cct.param_tuning.get_lambda_range(1e-3, 1e1, num_per_order=4)

    # Compute the loss function values for all the regularization weights
    f_avgs, f_stds, _ = hpt_cv.compute_loss_values(lams_reg)
    # Compute the error values for all the regularization weights
    err_l1, err_l2 = hpt_cv.compute_reconstruction_error(lams_reg, expected_ph)

    # parabolic fit of minimum over the computer curve
    lam_min, _ = hpt_cv.fit_loss_min(lams_reg, f_avgs)

    # Regularized weighted least-squares solver (PDHG), on the whole dataset
    solver = solver_init(lam_min)
    rec, rec_info = solver_exec(solver)

    with cct.projectors.ProjectorUncorrected(ph.shape, angles) as A:
        # Unregularized weighted least-squares solver (PDHG), for reference
        solver_wls = cct.solvers.PDHG(verbose=True, data_term=data_term_lsw)
        rec_wls, _ = solver_wls(A, sino_substr, iterations, x_mask=vol_mask, lower_limit=0)

    # Create the regularization weight finding helper object (using L-curve)
    hpt_lc = cct.param_tuning.LCurve(iso_tv_seminorm, verbose=True, plot_result=True)
    hpt_lc.task_exec_function = solve_reg

    f_vals_lc = hpt_lc.compute_loss_values(lams_reg)

    save_figs = False

    if save_figs:
        base_fig_dir = "figures/aided-reg"

        # Making figures
        if not os.path.exists(base_fig_dir):
            os.makedirs(base_fig_dir)

    fig = plot_loss_figure(lams_reg, f_avgs, f_stds, "Cross-Validation error - wl2-norm", err_l1, err_l2)

    if save_figs:
        fig.savefig(os.path.join(base_fig_dir, "reg-error-cv.eps"))
        fig.savefig(os.path.join(base_fig_dir, "reg-error-cv.png"))

    fig = plot_loss_figure(lams_reg, f_vals_lc, None, "L-curve values - TV-seminorm", err_l1, err_l2)

    if save_figs:
        fig.savefig(os.path.join(base_fig_dir, "reg-error-lc.eps"))
        fig.savefig(os.path.join(base_fig_dir, "reg-error-lc.png"))

    # Reconstructions
    fig = plt.figure(figsize=cm2inch([48, 12]))
    gs = fig.add_gridspec(4, 4)
    ax_ph = fig.add_subplot(gs[:, 0])
    im_ph = ax_ph.imshow(expected_ph)
    ax_ph.set_title("Phantom")
    fig.colorbar(im_ph, ax=ax_ph)

    ax_sino_clean = fig.add_subplot(gs[0, 1])
    with cct.projectors.ProjectorUncorrected(ph_or.shape, angles) as p:
        sino_clean = p.fp(expected_ph)
    im_sino_clean = ax_sino_clean.imshow(sino_clean)
    ax_sino_clean.set_title("Clean sinogram")

    ax_sino_noise = fig.add_subplot(gs[1, 1])
    im_sino_noise = ax_sino_noise.imshow(sino_substr)
    ax_sino_noise.set_title("Noisy sinogram")

    ax_sino_lines = fig.add_subplot(gs[2:, 1])
    im_sino_lines = ax_sino_lines.plot(sino_substr[9, :], label="Noisy")
    im_sino_lines = ax_sino_lines.plot(sino_clean[9, :], label="Clean")
    ax_sino_lines.set_title("Sinograms - angle: 10")
    ax_sino_lines.grid()
    ax_sino_lines.legend()

    ax_ls = fig.add_subplot(gs[:, 2])
    im_ls = ax_ls.imshow(np.squeeze(rec_wls))
    ax_ls.set_title(solver_wls.info().upper())
    fig.colorbar(im_ls, ax=ax_ls)

    ax_reg = fig.add_subplot(gs[:, 3])
    label_2 = solver.info().upper()
    im_reg = ax_reg.imshow(np.squeeze(rec))
    ax_reg.set_title(solver.info().upper())
    fig.colorbar(im_reg, ax=ax_reg)

    fig.tight_layout()

    if save_figs:
        fig.savefig(os.path.join(base_fig_dir, "rec-comparison.eps"))
        fig.savefig(os.path.join(base_fig_dir, "rec-comparison.png"))

    plt.tight_layout()
    plt.show(block=False)

    print(np.std((expected_ph - rec) / (expected_ph + (expected_ph == 0))))
