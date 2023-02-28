# -*- coding: utf-8 -*-
"""
Ghost-imaging reconstruction utility functions.

@author: Nicola VIGANÒ, The European Synchrotron, Grenoble, France
"""

import numpy as np
from numpy.typing import NDArray

from typing import Union, Tuple, Callable

import corrct as cct


def find_reg_weight(
    A: cct.operators.ProjectorOperator,
    data: NDArray,
    iterations: int,
    reg: Callable[[float, NDArray], cct.regularizers.BaseRegularizer],
    lambda_range: Union[Tuple[float, float], NDArray],
    data_term: Union[cct.data_terms.DataFidelityBase, str] = "l2",
    parallel_eval: bool = False,
) -> float:
    sino_ones_mask = np.ones_like(data)
    bwd_prj_weights = A.bp(sino_ones_mask)
    bwd_prj_weights /= bwd_prj_weights.mean()

    # Instantiates the solver object, that is later used for computing the reconstruction
    def solver_spawn(lam_reg):
        # Using the PDHG solver from Chambolle and Pock
        return cct.solvers.PDHG(
            verbose=True, data_term=data_term, regularizer=reg(lam_reg, bwd_prj_weights), data_term_test=data_term
        )

    # Computes the reconstruction for a given solver and a given cross-validation data mask
    def solver_call(solver, b_test_mask=None):
        return solver(A, data, iterations, lower_limit=0, precondition=True, b_test_mask=b_test_mask)

    # Create the regularization weight finding helper object (using cross-validation)
    reg_help_cv = cct.param_tuning.CrossValidation(
        data.shape, num_averages=3, verbose=True, plot_result=True, parallel_eval=parallel_eval
    )
    reg_help_cv.solver_spawning_function = solver_spawn
    reg_help_cv.solver_calling_function = solver_call

    # Define the regularization weight range
    lams_reg = reg_help_cv.get_lambda_range(lambda_range[0], lambda_range[1], num_per_order=4)

    # Compute the loss function values for all the regularization weights
    f_avgs, f_stds, _ = reg_help_cv.compute_loss_values(lams_reg)

    # parabolic fit of minimum over the computer curve
    lam_min, _ = reg_help_cv.fit_loss_min(lams_reg, f_avgs)

    return lam_min
