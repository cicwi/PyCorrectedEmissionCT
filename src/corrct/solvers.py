#!/usr/bin/env python3
"""
Solvers for the tomographic reconstruction problem.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import copy as cp
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray
from tqdm.auto import tqdm

from . import data_terms, filters, operators, projectors, regularizers

eps = np.finfo(np.float32).eps


NDArrayFloat = NDArray[np.floating]


def power_method(A: operators.BaseTransform, b: NDArrayFloat, iterations: int = 5) -> tuple[float, tuple[int, ...], DTypeLike]:
    """
    Compute the l2-norm of the operator A, with the power method.

    Parameters
    ----------
    A : BaseTransform
        The forward operator whose l2-norm needs to be computed.
    b : NDArrayFloat
        The data vector (used only for shape and dtype).
    iterations : int, optional
        Number of power-method iterations. The default is 5.

    Returns
    -------
    Tuple[float, Tuple[int], DTypeLike]
        The l2-norm of A, and the shape and type of the solution.
    """
    x: NDArrayFloat = np.random.rand(*b.shape).astype(b.dtype)
    x /= np.linalg.norm(x)
    x = A.T(x)

    x_norm = np.linalg.norm(x)
    L = x_norm

    for _ in range(iterations):
        x /= x_norm
        x = A.T(A(x))
        x_norm = np.linalg.norm(x)
        L = np.sqrt(x_norm)

    return float(L), x.shape, x.dtype


def compute_diagonal_scaling(
    A_abs: operators.BaseTransform,
    At_abs: operators.BaseTransform,
    b: NDArrayFloat,
    regs: Sequence[regularizers.BaseRegularizer],
    relaxation_sigma: float = 1.0,
    relaxation_tau: float = 1.0,
    x_mask: NDArray | None = None,
    b_mask: NDArray | None = None,
) -> tuple[NDArray, NDArray, tuple[int, ...], DTypeLike]:
    # Back-projection diagonal re-scaling
    # Tau = |A^T| 1_b  (row sums -> image-space diagonal)
    b_ones = np.ones_like(b)
    if b_mask is not None:
        b_ones *= b_mask
    tau = np.abs(At_abs(b_ones))
    for reg in regs:
        tau += reg.initialize_sigma_tau(tau)
    tau[(tau / np.max(tau)) < 1e-5] = 1
    tau = relaxation_tau / tau

    # Forward-projection diagonal re-scaling
    # Sigma = |A| 1_x  (column sums -> measurement-space diagonal)
    x_ones = np.ones_like(tau)
    if x_mask is not None:
        x_ones *= x_mask
    sigma = np.abs(A_abs(x_ones))
    sigma[(sigma / np.max(sigma)) < 1e-5] = 1.0
    sigma = relaxation_sigma / sigma

    return sigma, tau, tau.shape, tau.dtype


def compute_Lipschitz_scaling(
    A: operators.BaseTransform,
    b: NDArrayFloat,
    regs: Sequence[regularizers.BaseRegularizer],
    relaxation_sigma: float = 1.0,
    relaxation_tau: float = 1.0,
) -> tuple[float, float | NDArray, tuple[int, ...], DTypeLike]:
    L, x_shape, x_dtype = power_method(A, b)
    tau = L

    dummy_x = np.empty(x_shape, dtype=x_dtype)
    for reg in regs:
        tau += reg.initialize_sigma_tau(dummy_x)

    tau = relaxation_tau / tau
    sigma = relaxation_sigma / L

    return sigma, tau, x_shape, x_dtype


class SolutionInfo:
    """Reconstruction info."""

    method: str
    iterations: int
    max_iterations: int

    residual0: float | np.floating
    residual0_cv: float | np.floating

    residuals: NDArrayFloat
    residuals_cv: NDArrayFloat
    tolerance: float | np.floating | None

    def __init__(
        self,
        method: str,
        max_iterations: int,
        tolerance: float | np.floating | None,
        residual0: float = np.inf,
        residual0_cv: float = np.inf,
    ) -> None:
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.residual0 = residual0
        self.residual0_cv = residual0_cv

        self.residuals = np.zeros(max_iterations)
        self.residuals_cv = np.zeros(max_iterations)

        self.iterations = 0

    @property
    def residuals_rel(self) -> NDArrayFloat:
        return self.residuals / self.residual0

    @property
    def residuals_cv_rel(self) -> NDArrayFloat:
        return self.residuals_cv / self.residual0_cv


class Solver(ABC):
    """
    Initialize the base solver class.

    Parameters
    ----------
    verbose : bool, optional
        Turn on verbose output. The default is False.
    tolerance : float | None, optional
        Tolerance on the data residual for computing when to stop iterations.
        The default is None.
    relaxation : float, optional
        The relaxation length. The default is 1.0.
    data_term : str | data_terms.DataFidelityBase, optional
        Data fidelity term for computing the data residual. The default is "l2".
    data_term_test : data_terms.DataFidelityBase | None, optional
        The data fidelity to be used for the test set.
        If None, it will use the same as for the rest of the data.
        The default is None.
    """

    def __init__(
        self,
        verbose: bool = False,
        leave_progress: bool = True,
        relaxation: float = 1.0,
        tolerance: float | None = None,
        data_term: str | data_terms.DataFidelityBase = "l2",
        data_term_test: str | data_terms.DataFidelityBase | None = None,
    ):
        self.verbose = verbose
        self.leave_progress = leave_progress
        self.relaxation = relaxation
        self.tolerance = tolerance

        self.data_term = self._initialize_data_fidelity_function(data_term)
        if data_term_test is None:
            data_term_test = self.data_term
        else:
            data_term_test = self._initialize_data_fidelity_function(data_term_test)
        self.data_term_test = cp.deepcopy(data_term_test)

    def info(self) -> str:
        """
        Return the solver info.

        Returns
        -------
        str
            Solver info string.
        """
        return type(self).__name__

    def upper(self) -> str:
        """
        Return the upper case name of the solver.

        Returns
        -------
        str
            Upper case string name of the solver.
        """
        return type(self).__name__.upper()

    def lower(self) -> str:
        """
        Return the lower case name of the solver.

        Returns
        -------
        str
            Lower case string name of the solver.
        """
        return type(self).__name__.lower()

    @abstractmethod
    def __call__(
        self, A: operators.BaseTransform, b: NDArrayFloat, *args: Any, **kwds: Any
    ) -> tuple[NDArrayFloat, SolutionInfo]:
        """Execute the reconstruction of the data.

        Parameters
        ----------
        A : operators.BaseTransform
            The projection operator.
        b : NDArrayFloat
            The data to be reconstructed.

        Returns
        -------
        Tuple[NDArrayFloat, SolutionInfo]
            The reconstruction and related information.
        """

    @staticmethod
    def _initialize_data_fidelity_function(data_term: str | data_terms.DataFidelityBase) -> data_terms.DataFidelityBase:
        if isinstance(data_term, str):
            if data_term.lower() == "l2":
                return data_terms.DataFidelity_l2()
            elif data_term.lower() == "kl":
                return data_terms.DataFidelity_KL()
            else:
                raise ValueError(f"Unknown data term: '{data_term}', only accepted terms are: 'l2' | 'kl'.")
        elif isinstance(data_term, (data_terms.DataFidelity_l2, data_terms.DataFidelity_KL)):
            return cp.deepcopy(data_term)
        else:
            raise ValueError(f"Unsupported data term: '{data_term.info()}', only accepted terms are 'kl' and 'l2'-based.")

    @staticmethod
    def _initialize_regularizer(
        regularizer: regularizers.BaseRegularizer | None | Sequence[regularizers.BaseRegularizer],
    ) -> Sequence[regularizers.BaseRegularizer]:
        if regularizer is None:
            return []
        elif isinstance(regularizer, regularizers.BaseRegularizer):
            return [regularizer]
        elif isinstance(regularizer, (list, tuple)):
            check_regs_ok = [isinstance(r, regularizers.BaseRegularizer) for r in regularizer]
            if not np.all(check_regs_ok):
                raise ValueError(
                    "The following regularizers are not derived from the regularizers.BaseRegularizer class: "
                    f"{np.array(np.arange(len(check_regs_ok))[np.array(check_regs_ok, dtype=bool)])}"
                )
            else:
                return list(regularizer)
        else:
            raise ValueError("Unknown regularizer type.")

    @staticmethod
    def _initialize_b_masks(
        b: NDArrayFloat, b_mask: NDArrayFloat | None, b_test_mask: NDArrayFloat | None
    ) -> tuple[NDArrayFloat | None, NDArrayFloat | None]:
        if b_test_mask is not None:
            if b_mask is None:
                b_mask = np.ones_like(b)
            # As we are being passed a test residual pixel mask, we need
            # to make sure to mask those pixels out from the reconstruction.
            # At the same time, we need to remove any masked pixel from the test count.
            b_mask, b_test_mask = b_mask * (1 - b_test_mask), b_test_mask * b_mask
        return (b_mask, b_test_mask)


class FBP(Solver):
    """Implementation of the Filtered Back-Projection (FBP) algorithm."""

    def __init__(
        self,
        verbose: bool = False,
        leave_progress: bool = False,
        regularizer: Sequence[regularizers.BaseRegularizer] | regularizers.BaseRegularizer | None = None,
        data_term: str | data_terms.DataFidelityBase = "l2",
        fbp_filter: str | NDArrayFloat | filters.Filter = "ramp",
        pad_mode: str = "constant",
    ):
        """Initialize the Filtered Back-Projection (FBP) algorithm.

        Parameters
        ----------
        verbose : bool, optional
            Turn on verbose output. The default is False.
        leave_progress: bool, optional
            Leave the progress bar after the computation is finished. The default is True.
        regularizer : Sequence[regularizers.BaseRegularizer] | regularizers.BaseRegularizer | None, optional
            NOT USED, only exposed for compatibility reasons.
        data_term : str | data_terms.DataFidelityBase, optional
            NOT USED, only exposed for compatibility reasons.
        fbp_filter : str | NDArrayFloat | filters.Filter, optional
            FBP filter to use. Either a string from scikit-image's list of `iradon` filters, or an array. The default is "ramp".
        pad_mode: str, optional
            The padding mode to use for the linear convolution. The default is "constant".
        """
        super().__init__(verbose=verbose)
        if isinstance(fbp_filter, str):
            fbp_filter = fbp_filter.lower()
        self.fbp_filter = fbp_filter
        self.pad_mode = pad_mode

    def info(self) -> str:
        """
        Return the solver info.

        Returns
        -------
        str
            Solver info string.
        """
        if isinstance(self.fbp_filter, str):
            return super().info() + "(F:" + self.fbp_filter.upper() + ")"
        elif isinstance(self.fbp_filter, np.ndarray):
            return super().info() + "(F:" + filters.FilterCustom.__name__.upper() + ")"
        else:
            return super().info() + "(F:" + type(self.fbp_filter).__name__.upper() + ")"

    def __call__(  # noqa: C901
        self,
        A: operators.BaseTransform,
        b: NDArrayFloat,
        iterations: int = 0,
        x0: NDArrayFloat | None = None,
        lower_limit: float | NDArrayFloat | None = None,
        upper_limit: float | NDArrayFloat | None = None,
        x_mask: NDArrayFloat | None = None,
        b_mask: NDArrayFloat | None = None,
    ) -> tuple[NDArrayFloat, SolutionInfo]:
        """
        Reconstruct the data, using the FBP algorithm.

        Parameters
        ----------
        A : BaseTransform
            Projection operator.
        b : NDArrayFloat
            Data to reconstruct.
        iterations : int
            Number of iterations.
        x0 : NDArrayFloat | None, optional
            Initial solution. The default is None.
        lower_limit : float | NDArrayFloat | None, optional
            Lower clipping value. The default is None.
        upper_limit : float | NDArrayFloat | None, optional
            Upper clipping value. The default is None.
        x_mask : NDArrayFloat | None, optional
            Solution mask. The default is None.
        b_mask : NDArrayFloat | None, optional
            Data mask. The default is None.

        Raises
        ------
        ValueError
            In case the data is 1D.

        Returns
        -------
        tuple[NDArrayFloat, SolutionInfo]
            The reconstruction, and None.
        """
        if len(b.shape) < 2:
            raise ValueError(f"Data should be at least 2-dimensional (b.shape = {b.shape})")

        info = SolutionInfo(self.info(), max_iterations=0, tolerance=0.0)

        if isinstance(self.fbp_filter, str):
            if self.fbp_filter in ("mr", "data"):
                local_filter = filters.FilterMR(projector=A)
            else:
                local_filter = filters.FilterFBP(filter_name=self.fbp_filter)
        elif isinstance(self.fbp_filter, np.ndarray):
            local_filter = filters.FilterCustom(self.fbp_filter)
        else:
            local_filter = self.fbp_filter
        local_filter.pad_mode = self.pad_mode

        if isinstance(A, operators.ProjectorOperator):
            pre_weights = A.get_pre_weights()
            if pre_weights is not None:
                b = b * pre_weights

        b_f = local_filter(b)

        x = A.T(b_f)

        if lower_limit is not None or upper_limit is not None:
            x = x.clip(lower_limit, upper_limit)
        if x_mask is not None:
            x *= x_mask

        return x, info


class SART(Solver):
    """Solver class implementing the Simultaneous Algebraic Reconstruction Technique (SART) algorithm."""

    def compute_residual(
        self,
        A: Callable,
        b: NDArrayFloat,
        x: NDArrayFloat,
        A_num_rows: int,
        b_mask: NDArrayFloat | None,
    ) -> NDArrayFloat:
        """Compute the solution residual.

        Parameters
        ----------
        A : Callable
            The forward projector.
        b : NDArrayFloat
            The detector data.
        x : NDArrayFloat
            The current solution
        A_num_rows : int
            The number of projections.
        b_mask : NDArrayFloat | None
            The mask to apply

        Returns
        -------
        NDArrayFloat
            The residual.
        """
        fp = np.stack([A(x, ii) for ii in range(A_num_rows)], axis=-1)
        fp = np.ascontiguousarray(fp, dtype=b.dtype)
        res = fp - b
        if b_mask is not None:
            res *= b_mask
        return res

    def __call__(  # noqa: C901
        self,
        A: Callable[[NDArray, int], NDArray] | projectors.ProjectorUncorrected,
        b: NDArrayFloat,
        iterations: int,
        A_num_rows: int | None = None,
        At: Callable | None = None,
        x0: NDArrayFloat | None = None,
        lower_limit: float | NDArrayFloat | None = None,
        upper_limit: float | NDArrayFloat | None = None,
        x_mask: NDArrayFloat | None = None,
        b_mask: NDArrayFloat | None = None,
    ) -> tuple[NDArrayFloat, SolutionInfo]:
        """
        Reconstruct the data, using the SART algorithm.

        Parameters
        ----------
        A : Callable[[NDArray, int], NDArray] | projectors.ProjectorUncorrected
            Projection operator.
        b : NDArrayFloat
            Data to reconstruct.
        iterations : int
            Number of iterations.
        A_num_rows : int | None
            Number of projections.
        x0 : NDArrayFloat | None, optional
            Initial solution. The default is None.
        At : Callable | None, optional
            The back-projection operator. This is only needed if the projection operator does not have an adjoint.
            The default is None.
        lower_limit : float | NDArrayFloat | None, optional
            Lower clipping value. The default is None.
        upper_limit : float | NDArrayFloat | None, optional
            Upper clipping value. The default is None.
        x_mask : NDArrayFloat | None, optional
            Solution mask. The default is None.
        b_mask : NDArrayFloat | None, optional
            Data mask. The default is None.

        Returns
        -------
        tuple[NDArrayFloat, SolutionInfo]
            The reconstruction, and the residuals.
        """
        if isinstance(A, projectors.ProjectorUncorrected):
            p = A

            if not p.projector_backend.has_individual_projs:
                raise ValueError("The projector needs to have enabled single projections.")

            A = lambda x, ii: p.fp_angle(x, ii)  # noqa: E731
            if isinstance(p, projectors.ProjectorAttenuationXRF):
                At = lambda y, ii: p.bp_angle(y, ii, single_line=True)  # noqa: E731
            else:
                At = lambda y, ii: p.bp_angle(y, ii)  # noqa: E731
            A_num_rows = len(p.angles_rot_rad)
        elif At is None:
            raise ValueError("Parameter `At` is required, if `A` is not a projector.")
        elif A_num_rows is None:
            raise ValueError("Parameter `A_num_rows` is required, if `A` is not a projector.")

        # Back-projection diagonal re-scaling
        b_ones = np.ones_like(b)
        if b_mask is not None:
            b_ones *= b_mask
        tau = [At(b_ones[..., ii, :], ii) for ii in range(A_num_rows)]
        tau = np.abs(np.stack(tau, axis=-2))
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = self.relaxation / tau

        # Forward-projection diagonal re-scaling
        x_ones = np.ones([*tau.shape[:-2], tau.shape[-1]], dtype=tau.dtype)
        if x_mask is not None:
            x_ones *= x_mask
        sigma = [A(x_ones, ii) for ii in range(A_num_rows)]
        sigma = np.abs(np.stack(sigma, axis=-2))
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        if x0 is None:
            x0 = np.zeros_like(x_ones)
        else:
            x0 = np.array(x0).copy()
        x = x0

        info = SolutionInfo(self.info(), max_iterations=iterations, tolerance=self.tolerance)

        if self.tolerance is not None:
            res = self.compute_residual(A, b, x, A_num_rows=A_num_rows, b_mask=b_mask)
            info.residual0 = np.linalg.norm(res.flatten())

        rows_sequence = np.random.permutation(A_num_rows)

        algo_info = f"- Performing {self.upper()} iterations: "

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose), leave=self.leave_progress):
            info.iterations += 1

            for ii_a in rows_sequence:
                res = A(x, ii_a) - b[..., ii_a, :]
                if b_mask is not None:
                    res *= b_mask[..., ii_a, :]

                x -= At(res * sigma[..., ii_a, :], ii_a) * tau[..., ii_a, :]

                if lower_limit is not None:
                    x = np.fmax(x, lower_limit)
                if upper_limit is not None:
                    x = np.fmin(x, upper_limit)
                if x_mask is not None:
                    x *= x_mask

            if self.tolerance is not None:
                res = self.compute_residual(A, b, x, A_num_rows=A_num_rows, b_mask=b_mask)
                info.residuals[ii] = np.linalg.norm(res)

                if self.tolerance > info.residuals[ii]:
                    break

        return x, info


class MLEM(Solver):
    """
    Initialize the MLEM solver class.

    This class implements the Maximul Likelihood Expectation Maximization (MLEM) algorithm.

    Parameters
    ----------
    verbose : bool, optional
        Turn on verbose output. The default is False.
    leave_progress: bool, optional
        Leave the progress bar after the computation is finished. The default is True.
    tolerance : float | None, optional
        Tolerance on the data residual for computing when to stop iterations.
        The default is None.
    regularizer : Sequence[regularizers.BaseRegularizer] | regularizers.BaseRegularizer | None, optional
        Regularizer to be used. The default is None.
    data_term : str | data_terms.DataFidelityBase, optional
        Data fidelity term for computing the data residual. The default is "l2".
    data_term_test : data_terms.DataFidelityBase | None, optional
        The data fidelity to be used for the test set.
        If None, it will use the same as for the rest of the data.
        The default is None.
    """

    def __init__(
        self,
        verbose: bool = False,
        leave_progress: bool = True,
        tolerance: float | None = None,
        regularizer: Sequence[regularizers.BaseRegularizer] | regularizers.BaseRegularizer | None = None,
        data_term: str | data_terms.DataFidelityBase = "kl",
        data_term_test: str | data_terms.DataFidelityBase | None = None,
    ):
        super().__init__(
            verbose=verbose,
            leave_progress=leave_progress,
            tolerance=tolerance,
            data_term=data_term,
            data_term_test=data_term_test,
        )
        self.regularizer = self._initialize_regularizer(regularizer)

    def info(self) -> str:
        """
        Return the MLEM info.

        Returns
        -------
        str
            info string.
        """
        return Solver.info(self) + f"(B:{self.data_term.background:g})" if self.data_term.background is not None else ""

    def __call__(  # noqa: C901
        self,
        A: operators.BaseTransform,
        b: NDArrayFloat,
        iterations: int,
        x0: NDArrayFloat | None = None,
        lower_limit: float | NDArrayFloat | None = None,
        upper_limit: float | NDArrayFloat | None = None,
        x_mask: NDArrayFloat | None = None,
        b_mask: NDArrayFloat | None = None,
        b_test_mask: NDArrayFloat | None = None,
    ) -> tuple[NDArrayFloat, SolutionInfo]:
        """
        Reconstruct the data, using the MLEM algorithm.

        Parameters
        ----------
        A : BaseTransform
            Projection operator.
        b : NDArrayFloat
            Data to reconstruct.
        iterations : int
            Number of iterations.
        x0 : NDArrayFloat | None, optional
            Initial solution. The default is None.
        lower_limit : float | NDArrayFloat | None, optional
            Lower clipping value. The default is None.
        upper_limit : float | NDArrayFloat | None, optional
            Upper clipping value. The default is None.
        x_mask : NDArrayFloat | None, optional
            Solution mask. The default is None.
        b_mask : NDArrayFloat | None, optional
            Data mask. The default is None.
        b_test_mask : NDArrayFloat | None, optional
            Test data mask. The default is None.

        Returns
        -------
        tuple[NDArrayFloat, SolutionInfo]
            The reconstruction, and the residuals.
        """
        b = np.array(b)

        b_mask, b_test_mask = self._initialize_b_masks(b, b_mask, b_test_mask)

        # Back-projection diagonal re-scaling
        b_ones = np.ones_like(b)
        if b_mask is not None:
            b_ones *= b_mask
        tau = A.T(b_ones)

        # Forward-projection diagonal re-scaling
        x_ones = np.ones_like(tau)
        if x_mask is not None:
            x_ones *= x_mask
        sigma = np.abs(A(x_ones))
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        if x0 is None:
            x = np.ones_like(tau)
        else:
            x = np.array(x0).copy()
        if x_mask is not None:
            x *= x_mask

        self.data_term.assign_data(b)

        info = SolutionInfo(self.info(), max_iterations=iterations, tolerance=self.tolerance)

        if b_test_mask is not None or self.tolerance is not None:
            Ax = A(x)

            if b_test_mask is not None:
                if self.data_term_test.background != self.data_term.background:
                    print("WARNING - the data_term and and data_term_test should have the same background. Making them equal.")
                    self.data_term_test.background = self.data_term.background
                self.data_term_test.assign_data(b)

                res_test_0 = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                info.residual0_cv = self.data_term_test.compute_residual_norm(res_test_0)

            if self.tolerance is not None:
                res_0 = self.data_term.compute_residual(Ax, mask=b_mask)
                info.residual0 = self.data_term.compute_residual_norm(res_0)

        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        algo_info = f"- Performing {self.upper()}-{self.data_term.upper()}{reg_info} iterations: "

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose), leave=self.leave_progress):
            info.iterations += 1

            # The MLEM update
            Ax = A(x)

            if b_test_mask is not None:
                res_test = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                info.residuals_cv[ii] = self.data_term_test.compute_residual_norm(res_test)

            if self.tolerance is not None:
                res = self.data_term.compute_residual(Ax, mask=b_mask)
                info.residuals[ii] = self.data_term.compute_residual_norm(res)
                if self.tolerance > info.residuals[ii]:
                    break

            if self.data_term.background is not None:
                Ax = Ax + self.data_term.background
            Ax = Ax.clip(eps, None)

            upd = A.T(b / Ax)
            x *= upd / tau

            if lower_limit is not None or upper_limit is not None:
                x = x.clip(lower_limit, upper_limit)
            if x_mask is not None:
                x *= x_mask

        return x, info


class SIRT(Solver):
    """
    Initialize the SIRT solver class.

    This class implements the Simultaneous Iterative Reconstruction Technique (SIRT) algorithm.

    Parameters
    ----------
    verbose : bool, optional
        Turn on verbose output. The default is False.
    leave_progress: bool, optional
        Leave the progress bar after the computation is finished. The default is True.
    tolerance : float | None, optional
        Tolerance on the data residual for computing when to stop iterations.
        The default is None.
    relaxation : float, optional
        The relaxation length. The default is 1.95.
    regularizer : Sequence[regularizers.BaseRegularizer] | regularizers.BaseRegularizer | None, optional
        Regularizer to be used. The default is None.
    data_term : str | data_terms.DataFidelityBase, optional
        Data fidelity term for computing the data residual. The default is "l2".
    data_term_test : data_terms.DataFidelityBase | None, optional
        The data fidelity to be used for the test set.
        If None, it will use the same as for the rest of the data.
        The default is None.
    """

    def __init__(
        self,
        verbose: bool = False,
        leave_progress: bool = True,
        relaxation: float = 1.95,
        tolerance: float | None = None,
        regularizer: Sequence[regularizers.BaseRegularizer] | regularizers.BaseRegularizer | None = None,
        data_term: str | data_terms.DataFidelityBase = "l2",
        data_term_test: str | data_terms.DataFidelityBase | None = None,
    ):
        super().__init__(
            verbose=verbose,
            leave_progress=leave_progress,
            relaxation=relaxation,
            tolerance=tolerance,
            data_term=data_term,
            data_term_test=data_term_test,
        )
        self.regularizer = self._initialize_regularizer(regularizer)

    def info(self) -> str:
        """
        Return the SIRT info.

        Returns
        -------
        str
            SIRT info string.
        """
        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        return Solver.info(self) + "-" + self.data_term.info() + reg_info

    def __call__(  # noqa: C901
        self,
        A: operators.BaseTransform,
        b: NDArrayFloat,
        iterations: int,
        x0: NDArrayFloat | None = None,
        lower_limit: float | NDArrayFloat | None = None,
        upper_limit: float | NDArrayFloat | None = None,
        x_mask: NDArrayFloat | None = None,
        b_mask: NDArrayFloat | None = None,
        b_test_mask: NDArrayFloat | None = None,
    ) -> tuple[NDArrayFloat, SolutionInfo]:
        """
        Reconstruct the data, using the SIRT algorithm.

        Parameters
        ----------
        A : BaseTransform
            Projection operator.
        b : NDArrayFloat
            Data to reconstruct.
        iterations : int
            Number of iterations.
        x0 : NDArrayFloat | None, optional
            Initial solution. The default is None.
        lower_limit : float | NDArrayFloat | None, optional
            Lower clipping value. The default is None.
        upper_limit : float | NDArrayFloat | None, optional
            Upper clipping value. The default is None.
        x_mask : NDArrayFloat | None, optional
            Solution mask. The default is None.
        b_mask : NDArrayFloat | None, optional
            Data mask. The default is None.
        b_test_mask : NDArrayFloat | None, optional
            Test data mask. The default is None.

        Returns
        -------
        tuple[NDArrayFloat, SolutionInfo]
            The reconstruction, and the residuals.
        """
        b = np.array(b)

        b_mask, b_test_mask = self._initialize_b_masks(b, b_mask, b_test_mask)

        try:
            At_abs = A.T.absolute()
            A_abs = A.absolute()
        except AttributeError:
            print("WARNING: operator does not support absolute(); Using the operator itself.")
            A_abs = A
            At_abs = A.T

        sigma, tau, x_shape, x_dtype = compute_diagonal_scaling(
            A_abs, At_abs, b, regs=self.regularizer, relaxation_tau=self.relaxation, x_mask=x_mask, b_mask=b_mask
        )

        if x0 is None:
            x = np.zeros_like(tau)
        else:
            x = np.array(x0).copy()

        self.data_term.assign_data(b, sigma)

        info = SolutionInfo(self.info(), max_iterations=iterations, tolerance=self.tolerance)

        if b_test_mask is not None or self.tolerance is not None:
            Ax = A(x)

            res_0 = self.data_term.compute_residual(Ax, mask=b_mask)
            info.residual0 = self.data_term.compute_residual_norm(res_0)

            if b_test_mask is not None:
                if self.data_term_test.background != self.data_term.background:
                    print("WARNING - the data_term and and data_term_test should have the same background. Making them equal.")
                    self.data_term_test.background = self.data_term.background
                self.data_term_test.assign_data(b, sigma)

                res_test_0 = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                info.residual0_cv = self.data_term_test.compute_residual_norm(res_test_0)

        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        algo_info = f"- Performing {self.upper()}-{self.data_term.upper()}{reg_info} iterations: "

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose), leave=self.leave_progress):
            info.iterations += 1

            Ax = A(x)
            res = self.data_term.compute_residual(Ax, mask=b_mask)

            if b_test_mask is not None or self.tolerance is not None:
                info.residuals[ii] = self.data_term.compute_residual_norm(res)

                if b_test_mask is not None:
                    res_test = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                    info.residuals_cv[ii] = self.data_term_test.compute_residual_norm(res_test)

                if self.tolerance is not None and self.tolerance > info.residuals[ii]:
                    if self.verbose:
                        print(f"Residual reached the desired tolerance of {self.tolerance}. Ending iterations..")
                    break

            q = [reg.initialize_dual() for reg in self.regularizer]
            for q_r, reg in zip(q, self.regularizer):
                reg.update_dual(q_r, x)
                reg.apply_proximal_dual(q_r)

            upd = A.T(res * sigma)
            for q_r, reg in zip(q, self.regularizer):
                upd -= reg.compute_update_primal(q_r)
            x += upd * tau

            if lower_limit is not None or upper_limit is not None:
                x = x.clip(lower_limit, upper_limit)
            if x_mask is not None:
                x *= x_mask

        return x, info


class PDHG(Solver):
    """
    Initialize the PDHG solver class.

    PDHG stands for primal-dual hybrid gradient algorithm from Chambolle and Pock.

    Parameters
    ----------
    verbose : bool, optional
        Turn on verbose output. The default is False.
    leave_progress: bool, optional
        Leave the progress bar after the computation is finished. The default is True.
    tolerance : float | None, optional
        Tolerance on the data residual for computing when to stop iterations.
        The default is None.
    relaxation : float, optional
        The relaxation length. The default is 0.95.
    regularizer : Sequence[regularizers.BaseRegularizer] | regularizers.BaseRegularizer | None, optional
        Regularizer to be used. The default is None.
    data_term : str | data_terms.DataFidelityBase, optional
        Data fidelity term for computing the data residual. The default is "l2".
    data_term_test : data_terms.DataFidelityBase | None, optional
        The data fidelity to be used for the test set.
        If None, it will use the same as for the rest of the data.
        The default is None.
    """

    def __init__(
        self,
        verbose: bool = False,
        leave_progress: bool = True,
        tolerance: float | None = None,
        relaxation: float = 0.95,
        regularizer: Sequence[regularizers.BaseRegularizer] | regularizers.BaseRegularizer | None = None,
        data_term: str | data_terms.DataFidelityBase = "l2",
        data_term_test: str | data_terms.DataFidelityBase | None = None,
    ):
        super().__init__(
            verbose=verbose,
            leave_progress=leave_progress,
            relaxation=relaxation,
            tolerance=tolerance,
            data_term=data_term,
            data_term_test=data_term_test,
        )
        self.regularizer = self._initialize_regularizer(regularizer)

    def info(self) -> str:
        """
        Return the PDHG info.

        Returns
        -------
        str
            PDHG info string.
        """
        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        return Solver.info(self) + "-" + self.data_term.info() + reg_info

    @staticmethod
    def _initialize_data_fidelity_function(data_term: str | data_terms.DataFidelityBase):
        if isinstance(data_term, str):
            if data_term.lower() == "l2":
                return data_terms.DataFidelity_l2()
            if data_term.lower() == "l1":
                return data_terms.DataFidelity_l1()
            if data_term.lower() == "kl":
                return data_terms.DataFidelity_KL()
            else:
                raise ValueError(f'Unknown data term: "{data_term}", accepted terms are: "l2" | "l1" | "kl".')
        else:
            return cp.deepcopy(data_term)

    def __call__(  # noqa: C901
        self,
        A: operators.BaseTransform,
        b: NDArrayFloat,
        iterations: int,
        x0: NDArrayFloat | None = None,
        lower_limit: float | NDArrayFloat | None = None,
        upper_limit: float | NDArrayFloat | None = None,
        x_mask: NDArrayFloat | None = None,
        b_mask: NDArrayFloat | None = None,
        b_test_mask: NDArrayFloat | None = None,
        precondition: bool = True,
    ) -> tuple[NDArrayFloat, SolutionInfo]:
        """
        Reconstruct the data, using the PDHG algorithm.

        Parameters
        ----------
        A : BaseTransform
            Projection operator.
        b : NDArrayFloat
            Data to reconstruct.
        iterations : int
            Number of iterations.
        x0 : NDArrayFloat | None, optional
            Initial solution. The default is None.
        lower_limit : float | NDArrayFloat | None, optional
            Lower clipping value. The default is None.
        upper_limit : float | NDArrayFloat | None, optional
            Upper clipping value. The default is None.
        x_mask : NDArrayFloat | None, optional
            Solution mask. The default is None.
        b_mask : NDArrayFloat | None, optional
            Data mask. The default is None.
        b_test_mask : NDArrayFloat | None, optional
            Test data mask. The default is None.
        precondition : bool, optional
            Whether to use the preconditioned version of the algorithm. The default is True.

        Returns
        -------
        tuple[NDArrayFloat, SolutionInfo]
            The reconstruction, and the residuals.
        """
        b = np.array(b)

        if precondition:
            try:
                At_abs = A.T.absolute()
                A_abs = A.absolute()
            except AttributeError:
                print(A)
                print("WARNING: Turning off preconditioning because system matrix does not support absolute")
                precondition = False

        b_mask, b_test_mask = self._initialize_b_masks(b, b_mask, b_test_mask)

        if precondition:
            sigma, tau, x_shape, x_dtype = compute_diagonal_scaling(
                A_abs,
                At_abs,
                b,
                regs=self.regularizer,
                relaxation_tau=self.relaxation,
                relaxation_sigma=self.relaxation,
                x_mask=x_mask,
                b_mask=b_mask,
            )
        else:
            sigma, tau, x_shape, x_dtype = compute_Lipschitz_scaling(
                A, b, regs=self.regularizer, relaxation_sigma=self.relaxation, relaxation_tau=self.relaxation
            )

        if x0 is None:
            x0 = np.zeros(x_shape, dtype=x_dtype)
        else:
            x0 = np.array(x0).copy()
        x = x0
        x_relax = x.copy()

        self.data_term.assign_data(b, sigma)
        p = self.data_term.initialize_dual()

        q = [reg.initialize_dual() for reg in self.regularizer]

        info = SolutionInfo(self.info(), max_iterations=iterations, tolerance=self.tolerance)

        if b_test_mask is not None or self.tolerance is not None:
            Ax = A(x)

            res_0 = self.data_term.compute_residual(Ax, mask=b_mask)
            info.residual0 = self.data_term.compute_residual_norm(res_0)

            if b_test_mask is not None:
                if self.data_term_test.background != self.data_term.background:
                    print("WARNING - the data_term and and data_term_test should have the same background. Making them equal.")
                    self.data_term_test.background = self.data_term.background
                self.data_term_test.assign_data(b, sigma)

                res_test_0 = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                info.residual0_cv = self.data_term_test.compute_residual_norm(res_test_0)

        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        algo_info = f"- Performing {self.upper()}-{self.data_term.upper()}{reg_info} iterations: "

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose), leave=self.leave_progress):
            info.iterations += 1

            Ax_rlx = A(x_relax)
            self.data_term.update_dual(p, Ax_rlx)
            self.data_term.apply_proximal_dual(p)

            if b_mask is not None:
                p *= b_mask

            for q_r, reg in zip(q, self.regularizer):
                reg.update_dual(q_r, x_relax)
                reg.apply_proximal_dual(q_r)

            upd = A.T(p)
            for q_r, reg in zip(q, self.regularizer):
                upd += reg.compute_update_primal(q_r)
            x_new = x - upd * tau

            if lower_limit is not None or upper_limit is not None:
                x_new = x_new.clip(lower_limit, upper_limit)
            if x_mask is not None:
                x_new *= x_mask

            x_relax = x_new + (x_new - x)
            x = x_new

            if b_test_mask is not None or self.tolerance is not None:
                Ax = A(x)
                res = self.data_term.compute_residual(Ax, mask=b_mask)
                info.residuals[ii] = self.data_term.compute_residual_norm(res)

                if b_test_mask is not None:
                    res_test = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                    info.residuals_cv[ii] = self.data_term_test.compute_residual_norm(res_test)

                if self.tolerance is not None and self.tolerance > info.residuals[ii]:
                    if self.verbose:
                        print(f"Residual reached the desired tolerance of {self.tolerance}. Ending iterations..")
                    break

        return x, info


class FISTA(Solver):
    """Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

    Implements the accelerated proximal gradient method from Beck & Teboulle (2009),
    "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems",
    SIAM Journal on Imaging Sciences.

    The algorithm minimizes:
        min_x  f(x)  +  g(x)

    where:
      - f(x) is the data-fidelity term, handled via its *gradient*:
            grad f(x) = A^T (A x - b)     for the l2 case
            grad f(x) = A^T (1 - b / Ax)  for the KL case
        (computed using ``data_term.compute_residual`` followed by ``A.T``).
      - g(x) is an optional regularizer with a tractable *primal proximal*,
        applied via ``regularizer.apply_proximal_primal``.

    **TV-min regularization (Regularizer_Grad and all TV subclasses) is not
    supported** because their primal proximal has no closed form.  Passing such
    a regularizer raises ``ValueError`` at construction time.

    **At most one regularizer** is accepted; passing a list with more than one
    raises ``ValueError``.

    Step-size strategy
    ------------------
    Two modes are available, selected by the ``precondition`` argument to
    ``__call__``:

    * ``precondition=False`` (default Lipschitz): the step size is
      ``tau = relaxation / L`` where ``L = ||A||^2`` is estimated via the power
      method.

    * ``precondition=True`` (SIRT-style diagonal preconditioning): following the
      same convention as SIRT,

          Sigma = diag(|A| 1_x)      (column sums of |A|, in measurement space)
          Tau   = diag(|A^T| 1_b)    (row sums of |A|, in image space)

      The gradient step then reads:
          x <- x - Tau * A^T(residual * Sigma)

      where ``residual = b - A x``.  This is the SIRT preconditioned gradient step;
      the ``relaxation`` scalar is folded into Tau as ``Tau = relaxation / Tau_raw``.

    Parameters
    ----------
    verbose : bool, optional
        Turn on verbose output. The default is False.
    leave_progress : bool, optional
        Leave the progress bar after computation. The default is True.
    tolerance : float | None, optional
        Stop early when the residual norm drops below this value. The default is None.
    relaxation : float, optional
        Step-size relaxation factor (scalar multiplier on tau). The default is 1.0.
    regularizer : BaseRegularizer | None, optional
        A single regularizer with a tractable primal proximal.  TV-type and
        Laplacian regularizers are rejected.  The default is None.
    data_term : str | DataFidelityBase, optional
        Data fidelity. Accepts ``"l2"`` or a ``DataFidelity_l2`` instance.
        The default is ``"l2"``.
    data_term_test : DataFidelityBase | None, optional
        Data fidelity for the held-out test set. Defaults to the same as
        ``data_term``.
    restart_period : int | None, optional
        If given, the FISTA momentum sequence is restarted every
        ``restart_period`` iterations.  If ``None`` (default), the pure
        Beck-Teboulle monotone sequence is used with no restart.
    """

    def __init__(
        self,
        verbose: bool = False,
        leave_progress: bool = True,
        tolerance: float | None = None,
        relaxation: float = 1.0,
        regularizer: regularizers.BaseRegularizer | None = None,
        data_term: str | data_terms.DataFidelityBase = "l2",
        data_term_test: str | data_terms.DataFidelityBase | None = None,
        restart_period: int | None = None,
    ):
        super().__init__(
            verbose=verbose,
            leave_progress=leave_progress,
            relaxation=relaxation,
            tolerance=tolerance,
            data_term=data_term,
            data_term_test=data_term_test,
        )

        # Validate and store the (single) regularizer
        regs = self._initialize_regularizer(regularizer)
        if len(regs) > 1:
            raise ValueError(
                "FISTA supports at most one regularizer, but "
                f"{len(regs)} were provided. "
                "To combine multiple regularizers use PDHG instead."
            )
        if len(regs) == 1:
            reg = regs[0]
            # Reject TV / gradient / Laplacian regularizers up front by
            # attempting a dry-run of apply_proximal_primal with a tiny dummy.
            # We call the method on a zero-size array; if it raises
            # NotImplementedError we know the regularizer is unsupported.
            try:
                reg.apply_proximal_primal(np.zeros(1, dtype=np.float32), 1.0)
            except NotImplementedError as exc:
                raise ValueError(
                    f"Regularizer '{reg.info()}' does not support a primal proximal "
                    "and cannot be used with FISTA. "
                    "Switch to PDHG for this regularizer, or use a wavelet / l1 "
                    "regularizer with FISTA.\n"
                    f"Original error: {exc}"
                ) from exc
            except Exception:
                # Any other exception (e.g. shape mismatch on the dummy) is fine;
                # it means the method exists and would be called correctly later.
                pass
        self.regularizer: list[regularizers.BaseRegularizer] = list(regs)
        self.restart_period = restart_period

    def info(self) -> str:
        """Return the FISTA info string."""
        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        return Solver.info(self) + "-" + self.data_term.info() + reg_info

    @staticmethod
    def _initialize_data_fidelity_function(data_term: str | data_terms.DataFidelityBase):
        """Accept l2, wl2, l2b and their subclasses for FISTA."""
        if isinstance(data_term, str):
            if data_term.lower() == "l2":
                return data_terms.DataFidelity_l2()
            else:
                raise ValueError(f"Unknown data term: '{data_term}'. FISTA only accepts 'l2' and derivatives.")
        elif isinstance(data_term, data_terms.DataFidelity_l2):
            return cp.deepcopy(data_term)
        else:
            raise ValueError(f"Unsupported data term type: {type(data_term)}. FISTA only accepts 'l2' and derivatives.")

    def __call__(  # noqa: C901
        self,
        A: operators.BaseTransform,
        b: NDArrayFloat,
        iterations: int,
        x0: NDArrayFloat | None = None,
        lower_limit: float | NDArrayFloat | None = None,
        upper_limit: float | NDArrayFloat | None = None,
        x_mask: NDArrayFloat | None = None,
        b_mask: NDArrayFloat | None = None,
        b_test_mask: NDArrayFloat | None = None,
        precondition: bool = True,
    ) -> tuple[NDArrayFloat, SolutionInfo]:
        """Run FISTA.

        Parameters
        ----------
        A : BaseTransform
            Forward operator.
        b : NDArrayFloat
            Measurement data.
        iterations : int
            Number of outer iterations.
        x0 : NDArrayFloat | None, optional
            Warm-start solution. The default is None (zero initialisation).
        lower_limit : float | NDArrayFloat | None, optional
            Hard lower bound applied after each proximal step (clipping).
            The default is None.
        upper_limit : float | NDArrayFloat | None, optional
            Hard upper bound applied after each proximal step (clipping).
            The default is None.
        x_mask : NDArrayFloat | None, optional
            Binary solution mask. The default is None.
        b_mask : NDArrayFloat | None, optional
            Binary data mask. The default is None.
        b_test_mask : NDArrayFloat | None, optional
            Held-out test set mask. The default is None.
        precondition : bool, optional
            If ``True`` (default), use SIRT-style diagonal preconditioning.
            If ``False``, use ``tau = relaxation / L`` where
            ``L`` is estimated by the power method.

        Returns
        -------
        tuple[NDArrayFloat, SolutionInfo]
            The reconstruction and associated iteration info.
        """
        b = np.array(b)

        if precondition:
            try:
                At_abs = A.T.absolute()
                A_abs = A.absolute()
            except AttributeError:
                print("WARNING: operator does not support absolute(); falling back to un-preconditioned Lipschitz step size.")
                precondition = False

        b_mask, b_test_mask = self._initialize_b_masks(b, b_mask, b_test_mask)

        if precondition:
            sigma, tau, x_shape, x_dtype = compute_diagonal_scaling(
                A_abs, At_abs, b, regs=[], relaxation_tau=self.relaxation, x_mask=x_mask, b_mask=b_mask
            )
        else:
            sigma, tau, x_shape, x_dtype = compute_Lipschitz_scaling(A, b, regs=[], relaxation_tau=self.relaxation)

        # We need x_shape to initialise the regularizer (also sets up self.op
        # inside the regularizer so that apply_proximal_primal works).
        dummy_x = np.zeros(x_shape, dtype=x_dtype)
        tau_reg: float | NDArrayFloat = tau  # default; overwritten below if regularizer present
        if self.regularizer:
            reg = self.regularizer[0]
            reg.initialize_sigma_tau(dummy_x)
            # In FISTA the proximal step size for g is the same tau as used for
            # the gradient step on f.  The regularizer's weight is already
            # folded into apply_proximal_primal via tau * self.weight inside
            # that method, so we pass tau directly.
            # tau_reg is kept as a reference to tau so that if tau is a
            # per-element array (preconditioned case) it is passed correctly.
            tau_reg = tau

        # sigma is passed for the test/residual tracking, but is NOT used in the
        # gradient step here; the gradient is computed analytically from compute_residual).
        self.data_term.assign_data(b)
        if b_test_mask is not None:
            if self.data_term_test.background != self.data_term.background:
                print("WARNING - the data_term and data_term_test should have the same background. Making them equal.")
                self.data_term_test.background = self.data_term.background
            self.data_term_test.assign_data(b)

        if x0 is None:
            x = np.zeros(x_shape, dtype=x_dtype)
        else:
            x = np.array(x0, dtype=x_dtype).copy()
        if x_mask is not None:
            x *= x_mask

        # y is the momentum-extrapolated point (Beck–Teboulle notation)
        y = x.copy()
        t = 1.0  # momentum parameter

        info = SolutionInfo(self.info(), max_iterations=iterations, tolerance=self.tolerance)

        if b_test_mask is not None or self.tolerance is not None:
            Ax0 = A(x)

            res_0 = self.data_term.compute_residual(Ax0, mask=b_mask)
            info.residual0 = self.data_term.compute_residual_norm(res_0)

            if b_test_mask is not None:
                res_test_0 = self.data_term_test.compute_residual(Ax0, mask=b_test_mask)
                info.residual0_cv = self.data_term_test.compute_residual_norm(res_test_0)

        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        algo_info = f"- Performing {self.upper()}-{self.data_term.upper()}{reg_info} iterations: "

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose), leave=self.leave_progress):
            info.iterations += 1

            # --- Gradient step on f at the momentum point y ---
            Ay = A(y)
            residual = self.data_term.compute_residual(Ay, mask=b_mask)
            grad = A.T(residual * sigma)

            x_new = y + tau * grad

            # --- Proximal step on g (regularizer) ---
            if self.regularizer:
                self.regularizer[0].apply_proximal_primal(x_new, tau_reg)

            # --- Hard clipping and solution mask ---
            if lower_limit is not None or upper_limit is not None:
                x_new = x_new.clip(lower_limit, upper_limit)
            if x_mask is not None:
                x_new *= x_mask

            # --- Beck–Teboulle momentum update ---
            t_new = float(1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0

            # Optional fixed-period restart: reset momentum sequence
            if self.restart_period is not None and (ii + 1) % self.restart_period == 0:
                t_new = 1.0

            momentum = (t - 1.0) / t_new
            y = x_new + momentum * (x_new - x)

            x = x_new
            t = t_new

            # --- Residual tracking ---
            if b_test_mask is not None or self.tolerance is not None:
                Ax = A(x)
                res = self.data_term.compute_residual(Ax, mask=b_mask)
                info.residuals[ii] = self.data_term.compute_residual_norm(res)

                if b_test_mask is not None:
                    res_test = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                    info.residuals_cv[ii] = self.data_term_test.compute_residual_norm(res_test)

                if self.tolerance is not None and self.tolerance > info.residuals[ii]:
                    if self.verbose:
                        print(f"Residual reached the desired tolerance of {self.tolerance}. Ending iterations..")
                    break

        return x, info
