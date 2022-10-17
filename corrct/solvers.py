#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solvers for the tomographic reconstruction problem.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from typing import Callable, Optional, Sequence, Union, Tuple, Any

import numpy as np
import numpy.random
from numpy.typing import DTypeLike, NDArray

import copy as cp

from . import operators
from . import data_terms
from . import regularizers
from . import filters
from . import projectors

from tqdm.auto import tqdm

from abc import ABC, abstractmethod


eps = np.finfo(np.float32).eps


# ---- Data Fidelity terms ----


DataFidelityBase = data_terms.DataFidelityBase
DataFidelity_l2 = data_terms.DataFidelity_l2
DataFidelity_wl2 = data_terms.DataFidelity_wl2
DataFidelity_l2b = data_terms.DataFidelity_l2b
DataFidelity_l12 = data_terms.DataFidelity_l12
DataFidelity_l1 = data_terms.DataFidelity_l1
DataFidelity_l1b = data_terms.DataFidelity_l1b
DataFidelity_Huber = data_terms.DataFidelity_Huber
DataFidelity_KL = data_terms.DataFidelity_KL

# Multi-channel
DataFidelity_ln = data_terms.DataFidelity_ln


# ---- Regularizers ----


BaseRegularizer = regularizers.BaseRegularizer
Regularizer_Grad = regularizers.Regularizer_Grad
Regularizer_TV2D = regularizers.Regularizer_TV2D
Regularizer_TV3D = regularizers.Regularizer_TV3D
Regularizer_HubTV2D = regularizers.Regularizer_HubTV2D
Regularizer_HubTV3D = regularizers.Regularizer_HubTV3D
Regularizer_smooth2D = regularizers.Regularizer_smooth2D
Regularizer_smooth3D = regularizers.Regularizer_smooth3D
Regularizer_lap = regularizers.Regularizer_lap
Regularizer_lap2D = regularizers.Regularizer_lap2D
Regularizer_lap3D = regularizers.Regularizer_lap3D
Regularizer_l1 = regularizers.Regularizer_l1
Regularizer_l1swl = regularizers.Regularizer_l1swl
Regularizer_Hub_swl = regularizers.Regularizer_Hub_swl
Regularizer_l1dwl = regularizers.Regularizer_l1dwl
Regularizer_Hub_dwl = regularizers.Regularizer_Hub_dwl
Regularizer_l1med = regularizers.Regularizer_l1med
Regularizer_l2med = regularizers.Regularizer_l2med
Regularizer_fft = regularizers.Regularizer_fft

# Multi-channel
Regularizer_TNV = regularizers.Regularizer_TNV
Regularizer_VTV = regularizers.Regularizer_VTV
Regularizer_lnswl = regularizers.Regularizer_lnswl
Regularizer_vl1wl = regularizers.Regularizer_vl1wl
Regularizer_vSVD = regularizers.Regularizer_vSVD


# ---- Constraints ----


Constraint_LowerLimit = regularizers.Constraint_LowerLimit
Constraint_UpperLimit = regularizers.Constraint_UpperLimit


# ---- Solvers ----


NDArrayFloat = NDArray[np.floating]


class SolutionInfo:
    """Reconstruction info."""

    method: str
    iterations: int
    max_iterations: int

    residual0: Union[float, np.floating]
    residual0_cv: Union[float, np.floating]

    residuals: NDArrayFloat
    residuals_cv: NDArrayFloat
    tolerance: Union[float, np.floating, None]

    def __init__(
        self,
        method: str,
        max_iterations: int,
        tolerance: Union[float, np.floating, None],
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
    tolerance : Optional[float], optional
        Tolerance on the data residual for computing when to stop iterations.
        The default is None.
    relaxation : float, optional
        The relaxation length. The default is 1.0.
    data_term : Union[str, DataFidelityBase], optional
        Data fidelity term for computing the data residual. The default is "l2".
    data_term_test : Optional[DataFidelityBase], optional
        The data fidelity to be used for the test set.
        If None, it will use the same as for the rest of the data.
        The default is None.
    """

    def __init__(
        self,
        verbose: bool = False,
        relaxation: float = 1.0,
        tolerance: Optional[float] = None,
        data_term: Union[str, DataFidelityBase] = "l2",
        data_term_test: Union[str, DataFidelityBase, None] = None,
    ):
        self.verbose = verbose
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
    ) -> Tuple[NDArrayFloat, SolutionInfo]:
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
    def _initialize_data_fidelity_function(data_term: Union[str, DataFidelityBase]) -> DataFidelityBase:
        if isinstance(data_term, str):
            if data_term.lower() == "l2":
                return data_terms.DataFidelity_l2()
            else:
                raise ValueError('Unknown data term: "%s", only accepted terms are: "l2".' % data_term)
        elif isinstance(data_term, (data_terms.DataFidelity_l2, data_terms.DataFidelity_KL)):
            return cp.deepcopy(data_term)
        else:
            raise ValueError('Unsupported data term: "%s", only accepted terms are "l2"-based.' % data_term.info())

    @staticmethod
    def _initialize_regularizer(
        regularizer: Union[BaseRegularizer, None, Sequence[BaseRegularizer]]
    ) -> Sequence[BaseRegularizer]:
        if regularizer is None:
            return []
        elif isinstance(regularizer, regularizers.BaseRegularizer):
            return [regularizer]
        elif isinstance(regularizer, (list, tuple)):
            check_regs_ok = [isinstance(r, BaseRegularizer) for r in regularizer]
            if not np.all(check_regs_ok):
                raise ValueError(
                    "The following regularizers are not derived from the BaseRegularizer class: %s"
                    % np.array(np.arange(len(check_regs_ok))[np.array(check_regs_ok, dtype=bool)])
                )
            else:
                return list(regularizer)
        else:
            raise ValueError("Unknown regularizer type.")

    @staticmethod
    def _initialize_b_masks(
        b: NDArrayFloat, b_mask: Optional[NDArrayFloat], b_test_mask: Optional[NDArrayFloat]
    ) -> Tuple[Optional[NDArrayFloat], Optional[NDArrayFloat]]:
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
        regularizer: Union[Sequence[BaseRegularizer], BaseRegularizer, None] = None,
        data_term: Union[str, DataFidelityBase] = "l2",
        fbp_filter: Union[str, NDArrayFloat, filters.Filter] = "ramp",
        pad_mode: str = "constant",
    ):
        """Initialize the Filtered Back-Projection (FBP) algorithm.

        Parameters
        ----------
        verbose : bool, optional
            Turn on verbose output. The default is False.
        regularizer : Sequence[BaseRegularizer] | BaseRegularizer | None, optional
            NOT USED, only exposed for compatibility reasons.
        data_term : Union[str, DataFidelityBase], optional
            NOT USED, only exposed for compatibility reasons.
        fbp_filter : Union[str, NDArrayFloat], optional
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
            return super().info() + "-" + self.fbp_filter.upper()
        elif isinstance(self.fbp_filter, np.ndarray):
            return super().info() + "-" + filters.FilterCustom.__name__.upper()
        else:
            return super().info() + "-" + type(self.fbp_filter).__name__.upper()

    def __call__(  # noqa: C901
        self,
        A: operators.BaseTransform,
        b: NDArrayFloat,
        iterations: int = 0,
        x0: Optional[NDArrayFloat] = None,
        lower_limit: Union[float, NDArrayFloat, None] = None,
        upper_limit: Union[float, NDArrayFloat, None] = None,
        x_mask: Optional[NDArrayFloat] = None,
        b_mask: Optional[NDArrayFloat] = None,
    ) -> Tuple[NDArrayFloat, SolutionInfo]:
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
        x0 : Optional[NDArrayFloat], optional
            Initial solution. The default is None.
        lower_limit : Union[float, NDArrayFloat], optional
            Lower clipping value. The default is None.
        upper_limit : Union[float, NDArrayFloat], optional
            Upper clipping value. The default is None.
        x_mask : Optional[NDArrayFloat], optional
            Solution mask. The default is None.
        b_mask : Optional[NDArrayFloat], optional
            Data mask. The default is None.

        Raises
        ------
        ValueError
            In case the data is 1D.

        Returns
        -------
        Tuple[NDArrayFloat, SolutionInfo]
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
        b_mask: Optional[NDArrayFloat],
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
        b_mask : Optional[NDArrayFloat]
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
        A: Union[Callable[[NDArray, int], NDArray], projectors.ProjectorUncorrected],
        b: NDArrayFloat,
        iterations: int,
        A_num_rows: Optional[int] = None,
        At: Optional[Callable] = None,
        x0: Optional[NDArrayFloat] = None,
        lower_limit: Union[float, NDArrayFloat, None] = None,
        upper_limit: Union[float, NDArrayFloat, None] = None,
        x_mask: Optional[NDArrayFloat] = None,
        b_mask: Optional[NDArrayFloat] = None,
    ) -> Tuple[NDArrayFloat, SolutionInfo]:
        """
        Reconstruct the data, using the SART algorithm.

        Parameters
        ----------
        A : Union[Callable, BaseTransform]
            Projection operator.
        b : NDArrayFloat
            Data to reconstruct.
        iterations : int
            Number of iterations.
        A_num_rows : int
            Number of projections.
        x0 : Optional[NDArrayFloat], optional
            Initial solution. The default is None.
        At : Callable, optional
            The back-projection operator. This is only needed if the projection operator does not have an adjoint.
            The default is None.
        lower_limit : Union[float, NDArrayFloat], optional
            Lower clipping value. The default is None.
        upper_limit : Union[float, NDArrayFloat], optional
            Upper clipping value. The default is None.
        x_mask : Optional[NDArrayFloat], optional
            Solution mask. The default is None.
        b_mask : Optional[NDArrayFloat], optional
            Data mask. The default is None.

        Returns
        -------
        Tuple[NDArrayFloat, SolutionInfo]
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

        algo_info = "- Performing %s iterations: " % self.upper()

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose)):
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


class SIRT(Solver):
    """
    Initialize the SIRT solver class.

    This class implements the Simultaneous Iterative Reconstruction Technique (SIRT) algorithm.

    Parameters
    ----------
    verbose : bool, optional
        Turn on verbose output. The default is False.
    tolerance : Optional[float], optional
        Tolerance on the data residual for computing when to stop iterations.
        The default is None.
    relaxation : float, optional
        The relaxation length. The default is 1.95.
    regularizer : Sequence[BaseRegularizer] | BaseRegularizer | None, optional
        Regularizer to be used. The default is None.
    data_term : Union[str, DataFidelityBase], optional
        Data fidelity term for computing the data residual. The default is "l2".
    data_term_test : Optional[DataFidelityBase], optional
        The data fidelity to be used for the test set.
        If None, it will use the same as for the rest of the data.
        The default is None.
    """

    def __init__(
        self,
        verbose: bool = False,
        relaxation: float = 1.95,
        tolerance: Optional[float] = None,
        regularizer: Union[Sequence[BaseRegularizer], BaseRegularizer, None] = None,
        data_term: Union[str, DataFidelityBase] = "l2",
        data_term_test: Union[str, DataFidelityBase, None] = None,
    ):
        super().__init__(
            verbose=verbose, relaxation=relaxation, tolerance=tolerance, data_term=data_term, data_term_test=data_term_test
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
        x0: Optional[NDArrayFloat] = None,
        lower_limit: Union[float, NDArrayFloat, None] = None,
        upper_limit: Union[float, NDArrayFloat, None] = None,
        x_mask: Optional[NDArrayFloat] = None,
        b_mask: Optional[NDArrayFloat] = None,
        b_test_mask: Optional[NDArrayFloat] = None,
    ) -> Tuple[NDArrayFloat, SolutionInfo]:
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
        x0 : Optional[NDArrayFloat], optional
            Initial solution. The default is None.
        lower_limit : Union[float, NDArrayFloat], optional
            Lower clipping value. The default is None.
        upper_limit : Union[float, NDArrayFloat], optional
            Upper clipping value. The default is None.
        x_mask : Optional[NDArrayFloat], optional
            Solution mask. The default is None.
        b_mask : Optional[NDArrayFloat], optional
            Data mask. The default is None.
        b_test_mask : Optional[NDArrayFloat], optional
            Test data mask. The default is None.

        Returns
        -------
        Tuple[NDArrayFloat, SolutionInfo]
            The reconstruction, and the residuals.
        """
        b = np.array(b)

        (b_mask, b_test_mask) = self._initialize_b_masks(b, b_mask, b_test_mask)

        # Back-projection diagonal re-scaling
        b_ones = np.ones_like(b)
        if b_mask is not None:
            b_ones *= b_mask
        tau = np.abs(A.T(b_ones))
        for reg in self.regularizer:
            tau += reg.initialize_sigma_tau(tau)
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = self.relaxation / tau

        # Forward-projection diagonal re-scaling
        x_ones = np.ones_like(tau)
        if x_mask is not None:
            x_ones *= x_mask
        sigma = np.abs(A(x_ones))
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        if x0 is None:
            x = np.zeros_like(x_ones)
        else:
            x = np.array(x0).copy()

        self.data_term.assign_data(b, sigma)

        info = SolutionInfo(self.info(), max_iterations=iterations, tolerance=self.tolerance)

        if b_test_mask is not None or self.tolerance is not None:
            Ax = A(x)

            if b_test_mask is not None:
                if self.data_term_test.background != self.data_term.background:
                    print("WARNING - the data_term and and data_term_test should have the same background. Making them equal.")
                    self.data_term_test.background = self.data_term.background
                self.data_term_test.assign_data(b, sigma)

                res_test_0 = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                info.residual0_cv = self.data_term_test.compute_residual_norm(res_test_0)

            if self.tolerance is not None:
                res_0 = self.data_term.compute_residual(Ax, mask=b_mask)
                info.residual0 = self.data_term.compute_residual_norm(res_0)

        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        algo_info = "- Performing %s-%s%s iterations: " % (self.upper(), self.data_term.upper(), reg_info)

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose)):
            info.iterations += 1

            Ax = A(x)
            res = self.data_term.compute_residual(Ax, mask=b_mask)

            if b_test_mask is not None:
                res_test = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                info.residuals_cv[ii] = self.data_term_test.compute_residual_norm(res_test)

            if self.tolerance is not None:
                info.residuals[ii] = self.data_term.compute_residual_norm(res)
                if self.tolerance > info.residuals[ii]:
                    break

            q = [reg.initialize_dual() for reg in self.regularizer]
            for q_r, reg in zip(q, self.regularizer):
                reg.update_dual(q_r, x)
                reg.apply_proximal(q_r)

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

    PDHG stands for primal-dual hybridg gradient algorithm from Chambolle and Pock.

    Parameters
    ----------
    verbose : bool, optional
        Turn on verbose output. The default is False.
    tolerance : Optional[float], optional
        Tolerance on the data residual for computing when to stop iterations.
        The default is None.
    relaxation : float, optional
        The relaxation length. The default is 0.95.
    regularizer : Sequence[BaseRegularizer] | BaseRegularizer | None, optional
        Regularizer to be used. The default is None.
    data_term : Union[str, DataFidelityBase], optional
        Data fidelity term for computing the data residual. The default is "l2".
    data_term_test : Optional[DataFidelityBase], optional
        The data fidelity to be used for the test set.
        If None, it will use the same as for the rest of the data.
        The default is None.
    """

    def __init__(
        self,
        verbose: bool = False,
        tolerance: Optional[float] = None,
        relaxation: float = 0.95,
        regularizer: Union[Sequence[BaseRegularizer], BaseRegularizer, None] = None,
        data_term: Union[str, DataFidelityBase] = "l2",
        data_term_test: Union[str, DataFidelityBase, None] = None,
    ):
        super().__init__(
            verbose=verbose, relaxation=relaxation, tolerance=tolerance, data_term=data_term, data_term_test=data_term_test
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
    def _initialize_data_fidelity_function(data_term: Union[str, DataFidelityBase]):
        if isinstance(data_term, str):
            if data_term.lower() == "l2":
                return DataFidelity_l2()
            if data_term.lower() == "l1":
                return DataFidelity_l1()
            if data_term.lower() == "kl":
                return DataFidelity_KL()
            else:
                raise ValueError('Unknown data term: "%s", accepted terms are: "l2" | "l1" | "kl".' % data_term)
        else:
            return cp.deepcopy(data_term)

    def power_method(
        self, A: operators.BaseTransform, b: NDArrayFloat, iterations: int = 5
    ) -> Tuple[np.floating, Sequence[int], DTypeLike]:
        """
        Compute the l2-norm of the operator A, with the power method.

        Parameters
        ----------
        A : BaseTransform
            Operator whose l2-norm needs to be computed.
        b : NDArrayFloat
            The data vector.
        iterations : int, optional
            Number of power method iterations. The default is 5.

        Returns
        -------
        Tuple[float, Tuple[int], DTypeLike]
            The l2-norm of A, and the shape and type of the solution.
        """
        x: NDArrayFloat = np.array(np.random.rand(*b.shape))
        x = x.astype(b.dtype)
        x /= np.linalg.norm(x)
        x = A.T(x)

        x_norm = np.linalg.norm(x)
        L = x_norm

        for ii in range(iterations):
            x /= x_norm
            x = A.T(A(x))

            x_norm = np.linalg.norm(x)
            L = np.sqrt(x_norm)

        return (L, x.shape, x.dtype)

    def _get_data_sigma_tau_unpreconditioned(self, A: operators.BaseTransform, b: NDArrayFloat):
        (L, x_shape, x_dtype) = self.power_method(A, b)
        tau = L

        dummy_x = np.empty(x_shape, dtype=x_dtype)
        for reg in self.regularizer:
            tau += reg.initialize_sigma_tau(dummy_x)

        tau = self.relaxation / tau
        sigma = self.relaxation / L
        return (x_shape, x_dtype, sigma, tau)

    def __call__(  # noqa: C901
        self,
        A: operators.BaseTransform,
        b: NDArrayFloat,
        iterations: int,
        x0: Optional[NDArrayFloat] = None,
        lower_limit: Union[float, NDArrayFloat, None] = None,
        upper_limit: Union[float, NDArrayFloat, None] = None,
        x_mask: Optional[NDArrayFloat] = None,
        b_mask: Optional[NDArrayFloat] = None,
        b_test_mask: Optional[NDArrayFloat] = None,
        precondition: bool = True,
    ) -> Tuple[NDArrayFloat, SolutionInfo]:
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
        x0 : Optional[NDArrayFloat], optional
            Initial solution. The default is None.
        lower_limit : Union[float, NDArrayFloat], optional
            Lower clipping value. The default is None.
        upper_limit : Union[float, NDArrayFloat], optional
            Upper clipping value. The default is None.
        x_mask : Optional[NDArrayFloat], optional
            Solution mask. The default is None.
        b_mask : Optional[NDArrayFloat], optional
            Data mask. The default is None.
        b_test_mask : Optional[NDArrayFloat], optional
            Test data mask. The default is None.
        precondition : bool, optional
            Whether to use the preconditioned version of the algorithm. The default is True.

        Returns
        -------
        Tuple[NDArrayFloat, SolutionInfo]
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

        (b_mask, b_test_mask) = self._initialize_b_masks(b, b_mask, b_test_mask)

        if precondition:
            tau = np.ones_like(b)
            if b_mask is not None:
                tau *= b_mask
            tau = np.abs(At_abs(tau))
            for reg in self.regularizer:
                tau += reg.initialize_sigma_tau(tau)
            tau[(tau / np.max(tau)) < 1e-5] = 1
            tau = self.relaxation / tau

            x_shape = tau.shape
            x_dtype = tau.dtype

            sigma = np.ones_like(tau)
            if x_mask is not None:
                sigma *= x_mask
            sigma = np.abs(A_abs(sigma))
            sigma[(sigma / np.max(sigma)) < 1e-5] = 1
            sigma = self.relaxation / sigma
        else:
            (x_shape, x_dtype, sigma, tau) = self._get_data_sigma_tau_unpreconditioned(A, b)

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

            if b_test_mask is not None:
                if self.data_term_test.background != self.data_term.background:
                    print("WARNING - the data_term and and data_term_test should have the same background. Making them equal.")
                    self.data_term_test.background = self.data_term.background
                self.data_term_test.assign_data(b, sigma)

                res_test_0 = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                info.residual0_cv = self.data_term_test.compute_residual_norm(res_test_0)

            if self.tolerance is not None:
                res_0 = self.data_term.compute_residual(Ax, mask=b_mask)
                info.residual0 = self.data_term.compute_residual_norm(res_0)

        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        algo_info = "- Performing %s-%s%s iterations: " % (self.upper(), self.data_term.upper(), reg_info)

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose)):
            info.iterations += 1

            Ax_rlx = A(x_relax)
            self.data_term.update_dual(p, Ax_rlx)
            self.data_term.apply_proximal(p)

            if b_mask is not None:
                p *= b_mask

            for q_r, reg in zip(q, self.regularizer):
                reg.update_dual(q_r, x_relax)
                reg.apply_proximal(q_r)

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

                if b_test_mask is not None:
                    res_test = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                    info.residuals_cv[ii] = self.data_term_test.compute_residual_norm(res_test)

                if self.tolerance is not None:
                    res = self.data_term.compute_residual(Ax, mask=b_mask)
                    info.residuals[ii] = self.data_term.compute_residual_norm(res)
                    if self.tolerance > info.residuals[ii]:
                        break

        return x, info


CP = PDHG
