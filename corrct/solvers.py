#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solvers for the tomographic reconstruction problem.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from typing import Optional, Union, Tuple

import numpy as np
import numpy.random
from numpy.typing import ArrayLike, DTypeLike

import scipy.sparse as sps

import copy as cp

from . import data_terms
from . import regularizers

from tqdm import tqdm


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


# ---- Constraints ----


Constraint_LowerLimit = regularizers.Constraint_LowerLimit
Constraint_UpperLimit = regularizers.Constraint_UpperLimit


# ---- Solvers ----


class Solver(object):
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
        tolerance: float = None,
        data_term: Union[str, DataFidelityBase] = "l2",
        data_term_test: Optional[DataFidelityBase] = None,
    ):
        self.verbose = verbose
        self.relaxation = relaxation
        self.tolerance = tolerance

        self.data_term = self._initialize_data_fidelity_function(data_term)
        if data_term_test is None:
            data_term_test = self.data_term
        self.data_term_test = cp.deepcopy(self.data_term)

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
    def _initialize_data_operators(
        A: Union[ArrayLike, sps.linalg.LinearOperator, sps.dia_matrix],
        At: Optional[Union[ArrayLike, sps.linalg.LinearOperator, sps.dia_matrix]],
    ):
        if At is None:
            if isinstance(A, np.ndarray):
                At = A.transpose((1, 0))
            elif isinstance(A, sps.dia_matrix) or isinstance(A, sps.linalg.LinearOperator):
                At = A.transpose()

        if isinstance(At, np.ndarray) or isinstance(At, sps.dia_matrix):
            At_m = At
            At = At_m.dot
        if isinstance(A, np.ndarray) or isinstance(A, sps.dia_matrix):
            A_m = A
            A = A_m.dot
        return (A, At)

    @staticmethod
    def _initialize_regularizer(regularizer: Optional[BaseRegularizer]) -> BaseRegularizer:
        if regularizer is None:
            return []
        elif isinstance(regularizer, regularizers.BaseRegularizer):
            return [regularizer]
        elif isinstance(regularizer, (list, tuple)):
            check_regs_ok = [isinstance(r, BaseRegularizer) for r in regularizer]
            if not np.all(check_regs_ok):
                raise ValueError(
                    "The following regularizers are not derived from the BaseRegularizer class: %s"
                    % np.array(np.arange(len(check_regs_ok))[np.array(check_regs_ok, dtype=np.bool)])
                )
            else:
                return list(regularizer)
        else:
            raise ValueError("Unknown regularizer type.")

    @staticmethod
    def _initialize_b_masks(b: ArrayLike, b_mask: Optional[ArrayLike], b_test_mask: Optional[ArrayLike]):
        if b_test_mask is not None:
            if b_mask is None:
                b_mask = np.ones_like(b)
            # As we are being passed a test residual pixel mask, we need
            # to make sure to mask those pixels out from the reconstruction.
            # At the same time, we need to remove any masked pixel from the test count.
            b_mask, b_test_mask = b_mask * (1 - b_test_mask), b_test_mask * b_mask
        return (b_mask, b_test_mask)


class Sart(Solver):
    """Solver class implementing the Simultaneous Algebraic Reconstruction Technique (SART) algorithm."""

    def __call__(  # noqa: C901
        self,
        A,
        b: ArrayLike,
        iterations: int,
        A_num_rows: int,
        x0: Optional[ArrayLike] = None,
        At=None,
        lower_limit: Union[float, ArrayLike] = None,
        upper_limit: Union[float, ArrayLike] = None,
        x_mask: Optional[ArrayLike] = None,
        b_mask: Optional[ArrayLike] = None,
    ) -> Tuple[ArrayLike, Optional[ArrayLike]]:
        """
        Reconstruct the data, using the SART algorithm.

        Parameters
        ----------
        A : Union[Callable, BaseTransform]
            Projection operator.
        b : ArrayLike
            Data to reconstruct.
        iterations : int
            Number of iterations.
        A_num_rows : int
            Number of projections.
        x0 : Optional[ArrayLike], optional
            Initial solution. The default is None.
        At : Callable, optional
            The back-projection operator. This is only needed if the projection operator does not have an adjoint.
            The default is None.
        lower_limit : Union[float, ArrayLike], optional
            Lower clipping value. The default is None.
        upper_limit : Union[float, ArrayLike], optional
            Upper clipping value. The default is None.
        x_mask : Optional[ArrayLike], optional
            Solution mask. The default is None.
        b_mask : Optional[ArrayLike], optional
            Data mask. The default is None.

        Returns
        -------
        Tuple[ArrayLike, Tuple[Optional[ArrayLike]]]
            The reconstruction, and the residuals.
        """
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
        x = x0

        if self.tolerance is not None:
            res_norm_0 = np.linalg.norm((b * b_mask).flatten())
            res_norm_rel = np.ones((iterations,)) * self.tolerance
        else:
            res_norm_rel = None

        rows_sequence = np.random.permutation(A_num_rows)

        algo_info = "- Performing %s iterations: " % self.upper()

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose)):

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
                res = np.empty_like(b)
                for ii_a in rows_sequence:
                    res[..., ii_a, :] = A(x, ii_a) - b[..., ii_a, :]
                if b_mask is not None:
                    res *= b_mask
                res_norm_rel[ii] = np.linalg.norm(res) / res_norm_0

                if self.tolerance > res_norm_rel[ii]:
                    break

        return (x, res_norm_rel)


class Sirt(Solver):
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
    regularizer : Optional[BaseRegularizer], optional
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
        regularizer: Optional[BaseRegularizer] = None,
        data_term: Union[str, DataFidelityBase] = "l2",
        data_term_test: Optional[DataFidelityBase] = None,
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
        A,
        b: ArrayLike,
        iterations: int,
        x0: Optional[ArrayLike] = None,
        At=None,
        lower_limit: Union[float, ArrayLike] = None,
        upper_limit: Union[float, ArrayLike] = None,
        x_mask: Optional[ArrayLike] = None,
        b_mask: Optional[ArrayLike] = None,
        b_test_mask: Optional[ArrayLike] = None,
    ) -> Tuple[ArrayLike, Tuple[Optional[ArrayLike], Optional[ArrayLike], int]]:
        """
        Reconstruct the data, using the SIRT algorithm.

        Parameters
        ----------
        A : Union[Callable, BaseTransform]
            Projection operator.
        b : ArrayLike
            Data to reconstruct.
        iterations : int
            Number of iterations.
        x0 : Optional[ArrayLike], optional
            Initial solution. The default is None.
        At : Callable, optional
            The back-projection operator. This is only needed if the projection operator does not have an adjoint.
            The default is None.
        lower_limit : Union[float, ArrayLike], optional
            Lower clipping value. The default is None.
        upper_limit : Union[float, ArrayLike], optional
            Upper clipping value. The default is None.
        x_mask : Optional[ArrayLike], optional
            Solution mask. The default is None.
        b_mask : Optional[ArrayLike], optional
            Data mask. The default is None.
        b_test_mask : Optional[ArrayLike], optional
            Test data mask. The default is None.

        Returns
        -------
        Tuple[ArrayLike, Tuple[Optional[ArrayLike], Optional[ArrayLike], int]]
            The reconstruction, and the residuals.
        """
        (A, At) = self._initialize_data_operators(A, At)

        (b_mask, b_test_mask) = self._initialize_b_masks(b, b_mask, b_test_mask)

        # Back-projection diagonal re-scaling
        tau = np.ones_like(b)
        if b_mask is not None:
            tau *= b_mask
        tau = np.abs(At(tau))
        for reg in self.regularizer:
            tau += reg.initialize_sigma_tau(tau)
        tau[(tau / np.max(tau)) < 1e-5] = 1
        tau = self.relaxation / tau

        # Forward-projection diagonal re-scaling
        sigma = np.ones_like(tau)
        if x_mask is not None:
            sigma *= x_mask
        sigma = np.abs(A(sigma))
        sigma[(sigma / np.max(sigma)) < 1e-5] = 1
        sigma = 1 / sigma

        if x0 is None:
            x0 = At(b * sigma) * tau
        else:
            x0 = x0.copy()
        x = x0

        self.data_term.assign_data(b, sigma)

        if b_test_mask is not None:
            if self.data_term_test.background != self.data_term.background:
                print("WARNING - the data_term and and data_term_test should have the same background. Making them equal.")
                self.data_term_test.background = self.data_term.background
            self.data_term_test.assign_data(b, sigma)

            res_test_0 = self.data_term_test.compute_residual(0, mask=b_test_mask)
            res_test_norm_0 = self.data_term_test.compute_residual_norm(res_test_0)
            res_test_norm_rel = np.ones((iterations,))
        else:
            res_test_norm_rel = None

        if self.tolerance is not None:
            res_0 = self.data_term.compute_residual(0, mask=b_mask)
            res_norm_0 = self.data_term.compute_residual_norm(res_0)
            res_norm_rel = np.ones((iterations,)) * self.tolerance
        else:
            res_norm_rel = None

        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        algo_info = "- Performing %s-%s%s iterations: " % (self.upper(), self.data_term.upper(), reg_info)

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose)):
            Ax = A(x)
            res = self.data_term.compute_residual(Ax, mask=b_mask)

            if b_test_mask is not None:
                res_test = self.data_term_test.compute_residual(Ax, mask=b_test_mask)
                res_test_norm_rel[ii] = self.data_term_test.compute_residual_norm(res_test.flatten()) / res_test_norm_0

            if self.tolerance is not None:
                res_norm_rel[ii] = self.data_term.compute_residual_norm(res.flatten()) / res_norm_0
                if self.tolerance > res_norm_rel[ii]:
                    break

            q = [reg.initialize_dual() for reg in self.regularizer]
            for q_r, reg in zip(q, self.regularizer):
                reg.update_dual(q_r, x)
                reg.apply_proximal(q_r)

            upd = At(res * sigma)
            for q_r, reg in zip(q, self.regularizer):
                upd -= reg.compute_update_primal(q_r)
            x += upd * tau

            if lower_limit is not None or upper_limit is not None:
                x = x.clip(lower_limit, upper_limit)
            if x_mask is not None:
                x *= x_mask

        return (x, (res_norm_rel, res_test_norm_rel, ii))


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
    regularizer : Optional[BaseRegularizer], optional
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
        regularizer: Optional[BaseRegularizer] = None,
        data_term: Union[str, DataFidelityBase] = "l2",
        data_term_test: Optional[DataFidelityBase] = None,
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

    def power_method(self, A, At, b: ArrayLike, iterations: int = 5) -> Tuple[float, Tuple[int], DTypeLike]:
        """
        Compute the l2-norm of the operator A, with the power method.

        Parameters
        ----------
        A : Callable | BaseTransform
            Operator whose l2-norm needs to be computed.
        At : Callable | BaseTransform
            Adjoint of the operator.
        b : ArrayLike
            The data vector.
        iterations : int, optional
            Number of power method iterations. The default is 5.

        Returns
        -------
        Tuple[float, Tuple[int], DTypeLike]
            The l2-norm of A, and the shape and type of the solution.
        """
        x = np.random.rand(*b.shape).astype(b.dtype)
        x /= np.linalg.norm(x)
        x = At(x)

        x_norm = np.linalg.norm(x)
        L = x_norm

        for ii in range(iterations):
            x /= x_norm
            x = At(A(x))

            x_norm = np.linalg.norm(x)
            L = np.sqrt(x_norm)

        return (L, x.shape, x.dtype)

    def _get_data_sigma_tau_unpreconditioned(self, A, At, b: ArrayLike):
        (L, x_shape, x_dtype) = self.power_method(A, At, b)
        tau = L

        dummy_x = np.empty(x_shape, dtype=x_dtype)
        for reg in self.regularizer:
            tau += reg.initialize_sigma_tau(dummy_x)

        tau = self.relaxation / tau
        sigma = self.relaxation / L
        return (x_shape, x_dtype, sigma, tau)

    def __call__(  # noqa: C901
        self,
        A,
        b: ArrayLike,
        iterations: int,
        x0: Optional[ArrayLike] = None,
        At=None,
        lower_limit: Union[float, ArrayLike] = None,
        upper_limit: Union[float, ArrayLike] = None,
        x_mask: Optional[ArrayLike] = None,
        b_mask: Optional[ArrayLike] = None,
        b_test_mask: Optional[ArrayLike] = None,
        precondition: bool = True,
    ) -> Tuple[ArrayLike, Tuple[Optional[ArrayLike], Optional[ArrayLike], int]]:
        """
        Reconstruct the data, using the PDHG algorithm.

        Parameters
        ----------
        A : Union[Callable, BaseTransform]
            Projection operator.
        b : ArrayLike
            Data to reconstruct.
        iterations : int
            Number of iterations.
        x0 : Optional[ArrayLike], optional
            Initial solution. The default is None.
        At : Callable, optional
            The back-projection operator. This is only needed if the projection operator does not have an adjoint.
            The default is None.
        lower_limit : Union[float, ArrayLike], optional
            Lower clipping value. The default is None.
        upper_limit : Union[float, ArrayLike], optional
            Upper clipping value. The default is None.
        x_mask : Optional[ArrayLike], optional
            Solution mask. The default is None.
        b_mask : Optional[ArrayLike], optional
            Data mask. The default is None.
        b_test_mask : Optional[ArrayLike], optional
            Test data mask. The default is None.
        precondition : bool, optional
            Whether to use the preconditioned version of the algorithm. The default is True.

        Returns
        -------
        Tuple[ArrayLike, Tuple[Optional[ArrayLike], Optional[ArrayLike], int]]
            The reconstruction, and the residuals.
        """
        (A, At) = self._initialize_data_operators(A, At)
        if precondition:
            try:
                At_abs = At.absolute()
                A_abs = A.absolute()
            except AttributeError:
                print(A, At)
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
            (x_shape, x_dtype, sigma, tau) = self._get_data_sigma_tau_unpreconditioned(A, At, b)

        if x0 is None:
            x0 = np.zeros(x_shape, dtype=x_dtype)
        else:
            x0 = x0.copy()
        x = x0
        x_relax = x.copy()

        self.data_term.assign_data(b, sigma)
        p = self.data_term.initialize_dual()

        q = [reg.initialize_dual() for reg in self.regularizer]

        if b_test_mask is not None:
            if self.data_term_test.background != self.data_term.background:
                print("WARNING - the data_term and and data_term_test should have the same background. Making them equal.")
                self.data_term_test.background = self.data_term.background
            self.data_term_test.assign_data(b, sigma)

            res_test_0 = self.data_term_test.compute_residual(0, mask=b_test_mask)
            res_test_norm_0 = self.data_term_test.compute_residual_norm(res_test_0)
            res_test_norm_rel = np.ones((iterations,))
        else:
            res_test_norm_rel = None

        if self.tolerance is not None:
            res_0 = self.data_term.compute_residual(0, mask=b_mask)
            res_norm_0 = self.data_term.compute_residual_norm(res_0)
            res_norm_rel = np.ones((iterations,)) * self.tolerance
        else:
            res_norm_rel = None

        reg_info = "".join(["-" + r.info().upper() for r in self.regularizer])
        algo_info = "- Performing %s-%s%s iterations: " % (self.upper(), self.data_term.upper(), reg_info)

        for ii in tqdm(range(iterations), desc=algo_info, disable=(not self.verbose)):
            Ax_rlx = A(x_relax)
            self.data_term.update_dual(p, Ax_rlx)
            self.data_term.apply_proximal(p)

            if b_mask is not None:
                p *= b_mask

            for q_r, reg in zip(q, self.regularizer):
                reg.update_dual(q_r, x_relax)
                reg.apply_proximal(q_r)

            upd = At(p)
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
                res_test_norm_rel[ii] = self.data_term_test.compute_residual_norm(res_test.flatten()) / res_test_norm_0

            if self.tolerance is not None:
                res = self.data_term.compute_residual(Ax, mask=b_mask)
                res_norm_rel[ii] = self.data_term.compute_residual_norm(res.flatten()) / res_norm_0
                if self.tolerance > res_norm_rel[ii]:
                    break

        return (x, (res_norm_rel, res_test_norm_rel, ii))


CP = PDHG
