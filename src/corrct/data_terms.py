#!/usr/bin/env python3
"""
Data fidelity classes.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import operators

eps = np.finfo(np.float32).eps


NDArrayFloat = NDArray[np.floating]


def _soft_threshold(values: NDArrayFloat, threshold: float | NDArrayFloat) -> None:
    abs_values = np.abs(values)
    valid_values = abs_values > 0
    if isinstance(threshold, (float, int)) or threshold.size == 1:
        local_threshold = threshold
    else:
        local_threshold = threshold[valid_values]
    values[valid_values] *= np.fmax((abs_values[valid_values] - local_threshold) / abs_values[valid_values], 0)


class DataFidelityBase(ABC):
    """Define the DataFidelity classes interface."""

    data: NDArrayFloat | None
    sigma: float | NDArrayFloat
    background: NDArrayFloat | None

    sigma_data: NDArrayFloat | None

    __data_fidelity_name__ = ""

    def __init__(self, background: float | NDArrayFloat | None = None) -> None:
        """
        Initialize the base data-fidelity class.

        Parameters
        ----------
        background : float | NDArrayFloat | None, optional
            The data background. The default is None.
        """
        self.background = np.array(background) if background is not None else None
        self.data = None
        self.sigma = 1.0
        self.sigma_data = None

    def _slice_attr(self, attr: str, ind: Any) -> None:
        attr_val = self.__getattribute__(attr)
        if attr_val is not None and isinstance(attr_val, np.ndarray) and attr_val.size > 1:
            self.__setattr__(attr, attr_val[ind])

    def __getitem__(self, ind: Any) -> "DataFidelityBase":
        """
        Slice the norm and all its attributes.

        Parameters
        ----------
        ind : Any
            Slicing indices.

        Returns
        -------
        DataFidelityBase
            The sliced norm.
        """
        new_self = deepcopy(self)
        for attr in self.__dict__.keys():
            new_self._slice_attr(attr, ind)
        return new_self

    def info(self) -> str:
        """
        Return the data-fidelity info.

        Returns
        -------
        str
            Data fidelity info string.
        """
        if self.background is not None:
            if np.array(self.background).size > 1:
                bckgrnd_str = "(B:<array>)"
            else:
                bckgrnd_str = "(B:%g)" % self.background
        else:
            bckgrnd_str = ""
        return self.__data_fidelity_name__ + bckgrnd_str

    def upper(self) -> str:
        """
        Return the upper case name of the data-fidelity.

        Returns
        -------
        str
            Upper case string name of the data-fidelity.
        """
        return self.info().upper()

    def lower(self) -> str:
        """
        Return the lower case name of the data-fidelity.

        Returns
        -------
        str
            Lower case string name of the data-fidelity.
        """
        return self.info().lower()

    def assign_data(self, data: float | NDArrayFloat | None = None, sigma: float | NDArrayFloat = 1.0) -> None:
        """Initialize the data bias, and sigma of the data term.

        Parameters
        ----------
        data : float | NDArrayFloat | None, optional
            The data bias, by default None
        sigma : float | NDArrayFloat, optional
            The sigma, by default 1.0
        """
        self.data = np.array(data) if data is not None else None
        self.sigma = sigma
        self.sigma_data = self._compute_sigma_data()
        if self.background is not None and self.data is not None:
            self.background = self.background.astype(self.data.dtype)

    def compute_residual(self, proj_primal: NDArrayFloat, mask: NDArrayFloat | None = None) -> NDArrayFloat:
        """Compute the residual in the dual domain.

        Parameters
        ----------
        proj_primal : NDArrayFloat
            Projection of the primal solution
        mask : NDArrayFloat | None, optional
            Mask of the dual domain, by default None

        Returns
        -------
        NDArrayFloat
            The residual
        """
        if self.background is not None:
            proj_primal = proj_primal + self.background

        if self.data is not None:
            residual = self.data - proj_primal
        else:
            residual = proj_primal.copy()

        if mask is not None:
            residual *= mask
        return residual

    @abstractmethod
    def compute_residual_norm(self, dual: NDArrayFloat) -> float:
        """Compute the norm of the residual.

        Parameters
        ----------
        dual : NDArrayFloat
            The residual in the dual domain.

        Returns
        -------
        float
            The residual norm.
        """

    def _compute_sigma_data(self):
        if self.data is None:
            return None
        else:
            return self.sigma * self.data

    def compute_data_dual_dot(self, dual: NDArrayFloat, mask: NDArrayFloat | None = None) -> float:
        """Compute the dot product of the data bias and the dual solution.

        Parameters
        ----------
        dual : NDArrayFloat
            The dual solution.
        mask : NDArrayFloat | None, optional
            Mask of the dual domain, by default None

        Returns
        -------
        float
            The dot product between the data bias and the dual solution
        """
        if self.data is not None:
            if mask is not None:
                dual = dual * mask

            return np.dot(dual.flatten(), self.data.flatten())
        else:
            return 0.0

    def initialize_dual(self) -> NDArrayFloat:
        """Initialize the dual domain solution.

        Returns
        -------
        NDArrayFloat
            A zero array with the dimensions of the dual domain.
        """
        return np.zeros_like(self.data)

    def update_dual(self, dual: NDArrayFloat, proj_primal: NDArrayFloat) -> None:
        """Update the dual solution.

        Parameters
        ----------
        dual : NDArrayFloat
            The current dual solution
        proj_primal : NDArrayFloat
            The projected primal solution
        """
        if self.background is None:
            dual += proj_primal * self.sigma
        else:
            dual += (proj_primal + self.background) * self.sigma

    @abstractmethod
    def apply_proximal_dual(self, dual: NDArrayFloat) -> None:
        """Apply the proximal in the dual domain.

        Parameters
        ----------
        dual : NDArrayFloat
            The dual solution
        """

    @abstractmethod
    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Apply the proximal operator in the primal domain, in-place.

        Computes prox_{tau * f}(primal), i.e. the proximal of the data-fidelity
        f with step size tau, and stores the result back into ``primal``.

        Note: in FISTA the data term is handled via a gradient step, so this
        method is exposed for standalone use or for algorithms that prefer a
        pure proximal approach (e.g. when A = I).

        Parameters
        ----------
        primal : NDArrayFloat
            The current primal variable (modified in-place).
        tau : float | NDArrayFloat
            The proximal step size.
        """

    @abstractmethod
    def compute_primal_dual_gap(
        self, proj_primal: NDArrayFloat, dual: NDArrayFloat, mask: NDArrayFloat | None = None
    ) -> float:
        """Compute the primal-dual gap of the current solution.

        Parameters
        ----------
        proj_primal : NDArrayFloat
            The projected primal solution (in the dual domain)
        dual : NDArrayFloat
            The dual solution
        mask : NDArrayFloat | None, optional
            Mask in the dual domain, by default None

        Returns
        -------
        float
            The primal-dual gap
        """


class DataFidelity_l2(DataFidelityBase):
    """l2-norm data-fidelity class."""

    __data_fidelity_name__ = "l2"

    sigma1: float | NDArrayFloat

    def __init__(self, background: float | NDArrayFloat | None = None) -> None:
        super().__init__(background=background)
        self.sigma1 = 1.0

    def assign_data(self, data: float | NDArrayFloat | None = None, sigma: float | NDArrayFloat = 1.0) -> None:
        super().assign_data(data=data, sigma=sigma)
        self.sigma1 = 1 / (1 + sigma)

    def compute_residual_norm(self, dual: NDArrayFloat) -> float:
        return float(np.linalg.norm(dual.flatten(), ord=2) ** 2)

    def apply_proximal_dual(self, dual: NDArrayFloat) -> None:
        if self.data is not None and self.sigma_data is not None:
            dual -= self.sigma_data
        dual *= self.sigma1

    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Apply prox_{tau * (1/2) * ||. - b||^2} in-place.

        The closed-form solution is:
            prox(x) = (x + tau * b) / (1 + tau)

        When no data has been assigned (b = 0):
            prox(x) = x / (1 + tau)

        Parameters
        ----------
        primal : NDArrayFloat
            The primal variable to update in-place.
        tau : float | NDArrayFloat
            The proximal step size.
        """
        if self.data is not None:
            primal += tau * self.data
        primal /= 1.0 + tau

    def compute_primal_dual_gap(
        self, proj_primal: NDArrayFloat, dual: NDArrayFloat, mask: NDArrayFloat | None = None
    ) -> float:
        return float(
            np.linalg.norm(self.compute_residual(proj_primal, mask), ord=2) + np.linalg.norm(dual, ord=2)
        ) / 2 + self.compute_data_dual_dot(dual)


class DataFidelity_wl2(DataFidelity_l2):
    """Weighted l2-norm data-fidelity class."""

    __data_fidelity_name__ = "wl2"

    sigma1: float | NDArrayFloat
    weights: NDArrayFloat

    def __init__(self, weights: float | NDArrayFloat, background: float | NDArrayFloat | None = None) -> None:
        super().__init__(background=background)
        self.sigma1 = 1.0
        self.weights = np.array(weights)

    def assign_data(self, data: float | NDArrayFloat | None, sigma: float | NDArrayFloat = 1.0):
        super().assign_data(data=data, sigma=sigma)
        if isinstance(self.sigma, np.ndarray):
            dtype = self.sigma.dtype
        else:
            dtype = type(self.sigma)
        invalid_weights = (self.weights == 0).astype(dtype)
        self.sigma1 = 1 / (1 + sigma / (self.weights + invalid_weights)) * (1 - invalid_weights)

    def compute_residual(self, proj_primal, mask: float | NDArrayFloat | None = None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        if self.data is not None:
            residual = (self.data - proj_primal) * self.weights
        else:
            residual = proj_primal * self.weights
        if mask is not None:
            residual *= mask
        return residual

    def compute_residual_norm(self, dual: float | NDArrayFloat) -> float:
        valid_weights = self.weights != 0
        if isinstance(dual, np.ndarray):
            dual = dual[valid_weights]
        weights = self.weights[valid_weights]
        return float(np.linalg.norm((dual / np.sqrt(weights)).flatten(), ord=2) ** 2)

    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Apply prox_{tau * (1/2) * ||. - b||^2_W} in-place.

        For the weighted l2 norm (1/2)||x - b||^2_W = (1/2) sum_i w_i (x_i - b_i)^2,
        the proximal is computed element-wise:
            prox(x)_i = (x_i + tau * w_i * b_i) / (1 + tau * w_i)

        Zero-weight entries are left unchanged (no constraint enforced there).

        Parameters
        ----------
        primal : NDArrayFloat
            The primal variable to update in-place.
        tau : float | NDArrayFloat
            The proximal step size.
        """
        tau_w = tau * self.weights
        if self.data is not None:
            primal += tau_w * self.data
        # For zero-weight entries: denominator = 1, so they pass through unchanged.
        denom = 1.0 + tau_w
        primal /= denom


class DataFidelity_l2b(DataFidelity_l2):
    """l2-norm ball data-fidelity class."""

    __data_fidelity_name__ = "l2b"

    sigma1: float | NDArrayFloat
    sigma_error: float | NDArrayFloat
    sigma_sqrt_error: float | NDArrayFloat

    def __init__(self, local_error: float | NDArrayFloat, background: float | NDArrayFloat | None = None):
        super().__init__(background=background)
        self.sigma1 = 1.0
        self.local_error = local_error
        self.sigma_error = 1.0 * self.local_error
        self.sigma_sqrt_error = 1.0 * np.sqrt(self.local_error)

    def assign_data(self, data: float | NDArrayFloat | None, sigma: float | NDArrayFloat = 1.0):
        self.sigma_error = sigma * self.local_error
        self.sigma_sqrt_error = sigma * np.sqrt(self.local_error)
        super().assign_data(data=data, sigma=sigma)
        self.sigma1 = 1 / (1 + self.sigma_error)

    def compute_residual(self, proj_primal: NDArrayFloat, mask: NDArrayFloat | None = None) -> NDArrayFloat:
        residual = super().compute_residual(proj_primal, mask)
        _soft_threshold(residual, self.sigma_sqrt_error)
        return residual

    def apply_proximal_dual(self, dual: NDArrayFloat) -> None:
        if self.data is not None and self.sigma_data is not None:
            dual -= self.sigma_data
        _soft_threshold(dual, self.sigma_sqrt_error)
        dual *= self.sigma1

    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Apply prox_{tau * f_{l2b}} in-place.

        The l2-ball data fidelity is:
            f(x) = max(||x - b|| - sqrt(epsilon), 0)^2 / 2

        whose proximal is a soft-thresholded then scaled l2 proximal.
        Specifically:
            v = x - b
            soft-threshold v by sqrt(epsilon) * tau / (1 + tau * epsilon)
            prox(x) = b + v_thresholded / (1 + tau * epsilon)

        Parameters
        ----------
        primal : NDArrayFloat
            The primal variable to update in-place.
        tau : float | NDArrayFloat
            The proximal step size.
        """
        if self.data is not None:
            primal -= self.data
        tau_eps = tau * self.local_error
        _soft_threshold(primal, np.sqrt(self.local_error) * tau / (1.0 + tau_eps))
        primal /= 1.0 + tau_eps
        if self.data is not None:
            primal += self.data

    def compute_primal_dual_gap(
        self, proj_primal: NDArrayFloat, dual: NDArrayFloat, mask: NDArrayFloat | None = None
    ) -> float:
        return float(
            np.linalg.norm(self.compute_residual(proj_primal, mask), ord=2)
            + np.linalg.norm(np.sqrt(self.local_error) * dual, ord=2)
        ) / 2 + self.compute_data_dual_dot(dual)


class DataFidelity_Huber(DataFidelityBase):
    """Huber-norm data-fidelity class. Given a parameter a: l2-norm for x < a, and l1-norm for x > a."""

    __data_fidelity_name__ = "Hub"

    one_sigma_error: float | NDArrayFloat

    def __init__(self, local_error, background=None, l2_axis=None):
        super().__init__(background=background)
        self.local_error = local_error
        self.l2_axis = l2_axis
        self.one_sigma_error = 1.0

    def assign_data(self, data, sigma=1.0):
        self.one_sigma_error = 1.0 / (1.0 + sigma * self.local_error)
        super().assign_data(data=data, sigma=sigma)

    def compute_residual_norm(self, dual):
        l2_points = dual <= self.local_error
        l1_points = 1 - l2_points
        return np.linalg.norm(dual[l2_points].flatten(), ord=2) ** 2 + np.linalg.norm(dual[l1_points].flatten(), ord=1)

    def apply_proximal_dual(self, dual):
        if self.data is not None and self.sigma_data is not None:
            dual -= self.sigma_data

        dual *= self.one_sigma_error

        if self.l2_axis is None:
            dual /= np.fmax(1, np.abs(dual))
        else:
            dual_dir_norm_l2 = np.linalg.norm(dual, ord=2, axis=self.l2_axis, keepdims=True)
            dual /= np.fmax(1, dual_dir_norm_l2)

    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Not implemented: the Huber proximal in the primal has no simple closed form for general data.

        The Huber function is f(x) = (1/2)||x-b||^2 if ||x-b|| <= a, else a*||x-b|| - a^2/2.
        Its proximal is a smooth interpolation between the l2 and l1 proximals, whose
        closed form depends on the norm of (x - b) relative to the threshold, making
        it straightforward only for the scalar case. For vector inputs with l2_axis,
        a Newton iteration would be required.

        Raises
        ------
        NotImplementedError
            Always raised; use a gradient step or PDHG instead.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.apply_proximal_primal: the Huber proximal has no simple "
            "closed-form solution for general (possibly vector-valued) inputs. "
            "Use a gradient step in the solver, or switch to PDHG for this data term."
        )

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background
        return (
            np.linalg.norm(self.compute_residual(proj_primal, mask), ord=2)
            + self.compute_data_dual_dot(dual)
            + self.local_error * np.linalg.norm(dual, ord=2)
        )


class DataFidelity_l1(DataFidelityBase):
    """l1-norm data-fidelity class."""

    __data_fidelity_name__ = "l1"

    def __init__(self, background=None):
        super().__init__(background=background)

    def _get_inner_norm(self, dual):
        return np.abs(dual)

    def _apply_threshold(self, dual):
        pass

    def apply_proximal_dual(self, dual, weight=1.0):
        if self.data is not None:
            dual -= self.sigma_data
        self._apply_threshold(dual)
        dual_inner_norm = self._get_inner_norm(dual)
        dual /= np.fmax(dual_inner_norm, weight)
        dual *= weight

    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Apply prox_{tau * ||. - b||_1} in-place via soft-thresholding.

        For f(x) = ||x - b||_1, the proximal is element-wise soft-thresholding
        centered on b:
            prox(x)_i = b_i + sign(x_i - b_i) * max(|x_i - b_i| - tau, 0)

        When no data has been assigned (b = 0), this reduces to plain
        soft-thresholding with threshold tau.

        Parameters
        ----------
        primal : NDArrayFloat
            The primal variable to update in-place.
        tau : float | NDArrayFloat
            The proximal step size (soft-threshold level).
        """
        if self.data is not None:
            primal -= self.data
        _soft_threshold(primal, tau)
        if self.data is not None:
            primal += self.data

    def compute_residual_norm(self, dual):
        dual = dual.copy()
        self._apply_threshold(dual)
        dual_inner_norm = self._get_inner_norm(dual)
        return np.linalg.norm(dual_inner_norm, ord=1)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background

        residual = self.compute_residual(proj_primal, mask)
        self._apply_threshold(residual)
        residual_inner_norm = self._get_inner_norm(residual)
        return np.linalg.norm(residual_inner_norm, ord=1) + self.compute_data_dual_dot(dual)


class DataFidelity_l12(DataFidelity_l1):
    """l12-norm data-fidelity class."""

    __data_fidelity_name__ = "l12"

    def __init__(self, background=None, l2_axis=0):
        super().__init__(background=background)
        self.l2_axis = l2_axis

    def _get_inner_norm(self, dual):
        return np.linalg.norm(dual, ord=2, axis=self.l2_axis, keepdims=True)

    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Not implemented: the l12 proximal in the primal domain requires a group soft-threshold
        along l2_axis, which is straightforward only when the l2_axis corresponds to independent
        groups that do not interact through A. For general use, apply PDHG or provide a
        custom group-soft-threshold.

        Raises
        ------
        NotImplementedError
            Always raised.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.apply_proximal_primal: the l12 norm proximal in the "
            "primal domain is axis-dependent and cannot be applied independently per element. "
            "Use PDHG for this data term, which handles it correctly via the dual."
        )


class DataFidelity_l1b(DataFidelity_l1):
    """l1-norm ball data-fidelity class."""

    __data_fidelity_name__ = "l1b"

    sigma_error: float | NDArrayFloat

    def __init__(self, local_error, background=None):
        super().__init__(background=background)
        self.local_error = local_error
        self.sigma_error = 1.0 * self.local_error

    def assign_data(self, data, sigma=1.0):
        self.sigma_error = sigma * self.local_error
        super().assign_data(data=data, sigma=sigma)

    def _apply_threshold(self, dual):
        _soft_threshold(dual, self.local_error)

    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Apply prox_{tau * f_{l1b}} in-place.

        The l1-ball data fidelity is:
            f(x) = max(||x - b||_1 - epsilon, 0)

        Its proximal is a two-stage soft-threshold:
            v = x - b
            soft-threshold v by (tau / (1 + tau)) elementwise, with level epsilon
            prox(x) = b + v_thresholded

        This is obtained via Moreau's identity applied to the indicator of the
        l1-ball of radius epsilon.

        Parameters
        ----------
        primal : NDArrayFloat
            The primal variable to update in-place.
        tau : float | NDArrayFloat
            The proximal step size.
        """
        if self.data is not None:
            primal -= self.data
        # Two-level soft threshold: first remove the ball radius, then scale
        _soft_threshold(primal, self.local_error)
        primal *= tau / (1.0 + tau)
        if self.data is not None:
            primal += self.data


class DataFidelity_KL(DataFidelityBase):
    """Kullback-Leibler data-fidelity class."""

    __data_fidelity_name__ = "KL"

    def _compute_sigma_data(self):
        if self.data is None:
            return None
        else:
            return 4 * self.sigma * np.fmax(self.data, 0.0)

    def apply_proximal_dual(self, dual):
        if self.sigma_data is not None:
            dual[:] = (1 + dual[:] - np.sqrt((dual[:] - 1) ** 2 + self.sigma_data[:])) / 2
        else:
            dual[:] = (1 + dual[:] - np.sqrt((dual[:] - 1) ** 2)) / 2

    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Apply prox_{tau * KL(b, .)} in-place.

        The Kullback-Leibler divergence (in the emission-CT convention) is:
            f(x) = sum_i (x_i - b_i * log(x_i))   (for x_i > 0)

        Its proximal has the closed-form solution:
            prox(x)_i = ((x_i - tau) + sqrt((x_i - tau)^2 + 4 * tau * b_i)) / 2

        This is always non-negative when b_i >= 0 and x_i > 0.

        Parameters
        ----------
        primal : NDArrayFloat
            The primal variable to update in-place (must be > 0).
        tau : float | NDArrayFloat
            The proximal step size.
        """
        if self.data is not None:
            b = np.fmax(self.data, 0.0)
            disc = (primal - tau) ** 2 + 4.0 * tau * b
            primal[:] = ((primal - tau) + np.sqrt(disc)) / 2.0
        else:
            # No data: f(x) = sum x_i, prox is max(x - tau, 0) clamped away from zero
            primal[:] = np.fmax(primal - tau, eps)

    def compute_residual(self, proj_primal: NDArray, mask: NDArray | None = None, use_proximal: bool = True) -> NDArrayFloat:
        if self.background is not None:
            proj_primal = proj_primal + self.background

        proj_primal = np.fmax(proj_primal, eps)

        if use_proximal:
            # we take the Moreau envelope here, and apply the proximal to it
            residual = np.fmax(proj_primal, eps) * self.sigma

            self.apply_proximal_dual(residual)
        else:
            if self.data is not None:
                residual = 1.0 - np.fmax(self.data, 0.0) / proj_primal
            else:
                residual = np.ones_like(proj_primal)

        if mask is not None:
            residual *= mask
        return -residual

    def compute_residual_norm(self, dual):
        return np.linalg.norm(dual.flatten(), ord=1)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background

        if self.data is not None:
            data_nn = np.fmax(self.data, eps)
            proj_primal_nn = np.fmax(proj_primal, eps)
            residual = proj_primal_nn - data_nn * (1 - np.log(data_nn) + np.log(proj_primal_nn))
        else:
            residual = np.copy(proj_primal)
        if mask is not None:
            residual *= mask

        return np.linalg.norm(residual, ord=1)


class DataFidelity_ln(DataFidelityBase):
    """nuclear-norm data-fidelity class."""

    __data_fidelity_name__ = "ln"

    def __init__(self, background=None, ln_axes: Sequence[int] = (1, -1), spectral_norm: DataFidelityBase = DataFidelity_l1()):
        super().__init__(background=background)
        self.ln_axes = ln_axes
        self.spectral_norm = spectral_norm
        self.use_fallback = False

    def apply_proximal_dual(self, dual):
        dual_tmp = dual.copy()

        if self.sigma_data is not None:
            # If we have a bias term, we interpret it as an addition to the rows in the SVD decomposition.
            # Performing this operation before the transpose is a waste of computation, but it simplifies the logic.
            dual_tmp = np.concatenate((dual_tmp, self.sigma_data), axis=self.ln_axes[0])

        if self.use_fallback:
            t_range = [*range(len(dual_tmp.shape))]
            t_range.append(t_range.pop(self.ln_axes[0]))
            t_range.append(t_range.pop(self.ln_axes[1]))
            dual_tmp = np.transpose(dual_tmp, t_range)

            U, s_p, Vt = np.linalg.svd(dual_tmp, full_matrices=False)

            self.spectral_norm.apply_proximal_dual(s_p)

            dual_tmp = np.matmul(U, s_p[..., None] * Vt)
            dual_tmp = np.transpose(dual_tmp, np.argsort(t_range))
        else:
            op_svd = operators.TransformSVD(dual_tmp.shape, axes_rows=self.ln_axes[0], axes_cols=self.ln_axes[1])
            s_p = op_svd(dual_tmp)

            self.spectral_norm.apply_proximal_dual(s_p)

            dual_tmp = op_svd.T(s_p)

        if self.data is not None:
            # We now strip the bias data, to make sure that we don't change dimensionality.
            # dual_tmp = dual_tmp[..., : dual_tmp.shape[-2] - 1 :, :]
            dual_tmp = np.take(dual_tmp, np.arange(dual_tmp.shape[self.ln_axes[0]] - 1), axis=self.ln_axes[0])

        dual[:] = dual_tmp[:]

    def apply_proximal_primal(self, primal: NDArrayFloat, tau: float | NDArrayFloat) -> None:
        """Not implemented: the nuclear-norm proximal requires a full SVD and singular-value
        soft-thresholding, which is only well-defined for the dual formulation used here
        (where the SVD axes and data structure are set up by the PDHG dual update).
        Applying it directly in the primal domain would require re-interpreting primal
        axes as matrix rows/columns, which depends on context not available here.

        Raises
        ------
        NotImplementedError
            Always raised; use PDHG for nuclear-norm data fidelity.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.apply_proximal_primal: the nuclear-norm proximal "
            "requires an SVD over axes that are determined by the dual formulation. "
            "It cannot be applied generically in the primal domain. Use PDHG instead."
        )

    def compute_residual_norm(self, dual):
        op_svd = operators.TransformSVD(dual.shape, axes_rows=self.ln_axes[0], axes_cols=self.ln_axes[1])
        s_p = op_svd(dual)
        return np.linalg.norm(s_p, ord=1)

    def compute_primal_dual_gap(self, proj_primal, dual, mask=None):
        if self.background is not None:
            proj_primal = proj_primal + self.background

        residual = self.compute_residual(proj_primal, mask)

        return self.compute_residual_norm(residual) + self.compute_data_dual_dot(dual)
