#!/usr/bin/env python3
"""
Aided regularization parameter estimation.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import concurrent.futures as cf
import inspect
import multiprocessing as mp
import time as tm
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Union, Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import ArrayLike, DTypeLike, NDArray
from matplotlib.ticker import StrMethodFormatter
from tqdm.auto import tqdm

from . import solvers

num_threads = round(np.log2(mp.cpu_count() + 1))


NDArrayFloat = NDArray[np.floating]


def create_random_test_mask(
    data_shape: Sequence[int], test_fraction: float = 0.05, dtype: DTypeLike = np.float32
) -> NDArrayFloat:
    """
    Create a random mask for cross-validation.

    Parameters
    ----------
    data_shape : Sequence[int]
        The shape of the data.
    test_fraction : float, optional
        The fraction of pixels to mask. The default is 0.05.
    data_dtype : DTypeLike, optional
        The data type of the mask. The default is np.float32.

    Returns
    -------
    data_test_mask : NDArrayFloat
        The pixel mask.
    """
    data_test_mask = np.zeros(data_shape, dtype=dtype)
    num_test_pixels = int(np.ceil(data_test_mask.size * test_fraction))
    test_pixels = np.random.permutation(data_test_mask.size)
    test_pixels = np.unravel_index(test_pixels[:num_test_pixels], data_shape)
    data_test_mask[test_pixels] = 1
    return data_test_mask


def get_lambda_range(start: float, end: float, num_per_order: int = 4, aligned_order: bool = True) -> NDArrayFloat:
    """Compute hyper-parameter values within an interval.

    Parameters
    ----------
    start : float
        First hyper-parameter value.
    end : float
        Last hyper-parameter value.
    num_per_order : int, optional
        Number of steps per order of magnitude. The default is 4.
    aligned_order : bool, optional
        Whether to align the 1 of each order of magnitude or to the given start value. The default is True.

    Returns
    -------
    NDArrayFloat
        List of hyper-parameter values.
    """
    step_size = 10 ** (1 / num_per_order)
    if aligned_order:
        order_start = 10 ** np.floor(np.log10(start))
        order_end = 10 ** np.ceil(np.log10(end))
        num_steps = np.ceil(num_per_order * np.log10(order_end / order_start) - 1e-3)
        tmp_steps = order_start * (step_size ** np.arange(num_steps + 1))
        return tmp_steps[np.logical_and(tmp_steps >= start, (tmp_steps * (1 - 1e-3)) <= end)]
    else:
        num_steps = np.ceil(num_per_order * np.log10(end / start) - 1e-3)
        return start * (step_size ** np.arange(num_steps + 1))


def fit_func_min(
    hp_vals: Union[ArrayLike, NDArrayFloat],
    f_vals: NDArrayFloat,
    f_stds: Optional[NDArrayFloat] = None,
    scale: Literal["linear", "log"] = "log",
    verbose: bool = False,
    plot_result: bool = False,
) -> tuple[float, float]:
    """Parabolic fit of objective function costs for the different hyper-parameter values.

    Parameters
    ----------
    hp_vals : Union[ArrayLike, NDArrayFloat]
        Hyper-parameter values.
    f_vals : NDArrayFloat
        Objective function costs of each hyper-parameter value.
    f_stds : NDArrayFloat, optional
        Objective function cost standard deviations of each hyper-parameter value.
        It is only used for plotting purposes. The default is None.
    scale : str, optional
        Scale of the fit. Options are: "log" | "linear". The default is "log".
    verbose : bool, optional
        Whether to produce verbose output, by default False
    plot_result : bool, optional
        Whether to plot the result, by default False

    Returns
    -------
    min_hp_val : float
        Expected hyper-parameter value of the fitted minimum.
    min_f_val : float
        Expected objective function cost of the fitted minimum.
    """
    hp_vals = np.array(hp_vals, ndmin=1)

    if len(hp_vals) < 3 or len(f_vals) < 3 or len(hp_vals) != len(f_vals):
        raise ValueError(
            "Lengths of the lambdas and function values should be identical and >= 3."
            f"Given: lams={len(hp_vals)}, vals={len(f_vals)}"
        )

    if scale.lower() == "log":
        to_fit = np.log10
        from_fit = lambda x: 10**x
    elif scale.lower() == "linear":
        to_fit = lambda x: x
        from_fit = to_fit
    else:
        raise ValueError(f"Parameter 'scale' should be either 'log' or 'linear', given '{scale}' instead")

    min_pos = np.argmin(f_vals)
    if min_pos == 0:
        warn("Minimum value at the beginning of the lambda range.")
        hp_inds_fit = list(np.arange(3))
    elif min_pos == (len(f_vals) - 1):
        warn("Minimum value at the end of the lambda range.")
        hp_inds_fit = list(np.mod(np.arange(-3, 0), hp_vals.size))
    else:
        hp_inds_fit = list(np.arange(min_pos - 1, min_pos + 2))
    lams_reg_fit = to_fit(hp_vals[hp_inds_fit])
    f_vals_fit = f_vals[hp_inds_fit]

    counter = tm.perf_counter()
    if verbose:
        print(
            f"Fitting minimum within the interval [{hp_vals[hp_inds_fit[0]]:.3e}, {hp_vals[hp_inds_fit[-1]]:.3e}]"
            f" (indices: [{hp_inds_fit[0]}, {hp_inds_fit[-1]}]): ",
            end="",
            flush=True,
        )

    # using Polynomial.fit, because it is supposed to be more numerically
    # stable than previous solutions (according to numpy).
    poly = Polynomial.fit(lams_reg_fit, f_vals_fit, deg=2)
    coeffs = poly.convert().coef
    if coeffs[2] <= 0:
        warn("Fitted curve is concave. Returning minimum measured point.")
        return hp_vals[min_pos], f_vals[min_pos]

    # For a 1D parabola `f(x) = c + bx + ax^2`, the vertex position is:
    # x_v = -b / 2a.
    vertex_pos = -coeffs[1] / (2 * coeffs[2])
    vertex_val = coeffs[0] + vertex_pos * coeffs[1] / 2

    min_hp_val, min_f_val = from_fit(vertex_pos), vertex_val
    if min_hp_val < hp_vals[0] or min_hp_val > hp_vals[-1]:
        warn(
            f"Fitted lambda {min_hp_val:.3e} is outside the bounds of input lambdas [{hp_vals[0]:.3e}, {hp_vals[-1]:.3e}]."
            " Returning minimum measured point."
        )
        res_hp_val, res_f_val = hp_vals[min_pos], f_vals[min_pos]
    else:
        res_hp_val, res_f_val = min_hp_val, min_f_val

    if verbose:
        print(f"Found at {min_hp_val:.3e}, in {tm.perf_counter() - counter:g} seconds.\n")

    if plot_result:
        fig, axs = plt.subplots()
        axs.set_xscale(scale, nonpositive="clip")
        if f_stds is None:
            axs.plot(hp_vals, f_vals)
        else:
            axs.errorbar(hp_vals, f_vals, yerr=f_stds, ecolor=(0.5, 0.5, 0.5), elinewidth=1, capsize=2)
        x = np.linspace(lams_reg_fit[0], lams_reg_fit[2])
        y = coeffs[0] + x * (coeffs[1] + x * coeffs[2])
        axs.plot(from_fit(x), y)
        axs.scatter(min_hp_val, min_f_val)
        axs.grid()
        for tl in axs.get_xticklabels():
            tl.set_fontsize(13)
        for tl in axs.get_yticklabels():
            tl.set_fontsize(13)
        axs.set_xlabel(r"$\lambda$ values", fontsize=16)
        axs.set_ylabel("Cross-validation loss values", fontsize=16)
        axs.yaxis.set_major_formatter(StrMethodFormatter("{x:.2e}"))
        fig.tight_layout()
        plt.show(block=False)

    return res_hp_val, res_f_val


class BaseParameterTuning(ABC):
    """Base class for parameter tuning classes."""

    _solver_spawning_functionls: Optional[Callable]
    _solver_calling_function: Optional[Callable[[Any], tuple[NDArrayFloat, solvers.SolutionInfo]]]

    def __init__(
        self, dtype: DTypeLike = np.float32, parallel_eval: bool = True, verbose: bool = False, plot_result: bool = False
    ) -> None:
        """Initialize a base helper class.

        Parameters
        ----------
        dtype : DTypeLike, optional
            Type of the data, by default np.float32
        parallel_eval : bool, optional
            Whether to evaluate results in parallel, by default True
        verbose : bool, optional
            Whether to produce verbose output, by default False
        plot_result : bool, optional
            Whether to plot the results, by default False
        """
        self.dtype = dtype

        self.parallel_eval = parallel_eval
        self.verbose = verbose
        self.plot_result = plot_result

        self._solver_spawning_function = None
        self._solver_calling_function = None

    @property
    def solver_spawning_function(self) -> Callable:
        """Return the locally stored solver spawning function."""
        if self._solver_spawning_function is None:
            raise ValueError("Solver spawning function not initialized!")
        return self._solver_spawning_function

    @property
    def solver_calling_function(self) -> Callable[[Any, ...], tuple[NDArrayFloat, solvers.SolutionInfo]]:
        """Return the locally stored solver calling function."""
        if self._solver_calling_function is None:
            raise ValueError("Solver spawning function not initialized!")
        return self._solver_calling_function

    @solver_spawning_function.setter
    def solver_spawning_function(self, solver_spawn: Callable) -> None:
        if not isinstance(solver_spawn, Callable):
            raise ValueError("Expected a solver spawning function (callable)")
        if len(inspect.signature(solver_spawn).parameters) != 1:
            raise ValueError(
                "Expected a solver spawning function (callable), whose only parameter is the regularization lambda"
            )
        self._solver_spawning_function = solver_spawn

    @solver_calling_function.setter
    def solver_calling_function(self, solver_call: Callable[[Any, ...], tuple[NDArrayFloat, solvers.SolutionInfo]]) -> None:
        if not isinstance(solver_call, Callable):
            raise ValueError("Expected a solver calling function (callable)")
        if not len(inspect.signature(solver_call).parameters) >= 1:
            raise ValueError("Expected a solver calling function (callable), with at least one parameter (solver)")
        self._solver_calling_function = solver_call

    @staticmethod
    def get_lambda_range(start: float, end: float, num_per_order: int = 4) -> NDArrayFloat:
        """Compute regularization weights within an interval.

        Parameters
        ----------
        start : float
            First regularization weight.
        end : float
            Last regularization weight.
        num_per_order : int, optional
            Number of steps per order of magnitude. The default is 4.

        Returns
        -------
        NDArrayFloat
            List of regularization weights.
        """
        return get_lambda_range(start=start, end=end, num_per_order=num_per_order)

    def compute_reconstruction_and_loss(self, hp_val: float, *args: Any, **kwds: Any) -> tuple[float, NDArrayFloat]:
        """Compute objective function cost for the given hyper-parameter value.

        Parameters
        ----------
        hp_val : float
            hyper-parameter value.
        *args : Any
            Optional positional arguments for the reconstruction.
        **kwds : Any
            Optional keyword arguments for the reconstruction.

        Returns
        -------
        cost : float
            Cost of the given regularization weight.
        rec : NDArray
            Reconstruction at the given weight.
        """
        solver = self.solver_spawning_function(hp_val)
        rec, rec_info = self.solver_calling_function(solver, *args, **kwds)

        # Output will be: test objective function cost, and reconstruction
        return float(rec_info.residuals_cv_rel[-1]), rec

    def compute_reconstruction_error(
        self, hp_vals: Union[ArrayLike, NDArrayFloat], gnd_truth: NDArrayFloat
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Compute the reconstruction errors for each hyper-parameter values against the ground truth.

        Parameters
        ----------
        hp_vals : Union[ArrayLike, NDArrayFloat]
            List of hyper-parameter values.
        gnd_truth : NDArrayFloat
            Expected reconstruction.

        Returns
        -------
        err_l1 : NDArrayFloat
            l1-norm errors for each reconstruction.
        err_l2 : NDArrayFloat
            l2-norm errors for each reconstruction.
        """
        hp_vals = np.array(hp_vals, ndmin=1)

        if self.verbose:
            print("Computing reconstruction error:")
            print(f"- Hyper-parameter values range: [{hp_vals[0]:.3e}, {hp_vals[-1]:.3e}] in {len(hp_vals)} steps")
            print(f"Parallel evaluation: {self.parallel_eval} (n. threads: {num_threads})")

        if self.parallel_eval:
            with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_lambda = {
                    executor.submit(self.compute_reconstruction_and_loss, l): (ii, l) for ii, l in enumerate(hp_vals)
                }

                recs = [np.array([])] * len(hp_vals)
                try:
                    for future in tqdm(
                        cf.as_completed(future_to_lambda),
                        desc="Hyper-parameter values",
                        disable=not self.verbose,
                        total=len(hp_vals),
                    ):
                        hp_ind, hp_val = future_to_lambda[future]
                        try:
                            recs[hp_ind] = future.result()[1]
                        except ValueError as exc:
                            print(f"Hyper-parameter value {hp_val} (#{hp_ind}) generated an exception: {exc}")
                            raise
                except:
                    print("Shutting down..", end="", flush=True)
                    executor.shutdown(cancel_futures=True)
                    print("\b\b: Done.")
                    raise
        else:
            recs = [
                self.compute_reconstruction_and_loss(l)[1]
                for l in tqdm(hp_vals, desc="Hyper-parameter values", disable=not self.verbose)
            ]

        err_l1 = np.zeros((len(hp_vals),), dtype=self.dtype)
        err_l2 = np.zeros((len(hp_vals),), dtype=self.dtype)

        for ii_l, rec in enumerate(recs):
            residual = np.abs(gnd_truth - rec)
            err_l1[ii_l] = np.linalg.norm(residual.ravel(), ord=1)
            err_l2[ii_l] = np.linalg.norm(residual.ravel(), ord=2) ** 2

        if self.plot_result:
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].set_xscale("log", nonpositive="clip")  # type: ignore
            axs[0].plot(hp_vals, err_l1, label="Error - l1-norm")  # type: ignore
            axs[0].legend()  # type: ignore
            axs[1].set_xscale("log", nonpositive="clip")  # type: ignore
            axs[1].plot(hp_vals, err_l2, label="Error - l2-norm ^ 2")  # type: ignore
            axs[1].legend()  # type: ignore
            fig.tight_layout()
            plt.show(block=False)

        return err_l1, err_l2

    @abstractmethod
    def compute_loss_values(self, hp_vals: Union[ArrayLike, NDArrayFloat]) -> NDArrayFloat:
        """Compute the objective function costs for a list of hyper-parameter values.

        Parameters
        ----------
        hp_vals : Union[ArrayLike, NDArrayFloat]
            List of hyper-parameter values.

        Returns
        -------
        NDArrayFloat
            Objective function cost for each hyper-parameter value.
        """


class LCurve(BaseParameterTuning):
    """L-curve regularization parameter estimation helper."""

    def __init__(
        self,
        loss_function: Callable,
        data_dtype: DTypeLike = np.float32,
        parallel_eval: bool = True,
        verbose: bool = False,
        plot_result: bool = False,
    ):
        """Create an L-curve regularization parameter estimation helper.

        Parameters
        ----------
        loss_function : Callable
            The loss function for the computation of the L-curve values.
        data_dtype : DTypeLike, optional
            Type of the input data. The default is np.float32.
        parallel_eval : bool, optional
            Compute loss and error values in parallel. The default is False.
        verbose : bool, optional
            Print verbose output. The default is False.
        plot_result : bool, optional
            Plot results. The default is False.

        Raises
        ------
        ValueError
            In case 'loss_function' is not callable or does not expose at least one argument.
        """
        super().__init__(dtype=data_dtype, parallel_eval=parallel_eval, verbose=verbose, plot_result=plot_result)

        if not isinstance(loss_function, Callable):
            raise ValueError(
                "Expected a callable with one argument for the argument 'loss_function',"
                " whose parameters are: the solver and the data test mask"
            )
        if len(inspect.signature(loss_function).parameters) != 1:
            raise ValueError("The callable 'loss_function', should have one parameter")
        self.loss_function = loss_function

    def compute_loss_values(self, hp_vals: Union[ArrayLike, NDArrayFloat]) -> NDArrayFloat:
        """Compute objective function values for all hyper-parameter values.

        Parameters
        ----------
        hp_vals : Union[ArrayLike, NDArrayFloat]
            Hyper-parameter values to use for computing the different objective function values.

        Returns
        -------
        f_vals : NDArrayFloat
            Objective function cost for each hyper-parameter value.
        """
        hp_vals = np.array(hp_vals, ndmin=1)

        counter = tm.perf_counter()
        if self.verbose:
            print("Computing L-curve loss values:")
            print(f"- Hyper-parameter values range: [{hp_vals[0]:.3e}, {hp_vals[-1]:.3e}] in {len(hp_vals)} steps")
            print(f"Parallel evaluation: {self.parallel_eval} (n. threads: {num_threads})")

        if self.parallel_eval:
            with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_lambda = {
                    executor.submit(self.compute_reconstruction_and_loss, l): (ii, l) for ii, l in enumerate(hp_vals)
                }

                recs = [np.array([])] * len(hp_vals)
                try:
                    for future in tqdm(
                        cf.as_completed(future_to_lambda),
                        desc="Hyper-parameter values",
                        disable=not self.verbose,
                        total=len(hp_vals),
                    ):
                        hp_ind, hp_val = future_to_lambda[future]
                        try:
                            recs[hp_ind] = future.result()[1]
                        except ValueError as exc:
                            print(f"Hyper-parameter value {hp_val} (#{hp_ind}) generated an exception: {exc}")
                            raise
                except:
                    print("Shutting down..", end="", flush=True)
                    executor.shutdown(cancel_futures=True)
                    print("\b\b: Done.")
                    raise
        else:
            recs = [
                self.compute_reconstruction_and_loss(l)[1]
                for l in tqdm(hp_vals, desc="Hyper-parameter values", disable=not self.verbose)
            ]

        f_vals = np.array([self.loss_function(rec) for rec in recs], dtype=self.dtype)

        if self.verbose:
            print(f"Done in {tm.perf_counter() - counter} seconds.\n")

        if self.plot_result:
            fig, axs = plt.subplots()
            axs.set_title("L-Curve loss values")
            axs.set_xscale("log", nonpositive="clip")
            axs.set_yscale("log", nonpositive="clip")
            axs.plot(hp_vals, f_vals)
            axs.grid()
            fig.tight_layout()
            plt.show(block=False)

        return f_vals


class CrossValidation(BaseParameterTuning):
    """Cross-validation hyper-parameter estimation class."""

    def __init__(
        self,
        data_shape: Sequence[int],
        dtype: DTypeLike = np.float32,
        cv_fraction: float = 0.1,
        num_averages: int = 7,
        parallel_eval: bool = True,
        verbose: bool = False,
        plot_result: bool = False,
    ):
        """Create a cross-validation hyper-parameter estimation helper.

        Parameters
        ----------
        data_shape : Sequence[int]
            Shape of the projection data.
        data_dtype : DTypeLike, optional
            Type of the input data. The default is np.float32.
        cv_fraction : float, optional
            Fraction of detector points to use for the leave-out set. The default is 0.1.
        num_averages : int, optional
            Number of averages random leave-out sets to use. The default is 7.
        parallel_eval : bool, optional
            Compute loss and error values in parallel. The default is False.
        verbose : bool, optional
            Print verbose output. The default is False.
        plot_result : bool, optional
            Plot results. The default is False.
        """
        super().__init__(dtype=dtype, parallel_eval=parallel_eval, verbose=verbose, plot_result=plot_result)
        self.data_shape = data_shape
        self.cv_fraction = cv_fraction
        self.num_averages = num_averages

        self.data_test_masks = [self._create_random_test_mask() for _ in range(self.num_averages)]

    def _create_random_test_mask(self) -> NDArrayFloat:
        return create_random_test_mask(self.data_shape, self.cv_fraction, self.dtype)

    def compute_loss_values(self, hp_vals: Union[ArrayLike, NDArrayFloat]) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Compute objective function values for all requested hyper-parameter values..

        Parameters
        ----------
        params : Union[ArrayLike, NDArrayFloat]
            Hyper-parameter values (e.g. regularization weight) to evaluate.

        Returns
        -------
        f_avgs : NDArrayFloat
            Average objective function costs for each regularization weight.
        f_stds : NDArrayFloat
            Standard deviation of objective function costs for each regularization weight.
        f_vals : NDArrayFloat
            Objective function costs for each regularization weight.
        """
        hp_vals = np.array(hp_vals, ndmin=1)

        counter = tm.perf_counter()
        if self.verbose:
            print("Computing cross-validation loss values:")
            print(f"- Hyper-parameter range: [{hp_vals[0]:.3e}, {hp_vals[-1]:.3e}] in {len(hp_vals)} steps")
            print(f"- Number of averages: {self.num_averages}")
            print(f"- Leave-out pixel fraction: {self.cv_fraction:%}")
            print(f"Parallel evaluation: {self.parallel_eval} (n. threads: {num_threads})")

        f_vals = np.empty((len(hp_vals), self.num_averages), dtype=self.dtype)
        for ii_avg in range(self.num_averages):
            if self.verbose:
                print(f"\nRound: {ii_avg + 1}/{self.num_averages}")

            if self.parallel_eval:
                with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    future_to_lambda = {
                        executor.submit(self.compute_reconstruction_and_loss, l, self.data_test_masks[ii_avg]): (ii, l)
                        for ii, l in enumerate(hp_vals)
                    }

                    f_vals_ii = [0.0] * len(hp_vals)
                    try:
                        for future in tqdm(
                            cf.as_completed(future_to_lambda),
                            desc="Hyper-parameter values",
                            disable=not self.verbose,
                            total=len(hp_vals),
                        ):
                            hp_ind, hp_val = future_to_lambda[future]
                            try:
                                f_vals_ii[hp_ind] = future.result()[0]
                            except ValueError as exc:
                                print(f"Hyper-parameter value {hp_val} (#{hp_ind}) generated an exception: {exc}")
                                raise
                    except:
                        print("Shutting down..", end="", flush=True)
                        executor.shutdown(cancel_futures=True)
                        print("\b\b: Done.")
                        raise
            else:
                f_vals_ii = [
                    self.compute_reconstruction_and_loss(l, self.data_test_masks[ii_avg])[0]
                    for l in tqdm(hp_vals, desc="Hyper-parameter values", disable=not self.verbose)
                ]

            f_vals[:, ii_avg] = np.array(f_vals_ii)

        f_avgs = np.mean(f_vals, axis=1)
        f_stds = np.std(f_vals, axis=1)

        if self.verbose:
            print(f"Done in {tm.perf_counter() - counter:g} seconds.\n")

        if self.plot_result:
            fig, axs = plt.subplots()
            axs.set_title(f"Cross-validation loss values (avgs: {self.num_averages})")
            axs.set_xscale("log", nonpositive="clip")
            axs.errorbar(hp_vals, f_avgs, yerr=f_stds, ecolor=(0.5, 0.5, 0.5), elinewidth=1, capsize=2)
            for f in f_vals.T:
                axs.plot(hp_vals, f, linewidth=1, linestyle="--")
            axs.grid()
            fig.tight_layout()
            plt.show(block=False)

        return f_avgs, f_stds, f_vals

    def fit_loss_min(
        self,
        hp_vals: Union[ArrayLike, NDArrayFloat],
        f_vals: NDArrayFloat,
        f_stds: Optional[NDArrayFloat] = None,
        scale: Literal["linear", "log"] = "log",
    ) -> tuple[float, float]:
        """Parabolic fit of objective function costs for the different hyper-parameter values.

        Parameters
        ----------
        hp_vals : Union[ArrayLike, NDArrayFloat]
            Hyper-parameter values.
        f_vals : NDArrayFloat
            Objective function costs of each hyper-parameter value.
        f_stds : NDArrayFloat, optional
            Objective function cost standard deviations of each hyper-parameter value.
            It is only used for plotting purposes. The default is None.
        scale : str, optional
            Scale of the fit. Options are: "log" | "linear". The default is "log".

        Returns
        -------
        min_hp_val : float
            Expected hyper-parameter value of the fitted minimum.
        min_f_val : float
            Expected objective function cost of the fitted minimum.
        """
        return fit_func_min(
            hp_vals=hp_vals, f_vals=f_vals, f_stds=f_stds, scale=scale, verbose=self.verbose, plot_result=self.plot_result
        )
