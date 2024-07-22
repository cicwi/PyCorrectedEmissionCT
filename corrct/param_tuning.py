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

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import ArrayLike, DTypeLike, NDArray
from matplotlib.ticker import StrMethodFormatter

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
    step_size = 10 ** (1 / num_per_order)
    num_steps = np.ceil(num_per_order * np.log10(end / start) - 1e-3)
    return start * (step_size ** np.arange(num_steps + 1))


class BaseRegularizationEstimation(ABC):
    """Base class for regularization parameter estimation class."""

    _solver_calling_function: Optional[Callable[[Any], tuple[NDArrayFloat, solvers.SolutionInfo]]]

    def __init__(
        self, dtype: DTypeLike = np.float32, parallel_eval: bool = True, verbose: bool = False, plot_result: bool = False
    ):
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
    def solver_calling_function(self) -> Callable[[Any], tuple[NDArrayFloat, solvers.SolutionInfo]]:
        """Return the locally stored solver calling function."""
        if self._solver_calling_function is None:
            raise ValueError("Solver spawning function not initialized!")
        return self._solver_calling_function

    @solver_spawning_function.setter
    def solver_spawning_function(self, solver_spawn: Callable):
        if not isinstance(solver_spawn, Callable):
            raise ValueError("Expected a solver spawning function (callable)")
        if len(inspect.signature(solver_spawn).parameters) != 1:
            raise ValueError(
                "Expected a solver spawning function (callable), whose only parameter is the regularization lambda"
            )
        self._solver_spawning_function = solver_spawn

    @solver_calling_function.setter
    def solver_calling_function(self, solver_call: Callable):
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

    def compute_reconstruction_and_loss(self, lam_reg: float, *args: Any, **kwds: Any) -> tuple[np.floating, NDArrayFloat]:
        """Compute objective function cost for the given regularization weight.

        Parameters
        ----------
        lam_reg : float
            Regularization weight.
        *args : Any
            Optional positional arguments for the reconstruction.
        **kwds : Any
            Optional keyword arguments for the reconstruction.

        Returns
        -------
        cost
            Cost of the given regularization weight.
        rec : ArrayLike
            Reconstruction at the given weight.
        """
        solver = self.solver_spawning_function(lam_reg)
        rec, rec_info = self.solver_calling_function(solver, *args, **kwds)

        # Output will be: test objective function cost, and reconstruction
        return rec_info.residuals_cv_rel[-1], rec

    def compute_reconstruction_error(
        self, lams_reg: Union[ArrayLike, NDArrayFloat], gnd_truth: NDArrayFloat
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Compute the reconstructions for each regularization weight error against the ground truth.

        Parameters
        ----------
        lams_reg : Union[ArrayLike, NDArrayFloat]
            List of regularization weights.
        gnd_truth : NDArrayFloat
            Expected reconstruction.

        Returns
        -------
        err_l1 : NDArrayFloat
            l1-norm errors for each reconstruction.
        err_l2 : NDArrayFloat
            l2-norm errors for each reconstruction.
        """
        lams_reg = np.array(lams_reg, ndmin=1)

        counter = tm.perf_counter()
        if self.verbose:
            print("Computing reconstruction error:")
            print(f"- Regularization weights range: [{lams_reg[0]}, {lams_reg[-1]}] in {len(lams_reg)} steps")
            print(f"Parallel evaluation: {self.parallel_eval} (n. threads: {num_threads})")

        if self.parallel_eval:
            with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                fr = [executor.submit(self.compute_reconstruction_and_loss, l) for l in lams_reg]

                recs = [r.result()[1] for r in fr]
        else:
            recs = [self.compute_reconstruction_and_loss(l)[1] for l in lams_reg]

        err_l1 = np.zeros((len(lams_reg),), dtype=self.dtype)
        err_l2 = np.zeros((len(lams_reg),), dtype=self.dtype)

        for ii_l, rec in enumerate(recs):
            residual = np.abs(gnd_truth - rec)
            err_l1[ii_l] = np.linalg.norm(residual.ravel(), ord=1)
            err_l2[ii_l] = np.linalg.norm(residual.ravel(), ord=2) ** 2

        if self.verbose:
            print(f"Done in {tm.perf_counter() - counter} seconds.\n")

        if self.plot_result:
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].set_xscale("log", nonpositive="clip")  # type: ignore
            axs[0].plot(lams_reg, err_l1, label="Error - l1-norm")  # type: ignore
            axs[0].legend()  # type: ignore
            axs[1].set_xscale("log", nonpositive="clip")  # type: ignore
            axs[1].plot(lams_reg, err_l2, label="Error - l2-norm ^ 2")  # type: ignore
            axs[1].legend()  # type: ignore
            fig.tight_layout()
            plt.show(block=False)

        return err_l1, err_l2

    @abstractmethod
    def compute_loss_values(self, lams_reg: Union[ArrayLike, NDArrayFloat]) -> NDArrayFloat:
        """Compute the objective function costs for a list of regularization weights.

        Parameters
        ----------
        lams_reg : Union[ArrayLike, NDArrayFloat]
            List of regularization weights.

        Returns
        -------
        NDArrayFloat
            Objective function cost for each regularization weight.
        """


class LCurve(BaseRegularizationEstimation):
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

    def compute_loss_values(self, lams_reg: Union[ArrayLike, NDArrayFloat]) -> NDArrayFloat:
        """Compute objective function values for all regularization weights.

        Parameters
        ----------
        lams_reg : Union[ArrayLike, NDArrayFloat]
            Regularization weights to use for computing the different objective function values.

        Returns
        -------
        f_vals : NDArrayFloat
            Objective function cost for each regularization weight.
        """
        lams_reg = np.array(lams_reg, ndmin=1)

        counter = tm.perf_counter()
        if self.verbose:
            print("Computing L-curve loss values:")
            print(f"- Regularization weights range: [{lams_reg[0]}, {lams_reg[-1]}] in {len(lams_reg)} steps")
            print(f"Parallel evaluation: {self.parallel_eval} (n. threads: {num_threads})")

        if self.parallel_eval:
            with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_lambda = {
                    executor.submit(self.compute_reconstruction_and_loss, l): (ii, l) for ii, l in enumerate(lams_reg)
                }

                recs = []
                for future in cf.as_completed(future_to_lambda):
                    lam_ind, lam = future_to_lambda[future]
                    try:
                        recs.append(future.result()[1])
                    except ValueError as exc:
                        print(f"Lambda {lam} (#{lam_ind}) generated an exception: {exc}")
                        raise
        else:
            recs = [self.compute_reconstruction_and_loss(l)[1] for l in lams_reg]

        f_vals = np.array([self.loss_function(rec) for rec in recs], dtype=self.dtype)

        if self.verbose:
            print(f"Done in {tm.perf_counter() - counter} seconds.\n")

        if self.plot_result:
            fig, axs = plt.subplots()
            axs.set_title("L-Curve loss values")
            axs.set_xscale("log", nonpositive="clip")
            axs.set_yscale("log", nonpositive="clip")
            axs.plot(lams_reg, f_vals)
            axs.grid()
            fig.tight_layout()
            plt.show(block=False)

        return f_vals


class CrossValidation(BaseRegularizationEstimation):
    """Cross-validation regularization parameter estimation helper."""

    def __init__(
        self,
        data_shape: Sequence[int],
        dtype: DTypeLike = np.float32,
        test_fraction: float = 0.1,
        num_averages: int = 7,
        parallel_eval: bool = True,
        verbose: bool = False,
        plot_result: bool = False,
    ):
        """Create a cross-validation regularization parameter estimation helper.

        Parameters
        ----------
        data_shape : Sequence[int]
            Shape of the projection data.
        data_dtype : DTypeLike, optional
            Type of the input data. The default is np.float32.
        test_fraction : float, optional
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
        self.test_fraction = test_fraction
        self.num_averages = num_averages

        self.data_test_masks = [self._create_random_test_mask() for ii in range(self.num_averages)]

    def _create_random_test_mask(self) -> NDArrayFloat:
        return create_random_test_mask(self.data_shape, self.test_fraction, self.dtype)

    def compute_loss_values(self, lams_reg: Union[ArrayLike, NDArrayFloat]) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Compute objective function values for all regularization weights.

        Parameters
        ----------
        lams_reg : Union[ArrayLike, NDArrayFloat]
            Regularization weights to use for computing the different objective function values.

        Returns
        -------
        f_avgs : NDArrayFloat
            Average objective function costs for each regularization weight.
        f_stds : NDArrayFloat
            Standard deviation of objective function costs for each regularization weight.
        f_vals : NDArrayFloat
            Objective function costs for each regularization weight.
        """
        lams_reg = np.array(lams_reg, ndmin=1)

        counter = tm.perf_counter()
        if self.verbose:
            print("Computing cross-validation loss values:")
            print(f"- Regularization weights range: [{lams_reg[0]}, {lams_reg[-1]}] in {len(lams_reg)} steps")
            print(f"- Number of averages: {self.num_averages}")
            print(f"- Leave-out pixel fraction: {self.test_fraction:%}")
            print(f"Parallel evaluation: {self.parallel_eval} (n. threads: {num_threads})")

        f_vals = np.empty((len(lams_reg), self.num_averages), dtype=self.dtype)
        for ii_avg in range(self.num_averages):
            c_round = tm.perf_counter()
            if self.verbose:
                print(f"\nRound: {ii_avg + 1}/{self.num_averages}")

            if self.parallel_eval:
                with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    fr = [
                        executor.submit(self.compute_reconstruction_and_loss, l, self.data_test_masks[ii_avg])
                        for l in lams_reg
                    ]

                    f_vals_ii = [r.result()[0] for r in fr]
            else:
                f_vals_ii = [self.compute_reconstruction_and_loss(l, self.data_test_masks[ii_avg])[0] for l in lams_reg]

            f_vals[:, ii_avg] = f_vals_ii

            if self.verbose:
                print(f" - Done in {tm.perf_counter() - c_round:g} seconds.\n")

        f_avgs = np.mean(f_vals, axis=1)
        f_stds = np.std(f_vals, axis=1)

        if self.verbose:
            print(f"Done in {tm.perf_counter() - counter:g} seconds.\n")

        if self.plot_result:
            fig, axs = plt.subplots()
            axs.set_title(f"Cross-validation loss values (avgs: {self.num_averages})")
            axs.set_xscale("log", nonpositive="clip")
            axs.errorbar(lams_reg, f_avgs, yerr=f_stds, ecolor=(0.5, 0.5, 0.5), elinewidth=1, capsize=2)
            axs.grid()
            fig.tight_layout()
            plt.show(block=False)

        return f_avgs, f_stds, f_vals

    def fit_loss_min(
        self,
        lams_reg: Union[ArrayLike, NDArrayFloat],
        f_vals: NDArrayFloat,
        f_stds: Optional[NDArrayFloat] = None,
        scale: Literal["linear", "log"] = "log",
    ) -> tuple[float, float]:
        """Parabolic fit of objective function costs for the different regularization weights.

        Parameters
        ----------
        lams_reg : Union[ArrayLike, NDArrayFloat]
            Regularization weights.
        f_vals : NDArrayFloat
            Objective function costs of each regularization weight.
        f_stds : NDArrayFloat, optional
            Objective function cost standard deviations of each regularization weight.
            It is only used for plotting purposes. The default is None.
        scale : str, optional
            Scale of the fit. Options are: "log" | "linear". The default is "log".

        Returns
        -------
        min_lam : float
            Expected regularization weight of the fitted minimum.
        min_val : float
            Expected objective function cost of the fitted minimum.
        """
        lams_reg = np.array(lams_reg, ndmin=1)

        if len(lams_reg) < 3 or len(f_vals) < 3 or len(lams_reg) != len(f_vals):
            raise ValueError(
                "Lengths of the lambdas and function values should be identical and >= 3."
                f"Given: lams={len(lams_reg)}, vals={len(f_vals)}"
            )

        if scale.lower() == "log":
            to_fit = lambda x: np.log10(x)
            from_fit = lambda x: 10**x
        elif scale.lower() == "linear":
            to_fit = lambda x: x
            from_fit = to_fit
        else:
            raise ValueError(f"Parameter 'scale' should be either 'log' or 'linear', given '{scale}' instead")

        min_pos = np.argmin(f_vals)
        if min_pos == 0:
            print("WARNING: minimum value at the beginning of the lambda range.")
            lams_reg_fit = to_fit(lams_reg[:3])
            f_vals_fit = f_vals[:3]
        elif min_pos == (len(f_vals) - 1):
            print("WARNING: minimum value at the end of the lambda range.")
            lams_reg_fit = to_fit(lams_reg[-3:])
            f_vals_fit = f_vals[-3:]
        else:
            lams_reg_fit = to_fit(lams_reg[min_pos - 1 : min_pos + 2])
            f_vals_fit = f_vals[min_pos - 1 : min_pos + 2]

        counter = tm.perf_counter()
        if self.verbose:
            print(
                f"Fitting minimum within the parameter interval [{from_fit(lams_reg_fit[0])}, {from_fit(lams_reg_fit[-1])}]: ",
                end="",
                flush=True,
            )

        # using Polynomial.fit, because it is supposed to be more numerically
        # stable than previous solutions (according to numpy).
        poly = Polynomial.fit(lams_reg_fit, f_vals_fit, deg=2)
        coeffs = poly.convert().coef
        if coeffs[2] <= 0:
            print("WARNING: fitted curve is concave. Returning minimum measured point.")
            return lams_reg[min_pos], f_vals[min_pos]

        # For a 1D parabola `f(x) = c + bx + ax^2`, the vertex position is:
        # x_v = -b / 2a.
        vertex_pos = -coeffs[1] / (2 * coeffs[2])
        vertex_val = coeffs[0] + vertex_pos * coeffs[1] / 2

        min_lam, min_val = from_fit(vertex_pos), vertex_val
        if min_lam < lams_reg[0] or min_lam > lams_reg[-1]:
            print(
                f"WARNING: fitted lambda {min_lam} is outside the bounds of input lambdas [{lams_reg[0]}, {lams_reg[-1]}]."
                + " Returning minimum measured point."
            )
            res_lam, res_val = lams_reg[min_pos], f_vals[min_pos]
        else:
            res_lam, res_val = min_lam, min_val

        if self.verbose:
            print(f"Found at {min_lam:g}, in {tm.perf_counter() - counter:g} seconds.\n")

        if self.plot_result:
            fig, axs = plt.subplots()
            axs.set_xscale(scale, nonpositive="clip")
            if f_stds is None:
                axs.plot(lams_reg, f_vals)
            else:
                axs.errorbar(lams_reg, f_vals, yerr=f_stds, ecolor=(0.5, 0.5, 0.5), elinewidth=1, capsize=2)
            x = np.linspace(lams_reg_fit[0], lams_reg_fit[2])
            y = coeffs[0] + x * (coeffs[1] + x * coeffs[2])
            axs.plot(from_fit(x), y)
            axs.scatter(min_lam, min_val)
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

        return res_lam, res_val
