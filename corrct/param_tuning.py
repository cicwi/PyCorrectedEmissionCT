#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aided regularization parameter estimation.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
from numpy.polynomial.polynomial import Polynomial

import matplotlib.pyplot as plt

from typing import Callable, Sequence, Optional, Tuple, Any, Union
from numpy.typing import ArrayLike, DTypeLike, NDArray
import inspect

from abc import ABC, abstractmethod

import time as tm

import concurrent.futures as cf
import multiprocessing as mp

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


class BaseRegularizationEstimation(ABC):
    """Base class for regularization parameter estimation class."""

    _solver_calling_function: Optional[Callable[[Any], Tuple[NDArrayFloat, solvers.SolutionInfo]]]

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
    def solver_calling_function(self) -> Callable[[Any], Tuple[NDArrayFloat, solvers.SolutionInfo]]:
        """Return the locally stored solver calling function."""
        if self._solver_calling_function is None:
            raise ValueError("Solver spawning function not initialized!")
        return self._solver_calling_function

    @solver_spawning_function.setter
    def solver_spawning_function(self, s: Callable):
        if not isinstance(s, Callable):
            raise ValueError("Expected a solver spawning function (callable)")
        if len(inspect.signature(s).parameters) != 1:
            raise ValueError(
                "Expected a solver spawning function (callable), whose only parameter is the regularization lambda"
            )
        self._solver_spawning_function = s

    @solver_calling_function.setter
    def solver_calling_function(self, c: Callable):
        if not isinstance(c, Callable):
            raise ValueError("Expected a solver calling function (callable)")
        if not len(inspect.signature(c).parameters) >= 1:
            raise ValueError("Expected a solver calling function (callable), with at least one parameter (solver)")
        self._solver_calling_function = c

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
        step_size = 10 ** (1 / num_per_order)
        num_steps = np.ceil(num_per_order * np.log10(end / start) - 1e-3)
        return start * (step_size ** np.arange(num_steps + 1))

    def compute_reconstruction_and_loss(self, lam_reg: float, *args: Any, **kwds: Any) -> Tuple[np.floating, NDArrayFloat]:
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
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
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

        c = tm.perf_counter()
        if self.verbose:
            print("Computing reconstruction error:")
            print("- Regularization weights range: [%g, %g] in %d steps" % (lams_reg[0], lams_reg[-1], len(lams_reg)))
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
            print("Done in %g seconds.\n" % (tm.perf_counter() - c))

        if self.plot_result:
            f, axs = plt.subplots(2, 1, sharex=True)
            axs[0].set_xscale("log", nonpositive="clip")  # type: ignore
            axs[0].plot(lams_reg, err_l1, label="Error - l1-norm")  # type: ignore
            axs[0].legend()  # type: ignore
            axs[1].set_xscale("log", nonpositive="clip")  # type: ignore
            axs[1].plot(lams_reg, err_l2, label="Error - l2-norm ^ 2")  # type: ignore
            axs[1].legend()  # type: ignore
            f.tight_layout()
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

        c = tm.perf_counter()
        if self.verbose:
            print("Computing L-curve loss values:")
            print("- Regularization weights range: [%g, %g] in %d steps" % (lams_reg[0], lams_reg[-1], len(lams_reg)))
            print(f"Parallel evaluation: {self.parallel_eval} (n. threads: {num_threads})")

        if self.parallel_eval:
            with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                fr = [executor.submit(self.compute_reconstruction_and_loss, l) for l in lams_reg]

                recs = [r.result()[1] for r in fr]
        else:
            recs = [self.compute_reconstruction_and_loss(l)[1] for l in lams_reg]

        f_vals = np.array([self.loss_function(rec) for rec in recs], dtype=self.dtype)

        if self.verbose:
            print("Done in %g seconds.\n" % (tm.perf_counter() - c))

        if self.plot_result:
            f, ax = plt.subplots()
            ax.set_title("L-Curve loss values")
            ax.set_xscale("log", nonpositive="clip")
            ax.set_yscale("log", nonpositive="clip")
            ax.plot(lams_reg, f_vals)
            ax.grid()
            f.tight_layout()
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

    def compute_loss_values(self, lams_reg: Union[ArrayLike, NDArrayFloat]) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
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

        c = tm.perf_counter()
        if self.verbose:
            print("Computing cross-validation loss values:")
            print("- Regularization weights range: [%g, %g] in %d steps" % (lams_reg[0], lams_reg[-1], len(lams_reg)))
            print("- Number of averages: %d" % self.num_averages)
            print("- Leave-out pixel fraction: %g%%" % (self.test_fraction * 100))
            print(f"Parallel evaluation: {self.parallel_eval} (n. threads: {num_threads})")

        f_vals = np.empty((len(lams_reg), self.num_averages), dtype=self.dtype)
        for ii_avg in range(self.num_averages):
            c_round = tm.perf_counter()
            if self.verbose:
                print("\nRound: %d/%d" % (ii_avg + 1, self.num_averages))

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
                print(" - Done in %g seconds.\n" % (tm.perf_counter() - c_round))

        f_avgs = np.mean(f_vals, axis=1)
        f_stds = np.std(f_vals, axis=1)

        if self.verbose:
            print("Done in %g seconds.\n" % (tm.perf_counter() - c))

        if self.plot_result:
            f, ax = plt.subplots()
            ax.set_title("Cross-validation loss values (avgs: %d)" % self.num_averages)
            ax.set_xscale("log", nonpositive="clip")
            ax.errorbar(lams_reg, f_avgs, yerr=f_stds, ecolor=(0.5, 0.5, 0.5), elinewidth=1, capsize=2)
            ax.grid()
            f.tight_layout()
            plt.show(block=False)

        return f_avgs, f_stds, f_vals

    def fit_loss_min(
        self,
        lams_reg: Union[ArrayLike, NDArrayFloat],
        f_vals: NDArrayFloat,
        f_stds: Optional[NDArrayFloat] = None,
        scale: str = "log",
    ) -> Tuple[float, float]:
        """Parabolic fit of objective function costs for the different regularization weights.

        Parameters
        ----------
        lams_reg : Union[ArrayLike, NDArrayFloat]
            Regularization weights.
        f_vals : NDArrayFloat
            Objective function costs of each regularization weight.
        f_stds : NDArrayFloat, optional
            Objective function cost standard deviations of each regularization weight.
            It is only used for plotting purpouses. The default is None.
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
                "Given: lams=%d, vals=%d" % (len(lams_reg), len(f_vals))
            )

        if scale.lower() == "log":
            to_fit = lambda x: np.log10(x)
            from_fit = lambda x: 10**x
        elif scale.lower() == "linear":
            to_fit = lambda x: x
            from_fit = to_fit
        else:
            raise ValueError("Parameter 'scale' should be either 'log' or 'linear', given '%s' instead" % scale)

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

        c = tm.perf_counter()
        if self.verbose:
            print(
                "Fitting minimum within the parameter interval [%g, %g]: "
                % (from_fit(lams_reg_fit[0]), from_fit(lams_reg_fit[-1])),
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
                "WARNING: fitted lambda %g is outside the bounds of input lambdas [%g, %g]."
                % (min_lam, lams_reg[0], lams_reg[-1])
                + " Returning minimum measured point."
            )
            res_lam, res_val = lams_reg[min_pos], f_vals[min_pos]
        else:
            res_lam, res_val = min_lam, min_val

        if self.verbose:
            print("Found at %g, in %g seconds.\n" % (min_lam, tm.perf_counter() - c))

        if self.plot_result:
            f, ax = plt.subplots()
            ax.set_xscale(scale, nonpositive="clip")
            if f_stds is None:
                ax.plot(lams_reg, f_vals)
            else:
                ax.errorbar(lams_reg, f_vals, yerr=f_stds, ecolor=(0.5, 0.5, 0.5), elinewidth=1, capsize=2)
            x = np.linspace(lams_reg_fit[0], lams_reg_fit[2])
            y = coeffs[0] + x * (coeffs[1] + x * coeffs[2])
            ax.plot(from_fit(x), y)
            ax.scatter(min_lam, min_val)
            ax.grid()
            f.tight_layout()
            plt.show(block=False)

        return res_lam, res_val
