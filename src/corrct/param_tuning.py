#!/usr/bin/env python3
"""
Aided regularization parameter estimation.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from os import cpu_count
from time import perf_counter
from typing import Any, Literal, overload
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import DTypeLike, NDArray
from tqdm.auto import tqdm

from corrct.solvers import SolutionInfo

NUM_CPUS = cpu_count() or 1
MAX_THREADS = int(round(np.log2(NUM_CPUS + 1)))


NDArrayFloat = NDArray[np.floating]


def format_time(seconds: float) -> str:
    """
    Convert seconds to a formatted string in the format <hours>:<minutes>:<seconds>.<milliseconds>.

    Parameters
    ----------
    seconds : float
        Time in seconds.

    Returns
    -------
    str
        Formatted time string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d}.{milliseconds:03d}"


@dataclass
class PerfMeterTask:
    """Performance tracking class for single reconstructions."""

    init_time_s: float
    exec_time_s: float
    total_time_s: float


@dataclass
class PerfMeterBatch:
    """Performance tracking class for batches of reconstructions."""

    init_time_s: float = 0.0
    proc_time_s: float = 0.0
    total_time_s: float = 0.0
    tasks_perf: list[PerfMeterTask] = field(default_factory=lambda: [])

    def append(self, task: PerfMeterTask) -> None:
        """Append a task's performance metrics to the batch.

        Parameters
        ----------
        task : PerfMeterTask
            The task to append to the batch.
        """
        self.proc_time_s += task.total_time_s
        self.total_time_s += task.total_time_s
        self.tasks_perf.append(task)

    def __add__(self, other: "PerfMeterBatch") -> "PerfMeterBatch":
        """Add two PerfMeterBatch instances together.

        Parameters
        ----------
        other : PerfMeterBatch
            The other PerfMeterBatch instance to add to this one.

        Returns
        -------
        PerfMeterBatch
            A new PerfMeterBatch instance with the sum of the dispatch times,
            processing times, total times, and concatenated task performance
            lists from both instances.
        """
        return PerfMeterBatch(
            init_time_s=self.init_time_s + other.init_time_s,
            proc_time_s=self.proc_time_s + other.proc_time_s,
            total_time_s=self.total_time_s + other.total_time_s,
            tasks_perf=self.tasks_perf + other.tasks_perf,
        )

    def __str__(self) -> str:
        """
        Return a formatted string representation of the performance statistics.

        Returns
        -------
        str
            Formatted string representation of the performance statistics.
        """
        stats_str = "Performance Statistics:\n"
        stats_str += f"- Initialization time: {format_time(self.init_time_s)}\n"
        stats_str += f"- Processing time: {format_time(self.proc_time_s)}\n"

        if self.tasks_perf:
            avg_init_time = sum(task.init_time_s for task in self.tasks_perf) / len(self.tasks_perf)
            avg_exec_time = sum(task.exec_time_s for task in self.tasks_perf) / len(self.tasks_perf)
            avg_total_time = sum(task.total_time_s for task in self.tasks_perf) / len(self.tasks_perf)

            # Calculate apparent speed-up
            if self.proc_time_s > 0:
                speed_up = avg_total_time * len(self.tasks_perf) / self.proc_time_s
                stats_str += f"- Total time: {format_time(self.total_time_s)} (Tasks/Total ratio: {speed_up:.2f})\n"
            else:
                stats_str += f"- Total time: {format_time(self.total_time_s)}\n"

            stats_str += "\nAverage Task Performance:\n"
            stats_str += f"- Initialization time: {format_time(avg_init_time)}\n"
            stats_str += f"- Execution time: {format_time(avg_exec_time)}\n"
            stats_str += f"- Total time: {format_time(avg_total_time)}\n"
        else:
            stats_str += f"- Total time: {format_time(self.total_time_s)}\n"

        return stats_str


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
    dtype : DTypeLike, optional
        The data type of the mask. The default is np.float32.

    Returns
    -------
    NDArrayFloat
        The pixel mask.
    """
    data_test_mask = np.zeros(data_shape, dtype=dtype)
    num_test_pixels = int(np.ceil(data_test_mask.size * test_fraction))
    test_pixels = np.random.permutation(data_test_mask.size)
    test_pixels = np.unravel_index(test_pixels[:num_test_pixels], data_shape)
    data_test_mask[test_pixels] = 1
    return data_test_mask


def create_k_fold_test_masks(
    data_shape: Sequence[int],
    k_folds: int,
    dtype: DTypeLike = np.float32,
    seed: int | None = None,
) -> list[NDArray]:
    """
    Create K random masks for K-fold cross-validation.

    Parameters
    ----------
    data_shape : Sequence[int]
        The shape of the data.
    k_folds : int
        The number of folds.
    dtype : DTypeLike, optional
        The data type of the masks. The default is np.float32.
    seed : int | None, optional
        Seed for the random number generator. The default is None.

    Returns
    -------
    list[NDArray]
        A list of K pixel masks.
    """
    # Create a random number generator
    rng = np.random.default_rng(seed)

    # Create a list of indices and shuffle it
    indices = rng.permutation(np.prod(data_shape))

    # Split the indices into K folds using strides
    folds = [indices[i::k_folds] for i in range(k_folds)]

    # Create the masks
    masks = []
    for fold in folds:
        mask = np.zeros(data_shape, dtype=dtype)
        mask.ravel()[fold] = 1
        masks.append(mask)

    return masks


def get_lambda_range(start: float, end: float, num_per_order: int = 4, aligned_order: bool = True) -> NDArrayFloat:
    """
    Compute hyper-parameter values within an interval.

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
    hp_vals: float | Sequence[float] | NDArrayFloat,
    f_vals: NDArrayFloat,
    f_stds: NDArrayFloat | None = None,
    scale: Literal["linear", "log"] = "log",
    verbose: bool = False,
    plot_result: bool = False,
) -> tuple[float, float]:
    """
    Parabolic fit of objective function costs for the different hyper-parameter values.

    Parameters
    ----------
    hp_vals : float | Sequence[float] | NDArrayFloat
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

    counter = perf_counter()
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
        print(f"Found at {min_hp_val:.3e}, in {perf_counter() - counter:g} seconds.\n")

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
        axs.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
        fig.tight_layout()
        plt.show(block=False)

    return res_hp_val, res_f_val


def _compute_reconstruction(
    init_fun: Callable | None,
    exec_fun: Callable[[Any], tuple[NDArrayFloat, SolutionInfo]],
    hp_val: float,
    *,
    init_fun_kwds: Mapping,
    exec_fun_kwds: Mapping,
) -> tuple[NDArrayFloat, SolutionInfo, PerfMeterTask]:
    """
    Compute a single reconstruction.

    Parameters
    ----------
    init_fun : Callable | None
        The task initialization function.
    exec_fun : Callable[[Any], tuple[NDArrayFloat, SolutionInfo]]
        The task execution function.
    hp_val : float
        The hyper-parameter value.
    init_fun_kwds : Mapping
        Additional keyword arguments to pass to the task initialization function.
    exec_fun_kwds : Mapping
        Additional keyword arguments to pass to the task execution function.

    Returns
    -------
    NDArrayFloat
        The reconstruction.
    SolutionInfo
        The solution information.
    PerfMeterTask
        The performance statistics for the reconstruction.
    """
    c0 = perf_counter()
    if init_fun is not None:
        solver = init_fun(hp_val, **init_fun_kwds)
        c1 = perf_counter()
        rec, rec_info = exec_fun(solver, **exec_fun_kwds)
    else:
        c1 = perf_counter()
        rec, rec_info = exec_fun(hp_val, **exec_fun_kwds)
    c2 = perf_counter()

    stats = PerfMeterTask(init_time_s=(c1 - c0), exec_time_s=(c2 - c1), total_time_s=(c2 - c0))

    return rec, rec_info, stats


def _parallel_compute(
    executor: Executor,
    init_fun: Callable | None,
    exec_fun: Callable[[Any], tuple[NDArrayFloat, SolutionInfo]],
    hp_vals: float | Sequence[float] | NDArrayFloat,
    *,
    init_fun_kwds: Mapping | None = None,
    exec_fun_kwds: Mapping | None = None,
    verbose: bool = True,
) -> tuple[list[NDArray], list[SolutionInfo], PerfMeterBatch]:
    """
    Compute reconstructions in parallel.

    Parameters
    ----------
    executor : Executor
        The executor to use for parallel computation.
    init_fun : Callable | None
        The task initialization function.
    exec_fun : Callable[[Any], tuple[NDArrayFloat, SolutionInfo]]
        The task execution function.
    hp_vals : float | Sequence[float] | NDArrayFloat
        A list or array of hyperparameter values to evaluate.
    init_fun_kwds : Mapping | None, optional
        Additional keyword arguments to pass to the task initialization function. By default None.
    exec_fun_kwds : Mapping | None, optional
        Additional keyword arguments to pass to the task execution function. By default None.
    verbose : bool, optional
        Whether to produce verbose output, by default True

    Returns
    -------
    list[NDArray]
        A list of reconstructions corresponding to each hyperparameter value.
    list[SolutionInfo]
        A list of solution information objects corresponding to each reconstruction.
    PerfMeterBatch
        A computing performance statistics object containing aggregated performance metrics.
    """
    c0 = perf_counter()

    if init_fun_kwds is None:
        init_fun_kwds = dict()
    if exec_fun_kwds is None:
        exec_fun_kwds = dict()

    hp_vals = np.array(hp_vals, ndmin=1)

    future_to_lambda = {
        executor.submit(
            _compute_reconstruction, init_fun, exec_fun, hp_val, init_fun_kwds=init_fun_kwds, exec_fun_kwds=exec_fun_kwds
        ): (ii, hp_val)
        for ii, hp_val in enumerate(hp_vals)
    }

    recs = [np.array([])] * len(hp_vals)
    recs_info = [SolutionInfo("", 1, None)] * len(hp_vals)
    perf_items = [PerfMeterTask(0, 0, 0)] * len(hp_vals)

    c1 = perf_counter()

    try:
        for future in tqdm(
            as_completed(future_to_lambda),
            desc="Hyper-parameter values",
            disable=not verbose,
            total=len(hp_vals),
        ):
            hp_ind, hp_val = future_to_lambda[future]
            try:
                recs[hp_ind], recs_info[hp_ind], perf_items[hp_ind] = future.result()
            except ValueError as exc:
                print(f"Hyper-parameter value {hp_val} (#{hp_ind}) generated an exception: {exc}")
                raise
    except:
        print("Shutting down..", end="", flush=True)
        if "cancel_futures" in inspect.signature(executor.shutdown).parameters.keys():
            executor.shutdown(cancel_futures=True)
        else:
            # This handles the case of Dask's ClientExecutor
            executor.shutdown()
        print("\b\b: Done.")
        raise

    c2 = perf_counter()
    perf_batch = PerfMeterBatch(init_time_s=(c1 - c0), proc_time_s=(c2 - c1), total_time_s=(c2 - c0), tasks_perf=perf_items)

    return recs, recs_info, perf_batch


def _serial_compute(
    init_fun: Callable | None,
    exec_fun: Callable[[Any], tuple[NDArrayFloat, SolutionInfo]],
    hp_vals: float | Sequence[float] | NDArrayFloat,
    *,
    init_fun_kwds: Mapping | None = None,
    exec_fun_kwds: Mapping | None = None,
    verbose: bool = True,
) -> tuple[list[NDArray], list[SolutionInfo], PerfMeterBatch]:
    """
    Compute reconstructions in serial.

    Parameters
    ----------
    init_fun : Callable | None
        The task initialization function.
    exec_fun : Callable[[Any], tuple[NDArrayFloat, SolutionInfo]]
        The task execution function.
    hp_vals : float | Sequence[float] | NDArrayFloat
        A list or array of hyperparameter values to evaluate.
    init_fun_kwds : Mapping | None, optional
        Additional keyword arguments to pass to the task initialization function. By default None.
    exec_fun_kwds : Mapping | None, optional
        Additional keyword arguments to pass to the task execution function. By default None.
    verbose : bool, optional
        Whether to produce verbose output, by default True

    Returns
    -------
    list[NDArray]
        A list of reconstructions corresponding to each hyperparameter value.
    list[SolutionInfo]
        A list of solution information objects corresponding to each reconstruction.
    PerfMeterBatch
        A computing performance statistics object containing aggregated performance metrics.
    """
    c0 = perf_counter()

    if init_fun_kwds is None:
        init_fun_kwds = dict()
    if exec_fun_kwds is None:
        exec_fun_kwds = dict()

    hp_vals = np.array(hp_vals, ndmin=1)

    recs = [np.array([])] * len(hp_vals)
    recs_info = [SolutionInfo("", 1, None)] * len(hp_vals)
    perf_items = [PerfMeterTask(0, 0, 0)] * len(hp_vals)

    c1 = perf_counter()

    for ii, l in enumerate(tqdm(hp_vals, desc="Hyper-parameter values", disable=not verbose)):
        recs[ii], recs_info[ii], perf_items[ii] = _compute_reconstruction(
            init_fun, exec_fun, l, init_fun_kwds=init_fun_kwds, exec_fun_kwds=exec_fun_kwds
        )

    c2 = perf_counter()
    perf_batch = PerfMeterBatch(init_time_s=(c1 - c0), proc_time_s=(c2 - c1), total_time_s=(c2 - c0), tasks_perf=perf_items)

    return recs, recs_info, perf_batch


def plot_cv_curves(solution_infos: list[SolutionInfo], hp_vals: Sequence[float]) -> None:
    """
    Plot the relative cross-validation curves for all hyper-parameter values.

    Parameters
    ----------
    solution_infos : list[SolutionInfo]
        List of SolutionInfo objects containing the relative cross-validation residuals.
    hp_vals : Sequence[float]
        Sequence of hyper-parameter values corresponding to the solution_infos.
    """
    fig, axs = plt.subplots()
    for hp_val, info in zip(hp_vals, solution_infos):
        axs.semilogy(info.residuals_cv_rel, label=f"CV residuals, HP val={hp_val:.3e}")
    axs.set_xlabel("Iterations", fontsize=16)
    axs.grid()
    axs.legend(fontsize=13)
    for tl in axs.get_xticklabels():
        tl.set_fontsize(13)
    for tl in axs.get_yticklabels():
        tl.set_fontsize(13)
    fig.tight_layout()
    plt.show(block=False)


class BaseParameterTuning(ABC):
    """Base class for parameter tuning classes."""

    _task_init_function: Callable | None
    _task_exec_function: Callable[[Any], tuple[NDArrayFloat, SolutionInfo]] | None

    parallel_eval: int | Executor

    dtype: DTypeLike
    verbose: bool
    plot_result: bool
    print_timings: bool

    def __init__(
        self,
        dtype: DTypeLike = np.float32,
        parallel_eval: Executor | int | bool = True,
        verbose: bool = False,
        plot_result: bool = False,
        print_timings: bool = False,
    ) -> None:
        """
        Initialize a base helper class.

        Parameters
        ----------
        dtype : DTypeLike, optional
            Type of the data, by default np.float32
        parallel_eval : Executor | int | bool, optional
            Whether to evaluate results in parallel, by default True
        verbose : bool, optional
            Whether to produce verbose output, by default False
        plot_result : bool, optional
            Whether to plot the results, by default False
        print_timings : bool, optional
            Whether to print the performance metrics, by default False
        """
        self.dtype = dtype

        if isinstance(parallel_eval, bool):
            parallel_eval = MAX_THREADS if parallel_eval else 0
        self.parallel_eval = parallel_eval
        self.verbose = verbose
        self.plot_result = plot_result
        self.print_timings = print_timings

        self._task_init_function = None
        self._task_exec_function = None

    @property
    def task_init_function(self) -> Callable | None:
        """
        Return the local reference to the task initialization function.

        Returns
        -------
        Callable | None
            The task initialization function, if initialized.
        """
        return self._task_init_function

    @property
    def task_exec_function(self) -> Callable[[Any], tuple[NDArrayFloat, SolutionInfo]]:
        """
        Return the local reference to the task execution function.

        Returns
        -------
        Callable[[Any], tuple[NDArrayFloat, SolutionInfo]]
            The task execution function.
        """
        if self._task_exec_function is None:
            raise ValueError("Task execution function not initialized!")
        return self._task_exec_function

    @task_init_function.setter
    def task_init_function(self, init_fun: Callable | None) -> None:
        """
        Set the task initialization function.

        Parameters
        ----------
        init_fun : Callable | None
            The task initialization function.
        """
        if init_fun is not None:
            if not isinstance(init_fun, Callable):
                raise ValueError("Expected a task initialization function (callable)")
            if len(inspect.signature(init_fun).parameters) != 1:
                raise ValueError(
                    "Expected a task initialization function (callable), whose only parameter is the hyper-parameter value"
                )
        self._task_init_function = init_fun

    @task_exec_function.setter
    def task_exec_function(self, exec_fun: Callable[[Any], tuple[NDArrayFloat, SolutionInfo]]) -> None:
        """
        Set the task execution function.

        Parameters
        ----------
        exec_fun : Callable[[Any], tuple[NDArrayFloat, SolutionInfo]]
            The task execution function.
        """
        if not isinstance(exec_fun, Callable):
            raise ValueError("Expected a task execution function (callable)")
        if not len(inspect.signature(exec_fun).parameters) >= 1:
            raise ValueError("Expected a task execution function (callable), with at least one parameter (solver)")
        self._task_exec_function = exec_fun

    @property
    def solver_spawning_function(self) -> Callable | None:
        """Return the local reference to the task initialization function."""
        warn("DEPRECATED: This property is deprecated, and will be removed. Please use the property `task_init_function`")
        return self.task_init_function

    @property
    def solver_calling_function(self) -> Callable[[Any], tuple[NDArrayFloat, SolutionInfo]]:
        """Return the local reference to the task execution function."""
        warn("DEPRECATED: This property is deprecated, and will be removed. Please use the property `task_exec_function`")
        return self.task_exec_function

    @solver_spawning_function.setter
    def solver_spawning_function(self, init_fun: Callable) -> None:
        warn("DEPRECATED: This property is deprecated, and will be removed. Please use the property `task_init_function`")
        self.task_init_function = init_fun

    @solver_calling_function.setter
    def solver_calling_function(self, exec_fun: Callable[[Any], tuple[NDArrayFloat, SolutionInfo]]) -> None:
        warn("DEPRECATED: This property is deprecated, and will be removed. Please use the property `task_exec_function`")
        self.task_exec_function = exec_fun

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
        warn("DEPRECATED: This method is deprecated, and will be removed. Please use the module function with the same name")
        return get_lambda_range(start=start, end=end, num_per_order=num_per_order)

    def process_hp_vals(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
    ) -> tuple[list[NDArray], list[SolutionInfo], PerfMeterBatch]:
        """
        Compute reconstructions, solution information, and computing performance statistics for all hyperparameter values.

        Parameters
        ----------
        hp_vals : float | Sequence[float] | NDArrayFloat
            A list or array of hyperparameter values to evaluate.
        init_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task initialization function. By default None.
        exec_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task execution function. By default None.

        Returns
        -------
        tuple[list[NDArray], list[SolutionInfo], PerfStatsBatch]
            A tuple containing:
            - A list of reconstructions corresponding to each hyperparameter value.
            - A list of solution information objects corresponding to each reconstruction.
            - A computing performance statistics object containing aggregated performance metrics.
        """
        if self._task_exec_function is None:
            raise ValueError("Task execution function not initialized!")

        if isinstance(self.parallel_eval, bool) and self.parallel_eval:
            self.parallel_eval = MAX_THREADS

        compute_args = [self._task_init_function, self._task_exec_function, hp_vals]
        compute_kwds = dict(init_fun_kwds=init_fun_kwds, exec_fun_kwds=exec_fun_kwds, verbose=self.verbose)

        if isinstance(self.parallel_eval, Executor):
            recs, recs_info, perf_batch = _parallel_compute(self.parallel_eval, *compute_args, **compute_kwds)
        elif isinstance(self.parallel_eval, int):
            if self.parallel_eval:
                with ThreadPoolExecutor(max_workers=self.parallel_eval) as executor:
                    recs, recs_info, perf_batch = _parallel_compute(executor, *compute_args, **compute_kwds)
            else:
                recs, recs_info, perf_batch = _serial_compute(*compute_args, **compute_kwds)
        else:
            raise ValueError(
                f"The variable `parallel_eval` should either be an Executor, a boolean, or an int. "
                f"A `{type(self.parallel_eval)}` was passed instead."
            )

        if self.print_timings:
            print(perf_batch)

        return recs, recs_info, perf_batch

    def compute_reconstruction_error(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        gnd_truth: NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """
        Compute the reconstruction errors for each hyper-parameter values against the ground truth.

        Parameters
        ----------
        hp_vals : float | Sequence[float] | NDArrayFloat
            List of hyper-parameter values.
        gnd_truth : NDArrayFloat
            Expected reconstruction.
        init_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task initialization function. By default None.
        exec_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task execution function. By default None.

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
            if isinstance(self.parallel_eval, Executor):
                print(f"Parallel evaluation with externally provided executor: {self.parallel_eval}")
            else:
                print(
                    f"Parallel evaluation: {self.parallel_eval > 0}",
                    f"(n. threads: {self.parallel_eval})" if self.parallel_eval > 0 else "",
                )

        recs, _, _ = self.process_hp_vals(hp_vals, init_fun_kwds=init_fun_kwds, exec_fun_kwds=exec_fun_kwds)

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
            for tl in axs.get_xticklabels():
                tl.set_fontsize(13)
            for tl in axs.get_yticklabels():
                tl.set_fontsize(13)
            axs.set_xlabel(r"$\lambda$ values", fontsize=16)
            axs.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
            fig.tight_layout()
            plt.show(block=False)

        return err_l1, err_l2

    @overload
    def compute_loss_values(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
        return_all: Literal[False] = False,
    ) -> NDArrayFloat: ...

    @overload
    def compute_loss_values(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
        return_all: Literal[True] = True,
    ) -> tuple[NDArrayFloat, list[NDArrayFloat], list[SolutionInfo], PerfMeterBatch]: ...

    @abstractmethod
    def compute_loss_values(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
        return_all: bool = False,
    ) -> NDArrayFloat | tuple[NDArrayFloat, list[NDArrayFloat], list[SolutionInfo], PerfMeterBatch]:
        """
        Compute the objective function costs for a list of hyper-parameter values.

        Parameters
        ----------
        hp_vals : float | Sequence[float] | NDArrayFloat
            List of hyper-parameter values.
        init_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task initialization function. By default None.
        exec_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task execution function. By default None.
        return_all : bool, optional
            If True, return all the computation information (reconstructions, all loss values, performance). Default is False.

        Returns
        -------
        NDArrayFloat
            Objective function cost for each hyper-parameter value.
        recs : NDArrayFloat, optional
            Reconstructions for each hyper-parameter value (returned only if `return_all` is True).
        recs_info : list[SolutionInfo], optional
            Solution information for each reconstruction (returned only if `return_all` is True).
        perf_batch : PerfStatsBatch, optional
            Performance statistics for the batch of reconstructions (returned only if `return_all` is True).
        """


class LCurve(BaseParameterTuning):
    """L-curve regularization parameter estimation helper."""

    def __init__(
        self,
        loss_function: Callable,
        data_dtype: DTypeLike = np.float32,
        parallel_eval: Executor | int | bool = True,
        verbose: bool = False,
        plot_result: bool = False,
        print_timings: bool = False,
    ) -> None:
        """
        Create an L-curve regularization parameter estimation helper.

        Parameters
        ----------
        loss_function : Callable
            The loss function for the computation of the L-curve values.
        data_dtype : DTypeLike, optional
            Type of the input data. The default is np.float32.
        parallel_eval : Executor | int | bool, optional
            Compute loss and error values in parallel. The default is True.
        verbose : bool, optional
            Print verbose output. The default is False.
        plot_result : bool, optional
            Plot results. The default is False.
        print_timings : bool, optional
            Whether to print the performance metrics, by default False
        """
        super().__init__(
            dtype=data_dtype,
            parallel_eval=parallel_eval,
            verbose=verbose,
            plot_result=plot_result,
            print_timings=print_timings,
        )

        if not isinstance(loss_function, Callable):
            raise ValueError(
                "Expected a callable with one argument for the argument 'loss_function',"
                " whose parameters are: the solver and the data test mask"
            )
        if len(inspect.signature(loss_function).parameters) != 1:
            raise ValueError("The callable 'loss_function', should have one parameter")
        self.loss_function = loss_function

    @overload
    def compute_loss_values(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
        return_all: Literal[False] = False,
    ) -> NDArrayFloat: ...

    @overload
    def compute_loss_values(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
        return_all: Literal[True] = True,
    ) -> tuple[NDArrayFloat, list[NDArrayFloat], list[SolutionInfo], PerfMeterBatch]: ...

    def compute_loss_values(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
        return_all: bool = False,
    ) -> NDArrayFloat | tuple[NDArrayFloat, list[NDArrayFloat], list[SolutionInfo], PerfMeterBatch]:
        """
        Compute the objective function costs for a list of hyper-parameter values.

        Parameters
        ----------
        hp_vals : float | Sequence[float] | NDArrayFloat
            List of hyper-parameter values.
        init_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task initialization function. By default None.
        exec_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task execution function. By default None.
        return_all : bool, optional
            If True, return all the computation information (reconstructions, all loss values, performance). Default is False.

        Returns
        -------
        f_vals : NDArrayFloat
            Objective function cost for each hyper-parameter value.
        recs : list[NDArrayFloat], optional
            Reconstructions for each hyper-parameter value (returned only if `return_all` is True).
        recs_info : list[SolutionInfo], optional
            Solution information for each reconstruction (returned only if `return_all` is True).
        perf_batch : PerfStatsBatch, optional
            Performance statistics for the batch of reconstructions (returned only if `return_all` is True).
        """
        hp_vals = np.array(hp_vals, ndmin=1)

        if self.verbose:
            print("Computing L-curve loss values:")
            print(f"- Hyper-parameter values range: [{hp_vals[0]:.3e}, {hp_vals[-1]:.3e}] in {len(hp_vals)} steps")
            if isinstance(self.parallel_eval, Executor):
                print(f"Parallel evaluation with externally provided executor: {self.parallel_eval}")
            else:
                print(
                    f"Parallel evaluation: {self.parallel_eval > 0}",
                    f"(n. threads: {self.parallel_eval})" if self.parallel_eval > 0 else "",
                )

        recs, recs_info, perf_batch = self.process_hp_vals(hp_vals, init_fun_kwds=init_fun_kwds, exec_fun_kwds=exec_fun_kwds)
        f_vals = np.array([self.loss_function(rec) for rec in recs], dtype=self.dtype)

        if self.plot_result:
            fig, axs = plt.subplots()
            axs.set_title("L-Curve loss values", fontsize=16)
            axs.set_xscale("log", nonpositive="clip")
            axs.set_yscale("log", nonpositive="clip")
            axs.plot(hp_vals, f_vals)
            axs.grid()
            for tl in axs.get_xticklabels():
                tl.set_fontsize(13)
            for tl in axs.get_yticklabels():
                tl.set_fontsize(13)
            axs.set_xlabel(r"$\lambda$ values", fontsize=16)
            axs.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
            fig.tight_layout()
            plt.show(block=False)

        if return_all:
            return f_vals, recs, recs_info, perf_batch
        else:
            return f_vals


class CrossValidation(BaseParameterTuning):
    """Cross-validation hyper-parameter estimation class."""

    data_shape: Sequence[int]
    cv_fraction: float | None
    num_averages: int

    data_cv_masks: list[NDArray]
    mask_param_name: str

    def __init__(
        self,
        data_shape: Sequence[int],
        cv_fraction: float | None = 0.1,
        num_averages: int = 5,
        mask_param_name: str = "b_test_mask",
        parallel_eval: Executor | int | bool = True,
        dtype: DTypeLike = np.float32,
        verbose: bool = False,
        plot_result: bool = False,
        print_timings: bool = False,
    ) -> None:
        """
        Create a cross-validation hyper-parameter estimation helper.

        Parameters
        ----------
        data_shape : Sequence[int]
            Shape of the projection data.
        cv_fraction : float | None, optional
            Fraction of detector points to use for the leave-out set.
            If None, K-fold cross-validation is used, where `num_averages` indicates the number of k-folds.
            The default is 0.1.
        num_averages : int, optional
            Number of averages random leave-out sets to use. The default is 5.
        mask_param_name: str, optional
            The parameter name in the task execution function that accepts the data masks. The default is "b_test_mask".
        parallel_eval : Executor | int | bool, optional
            Compute loss and error values in parallel. The default is True.
        dtype : DTypeLike, optional
            Type of the input data. The default is np.float32.
        verbose : bool, optional
            Print verbose output. The default is False.
        plot_result : bool, optional
            Plot results. The default is False.
        print_timings : bool, optional
            Whether to print the performance metrics, by default False
        """
        super().__init__(
            dtype=dtype, parallel_eval=parallel_eval, verbose=verbose, plot_result=plot_result, print_timings=print_timings
        )
        self.data_shape = data_shape
        self.cv_fraction = cv_fraction
        self.num_averages = num_averages

        self.mask_param_name = mask_param_name

        if self.cv_fraction is not None:
            self.data_cv_masks = [
                create_random_test_mask(self.data_shape, self.cv_fraction, self.dtype) for _ in range(self.num_averages)
            ]
        else:
            self.data_cv_masks = create_k_fold_test_masks(self.data_shape, self.num_averages)

    @overload
    def compute_loss_values(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
        return_all: Literal[False] = False,
    ) -> tuple[NDArrayFloat, NDArrayFloat, list[NDArrayFloat]]: ...

    @overload
    def compute_loss_values(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
        return_all: bool = False,
    ) -> tuple[NDArrayFloat, NDArrayFloat, list[tuple[list[NDArrayFloat], list[SolutionInfo], PerfMeterBatch]]]: ...

    def compute_loss_values(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        *,
        init_fun_kwds: Mapping | None = None,
        exec_fun_kwds: Mapping | None = None,
        return_all: bool = False,
    ) -> (
        tuple[NDArrayFloat, NDArrayFloat, list[NDArrayFloat]]
        | tuple[NDArrayFloat, NDArrayFloat, list[tuple[list[NDArrayFloat], list[SolutionInfo], PerfMeterBatch]]]
    ):
        """
        Compute objective function values for all requested hyper-parameter values.

        Parameters
        ----------
        hp_vals : float | Sequence[float] | NDArrayFloat
            Hyper-parameter values (e.g., regularization weight) to evaluate.
        init_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task initialization function. By default None.
        exec_fun_kwds : Mapping | None, optional
            Additional keyword arguments to pass to the task execution function. By default None.
        return_all : bool, optional
            If True, return the reconstructions along with the loss values. Default is False.

        Returns
        -------
        f_avgs : NDArrayFloat
            Average objective function costs for each hyper-parameter value.
        f_stds : NDArrayFloat
            Standard deviation of objective function costs for each hyper-parameter value.
        f_vals : NDArrayFloat
            Objective function costs for each hyper-parameter value.
        recs : list[NDArrayFloat], optional
            Reconstructions for each hyper-parameter value (returned only if `return_all` is True).
        """
        if self._task_exec_function is None:
            raise ValueError("Task execution function not initialized!")

        f_sig = inspect.signature(self._task_exec_function)
        if self.mask_param_name not in f_sig.parameters.keys():
            raise ValueError(
                f"The task execution function should have a parameter called `{self.mask_param_name}`, "
                f"which is defined by the property `mask_param_name`. Please adjust accordingly."
            )

        if init_fun_kwds is None:
            init_fun_kwds = dict()
        if exec_fun_kwds is None:
            exec_fun_kwds = dict()
        exec_fun_kwds = dict(**exec_fun_kwds)

        hp_vals = np.array(hp_vals, ndmin=1)

        counter = perf_counter()
        if self.verbose:
            is_kfold = self.cv_fraction is None
            print("Computing cross-validation loss values:")
            print(f"- Hyper-parameter range: [{hp_vals[0]:.3e}, {hp_vals[-1]:.3e}] in {len(hp_vals)} steps")
            print(f"- Number of averages: {self.num_averages}" + (" (K-folds)" if is_kfold else ""))
            print(f"- Leave-out pixel fraction: {1 / self.num_averages if is_kfold else self.cv_fraction:.3%}")
            if isinstance(self.parallel_eval, Executor):
                print(f"Parallel evaluation with externally provided executor: {self.parallel_eval}")
            else:
                print(
                    f"Parallel evaluation: {self.parallel_eval > 0}",
                    f"(n. threads: {self.parallel_eval})" if self.parallel_eval > 0 else "",
                )

        f_vals = [np.array([])] * self.num_averages
        results = []

        for ii_avg in range(self.num_averages):
            if self.verbose:
                print(f"\nRound: {ii_avg + 1}/{self.num_averages}")

            exec_fun_kwds[self.mask_param_name] = self.data_cv_masks[ii_avg]

            recs_ii, recs_info_ii, perf_batch_ii = self.process_hp_vals(
                hp_vals, init_fun_kwds=init_fun_kwds, exec_fun_kwds=exec_fun_kwds
            )
            f_vals[ii_avg] = np.array([info.residuals_cv_rel[-1] for info in recs_info_ii])

            if return_all:
                results.append((recs_ii, recs_info_ii, perf_batch_ii))
            else:
                results.append(f_vals[ii_avg])

        f_avgs: NDArrayFloat = np.mean(f_vals, axis=0)
        f_stds: NDArrayFloat = np.std(f_vals, axis=0)

        if self.verbose:
            print(f"Done in {perf_counter() - counter:g} seconds.\n")

        if self.plot_result:
            fig, axs = plt.subplots()
            axs.set_title(f"Cross-validation loss values (avgs: {self.num_averages})", fontsize=16)
            axs.set_xscale("log", nonpositive="clip")
            axs.errorbar(hp_vals, f_avgs, yerr=f_stds, ecolor=(0.5, 0.5, 0.5), elinewidth=1, capsize=2)
            for f_vals_ii in f_vals:
                axs.plot(hp_vals, f_vals_ii, linewidth=1, linestyle="--")
            axs.grid()
            for tl in axs.get_xticklabels():
                tl.set_fontsize(13)
            for tl in axs.get_yticklabels():
                tl.set_fontsize(13)
            axs.set_xlabel(r"$\lambda$ values", fontsize=16)
            axs.set_ylabel("Loss values", fontsize=16)
            axs.yaxis.set_major_formatter(StrMethodFormatter("{x:.1e}"))
            fig.tight_layout()
            plt.show(block=False)

        return f_avgs, f_stds, results

    def fit_loss_min(
        self,
        hp_vals: float | Sequence[float] | NDArrayFloat,
        f_vals: NDArrayFloat,
        f_stds: NDArrayFloat | None = None,
        scale: Literal["linear", "log"] = "log",
    ) -> tuple[float, float]:
        """
        Parabolic fit of objective function costs for the different hyper-parameter values.

        Parameters
        ----------
        hp_vals : float | Sequence[float] | NDArrayFloat
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
