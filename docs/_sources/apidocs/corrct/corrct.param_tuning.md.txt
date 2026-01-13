# {py:mod}`corrct.param_tuning`

```{py:module} corrct.param_tuning
```

```{autodoc2-docstring} corrct.param_tuning
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PerfMeterTask <corrct.param_tuning.PerfMeterTask>`
  - ```{autodoc2-docstring} corrct.param_tuning.PerfMeterTask
    :summary:
    ```
* - {py:obj}`PerfMeterBatch <corrct.param_tuning.PerfMeterBatch>`
  - ```{autodoc2-docstring} corrct.param_tuning.PerfMeterBatch
    :summary:
    ```
* - {py:obj}`BaseParameterTuning <corrct.param_tuning.BaseParameterTuning>`
  - ```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning
    :summary:
    ```
* - {py:obj}`LCurve <corrct.param_tuning.LCurve>`
  - ```{autodoc2-docstring} corrct.param_tuning.LCurve
    :summary:
    ```
* - {py:obj}`CrossValidation <corrct.param_tuning.CrossValidation>`
  - ```{autodoc2-docstring} corrct.param_tuning.CrossValidation
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`format_time <corrct.param_tuning.format_time>`
  - ```{autodoc2-docstring} corrct.param_tuning.format_time
    :summary:
    ```
* - {py:obj}`create_random_test_mask <corrct.param_tuning.create_random_test_mask>`
  - ```{autodoc2-docstring} corrct.param_tuning.create_random_test_mask
    :summary:
    ```
* - {py:obj}`create_k_fold_test_masks <corrct.param_tuning.create_k_fold_test_masks>`
  - ```{autodoc2-docstring} corrct.param_tuning.create_k_fold_test_masks
    :summary:
    ```
* - {py:obj}`get_lambda_range <corrct.param_tuning.get_lambda_range>`
  - ```{autodoc2-docstring} corrct.param_tuning.get_lambda_range
    :summary:
    ```
* - {py:obj}`fit_func_min <corrct.param_tuning.fit_func_min>`
  - ```{autodoc2-docstring} corrct.param_tuning.fit_func_min
    :summary:
    ```
* - {py:obj}`_compute_reconstruction <corrct.param_tuning._compute_reconstruction>`
  - ```{autodoc2-docstring} corrct.param_tuning._compute_reconstruction
    :summary:
    ```
* - {py:obj}`_parallel_compute <corrct.param_tuning._parallel_compute>`
  - ```{autodoc2-docstring} corrct.param_tuning._parallel_compute
    :summary:
    ```
* - {py:obj}`_serial_compute <corrct.param_tuning._serial_compute>`
  - ```{autodoc2-docstring} corrct.param_tuning._serial_compute
    :summary:
    ```
* - {py:obj}`plot_cv_curves <corrct.param_tuning.plot_cv_curves>`
  - ```{autodoc2-docstring} corrct.param_tuning.plot_cv_curves
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NUM_CPUS <corrct.param_tuning.NUM_CPUS>`
  - ```{autodoc2-docstring} corrct.param_tuning.NUM_CPUS
    :summary:
    ```
* - {py:obj}`MAX_THREADS <corrct.param_tuning.MAX_THREADS>`
  - ```{autodoc2-docstring} corrct.param_tuning.MAX_THREADS
    :summary:
    ```
* - {py:obj}`NDArrayFloat <corrct.param_tuning.NDArrayFloat>`
  - ```{autodoc2-docstring} corrct.param_tuning.NDArrayFloat
    :summary:
    ```
````

### API

````{py:data} NUM_CPUS
:canonical: corrct.param_tuning.NUM_CPUS
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.NUM_CPUS
```

````

````{py:data} MAX_THREADS
:canonical: corrct.param_tuning.MAX_THREADS
:value: >
   'int(...)'

```{autodoc2-docstring} corrct.param_tuning.MAX_THREADS
```

````

````{py:data} NDArrayFloat
:canonical: corrct.param_tuning.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.NDArrayFloat
```

````

````{py:function} format_time(seconds: float) -> str
:canonical: corrct.param_tuning.format_time

```{autodoc2-docstring} corrct.param_tuning.format_time
```
````

`````{py:class} PerfMeterTask
:canonical: corrct.param_tuning.PerfMeterTask

```{autodoc2-docstring} corrct.param_tuning.PerfMeterTask
```

````{py:attribute} init_time_s
:canonical: corrct.param_tuning.PerfMeterTask.init_time_s
:type: float
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.PerfMeterTask.init_time_s
```

````

````{py:attribute} exec_time_s
:canonical: corrct.param_tuning.PerfMeterTask.exec_time_s
:type: float
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.PerfMeterTask.exec_time_s
```

````

````{py:attribute} total_time_s
:canonical: corrct.param_tuning.PerfMeterTask.total_time_s
:type: float
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.PerfMeterTask.total_time_s
```

````

`````

`````{py:class} PerfMeterBatch
:canonical: corrct.param_tuning.PerfMeterBatch

```{autodoc2-docstring} corrct.param_tuning.PerfMeterBatch
```

````{py:attribute} init_time_s
:canonical: corrct.param_tuning.PerfMeterBatch.init_time_s
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.param_tuning.PerfMeterBatch.init_time_s
```

````

````{py:attribute} proc_time_s
:canonical: corrct.param_tuning.PerfMeterBatch.proc_time_s
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.param_tuning.PerfMeterBatch.proc_time_s
```

````

````{py:attribute} total_time_s
:canonical: corrct.param_tuning.PerfMeterBatch.total_time_s
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.param_tuning.PerfMeterBatch.total_time_s
```

````

````{py:attribute} tasks_perf
:canonical: corrct.param_tuning.PerfMeterBatch.tasks_perf
:type: list[corrct.param_tuning.PerfMeterTask]
:value: >
   'field(...)'

```{autodoc2-docstring} corrct.param_tuning.PerfMeterBatch.tasks_perf
```

````

````{py:method} append(task: corrct.param_tuning.PerfMeterTask) -> None
:canonical: corrct.param_tuning.PerfMeterBatch.append

```{autodoc2-docstring} corrct.param_tuning.PerfMeterBatch.append
```

````

````{py:method} __add__(other: corrct.param_tuning.PerfMeterBatch) -> corrct.param_tuning.PerfMeterBatch
:canonical: corrct.param_tuning.PerfMeterBatch.__add__

```{autodoc2-docstring} corrct.param_tuning.PerfMeterBatch.__add__
```

````

````{py:method} __str__() -> str
:canonical: corrct.param_tuning.PerfMeterBatch.__str__

```{autodoc2-docstring} corrct.param_tuning.PerfMeterBatch.__str__
```

````

`````

````{py:function} create_random_test_mask(data_shape: collections.abc.Sequence[int], test_fraction: float = 0.05, dtype: numpy.typing.DTypeLike = np.float32) -> corrct.param_tuning.NDArrayFloat
:canonical: corrct.param_tuning.create_random_test_mask

```{autodoc2-docstring} corrct.param_tuning.create_random_test_mask
```
````

````{py:function} create_k_fold_test_masks(data_shape: collections.abc.Sequence[int], k_folds: int, dtype: numpy.typing.DTypeLike = np.float32, seed: int | None = None) -> list[numpy.typing.NDArray]
:canonical: corrct.param_tuning.create_k_fold_test_masks

```{autodoc2-docstring} corrct.param_tuning.create_k_fold_test_masks
```
````

````{py:function} get_lambda_range(start: float, end: float, num_per_order: int = 4, aligned_order: bool = True) -> corrct.param_tuning.NDArrayFloat
:canonical: corrct.param_tuning.get_lambda_range

```{autodoc2-docstring} corrct.param_tuning.get_lambda_range
```
````

````{py:function} fit_func_min(hp_vals: float | collections.abc.Sequence[float] | corrct.param_tuning.NDArrayFloat, f_vals: corrct.param_tuning.NDArrayFloat, f_stds: corrct.param_tuning.NDArrayFloat | None = None, scale: typing.Literal[linear, log] = 'log', verbose: bool = False, plot_result: bool = False) -> tuple[float, float]
:canonical: corrct.param_tuning.fit_func_min

```{autodoc2-docstring} corrct.param_tuning.fit_func_min
```
````

````{py:function} _compute_reconstruction(init_fun: collections.abc.Callable | None, exec_fun: collections.abc.Callable[[typing.Any], tuple[corrct.param_tuning.NDArrayFloat, corrct.solvers.SolutionInfo]], hp_val: float, *, init_fun_kwds: collections.abc.Mapping, exec_fun_kwds: collections.abc.Mapping) -> tuple[corrct.param_tuning.NDArrayFloat, corrct.solvers.SolutionInfo, corrct.param_tuning.PerfMeterTask]
:canonical: corrct.param_tuning._compute_reconstruction

```{autodoc2-docstring} corrct.param_tuning._compute_reconstruction
```
````

````{py:function} _parallel_compute(executor: concurrent.futures.Executor, init_fun: collections.abc.Callable | None, exec_fun: collections.abc.Callable[[typing.Any], tuple[corrct.param_tuning.NDArrayFloat, corrct.solvers.SolutionInfo]], hp_vals: float | collections.abc.Sequence[float] | corrct.param_tuning.NDArrayFloat, *, init_fun_kwds: collections.abc.Mapping | None = None, exec_fun_kwds: collections.abc.Mapping | None = None, verbose: bool = True) -> tuple[list[numpy.typing.NDArray], list[corrct.solvers.SolutionInfo], corrct.param_tuning.PerfMeterBatch]
:canonical: corrct.param_tuning._parallel_compute

```{autodoc2-docstring} corrct.param_tuning._parallel_compute
```
````

````{py:function} _serial_compute(init_fun: collections.abc.Callable | None, exec_fun: collections.abc.Callable[[typing.Any], tuple[corrct.param_tuning.NDArrayFloat, corrct.solvers.SolutionInfo]], hp_vals: float | collections.abc.Sequence[float] | corrct.param_tuning.NDArrayFloat, *, init_fun_kwds: collections.abc.Mapping | None = None, exec_fun_kwds: collections.abc.Mapping | None = None, verbose: bool = True) -> tuple[list[numpy.typing.NDArray], list[corrct.solvers.SolutionInfo], corrct.param_tuning.PerfMeterBatch]
:canonical: corrct.param_tuning._serial_compute

```{autodoc2-docstring} corrct.param_tuning._serial_compute
```
````

````{py:function} plot_cv_curves(solution_infos: list[corrct.solvers.SolutionInfo], hp_vals: collections.abc.Sequence[float]) -> None
:canonical: corrct.param_tuning.plot_cv_curves

```{autodoc2-docstring} corrct.param_tuning.plot_cv_curves
```
````

`````{py:class} BaseParameterTuning(dtype: numpy.typing.DTypeLike = np.float32, parallel_eval: concurrent.futures.Executor | int | bool = True, verbose: bool = False, plot_result: bool = False, print_timings: bool = False)
:canonical: corrct.param_tuning.BaseParameterTuning

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.__init__
```

````{py:attribute} _task_init_function
:canonical: corrct.param_tuning.BaseParameterTuning._task_init_function
:type: collections.abc.Callable | None
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning._task_init_function
```

````

````{py:attribute} _task_exec_function
:canonical: corrct.param_tuning.BaseParameterTuning._task_exec_function
:type: collections.abc.Callable[[typing.Any], tuple[corrct.param_tuning.NDArrayFloat, corrct.solvers.SolutionInfo]] | None
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning._task_exec_function
```

````

````{py:attribute} parallel_eval
:canonical: corrct.param_tuning.BaseParameterTuning.parallel_eval
:type: int | concurrent.futures.Executor
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.parallel_eval
```

````

````{py:attribute} dtype
:canonical: corrct.param_tuning.BaseParameterTuning.dtype
:type: numpy.typing.DTypeLike
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.dtype
```

````

````{py:attribute} verbose
:canonical: corrct.param_tuning.BaseParameterTuning.verbose
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.verbose
```

````

````{py:attribute} plot_result
:canonical: corrct.param_tuning.BaseParameterTuning.plot_result
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.plot_result
```

````

````{py:attribute} print_timings
:canonical: corrct.param_tuning.BaseParameterTuning.print_timings
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.print_timings
```

````

````{py:property} task_init_function
:canonical: corrct.param_tuning.BaseParameterTuning.task_init_function
:type: collections.abc.Callable | None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.task_init_function
```

````

````{py:property} task_exec_function
:canonical: corrct.param_tuning.BaseParameterTuning.task_exec_function
:type: collections.abc.Callable[[typing.Any], tuple[corrct.param_tuning.NDArrayFloat, corrct.solvers.SolutionInfo]]

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.task_exec_function
```

````

````{py:property} solver_spawning_function
:canonical: corrct.param_tuning.BaseParameterTuning.solver_spawning_function
:type: collections.abc.Callable | None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.solver_spawning_function
```

````

````{py:property} solver_calling_function
:canonical: corrct.param_tuning.BaseParameterTuning.solver_calling_function
:type: collections.abc.Callable[[typing.Any], tuple[corrct.param_tuning.NDArrayFloat, corrct.solvers.SolutionInfo]]

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.solver_calling_function
```

````

````{py:method} get_lambda_range(start: float, end: float, num_per_order: int = 4) -> corrct.param_tuning.NDArrayFloat
:canonical: corrct.param_tuning.BaseParameterTuning.get_lambda_range
:staticmethod:

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.get_lambda_range
```

````

````{py:method} process_hp_vals(hp_vals: float | collections.abc.Sequence[float] | corrct.param_tuning.NDArrayFloat, *, init_fun_kwds: collections.abc.Mapping | None = None, exec_fun_kwds: collections.abc.Mapping | None = None) -> tuple[list[numpy.typing.NDArray], list[corrct.solvers.SolutionInfo], corrct.param_tuning.PerfMeterBatch]
:canonical: corrct.param_tuning.BaseParameterTuning.process_hp_vals

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.process_hp_vals
```

````

````{py:method} compute_reconstruction_error(hp_vals: float | collections.abc.Sequence[float] | corrct.param_tuning.NDArrayFloat, gnd_truth: corrct.param_tuning.NDArrayFloat, *, init_fun_kwds: collections.abc.Mapping | None = None, exec_fun_kwds: collections.abc.Mapping | None = None) -> tuple[corrct.param_tuning.NDArrayFloat, corrct.param_tuning.NDArrayFloat]
:canonical: corrct.param_tuning.BaseParameterTuning.compute_reconstruction_error

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.compute_reconstruction_error
```

````

````{py:method} compute_loss_values(hp_vals: float | collections.abc.Sequence[float] | corrct.param_tuning.NDArrayFloat, *, init_fun_kwds: collections.abc.Mapping | None = None, exec_fun_kwds: collections.abc.Mapping | None = None, return_all: bool = False) -> corrct.param_tuning.NDArrayFloat | tuple[corrct.param_tuning.NDArrayFloat, list[corrct.param_tuning.NDArrayFloat], list[corrct.solvers.SolutionInfo], corrct.param_tuning.PerfMeterBatch]
:canonical: corrct.param_tuning.BaseParameterTuning.compute_loss_values
:abstractmethod:

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.compute_loss_values
```

````

`````

`````{py:class} LCurve(loss_function: collections.abc.Callable, data_dtype: numpy.typing.DTypeLike = np.float32, parallel_eval: concurrent.futures.Executor | int | bool = True, verbose: bool = False, plot_result: bool = False, print_timings: bool = False)
:canonical: corrct.param_tuning.LCurve

Bases: {py:obj}`corrct.param_tuning.BaseParameterTuning`

```{autodoc2-docstring} corrct.param_tuning.LCurve
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.param_tuning.LCurve.__init__
```

````{py:method} compute_loss_values(hp_vals: float | collections.abc.Sequence[float] | corrct.param_tuning.NDArrayFloat, *, init_fun_kwds: collections.abc.Mapping | None = None, exec_fun_kwds: collections.abc.Mapping | None = None, return_all: bool = False) -> corrct.param_tuning.NDArrayFloat | tuple[corrct.param_tuning.NDArrayFloat, list[corrct.param_tuning.NDArrayFloat], list[corrct.solvers.SolutionInfo], corrct.param_tuning.PerfMeterBatch]
:canonical: corrct.param_tuning.LCurve.compute_loss_values

```{autodoc2-docstring} corrct.param_tuning.LCurve.compute_loss_values
```

````

`````

`````{py:class} CrossValidation(data_shape: collections.abc.Sequence[int], cv_fraction: float | None = 0.1, num_averages: int = 5, mask_param_name: str = 'b_test_mask', parallel_eval: concurrent.futures.Executor | int | bool = True, dtype: numpy.typing.DTypeLike = np.float32, verbose: bool = False, plot_result: bool = False, print_timings: bool = False)
:canonical: corrct.param_tuning.CrossValidation

Bases: {py:obj}`corrct.param_tuning.BaseParameterTuning`

```{autodoc2-docstring} corrct.param_tuning.CrossValidation
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.__init__
```

````{py:attribute} data_shape
:canonical: corrct.param_tuning.CrossValidation.data_shape
:type: collections.abc.Sequence[int]
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.data_shape
```

````

````{py:attribute} cv_fraction
:canonical: corrct.param_tuning.CrossValidation.cv_fraction
:type: float | None
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.cv_fraction
```

````

````{py:attribute} num_averages
:canonical: corrct.param_tuning.CrossValidation.num_averages
:type: int
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.num_averages
```

````

````{py:attribute} data_cv_masks
:canonical: corrct.param_tuning.CrossValidation.data_cv_masks
:type: list[numpy.typing.NDArray]
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.data_cv_masks
```

````

````{py:attribute} mask_param_name
:canonical: corrct.param_tuning.CrossValidation.mask_param_name
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.mask_param_name
```

````

````{py:method} compute_loss_values(hp_vals: float | collections.abc.Sequence[float] | corrct.param_tuning.NDArrayFloat, *, init_fun_kwds: collections.abc.Mapping | None = None, exec_fun_kwds: collections.abc.Mapping | None = None, return_all: bool = False) -> tuple[corrct.param_tuning.NDArrayFloat, corrct.param_tuning.NDArrayFloat, list[corrct.param_tuning.NDArrayFloat]] | tuple[corrct.param_tuning.NDArrayFloat, corrct.param_tuning.NDArrayFloat, list[tuple[list[corrct.param_tuning.NDArrayFloat], list[corrct.solvers.SolutionInfo], corrct.param_tuning.PerfMeterBatch]]]
:canonical: corrct.param_tuning.CrossValidation.compute_loss_values

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.compute_loss_values
```

````

````{py:method} fit_loss_min(hp_vals: float | collections.abc.Sequence[float] | corrct.param_tuning.NDArrayFloat, f_vals: corrct.param_tuning.NDArrayFloat, f_stds: corrct.param_tuning.NDArrayFloat | None = None, scale: typing.Literal[linear, log] = 'log') -> tuple[float, float]
:canonical: corrct.param_tuning.CrossValidation.fit_loss_min

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.fit_loss_min
```

````

`````
