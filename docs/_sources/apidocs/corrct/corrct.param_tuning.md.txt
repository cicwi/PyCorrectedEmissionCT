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

* - {py:obj}`create_random_test_mask <corrct.param_tuning.create_random_test_mask>`
  - ```{autodoc2-docstring} corrct.param_tuning.create_random_test_mask
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
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`num_threads <corrct.param_tuning.num_threads>`
  - ```{autodoc2-docstring} corrct.param_tuning.num_threads
    :summary:
    ```
* - {py:obj}`NDArrayFloat <corrct.param_tuning.NDArrayFloat>`
  - ```{autodoc2-docstring} corrct.param_tuning.NDArrayFloat
    :summary:
    ```
````

### API

````{py:data} num_threads
:canonical: corrct.param_tuning.num_threads
:value: >
   'round(...)'

```{autodoc2-docstring} corrct.param_tuning.num_threads
```

````

````{py:data} NDArrayFloat
:canonical: corrct.param_tuning.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.NDArrayFloat
```

````

````{py:function} create_random_test_mask(data_shape: typing.Sequence[int], test_fraction: float = 0.05, dtype: numpy.typing.DTypeLike = np.float32) -> corrct.param_tuning.NDArrayFloat
:canonical: corrct.param_tuning.create_random_test_mask

```{autodoc2-docstring} corrct.param_tuning.create_random_test_mask
```
````

````{py:function} get_lambda_range(start: float, end: float, num_per_order: int = 4, aligned_order: bool = True) -> corrct.param_tuning.NDArrayFloat
:canonical: corrct.param_tuning.get_lambda_range

```{autodoc2-docstring} corrct.param_tuning.get_lambda_range
```
````

````{py:function} fit_func_min(hp_vals: typing.Union[numpy.typing.ArrayLike, corrct.param_tuning.NDArrayFloat], f_vals: corrct.param_tuning.NDArrayFloat, f_stds: typing.Optional[corrct.param_tuning.NDArrayFloat] = None, scale: typing.Literal[linear, log] = 'log', verbose: bool = False, plot_result: bool = False) -> tuple[float, float]
:canonical: corrct.param_tuning.fit_func_min

```{autodoc2-docstring} corrct.param_tuning.fit_func_min
```
````

`````{py:class} BaseParameterTuning(dtype: numpy.typing.DTypeLike = np.float32, parallel_eval: bool = True, verbose: bool = False, plot_result: bool = False)
:canonical: corrct.param_tuning.BaseParameterTuning

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.__init__
```

````{py:attribute} _solver_spawning_functionls
:canonical: corrct.param_tuning.BaseParameterTuning._solver_spawning_functionls
:type: typing.Optional[typing.Callable]
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning._solver_spawning_functionls
```

````

````{py:attribute} _solver_calling_function
:canonical: corrct.param_tuning.BaseParameterTuning._solver_calling_function
:type: typing.Optional[typing.Callable[[typing.Any], tuple[corrct.param_tuning.NDArrayFloat, corrct.solvers.SolutionInfo]]]
:value: >
   None

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning._solver_calling_function
```

````

````{py:property} solver_spawning_function
:canonical: corrct.param_tuning.BaseParameterTuning.solver_spawning_function
:type: typing.Callable

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.solver_spawning_function
```

````

````{py:property} solver_calling_function
:canonical: corrct.param_tuning.BaseParameterTuning.solver_calling_function
:type: typing.Callable[[typing.Any, ...], tuple[corrct.param_tuning.NDArrayFloat, corrct.solvers.SolutionInfo]]

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.solver_calling_function
```

````

````{py:method} get_lambda_range(start: float, end: float, num_per_order: int = 4) -> corrct.param_tuning.NDArrayFloat
:canonical: corrct.param_tuning.BaseParameterTuning.get_lambda_range
:staticmethod:

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.get_lambda_range
```

````

````{py:method} compute_reconstruction_and_loss(hp_val: float, *args: typing.Any, **kwds: typing.Any) -> tuple[float, corrct.param_tuning.NDArrayFloat]
:canonical: corrct.param_tuning.BaseParameterTuning.compute_reconstruction_and_loss

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.compute_reconstruction_and_loss
```

````

````{py:method} compute_reconstruction_error(hp_vals: typing.Union[numpy.typing.ArrayLike, corrct.param_tuning.NDArrayFloat], gnd_truth: corrct.param_tuning.NDArrayFloat) -> tuple[corrct.param_tuning.NDArrayFloat, corrct.param_tuning.NDArrayFloat]
:canonical: corrct.param_tuning.BaseParameterTuning.compute_reconstruction_error

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.compute_reconstruction_error
```

````

````{py:method} compute_loss_values(hp_vals: typing.Union[numpy.typing.ArrayLike, corrct.param_tuning.NDArrayFloat]) -> corrct.param_tuning.NDArrayFloat
:canonical: corrct.param_tuning.BaseParameterTuning.compute_loss_values
:abstractmethod:

```{autodoc2-docstring} corrct.param_tuning.BaseParameterTuning.compute_loss_values
```

````

`````

`````{py:class} LCurve(loss_function: typing.Callable, data_dtype: numpy.typing.DTypeLike = np.float32, parallel_eval: bool = True, verbose: bool = False, plot_result: bool = False)
:canonical: corrct.param_tuning.LCurve

Bases: {py:obj}`corrct.param_tuning.BaseParameterTuning`

```{autodoc2-docstring} corrct.param_tuning.LCurve
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.param_tuning.LCurve.__init__
```

````{py:method} compute_loss_values(hp_vals: typing.Union[numpy.typing.ArrayLike, corrct.param_tuning.NDArrayFloat]) -> corrct.param_tuning.NDArrayFloat
:canonical: corrct.param_tuning.LCurve.compute_loss_values

```{autodoc2-docstring} corrct.param_tuning.LCurve.compute_loss_values
```

````

`````

`````{py:class} CrossValidation(data_shape: typing.Sequence[int], dtype: numpy.typing.DTypeLike = np.float32, cv_fraction: float = 0.1, num_averages: int = 7, parallel_eval: bool = True, verbose: bool = False, plot_result: bool = False)
:canonical: corrct.param_tuning.CrossValidation

Bases: {py:obj}`corrct.param_tuning.BaseParameterTuning`

```{autodoc2-docstring} corrct.param_tuning.CrossValidation
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.__init__
```

````{py:method} _create_random_test_mask() -> corrct.param_tuning.NDArrayFloat
:canonical: corrct.param_tuning.CrossValidation._create_random_test_mask

```{autodoc2-docstring} corrct.param_tuning.CrossValidation._create_random_test_mask
```

````

````{py:method} compute_loss_values(hp_vals: typing.Union[numpy.typing.ArrayLike, corrct.param_tuning.NDArrayFloat]) -> tuple[corrct.param_tuning.NDArrayFloat, corrct.param_tuning.NDArrayFloat, corrct.param_tuning.NDArrayFloat]
:canonical: corrct.param_tuning.CrossValidation.compute_loss_values

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.compute_loss_values
```

````

````{py:method} fit_loss_min(hp_vals: typing.Union[numpy.typing.ArrayLike, corrct.param_tuning.NDArrayFloat], f_vals: corrct.param_tuning.NDArrayFloat, f_stds: typing.Optional[corrct.param_tuning.NDArrayFloat] = None, scale: typing.Literal[linear, log] = 'log') -> tuple[float, float]
:canonical: corrct.param_tuning.CrossValidation.fit_loss_min

```{autodoc2-docstring} corrct.param_tuning.CrossValidation.fit_loss_min
```

````

`````
