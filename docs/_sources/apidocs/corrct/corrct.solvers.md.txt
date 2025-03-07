# {py:mod}`corrct.solvers`

```{py:module} corrct.solvers
```

```{autodoc2-docstring} corrct.solvers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SolutionInfo <corrct.solvers.SolutionInfo>`
  - ```{autodoc2-docstring} corrct.solvers.SolutionInfo
    :summary:
    ```
* - {py:obj}`Solver <corrct.solvers.Solver>`
  - ```{autodoc2-docstring} corrct.solvers.Solver
    :summary:
    ```
* - {py:obj}`FBP <corrct.solvers.FBP>`
  - ```{autodoc2-docstring} corrct.solvers.FBP
    :summary:
    ```
* - {py:obj}`SART <corrct.solvers.SART>`
  - ```{autodoc2-docstring} corrct.solvers.SART
    :summary:
    ```
* - {py:obj}`MLEM <corrct.solvers.MLEM>`
  - ```{autodoc2-docstring} corrct.solvers.MLEM
    :summary:
    ```
* - {py:obj}`SIRT <corrct.solvers.SIRT>`
  - ```{autodoc2-docstring} corrct.solvers.SIRT
    :summary:
    ```
* - {py:obj}`PDHG <corrct.solvers.PDHG>`
  - ```{autodoc2-docstring} corrct.solvers.PDHG
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`eps <corrct.solvers.eps>`
  - ```{autodoc2-docstring} corrct.solvers.eps
    :summary:
    ```
* - {py:obj}`NDArrayFloat <corrct.solvers.NDArrayFloat>`
  - ```{autodoc2-docstring} corrct.solvers.NDArrayFloat
    :summary:
    ```
````

### API

````{py:data} eps
:canonical: corrct.solvers.eps
:value: >
   None

```{autodoc2-docstring} corrct.solvers.eps
```

````

````{py:data} NDArrayFloat
:canonical: corrct.solvers.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.solvers.NDArrayFloat
```

````

`````{py:class} SolutionInfo(method: str, max_iterations: int, tolerance: typing.Union[float, numpy.floating, None], residual0: float = np.inf, residual0_cv: float = np.inf)
:canonical: corrct.solvers.SolutionInfo

```{autodoc2-docstring} corrct.solvers.SolutionInfo
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.SolutionInfo.__init__
```

````{py:attribute} method
:canonical: corrct.solvers.SolutionInfo.method
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.method
```

````

````{py:attribute} iterations
:canonical: corrct.solvers.SolutionInfo.iterations
:type: int
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.iterations
```

````

````{py:attribute} max_iterations
:canonical: corrct.solvers.SolutionInfo.max_iterations
:type: int
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.max_iterations
```

````

````{py:attribute} residual0
:canonical: corrct.solvers.SolutionInfo.residual0
:type: typing.Union[float, numpy.floating]
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residual0
```

````

````{py:attribute} residual0_cv
:canonical: corrct.solvers.SolutionInfo.residual0_cv
:type: typing.Union[float, numpy.floating]
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residual0_cv
```

````

````{py:attribute} residuals
:canonical: corrct.solvers.SolutionInfo.residuals
:type: corrct.solvers.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residuals
```

````

````{py:attribute} residuals_cv
:canonical: corrct.solvers.SolutionInfo.residuals_cv
:type: corrct.solvers.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residuals_cv
```

````

````{py:attribute} tolerance
:canonical: corrct.solvers.SolutionInfo.tolerance
:type: typing.Union[float, numpy.floating, None]
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.tolerance
```

````

````{py:property} residuals_rel
:canonical: corrct.solvers.SolutionInfo.residuals_rel
:type: corrct.solvers.NDArrayFloat

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residuals_rel
```

````

````{py:property} residuals_cv_rel
:canonical: corrct.solvers.SolutionInfo.residuals_cv_rel
:type: corrct.solvers.NDArrayFloat

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residuals_cv_rel
```

````

`````

`````{py:class} Solver(verbose: bool = False, leave_progress: bool = True, relaxation: float = 1.0, tolerance: typing.Optional[float] = None, data_term: typing.Union[str, corrct.data_terms.DataFidelityBase] = 'l2', data_term_test: typing.Union[str, corrct.data_terms.DataFidelityBase, None] = None)
:canonical: corrct.solvers.Solver

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.solvers.Solver
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.Solver.__init__
```

````{py:method} info() -> str
:canonical: corrct.solvers.Solver.info

```{autodoc2-docstring} corrct.solvers.Solver.info
```

````

````{py:method} upper() -> str
:canonical: corrct.solvers.Solver.upper

```{autodoc2-docstring} corrct.solvers.Solver.upper
```

````

````{py:method} lower() -> str
:canonical: corrct.solvers.Solver.lower

```{autodoc2-docstring} corrct.solvers.Solver.lower
```

````

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, *args: typing.Any, **kwds: typing.Any) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.Solver.__call__
:abstractmethod:

```{autodoc2-docstring} corrct.solvers.Solver.__call__
```

````

````{py:method} _initialize_data_fidelity_function(data_term: typing.Union[str, corrct.data_terms.DataFidelityBase]) -> corrct.data_terms.DataFidelityBase
:canonical: corrct.solvers.Solver._initialize_data_fidelity_function
:staticmethod:

```{autodoc2-docstring} corrct.solvers.Solver._initialize_data_fidelity_function
```

````

````{py:method} _initialize_regularizer(regularizer: typing.Union[corrct.regularizers.BaseRegularizer, None, collections.abc.Sequence[corrct.regularizers.BaseRegularizer]]) -> collections.abc.Sequence[corrct.regularizers.BaseRegularizer]
:canonical: corrct.solvers.Solver._initialize_regularizer
:staticmethod:

```{autodoc2-docstring} corrct.solvers.Solver._initialize_regularizer
```

````

````{py:method} _initialize_b_masks(b: corrct.solvers.NDArrayFloat, b_mask: typing.Optional[corrct.solvers.NDArrayFloat], b_test_mask: typing.Optional[corrct.solvers.NDArrayFloat]) -> tuple[typing.Optional[corrct.solvers.NDArrayFloat], typing.Optional[corrct.solvers.NDArrayFloat]]
:canonical: corrct.solvers.Solver._initialize_b_masks
:staticmethod:

```{autodoc2-docstring} corrct.solvers.Solver._initialize_b_masks
```

````

`````

`````{py:class} FBP(verbose: bool = False, leave_progress: bool = False, regularizer: typing.Union[collections.abc.Sequence[corrct.regularizers.BaseRegularizer], corrct.regularizers.BaseRegularizer, None] = None, data_term: typing.Union[str, corrct.data_terms.DataFidelityBase] = 'l2', fbp_filter: typing.Union[str, corrct.solvers.NDArrayFloat, corrct.filters.Filter] = 'ramp', pad_mode: str = 'constant')
:canonical: corrct.solvers.FBP

Bases: {py:obj}`corrct.solvers.Solver`

```{autodoc2-docstring} corrct.solvers.FBP
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.FBP.__init__
```

````{py:method} info() -> str
:canonical: corrct.solvers.FBP.info

```{autodoc2-docstring} corrct.solvers.FBP.info
```

````

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int = 0, x0: typing.Optional[corrct.solvers.NDArrayFloat] = None, lower_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, upper_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, x_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None, b_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.FBP.__call__

```{autodoc2-docstring} corrct.solvers.FBP.__call__
```

````

`````

`````{py:class} SART(verbose: bool = False, leave_progress: bool = True, relaxation: float = 1.0, tolerance: typing.Optional[float] = None, data_term: typing.Union[str, corrct.data_terms.DataFidelityBase] = 'l2', data_term_test: typing.Union[str, corrct.data_terms.DataFidelityBase, None] = None)
:canonical: corrct.solvers.SART

Bases: {py:obj}`corrct.solvers.Solver`

```{autodoc2-docstring} corrct.solvers.SART
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.SART.__init__
```

````{py:method} compute_residual(A: typing.Callable, b: corrct.solvers.NDArrayFloat, x: corrct.solvers.NDArrayFloat, A_num_rows: int, b_mask: typing.Optional[corrct.solvers.NDArrayFloat]) -> corrct.solvers.NDArrayFloat
:canonical: corrct.solvers.SART.compute_residual

```{autodoc2-docstring} corrct.solvers.SART.compute_residual
```

````

````{py:method} __call__(A: typing.Union[typing.Callable[[numpy.typing.NDArray, int], numpy.typing.NDArray], corrct.projectors.ProjectorUncorrected], b: corrct.solvers.NDArrayFloat, iterations: int, A_num_rows: typing.Optional[int] = None, At: typing.Optional[typing.Callable] = None, x0: typing.Optional[corrct.solvers.NDArrayFloat] = None, lower_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, upper_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, x_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None, b_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.SART.__call__

```{autodoc2-docstring} corrct.solvers.SART.__call__
```

````

`````

`````{py:class} MLEM(verbose: bool = False, leave_progress: bool = True, tolerance: typing.Optional[float] = None, regularizer: typing.Union[collections.abc.Sequence[corrct.regularizers.BaseRegularizer], corrct.regularizers.BaseRegularizer, None] = None, data_term: typing.Union[str, corrct.data_terms.DataFidelityBase] = 'kl', data_term_test: typing.Union[str, corrct.data_terms.DataFidelityBase, None] = None)
:canonical: corrct.solvers.MLEM

Bases: {py:obj}`corrct.solvers.Solver`

```{autodoc2-docstring} corrct.solvers.MLEM
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.MLEM.__init__
```

````{py:method} info() -> str
:canonical: corrct.solvers.MLEM.info

```{autodoc2-docstring} corrct.solvers.MLEM.info
```

````

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int, x0: typing.Optional[corrct.solvers.NDArrayFloat] = None, lower_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, upper_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, x_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None, b_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None, b_test_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.MLEM.__call__

```{autodoc2-docstring} corrct.solvers.MLEM.__call__
```

````

`````

`````{py:class} SIRT(verbose: bool = False, leave_progress: bool = True, relaxation: float = 1.95, tolerance: typing.Optional[float] = None, regularizer: typing.Union[collections.abc.Sequence[corrct.regularizers.BaseRegularizer], corrct.regularizers.BaseRegularizer, None] = None, data_term: typing.Union[str, corrct.data_terms.DataFidelityBase] = 'l2', data_term_test: typing.Union[str, corrct.data_terms.DataFidelityBase, None] = None)
:canonical: corrct.solvers.SIRT

Bases: {py:obj}`corrct.solvers.Solver`

```{autodoc2-docstring} corrct.solvers.SIRT
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.SIRT.__init__
```

````{py:method} info() -> str
:canonical: corrct.solvers.SIRT.info

```{autodoc2-docstring} corrct.solvers.SIRT.info
```

````

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int, x0: typing.Optional[corrct.solvers.NDArrayFloat] = None, lower_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, upper_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, x_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None, b_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None, b_test_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.SIRT.__call__

```{autodoc2-docstring} corrct.solvers.SIRT.__call__
```

````

`````

`````{py:class} PDHG(verbose: bool = False, leave_progress: bool = True, tolerance: typing.Optional[float] = None, relaxation: float = 0.95, regularizer: typing.Union[collections.abc.Sequence[corrct.regularizers.BaseRegularizer], corrct.regularizers.BaseRegularizer, None] = None, data_term: typing.Union[str, corrct.data_terms.DataFidelityBase] = 'l2', data_term_test: typing.Union[str, corrct.data_terms.DataFidelityBase, None] = None)
:canonical: corrct.solvers.PDHG

Bases: {py:obj}`corrct.solvers.Solver`

```{autodoc2-docstring} corrct.solvers.PDHG
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.PDHG.__init__
```

````{py:method} info() -> str
:canonical: corrct.solvers.PDHG.info

```{autodoc2-docstring} corrct.solvers.PDHG.info
```

````

````{py:method} _initialize_data_fidelity_function(data_term: typing.Union[str, corrct.data_terms.DataFidelityBase])
:canonical: corrct.solvers.PDHG._initialize_data_fidelity_function
:staticmethod:

```{autodoc2-docstring} corrct.solvers.PDHG._initialize_data_fidelity_function
```

````

````{py:method} power_method(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int = 5) -> tuple[numpy.floating, collections.abc.Sequence[int], numpy.typing.DTypeLike]
:canonical: corrct.solvers.PDHG.power_method

```{autodoc2-docstring} corrct.solvers.PDHG.power_method
```

````

````{py:method} _get_data_sigma_tau_unpreconditioned(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat)
:canonical: corrct.solvers.PDHG._get_data_sigma_tau_unpreconditioned

```{autodoc2-docstring} corrct.solvers.PDHG._get_data_sigma_tau_unpreconditioned
```

````

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int, x0: typing.Optional[corrct.solvers.NDArrayFloat] = None, lower_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, upper_limit: typing.Union[float, corrct.solvers.NDArrayFloat, None] = None, x_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None, b_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None, b_test_mask: typing.Optional[corrct.solvers.NDArrayFloat] = None, precondition: bool = True) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.PDHG.__call__

```{autodoc2-docstring} corrct.solvers.PDHG.__call__
```

````

`````
