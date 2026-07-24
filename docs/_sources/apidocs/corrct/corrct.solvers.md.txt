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
* - {py:obj}`FISTA <corrct.solvers.FISTA>`
  - ```{autodoc2-docstring} corrct.solvers.FISTA
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`power_method <corrct.solvers.power_method>`
  - ```{autodoc2-docstring} corrct.solvers.power_method
    :summary:
    ```
* - {py:obj}`compute_diagonal_scaling <corrct.solvers.compute_diagonal_scaling>`
  - ```{autodoc2-docstring} corrct.solvers.compute_diagonal_scaling
    :summary:
    ```
* - {py:obj}`compute_Lipschitz_scaling <corrct.solvers.compute_Lipschitz_scaling>`
  - ```{autodoc2-docstring} corrct.solvers.compute_Lipschitz_scaling
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

````{py:function} power_method(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int = 5) -> tuple[float, tuple[int, ...], numpy.typing.DTypeLike]
:canonical: corrct.solvers.power_method

```{autodoc2-docstring} corrct.solvers.power_method
```
````

````{py:function} compute_diagonal_scaling(A_abs: corrct.operators.BaseTransform, At_abs: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, regs: collections.abc.Sequence[corrct.regularizers.BaseRegularizer], relaxation_sigma: float = 1.0, relaxation_tau: float = 1.0, x_mask: numpy.typing.NDArray | None = None, b_mask: numpy.typing.NDArray | None = None) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray, tuple[int, ...], numpy.typing.DTypeLike]
:canonical: corrct.solvers.compute_diagonal_scaling

```{autodoc2-docstring} corrct.solvers.compute_diagonal_scaling
```
````

````{py:function} compute_Lipschitz_scaling(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, regs: collections.abc.Sequence[corrct.regularizers.BaseRegularizer], relaxation_sigma: float = 1.0, relaxation_tau: float = 1.0) -> tuple[float, float | numpy.typing.NDArray, tuple[int, ...], numpy.typing.DTypeLike]
:canonical: corrct.solvers.compute_Lipschitz_scaling

```{autodoc2-docstring} corrct.solvers.compute_Lipschitz_scaling
```
````

`````{py:class} SolutionInfo(method: str, max_iterations: int, tolerance: float | None, residual0_rec: float = np.inf, residual0_val: float = np.inf)
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

````{py:attribute} residual0_rec
:canonical: corrct.solvers.SolutionInfo.residual0_rec
:type: float
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residual0_rec
```

````

````{py:attribute} residual0_val
:canonical: corrct.solvers.SolutionInfo.residual0_val
:type: float
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residual0_val
```

````

````{py:attribute} residuals_rec
:canonical: corrct.solvers.SolutionInfo.residuals_rec
:type: corrct.solvers.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residuals_rec
```

````

````{py:attribute} residuals_val
:canonical: corrct.solvers.SolutionInfo.residuals_val
:type: corrct.solvers.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residuals_val
```

````

````{py:attribute} tolerance
:canonical: corrct.solvers.SolutionInfo.tolerance
:type: float | None
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.tolerance
```

````

````{py:attribute} best_residual_ind_rec
:canonical: corrct.solvers.SolutionInfo.best_residual_ind_rec
:type: int
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.best_residual_ind_rec
```

````

````{py:attribute} best_residual_ind_val
:canonical: corrct.solvers.SolutionInfo.best_residual_ind_val
:type: int
:value: >
   None

```{autodoc2-docstring} corrct.solvers.SolutionInfo.best_residual_ind_val
```

````

````{py:property} residuals_rec_rel
:canonical: corrct.solvers.SolutionInfo.residuals_rec_rel
:type: corrct.solvers.NDArrayFloat

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residuals_rec_rel
```

````

````{py:property} residuals_val_rel
:canonical: corrct.solvers.SolutionInfo.residuals_val_rel
:type: corrct.solvers.NDArrayFloat

```{autodoc2-docstring} corrct.solvers.SolutionInfo.residuals_val_rel
```

````

````{py:method} set_residual_rec(res: float) -> None
:canonical: corrct.solvers.SolutionInfo.set_residual_rec

```{autodoc2-docstring} corrct.solvers.SolutionInfo.set_residual_rec
```

````

````{py:method} set_residual_val(res: float) -> None
:canonical: corrct.solvers.SolutionInfo.set_residual_val

```{autodoc2-docstring} corrct.solvers.SolutionInfo.set_residual_val
```

````

````{py:method} get_best_residual_rec(is_relative: bool = True) -> float
:canonical: corrct.solvers.SolutionInfo.get_best_residual_rec

```{autodoc2-docstring} corrct.solvers.SolutionInfo.get_best_residual_rec
```

````

````{py:method} get_best_residual_val(is_relative: bool = True) -> float
:canonical: corrct.solvers.SolutionInfo.get_best_residual_val

```{autodoc2-docstring} corrct.solvers.SolutionInfo.get_best_residual_val
```

````

````{py:method} __repr__() -> str
:canonical: corrct.solvers.SolutionInfo.__repr__

````

`````

`````{py:class} Solver(verbose: bool = False, leave_progress: bool = True, relaxation: float = 1.0, tolerance: float | None = None, data_term: str | corrct.data_terms.DataFidelityBase = 'l2', data_term_val: str | corrct.data_terms.DataFidelityBase | None = None, criterion: typing.Literal[max_iter, loss_rec, loss_val] = 'max_iter')
:canonical: corrct.solvers.Solver

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.solvers.Solver
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.Solver.__init__
```

````{py:attribute} verbose
:canonical: corrct.solvers.Solver.verbose
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.solvers.Solver.verbose
```

````

````{py:attribute} leave_progress
:canonical: corrct.solvers.Solver.leave_progress
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.solvers.Solver.leave_progress
```

````

````{py:attribute} relaxation
:canonical: corrct.solvers.Solver.relaxation
:type: float
:value: >
   None

```{autodoc2-docstring} corrct.solvers.Solver.relaxation
```

````

````{py:attribute} tolerance
:canonical: corrct.solvers.Solver.tolerance
:type: float | None
:value: >
   None

```{autodoc2-docstring} corrct.solvers.Solver.tolerance
```

````

````{py:attribute} criterion
:canonical: corrct.solvers.Solver.criterion
:type: typing.Literal[max_iter, loss_rec, loss_val]
:value: >
   None

```{autodoc2-docstring} corrct.solvers.Solver.criterion
```

````

````{py:attribute} data_term
:canonical: corrct.solvers.Solver.data_term
:type: corrct.data_terms.DataFidelityBase
:value: >
   None

```{autodoc2-docstring} corrct.solvers.Solver.data_term
```

````

````{py:attribute} data_term_val
:canonical: corrct.solvers.Solver.data_term_val
:type: corrct.data_terms.DataFidelityBase
:value: >
   None

```{autodoc2-docstring} corrct.solvers.Solver.data_term_val
```

````

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

````{py:method} _initialize_data_fidelity_function(data_term: str | corrct.data_terms.DataFidelityBase) -> corrct.data_terms.DataFidelityBase
:canonical: corrct.solvers.Solver._initialize_data_fidelity_function
:staticmethod:

```{autodoc2-docstring} corrct.solvers.Solver._initialize_data_fidelity_function
```

````

````{py:method} _initialize_regularizer(regularizer: corrct.regularizers.BaseRegularizer | None | collections.abc.Sequence[corrct.regularizers.BaseRegularizer]) -> collections.abc.Sequence[corrct.regularizers.BaseRegularizer]
:canonical: corrct.solvers.Solver._initialize_regularizer
:staticmethod:

```{autodoc2-docstring} corrct.solvers.Solver._initialize_regularizer
```

````

````{py:method} _initialize_b_masks(b: corrct.solvers.NDArrayFloat, b_mask: corrct.solvers.NDArrayFloat | None, b_val_mask: corrct.solvers.NDArrayFloat | None) -> tuple[corrct.solvers.NDArrayFloat | None, corrct.solvers.NDArrayFloat | None]
:canonical: corrct.solvers.Solver._initialize_b_masks
:staticmethod:

```{autodoc2-docstring} corrct.solvers.Solver._initialize_b_masks
```

````

````{py:method} _check_require_residual(b_val_mask: corrct.solvers.NDArrayFloat | None) -> bool
:canonical: corrct.solvers.Solver._check_require_residual

```{autodoc2-docstring} corrct.solvers.Solver._check_require_residual
```

````

````{py:method} _select_best_solution(info: corrct.solvers.SolutionInfo, curr_best_x: numpy.typing.NDArray, new_x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.solvers.Solver._select_best_solution

```{autodoc2-docstring} corrct.solvers.Solver._select_best_solution
```

````

`````

`````{py:class} FBP(verbose: bool = False, leave_progress: bool = False, regularizer: collections.abc.Sequence[corrct.regularizers.BaseRegularizer] | corrct.regularizers.BaseRegularizer | None = None, data_term: str | corrct.data_terms.DataFidelityBase = 'l2', fbp_filter: str | corrct.solvers.NDArrayFloat | corrct.filters.Filter = 'ramp', pad_mode: str = 'constant')
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

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int = 0, x0: corrct.solvers.NDArrayFloat | None = None, lower_limit: float | corrct.solvers.NDArrayFloat | None = None, upper_limit: float | corrct.solvers.NDArrayFloat | None = None, x_mask: corrct.solvers.NDArrayFloat | None = None, b_mask: corrct.solvers.NDArrayFloat | None = None) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.FBP.__call__

```{autodoc2-docstring} corrct.solvers.FBP.__call__
```

````

`````

`````{py:class} SART(verbose: bool = False, leave_progress: bool = True, relaxation: float = 1.0, tolerance: float | None = None, data_term: str | corrct.data_terms.DataFidelityBase = 'l2', data_term_val: str | corrct.data_terms.DataFidelityBase | None = None, criterion: typing.Literal[max_iter, loss_rec, loss_val] = 'max_iter')
:canonical: corrct.solvers.SART

Bases: {py:obj}`corrct.solvers.Solver`

```{autodoc2-docstring} corrct.solvers.SART
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.SART.__init__
```

````{py:method} compute_residual(A: collections.abc.Callable, b: corrct.solvers.NDArrayFloat, x: corrct.solvers.NDArrayFloat, A_num_rows: int, b_mask: corrct.solvers.NDArrayFloat | None) -> corrct.solvers.NDArrayFloat
:canonical: corrct.solvers.SART.compute_residual

```{autodoc2-docstring} corrct.solvers.SART.compute_residual
```

````

````{py:method} __call__(A: collections.abc.Callable[[numpy.typing.NDArray, int], numpy.typing.NDArray] | corrct.projectors.ProjectorUncorrected, b: corrct.solvers.NDArrayFloat, iterations: int, A_num_rows: int | None = None, At: collections.abc.Callable | None = None, x0: corrct.solvers.NDArrayFloat | None = None, lower_limit: float | corrct.solvers.NDArrayFloat | None = None, upper_limit: float | corrct.solvers.NDArrayFloat | None = None, x_mask: corrct.solvers.NDArrayFloat | None = None, b_mask: corrct.solvers.NDArrayFloat | None = None) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.SART.__call__

```{autodoc2-docstring} corrct.solvers.SART.__call__
```

````

`````

`````{py:class} MLEM(verbose: bool = False, leave_progress: bool = True, tolerance: float | None = None, regularizer: collections.abc.Sequence[corrct.regularizers.BaseRegularizer] | corrct.regularizers.BaseRegularizer | None = None, data_term: str | corrct.data_terms.DataFidelityBase = 'kl', data_term_val: str | corrct.data_terms.DataFidelityBase | None = None, criterion: typing.Literal[max_iter, loss_rec, loss_val] = 'max_iter')
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

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int, x0: corrct.solvers.NDArrayFloat | None = None, lower_limit: float | corrct.solvers.NDArrayFloat | None = None, upper_limit: float | corrct.solvers.NDArrayFloat | None = None, x_mask: corrct.solvers.NDArrayFloat | None = None, b_mask: corrct.solvers.NDArrayFloat | None = None, b_val_mask: corrct.solvers.NDArrayFloat | None = None) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.MLEM.__call__

```{autodoc2-docstring} corrct.solvers.MLEM.__call__
```

````

`````

`````{py:class} SIRT(verbose: bool = False, leave_progress: bool = True, relaxation: float = 1.95, tolerance: float | None = None, regularizer: collections.abc.Sequence[corrct.regularizers.BaseRegularizer] | corrct.regularizers.BaseRegularizer | None = None, data_term: str | corrct.data_terms.DataFidelityBase = 'l2', data_term_val: str | corrct.data_terms.DataFidelityBase | None = None, criterion: typing.Literal[max_iter, loss_rec, loss_val] = 'max_iter')
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

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int, x0: corrct.solvers.NDArrayFloat | None = None, lower_limit: float | corrct.solvers.NDArrayFloat | None = None, upper_limit: float | corrct.solvers.NDArrayFloat | None = None, x_mask: corrct.solvers.NDArrayFloat | None = None, b_mask: corrct.solvers.NDArrayFloat | None = None, b_val_mask: corrct.solvers.NDArrayFloat | None = None) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.SIRT.__call__

```{autodoc2-docstring} corrct.solvers.SIRT.__call__
```

````

`````

`````{py:class} PDHG(verbose: bool = False, leave_progress: bool = True, tolerance: float | None = None, relaxation: float = 0.95, regularizer: collections.abc.Sequence[corrct.regularizers.BaseRegularizer] | corrct.regularizers.BaseRegularizer | None = None, data_term: str | corrct.data_terms.DataFidelityBase = 'l2', data_term_val: str | corrct.data_terms.DataFidelityBase | None = None, criterion: typing.Literal[max_iter, loss_rec, loss_val] = 'max_iter')
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

````{py:method} _initialize_data_fidelity_function(data_term: str | corrct.data_terms.DataFidelityBase)
:canonical: corrct.solvers.PDHG._initialize_data_fidelity_function
:staticmethod:

```{autodoc2-docstring} corrct.solvers.PDHG._initialize_data_fidelity_function
```

````

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int, x0: corrct.solvers.NDArrayFloat | None = None, lower_limit: float | corrct.solvers.NDArrayFloat | None = None, upper_limit: float | corrct.solvers.NDArrayFloat | None = None, x_mask: corrct.solvers.NDArrayFloat | None = None, b_mask: corrct.solvers.NDArrayFloat | None = None, b_val_mask: corrct.solvers.NDArrayFloat | None = None, precondition: bool = True) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.PDHG.__call__

```{autodoc2-docstring} corrct.solvers.PDHG.__call__
```

````

`````

`````{py:class} FISTA(verbose: bool = False, leave_progress: bool = True, tolerance: float | None = None, relaxation: float = 1.0, regularizer: corrct.regularizers.BaseRegularizer | None = None, data_term: str | corrct.data_terms.DataFidelityBase = 'l2', data_term_val: str | corrct.data_terms.DataFidelityBase | None = None, criterion: typing.Literal[max_iter, loss_rec, loss_val] = 'max_iter', restart_period: int | None = None)
:canonical: corrct.solvers.FISTA

Bases: {py:obj}`corrct.solvers.Solver`

```{autodoc2-docstring} corrct.solvers.FISTA
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.solvers.FISTA.__init__
```

````{py:method} info() -> str
:canonical: corrct.solvers.FISTA.info

```{autodoc2-docstring} corrct.solvers.FISTA.info
```

````

````{py:method} _initialize_data_fidelity_function(data_term: str | corrct.data_terms.DataFidelityBase)
:canonical: corrct.solvers.FISTA._initialize_data_fidelity_function
:staticmethod:

```{autodoc2-docstring} corrct.solvers.FISTA._initialize_data_fidelity_function
```

````

````{py:method} __call__(A: corrct.operators.BaseTransform, b: corrct.solvers.NDArrayFloat, iterations: int, x0: corrct.solvers.NDArrayFloat | None = None, lower_limit: float | corrct.solvers.NDArrayFloat | None = None, upper_limit: float | corrct.solvers.NDArrayFloat | None = None, x_mask: corrct.solvers.NDArrayFloat | None = None, b_mask: corrct.solvers.NDArrayFloat | None = None, b_val_mask: corrct.solvers.NDArrayFloat | None = None, precondition: bool = True) -> tuple[corrct.solvers.NDArrayFloat, corrct.solvers.SolutionInfo]
:canonical: corrct.solvers.FISTA.__call__

```{autodoc2-docstring} corrct.solvers.FISTA.__call__
```

````

`````
