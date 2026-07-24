# {py:mod}`corrct.data_terms`

```{py:module} corrct.data_terms
```

```{autodoc2-docstring} corrct.data_terms
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DataFidelityBase <corrct.data_terms.DataFidelityBase>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelityBase
    :summary:
    ```
* - {py:obj}`DataFidelity_l2 <corrct.data_terms.DataFidelity_l2>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2
    :summary:
    ```
* - {py:obj}`DataFidelity_l2w <corrct.data_terms.DataFidelity_l2w>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2w
    :summary:
    ```
* - {py:obj}`DataFidelity_l2b <corrct.data_terms.DataFidelity_l2b>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2b
    :summary:
    ```
* - {py:obj}`DataFidelity_Huber <corrct.data_terms.DataFidelity_Huber>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_Huber
    :summary:
    ```
* - {py:obj}`DataFidelity_l1 <corrct.data_terms.DataFidelity_l1>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1
    :summary:
    ```
* - {py:obj}`DataFidelity_l21 <corrct.data_terms.DataFidelity_l21>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21
    :summary:
    ```
* - {py:obj}`DataFidelity_l21w <corrct.data_terms.DataFidelity_l21w>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21w
    :summary:
    ```
* - {py:obj}`DataFidelity_l1b <corrct.data_terms.DataFidelity_l1b>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1b
    :summary:
    ```
* - {py:obj}`DataFidelity_KL <corrct.data_terms.DataFidelity_KL>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_KL
    :summary:
    ```
* - {py:obj}`DataFidelity_ln <corrct.data_terms.DataFidelity_ln>`
  - ```{autodoc2-docstring} corrct.data_terms.DataFidelity_ln
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_soft_threshold <corrct.data_terms._soft_threshold>`
  - ```{autodoc2-docstring} corrct.data_terms._soft_threshold
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`eps <corrct.data_terms.eps>`
  - ```{autodoc2-docstring} corrct.data_terms.eps
    :summary:
    ```
* - {py:obj}`NDArrayFloat <corrct.data_terms.NDArrayFloat>`
  - ```{autodoc2-docstring} corrct.data_terms.NDArrayFloat
    :summary:
    ```
````

### API

````{py:data} eps
:canonical: corrct.data_terms.eps
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.eps
```

````

````{py:data} NDArrayFloat
:canonical: corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.NDArrayFloat
```

````

````{py:function} _soft_threshold(values: corrct.data_terms.NDArrayFloat, threshold: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms._soft_threshold

```{autodoc2-docstring} corrct.data_terms._soft_threshold
```
````

`````{py:class} DataFidelityBase(background: float | corrct.data_terms.NDArrayFloat | None = None)
:canonical: corrct.data_terms.DataFidelityBase

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.__init__
```

````{py:attribute} data
:canonical: corrct.data_terms.DataFidelityBase.data
:type: corrct.data_terms.NDArrayFloat | None
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.data
```

````

````{py:attribute} sigma
:canonical: corrct.data_terms.DataFidelityBase.sigma
:type: float | corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.sigma
```

````

````{py:attribute} background
:canonical: corrct.data_terms.DataFidelityBase.background
:type: corrct.data_terms.NDArrayFloat | None
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.background
```

````

````{py:attribute} sigma_data
:canonical: corrct.data_terms.DataFidelityBase.sigma_data
:type: corrct.data_terms.NDArrayFloat | None
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.sigma_data
```

````

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelityBase.__data_fidelity_name__
:value: <Multiline-String>

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.__data_fidelity_name__
```

````

````{py:method} _slice_attr(attr: str, ind: typing.Any) -> None
:canonical: corrct.data_terms.DataFidelityBase._slice_attr

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase._slice_attr
```

````

````{py:method} __getitem__(ind: typing.Any) -> corrct.data_terms.DataFidelityBase
:canonical: corrct.data_terms.DataFidelityBase.__getitem__

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.__getitem__
```

````

````{py:method} info() -> str
:canonical: corrct.data_terms.DataFidelityBase.info

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.info
```

````

````{py:method} upper() -> str
:canonical: corrct.data_terms.DataFidelityBase.upper

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.upper
```

````

````{py:method} lower() -> str
:canonical: corrct.data_terms.DataFidelityBase.lower

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.lower
```

````

````{py:method} assign_data(data: float | corrct.data_terms.NDArrayFloat | None = None, sigma: float | corrct.data_terms.NDArrayFloat = 1.0) -> None
:canonical: corrct.data_terms.DataFidelityBase.assign_data

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.assign_data
```

````

````{py:method} compute_residual(proj_primal: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None) -> corrct.data_terms.NDArrayFloat
:canonical: corrct.data_terms.DataFidelityBase.compute_residual

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.compute_residual
```

````

````{py:method} compute_residual_norm(dual: corrct.data_terms.NDArrayFloat) -> float
:canonical: corrct.data_terms.DataFidelityBase.compute_residual_norm
:abstractmethod:

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.compute_residual_norm
```

````

````{py:method} _compute_sigma_data()
:canonical: corrct.data_terms.DataFidelityBase._compute_sigma_data

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase._compute_sigma_data
```

````

````{py:method} compute_data_dual_dot(dual: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None) -> float
:canonical: corrct.data_terms.DataFidelityBase.compute_data_dual_dot

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.compute_data_dual_dot
```

````

````{py:method} initialize_dual() -> corrct.data_terms.NDArrayFloat
:canonical: corrct.data_terms.DataFidelityBase.initialize_dual

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.initialize_dual
```

````

````{py:method} update_dual(dual: corrct.data_terms.NDArrayFloat, proj_primal: corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelityBase.update_dual

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.update_dual
```

````

````{py:method} apply_proximal_dual(dual: corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelityBase.apply_proximal_dual
:abstractmethod:

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.apply_proximal_dual
```

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelityBase.apply_proximal_primal
:abstractmethod:

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.apply_proximal_primal
```

````

````{py:method} compute_primal_dual_gap(proj_primal: corrct.data_terms.NDArrayFloat, dual: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None) -> float
:canonical: corrct.data_terms.DataFidelityBase.compute_primal_dual_gap
:abstractmethod:

```{autodoc2-docstring} corrct.data_terms.DataFidelityBase.compute_primal_dual_gap
```

````

`````

`````{py:class} DataFidelity_l2(background: float | corrct.data_terms.NDArrayFloat | None = None)
:canonical: corrct.data_terms.DataFidelity_l2

Bases: {py:obj}`corrct.data_terms.DataFidelityBase`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_l2.__data_fidelity_name__
:value: >
   'l2'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2.__data_fidelity_name__
```

````

````{py:attribute} sigma1
:canonical: corrct.data_terms.DataFidelity_l2.sigma1
:type: float | corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2.sigma1
```

````

````{py:method} assign_data(data: float | corrct.data_terms.NDArrayFloat | None = None, sigma: float | corrct.data_terms.NDArrayFloat = 1.0) -> None
:canonical: corrct.data_terms.DataFidelity_l2.assign_data

````

````{py:method} compute_residual_norm(dual: corrct.data_terms.NDArrayFloat) -> float
:canonical: corrct.data_terms.DataFidelity_l2.compute_residual_norm

````

````{py:method} apply_proximal_dual(dual: corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_l2.apply_proximal_dual

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_l2.apply_proximal_primal

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2.apply_proximal_primal
```

````

````{py:method} compute_primal_dual_gap(proj_primal: corrct.data_terms.NDArrayFloat, dual: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None) -> float
:canonical: corrct.data_terms.DataFidelity_l2.compute_primal_dual_gap

````

`````

`````{py:class} DataFidelity_l2w(weights: float | corrct.data_terms.NDArrayFloat, background: float | corrct.data_terms.NDArrayFloat | None = None)
:canonical: corrct.data_terms.DataFidelity_l2w

Bases: {py:obj}`corrct.data_terms.DataFidelity_l2`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2w
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2w.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_l2w.__data_fidelity_name__
:value: >
   'l2w'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2w.__data_fidelity_name__
```

````

````{py:attribute} sigma1
:canonical: corrct.data_terms.DataFidelity_l2w.sigma1
:type: float | corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2w.sigma1
```

````

````{py:attribute} weights
:canonical: corrct.data_terms.DataFidelity_l2w.weights
:type: corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2w.weights
```

````

````{py:method} assign_data(data: float | corrct.data_terms.NDArrayFloat | None, sigma: float | corrct.data_terms.NDArrayFloat = 1.0)
:canonical: corrct.data_terms.DataFidelity_l2w.assign_data

````

````{py:method} compute_residual(proj_primal, mask: float | corrct.data_terms.NDArrayFloat | None = None)
:canonical: corrct.data_terms.DataFidelity_l2w.compute_residual

````

````{py:method} compute_residual_norm(dual: float | corrct.data_terms.NDArrayFloat) -> float
:canonical: corrct.data_terms.DataFidelity_l2w.compute_residual_norm

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_l2w.apply_proximal_primal

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2w.apply_proximal_primal
```

````

`````

`````{py:class} DataFidelity_l2b(local_error: float | corrct.data_terms.NDArrayFloat, background: float | corrct.data_terms.NDArrayFloat | None = None)
:canonical: corrct.data_terms.DataFidelity_l2b

Bases: {py:obj}`corrct.data_terms.DataFidelity_l2`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2b
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2b.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_l2b.__data_fidelity_name__
:value: >
   'l2b'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2b.__data_fidelity_name__
```

````

````{py:attribute} sigma1
:canonical: corrct.data_terms.DataFidelity_l2b.sigma1
:type: float | corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2b.sigma1
```

````

````{py:attribute} sigma_error
:canonical: corrct.data_terms.DataFidelity_l2b.sigma_error
:type: float | corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2b.sigma_error
```

````

````{py:attribute} sigma_sqrt_error
:canonical: corrct.data_terms.DataFidelity_l2b.sigma_sqrt_error
:type: float | corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2b.sigma_sqrt_error
```

````

````{py:method} assign_data(data: float | corrct.data_terms.NDArrayFloat | None, sigma: float | corrct.data_terms.NDArrayFloat = 1.0)
:canonical: corrct.data_terms.DataFidelity_l2b.assign_data

````

````{py:method} compute_residual(proj_primal: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None) -> corrct.data_terms.NDArrayFloat
:canonical: corrct.data_terms.DataFidelity_l2b.compute_residual

````

````{py:method} apply_proximal_dual(dual: corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_l2b.apply_proximal_dual

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_l2b.apply_proximal_primal

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l2b.apply_proximal_primal
```

````

````{py:method} compute_primal_dual_gap(proj_primal: corrct.data_terms.NDArrayFloat, dual: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None) -> float
:canonical: corrct.data_terms.DataFidelity_l2b.compute_primal_dual_gap

````

`````

`````{py:class} DataFidelity_Huber(local_error: float | corrct.data_terms.NDArrayFloat, background: float | corrct.data_terms.NDArrayFloat | None = None, l2_axis: int | None = None)
:canonical: corrct.data_terms.DataFidelity_Huber

Bases: {py:obj}`corrct.data_terms.DataFidelityBase`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_Huber
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_Huber.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_Huber.__data_fidelity_name__
:value: >
   'Hub'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_Huber.__data_fidelity_name__
```

````

````{py:attribute} one_sigma_error
:canonical: corrct.data_terms.DataFidelity_Huber.one_sigma_error
:type: float | corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_Huber.one_sigma_error
```

````

````{py:method} assign_data(data: corrct.data_terms.NDArrayFloat, sigma: float | corrct.data_terms.NDArrayFloat = 1.0)
:canonical: corrct.data_terms.DataFidelity_Huber.assign_data

````

````{py:method} compute_residual_norm(dual: corrct.data_terms.NDArrayFloat) -> float
:canonical: corrct.data_terms.DataFidelity_Huber.compute_residual_norm

````

````{py:method} apply_proximal_dual(dual: corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_Huber.apply_proximal_dual

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_Huber.apply_proximal_primal
:abstractmethod:

```{autodoc2-docstring} corrct.data_terms.DataFidelity_Huber.apply_proximal_primal
```

````

````{py:method} compute_primal_dual_gap(proj_primal: corrct.data_terms.NDArrayFloat, dual: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None) -> float
:canonical: corrct.data_terms.DataFidelity_Huber.compute_primal_dual_gap

````

`````

`````{py:class} DataFidelity_l1(background=None)
:canonical: corrct.data_terms.DataFidelity_l1

Bases: {py:obj}`corrct.data_terms.DataFidelityBase`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_l1.__data_fidelity_name__
:value: >
   'l1'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1.__data_fidelity_name__
```

````

````{py:method} _get_inner_norm(dual: corrct.data_terms.NDArrayFloat) -> corrct.data_terms.NDArrayFloat
:canonical: corrct.data_terms.DataFidelity_l1._get_inner_norm

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1._get_inner_norm
```

````

````{py:method} _apply_threshold(dual: corrct.data_terms.NDArrayFloat)
:canonical: corrct.data_terms.DataFidelity_l1._apply_threshold

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1._apply_threshold
```

````

````{py:method} apply_proximal_dual(dual: corrct.data_terms.NDArrayFloat, weight: float | corrct.data_terms.NDArrayFloat = 1.0)
:canonical: corrct.data_terms.DataFidelity_l1.apply_proximal_dual

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_l1.apply_proximal_primal

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1.apply_proximal_primal
```

````

````{py:method} compute_residual_norm(dual: corrct.data_terms.NDArrayFloat) -> float
:canonical: corrct.data_terms.DataFidelity_l1.compute_residual_norm

````

````{py:method} compute_primal_dual_gap(proj_primal: corrct.data_terms.NDArrayFloat, dual: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None) -> float
:canonical: corrct.data_terms.DataFidelity_l1.compute_primal_dual_gap

````

`````

`````{py:class} DataFidelity_l21(background: float | corrct.data_terms.NDArrayFloat | None = None, l2_axis: int = 0)
:canonical: corrct.data_terms.DataFidelity_l21

Bases: {py:obj}`corrct.data_terms.DataFidelity_l1`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_l21.__data_fidelity_name__
:value: >
   'l12'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21.__data_fidelity_name__
```

````

````{py:attribute} l2_axis
:canonical: corrct.data_terms.DataFidelity_l21.l2_axis
:type: int
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21.l2_axis
```

````

````{py:method} _get_inner_norm(dual: corrct.data_terms.NDArrayFloat) -> corrct.data_terms.NDArrayFloat
:canonical: corrct.data_terms.DataFidelity_l21._get_inner_norm

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21._get_inner_norm
```

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_l21.apply_proximal_primal
:abstractmethod:

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21.apply_proximal_primal
```

````

`````

`````{py:class} DataFidelity_l21w(axis_weights: corrct.data_terms.NDArrayFloat, background: float | corrct.data_terms.NDArrayFloat | None = None, l2_axis: int = 0, inner_norm: float = 2)
:canonical: corrct.data_terms.DataFidelity_l21w

Bases: {py:obj}`corrct.data_terms.DataFidelity_l21`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21w
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21w.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_l21w.__data_fidelity_name__
:value: >
   'l21w'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21w.__data_fidelity_name__
```

````

````{py:attribute} axis_weights
:canonical: corrct.data_terms.DataFidelity_l21w.axis_weights
:type: corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21w.axis_weights
```

````

````{py:attribute} inner_norm
:canonical: corrct.data_terms.DataFidelity_l21w.inner_norm
:type: float
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21w.inner_norm
```

````

````{py:method} _get_inner_norm(dual: corrct.data_terms.NDArrayFloat) -> corrct.data_terms.NDArrayFloat
:canonical: corrct.data_terms.DataFidelity_l21w._get_inner_norm

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l21w._get_inner_norm
```

````

`````

`````{py:class} DataFidelity_l1b(local_error: float | corrct.data_terms.NDArrayFloat, background: float | corrct.data_terms.NDArrayFloat | None = None)
:canonical: corrct.data_terms.DataFidelity_l1b

Bases: {py:obj}`corrct.data_terms.DataFidelity_l1`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1b
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1b.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_l1b.__data_fidelity_name__
:value: >
   'l1b'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1b.__data_fidelity_name__
```

````

````{py:attribute} sigma_error
:canonical: corrct.data_terms.DataFidelity_l1b.sigma_error
:type: float | corrct.data_terms.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1b.sigma_error
```

````

````{py:method} assign_data(data: corrct.data_terms.NDArrayFloat, sigma: float | corrct.data_terms.NDArrayFloat = 1.0) -> None
:canonical: corrct.data_terms.DataFidelity_l1b.assign_data

````

````{py:method} _apply_threshold(dual: corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_l1b._apply_threshold

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1b._apply_threshold
```

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_l1b.apply_proximal_primal

```{autodoc2-docstring} corrct.data_terms.DataFidelity_l1b.apply_proximal_primal
```

````

`````

`````{py:class} DataFidelity_KL(background: float | corrct.data_terms.NDArrayFloat | None = None)
:canonical: corrct.data_terms.DataFidelity_KL

Bases: {py:obj}`corrct.data_terms.DataFidelityBase`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_KL
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_KL.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_KL.__data_fidelity_name__
:value: >
   'KL'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_KL.__data_fidelity_name__
```

````

````{py:method} _compute_sigma_data()
:canonical: corrct.data_terms.DataFidelity_KL._compute_sigma_data

```{autodoc2-docstring} corrct.data_terms.DataFidelity_KL._compute_sigma_data
```

````

````{py:method} apply_proximal_dual(dual: corrct.data_terms.NDArrayFloat)
:canonical: corrct.data_terms.DataFidelity_KL.apply_proximal_dual

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_KL.apply_proximal_primal

```{autodoc2-docstring} corrct.data_terms.DataFidelity_KL.apply_proximal_primal
```

````

````{py:method} compute_residual(proj_primal: numpy.typing.NDArray, mask: numpy.typing.NDArray | None = None, use_proximal: bool = True) -> corrct.data_terms.NDArrayFloat
:canonical: corrct.data_terms.DataFidelity_KL.compute_residual

````

````{py:method} compute_residual_norm(dual: corrct.data_terms.NDArrayFloat) -> float
:canonical: corrct.data_terms.DataFidelity_KL.compute_residual_norm

````

````{py:method} compute_primal_dual_gap(proj_primal: corrct.data_terms.NDArrayFloat, dual: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None)
:canonical: corrct.data_terms.DataFidelity_KL.compute_primal_dual_gap

````

`````

`````{py:class} DataFidelity_ln(background: float | corrct.data_terms.NDArrayFloat | None = None, ln_axes: collections.abc.Sequence[int] = (1, -1), spectral_norm: corrct.data_terms.DataFidelityBase = DataFidelity_l1())
:canonical: corrct.data_terms.DataFidelity_ln

Bases: {py:obj}`corrct.data_terms.DataFidelityBase`

```{autodoc2-docstring} corrct.data_terms.DataFidelity_ln
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.data_terms.DataFidelity_ln.__init__
```

````{py:attribute} __data_fidelity_name__
:canonical: corrct.data_terms.DataFidelity_ln.__data_fidelity_name__
:value: >
   'ln'

```{autodoc2-docstring} corrct.data_terms.DataFidelity_ln.__data_fidelity_name__
```

````

````{py:attribute} ln_axes
:canonical: corrct.data_terms.DataFidelity_ln.ln_axes
:type: collections.abc.Sequence[int]
:value: >
   None

```{autodoc2-docstring} corrct.data_terms.DataFidelity_ln.ln_axes
```

````

````{py:method} apply_proximal_dual(dual: corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_ln.apply_proximal_dual

````

````{py:method} apply_proximal_primal(primal: corrct.data_terms.NDArrayFloat, tau: float | corrct.data_terms.NDArrayFloat) -> None
:canonical: corrct.data_terms.DataFidelity_ln.apply_proximal_primal
:abstractmethod:

```{autodoc2-docstring} corrct.data_terms.DataFidelity_ln.apply_proximal_primal
```

````

````{py:method} compute_residual_norm(dual: corrct.data_terms.NDArrayFloat) -> float
:canonical: corrct.data_terms.DataFidelity_ln.compute_residual_norm

````

````{py:method} compute_primal_dual_gap(proj_primal: corrct.data_terms.NDArrayFloat, dual: corrct.data_terms.NDArrayFloat, mask: corrct.data_terms.NDArrayFloat | None = None)
:canonical: corrct.data_terms.DataFidelity_ln.compute_primal_dual_gap

````

`````
