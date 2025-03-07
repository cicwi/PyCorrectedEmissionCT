# {py:mod}`corrct.processing.noise`

```{py:module} corrct.processing.noise
```

```{autodoc2-docstring} corrct.processing.noise
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_variance_poisson <corrct.processing.noise.compute_variance_poisson>`
  - ```{autodoc2-docstring} corrct.processing.noise.compute_variance_poisson
    :summary:
    ```
* - {py:obj}`compute_variance_transmission <corrct.processing.noise.compute_variance_transmission>`
  - ```{autodoc2-docstring} corrct.processing.noise.compute_variance_transmission
    :summary:
    ```
* - {py:obj}`compute_variance_weight <corrct.processing.noise.compute_variance_weight>`
  - ```{autodoc2-docstring} corrct.processing.noise.compute_variance_weight
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`eps <corrct.processing.noise.eps>`
  - ```{autodoc2-docstring} corrct.processing.noise.eps
    :summary:
    ```
````

### API

````{py:data} eps
:canonical: corrct.processing.noise.eps
:value: >
   None

```{autodoc2-docstring} corrct.processing.noise.eps
```

````

````{py:function} compute_variance_poisson(Is: numpy.typing.NDArray, I0: typing.Optional[numpy.typing.NDArray] = None, var_I0: typing.Optional[numpy.typing.NDArray] = None, normalized: bool = True) -> numpy.typing.NDArray
:canonical: corrct.processing.noise.compute_variance_poisson

```{autodoc2-docstring} corrct.processing.noise.compute_variance_poisson
```
````

````{py:function} compute_variance_transmission(Is: numpy.typing.NDArray, I0: numpy.typing.NDArray, var_I0: typing.Optional[numpy.typing.NDArray] = None, normalized: bool = True) -> numpy.typing.NDArray
:canonical: corrct.processing.noise.compute_variance_transmission

```{autodoc2-docstring} corrct.processing.noise.compute_variance_transmission
```
````

````{py:function} compute_variance_weight(variance: numpy.typing.NDArray, *, percentile: float = 0.001, mask: typing.Optional[numpy.typing.NDArray] = None, normalized: bool = False, use_std: bool = False, semilog: bool = False) -> numpy.typing.NDArray
:canonical: corrct.processing.noise.compute_variance_weight

```{autodoc2-docstring} corrct.processing.noise.compute_variance_weight
```
````
