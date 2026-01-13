# {py:mod}`corrct.denoisers`

```{py:module} corrct.denoisers
```

```{autodoc2-docstring} corrct.denoisers
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_default_regularizer_l1dwl <corrct.denoisers._default_regularizer_l1dwl>`
  - ```{autodoc2-docstring} corrct.denoisers._default_regularizer_l1dwl
    :summary:
    ```
* - {py:obj}`denoise_image <corrct.denoisers.denoise_image>`
  - ```{autodoc2-docstring} corrct.denoisers.denoise_image
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`eps <corrct.denoisers.eps>`
  - ```{autodoc2-docstring} corrct.denoisers.eps
    :summary:
    ```
````

### API

````{py:data} eps
:canonical: corrct.denoisers.eps
:value: >
   None

```{autodoc2-docstring} corrct.denoisers.eps
```

````

````{py:function} _default_regularizer_l1dwl(r_w: float | numpy.typing.NDArray) -> corrct.regularizers.BaseRegularizer
:canonical: corrct.denoisers._default_regularizer_l1dwl

```{autodoc2-docstring} corrct.denoisers._default_regularizer_l1dwl
```
````

````{py:function} denoise_image(img: numpy.typing.NDArray, reg_weight: float | collections.abc.Sequence[float] | numpy.typing.NDArray = 0.01, psf: numpy.typing.NDArray | None = None, pix_weights: numpy.typing.NDArray | None = None, iterations: int = 250, regularizer: collections.abc.Callable = _default_regularizer_l1dwl, lower_limit: float | None = None, verbose: bool = True) -> numpy.typing.NDArray | tuple[numpy.typing.NDArray, float]
:canonical: corrct.denoisers.denoise_image

```{autodoc2-docstring} corrct.denoisers.denoise_image
```
````
