# {py:mod}`corrct.physics.phase`

```{py:module} corrct.physics.phase
```

```{autodoc2-docstring} corrct.physics.phase
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_delta_beta <corrct.physics.phase.get_delta_beta>`
  - ```{autodoc2-docstring} corrct.physics.phase.get_delta_beta
    :summary:
    ```
* - {py:obj}`get_delta_beta_curves <corrct.physics.phase.get_delta_beta_curves>`
  - ```{autodoc2-docstring} corrct.physics.phase.get_delta_beta_curves
    :summary:
    ```
* - {py:obj}`_tie_freq_response <corrct.physics.phase._tie_freq_response>`
  - ```{autodoc2-docstring} corrct.physics.phase._tie_freq_response
    :summary:
    ```
* - {py:obj}`_ctf_freq_response <corrct.physics.phase._ctf_freq_response>`
  - ```{autodoc2-docstring} corrct.physics.phase._ctf_freq_response
    :summary:
    ```
* - {py:obj}`plot_filter_responses <corrct.physics.phase.plot_filter_responses>`
  - ```{autodoc2-docstring} corrct.physics.phase.plot_filter_responses
    :summary:
    ```
* - {py:obj}`get_propagation_filter <corrct.physics.phase.get_propagation_filter>`
  - ```{autodoc2-docstring} corrct.physics.phase.get_propagation_filter
    :summary:
    ```
* - {py:obj}`apply_propagation_filter <corrct.physics.phase.apply_propagation_filter>`
  - ```{autodoc2-docstring} corrct.physics.phase.apply_propagation_filter
    :summary:
    ```
````

### API

````{py:function} get_delta_beta(cmp_name: str, energy_keV: float, density: typing.Union[float, None] = None) -> float
:canonical: corrct.physics.phase.get_delta_beta

```{autodoc2-docstring} corrct.physics.phase.get_delta_beta
```
````

````{py:function} get_delta_beta_curves(compounds: collections.abc.Sequence[str], energy_keV_range: tuple[float, float, int] = (1.0, 800.0, 500), plot: bool = True) -> collections.abc.Sequence[numpy.typing.NDArray]
:canonical: corrct.physics.phase.get_delta_beta_curves

```{autodoc2-docstring} corrct.physics.phase.get_delta_beta_curves
```
````

````{py:function} _tie_freq_response(k2: numpy.typing.NDArray, dist_um: float, wlength_um: float, delta_beta: float) -> numpy.typing.NDArray
:canonical: corrct.physics.phase._tie_freq_response

```{autodoc2-docstring} corrct.physics.phase._tie_freq_response
```
````

````{py:function} _ctf_freq_response(k2: numpy.typing.NDArray, dist_um: float, wlength_um: float, delta_beta: float) -> numpy.typing.NDArray
:canonical: corrct.physics.phase._ctf_freq_response

```{autodoc2-docstring} corrct.physics.phase._ctf_freq_response
```
````

````{py:function} plot_filter_responses(filter_length: int, pix_size_um: float, dist_um: float, wlength_um: float, delta_beta: float, domain: str = 'fourier') -> tuple
:canonical: corrct.physics.phase.plot_filter_responses

```{autodoc2-docstring} corrct.physics.phase.plot_filter_responses
```
````

````{py:function} get_propagation_filter(img_shape: typing.Union[collections.abc.Sequence[int], numpy.typing.NDArray], pix_size_um: float, dist_um: float, wlength_um: float, delta_beta: float, filter_type: str = 'ctf', use_rfft: bool = False, plot_result: bool = False, dtype: numpy.typing.DTypeLike = np.float32) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
:canonical: corrct.physics.phase.get_propagation_filter

```{autodoc2-docstring} corrct.physics.phase.get_propagation_filter
```
````

````{py:function} apply_propagation_filter(data_wvu: numpy.typing.NDArray, pix_size_um: float, dist_um: float, wlength_um: float, delta_beta: float, filter_type: str = 'tie') -> numpy.typing.NDArray
:canonical: corrct.physics.phase.apply_propagation_filter

```{autodoc2-docstring} corrct.physics.phase.apply_propagation_filter
```
````
