# {py:mod}`corrct.processing.pre`

```{py:module} corrct.processing.pre
```

```{autodoc2-docstring} corrct.processing.pre
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`pad_sinogram <corrct.processing.pre.pad_sinogram>`
  - ```{autodoc2-docstring} corrct.processing.pre.pad_sinogram
    :summary:
    ```
* - {py:obj}`apply_flat_field <corrct.processing.pre.apply_flat_field>`
  - ```{autodoc2-docstring} corrct.processing.pre.apply_flat_field
    :summary:
    ```
* - {py:obj}`apply_minus_log <corrct.processing.pre.apply_minus_log>`
  - ```{autodoc2-docstring} corrct.processing.pre.apply_minus_log
    :summary:
    ```
* - {py:obj}`rotate_proj_stack <corrct.processing.pre.rotate_proj_stack>`
  - ```{autodoc2-docstring} corrct.processing.pre.rotate_proj_stack
    :summary:
    ```
* - {py:obj}`shift_proj_stack <corrct.processing.pre.shift_proj_stack>`
  - ```{autodoc2-docstring} corrct.processing.pre.shift_proj_stack
    :summary:
    ```
* - {py:obj}`bin_imgs <corrct.processing.pre.bin_imgs>`
  - ```{autodoc2-docstring} corrct.processing.pre.bin_imgs
    :summary:
    ```
* - {py:obj}`background_from_margin <corrct.processing.pre.background_from_margin>`
  - ```{autodoc2-docstring} corrct.processing.pre.background_from_margin
    :summary:
    ```
* - {py:obj}`snip <corrct.processing.pre.snip>`
  - ```{autodoc2-docstring} corrct.processing.pre.snip
    :summary:
    ```
* - {py:obj}`background_from_snip <corrct.processing.pre.background_from_snip>`
  - ```{autodoc2-docstring} corrct.processing.pre.background_from_snip
    :summary:
    ```
* - {py:obj}`destripe_wlf_vwu <corrct.processing.pre.destripe_wlf_vwu>`
  - ```{autodoc2-docstring} corrct.processing.pre.destripe_wlf_vwu
    :summary:
    ```
* - {py:obj}`compute_eigen_flats <corrct.processing.pre.compute_eigen_flats>`
  - ```{autodoc2-docstring} corrct.processing.pre.compute_eigen_flats
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`eps <corrct.processing.pre.eps>`
  - ```{autodoc2-docstring} corrct.processing.pre.eps
    :summary:
    ```
````

### API

````{py:data} eps
:canonical: corrct.processing.pre.eps
:value: >
   None

```{autodoc2-docstring} corrct.processing.pre.eps
```

````

````{py:function} pad_sinogram(sinogram: numpy.typing.NDArray, width: typing.Union[int, collections.abc.Sequence[int], numpy.typing.NDArray], pad_axis: int = -1, mode: str = 'edge', **kwds) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.pad_sinogram

```{autodoc2-docstring} corrct.processing.pre.pad_sinogram
```
````

````{py:function} apply_flat_field(projs_wvu: numpy.typing.NDArray, flats_wvu: numpy.typing.NDArray, darks_wvu: typing.Optional[numpy.typing.NDArray] = None, crop: typing.Union[numpy.typing.NDArray, collections.abc.Sequence[int], None] = None, cap_intensity: typing.Optional[float] = None, dtype: numpy.typing.DTypeLike = np.float32) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.apply_flat_field

```{autodoc2-docstring} corrct.processing.pre.apply_flat_field
```
````

````{py:function} apply_minus_log(projs: numpy.typing.NDArray, lower_limit: float = -np.inf) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.apply_minus_log

```{autodoc2-docstring} corrct.processing.pre.apply_minus_log
```
````

````{py:function} rotate_proj_stack(data_vwu: numpy.typing.NDArray, rot_angle_deg: float) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.rotate_proj_stack

```{autodoc2-docstring} corrct.processing.pre.rotate_proj_stack
```
````

````{py:function} shift_proj_stack(data_vwu: numpy.typing.NDArray, shifts: numpy.typing.NDArray, use_fft: bool = False) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.shift_proj_stack

```{autodoc2-docstring} corrct.processing.pre.shift_proj_stack
```
````

````{py:function} bin_imgs(imgs: numpy.typing.NDArray, binning: typing.Union[int, float], axes: collections.abc.Sequence[int] = (-2, -1), auto_crop: bool = False, verbose: bool = True) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.bin_imgs

```{autodoc2-docstring} corrct.processing.pre.bin_imgs
```
````

````{py:function} background_from_margin(data_vwu: numpy.typing.NDArray, margin: typing.Union[int, collections.abc.Sequence[int], numpy.typing.NDArray[numpy.integer]] = 4, poly_order: int = 0, plot: bool = False) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.background_from_margin

```{autodoc2-docstring} corrct.processing.pre.background_from_margin
```
````

````{py:function} snip(img: numpy.typing.NDArray, kernel_dims: typing.Union[int, None] = None, iterations: int = 1000, window: int = 3, verbose: bool = False) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.snip

```{autodoc2-docstring} corrct.processing.pre.snip
```
````

````{py:function} background_from_snip(data_vwu: numpy.typing.NDArray, snip_iterations: int = 6, smooth_std: float = 0.0) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.background_from_snip

```{autodoc2-docstring} corrct.processing.pre.background_from_snip
```
````

````{py:function} destripe_wlf_vwu(data: numpy.typing.NDArray, sigma: float = 0.005, level: int = 1, wavelet: str = 'bior2.2', angle_axis: int = -2, other_axes: typing.Union[collections.abc.Sequence[int], numpy.typing.NDArray, None] = None) -> numpy.typing.NDArray
:canonical: corrct.processing.pre.destripe_wlf_vwu

```{autodoc2-docstring} corrct.processing.pre.destripe_wlf_vwu
```
````

````{py:function} compute_eigen_flats(trans_wvu: numpy.typing.NDArray, flats_wvu: typing.Optional[numpy.typing.NDArray] = None, darks_wvu: typing.Optional[numpy.typing.NDArray] = None, ndim: int = 2, plot: bool = False) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
:canonical: corrct.processing.pre.compute_eigen_flats

```{autodoc2-docstring} corrct.processing.pre.compute_eigen_flats
```
````
