# {py:mod}`corrct.alignment.fitting`

```{py:module} corrct.alignment.fitting
```

```{autodoc2-docstring} corrct.alignment.fitting
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`fit_shifts_u_sad <corrct.alignment.fitting.fit_shifts_u_sad>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.fit_shifts_u_sad
    :summary:
    ```
* - {py:obj}`fit_shifts_vu_xc <corrct.alignment.fitting.fit_shifts_vu_xc>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.fit_shifts_vu_xc
    :summary:
    ```
* - {py:obj}`fit_shifts_zyx_xc <corrct.alignment.fitting.fit_shifts_zyx_xc>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.fit_shifts_zyx_xc
    :summary:
    ```
* - {py:obj}`fit_image_rotation_and_scale <corrct.alignment.fitting.fit_image_rotation_and_scale>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.fit_image_rotation_and_scale
    :summary:
    ```
* - {py:obj}`fit_camera_tilt_angle <corrct.alignment.fitting.fit_camera_tilt_angle>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.fit_camera_tilt_angle
    :summary:
    ```
* - {py:obj}`sinusoid <corrct.alignment.fitting.sinusoid>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.sinusoid
    :summary:
    ```
* - {py:obj}`fit_sinusoid <corrct.alignment.fitting.fit_sinusoid>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.fit_sinusoid
    :summary:
    ```
* - {py:obj}`extract_peak_regions_1d <corrct.alignment.fitting.extract_peak_regions_1d>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.extract_peak_regions_1d
    :summary:
    ```
* - {py:obj}`refine_max_position_1d <corrct.alignment.fitting.refine_max_position_1d>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.refine_max_position_1d
    :summary:
    ```
* - {py:obj}`extract_peak_region_nd <corrct.alignment.fitting.extract_peak_region_nd>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.extract_peak_region_nd
    :summary:
    ```
* - {py:obj}`refine_max_position_2d <corrct.alignment.fitting.refine_max_position_2d>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.refine_max_position_2d
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NDArrayFloat <corrct.alignment.fitting.NDArrayFloat>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.NDArrayFloat
    :summary:
    ```
* - {py:obj}`eps <corrct.alignment.fitting.eps>`
  - ```{autodoc2-docstring} corrct.alignment.fitting.eps
    :summary:
    ```
````

### API

````{py:data} NDArrayFloat
:canonical: corrct.alignment.fitting.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.alignment.fitting.NDArrayFloat
```

````

````{py:data} eps
:canonical: corrct.alignment.fitting.eps
:value: >
   None

```{autodoc2-docstring} corrct.alignment.fitting.eps
```

````

````{py:function} fit_shifts_u_sad(data_wu: corrct.alignment.fitting.NDArrayFloat, proj_wu: corrct.alignment.fitting.NDArrayFloat, search_range: int = 16, pad_u: bool = False, error_norm: int = 1, decimals: int = 2) -> corrct.alignment.fitting.NDArrayFloat
:canonical: corrct.alignment.fitting.fit_shifts_u_sad

```{autodoc2-docstring} corrct.alignment.fitting.fit_shifts_u_sad
```
````

````{py:function} fit_shifts_vu_xc(data_vwu: corrct.alignment.fitting.NDArrayFloat, proj_vwu: corrct.alignment.fitting.NDArrayFloat, pad_u: bool = False, normalize_fourier: bool = False, margin: int = 0, use_rfft: bool = True, stack_axis: int = -2, decimals: int = 2) -> corrct.alignment.fitting.NDArrayFloat
:canonical: corrct.alignment.fitting.fit_shifts_vu_xc

```{autodoc2-docstring} corrct.alignment.fitting.fit_shifts_vu_xc
```
````

````{py:function} fit_shifts_zyx_xc(ref_vol_zyx: corrct.alignment.fitting.NDArrayFloat, rec_vol_zyx: corrct.alignment.fitting.NDArrayFloat, pad_zyx: bool = False, normalize_fourier: bool = True, use_rfft: bool = True, decimals: int = 2) -> corrct.alignment.fitting.NDArrayFloat
:canonical: corrct.alignment.fitting.fit_shifts_zyx_xc

```{autodoc2-docstring} corrct.alignment.fitting.fit_shifts_zyx_xc
```
````

````{py:function} fit_image_rotation_and_scale(img_1_vu: numpy.typing.NDArray, img_2_vu: numpy.typing.NDArray, pad_mode: typing.Union[str, None] = None, window_type: str = 'hann', verbose: bool = False) -> tuple[float, float]
:canonical: corrct.alignment.fitting.fit_image_rotation_and_scale

```{autodoc2-docstring} corrct.alignment.fitting.fit_image_rotation_and_scale
```
````

````{py:function} fit_camera_tilt_angle(img_1: numpy.typing.NDArray, img_2: numpy.typing.NDArray, pad_u: bool = False, fit_l1: bool = True, verbose: bool = False)
:canonical: corrct.alignment.fitting.fit_camera_tilt_angle

```{autodoc2-docstring} corrct.alignment.fitting.fit_camera_tilt_angle
```
````

````{py:function} sinusoid(x: typing.Union[corrct.alignment.fitting.NDArrayFloat, float], a: typing.Union[corrct.alignment.fitting.NDArrayFloat, float], p: typing.Union[corrct.alignment.fitting.NDArrayFloat, float], b: typing.Union[corrct.alignment.fitting.NDArrayFloat, float]) -> corrct.alignment.fitting.NDArrayFloat
:canonical: corrct.alignment.fitting.sinusoid

```{autodoc2-docstring} corrct.alignment.fitting.sinusoid
```
````

````{py:function} fit_sinusoid(angles: corrct.alignment.fitting.NDArrayFloat, values: corrct.alignment.fitting.NDArrayFloat, fit_l1: bool = False) -> tuple[float, float, float]
:canonical: corrct.alignment.fitting.fit_sinusoid

```{autodoc2-docstring} corrct.alignment.fitting.fit_sinusoid
```
````

````{py:function} extract_peak_regions_1d(cc: corrct.alignment.fitting.NDArrayFloat, axis: int = -1, peak_radius: int = 1, cc_coords: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None) -> tuple[corrct.alignment.fitting.NDArrayFloat, typing.Optional[numpy.typing.NDArray]]
:canonical: corrct.alignment.fitting.extract_peak_regions_1d

```{autodoc2-docstring} corrct.alignment.fitting.extract_peak_regions_1d
```
````

````{py:function} refine_max_position_1d(f_vals: corrct.alignment.fitting.NDArrayFloat, f_x: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None, return_vertex_val: bool = False, decimals: int = 2) -> typing.Union[corrct.alignment.fitting.NDArrayFloat, tuple[corrct.alignment.fitting.NDArrayFloat, corrct.alignment.fitting.NDArrayFloat]]
:canonical: corrct.alignment.fitting.refine_max_position_1d

```{autodoc2-docstring} corrct.alignment.fitting.refine_max_position_1d
```
````

````{py:function} extract_peak_region_nd(cc: corrct.alignment.fitting.NDArrayFloat, peak_radius: int = 1, cc_coords: typing.Optional[typing.Sequence[typing.Union[typing.Sequence, numpy.typing.NDArray]]] = None) -> tuple[numpy.typing.NDArray, typing.Optional[typing.Sequence[numpy.typing.NDArray]]]
:canonical: corrct.alignment.fitting.extract_peak_region_nd

```{autodoc2-docstring} corrct.alignment.fitting.extract_peak_region_nd
```
````

````{py:function} refine_max_position_2d(f_vals: corrct.alignment.fitting.NDArrayFloat, fy: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None, fx: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None) -> numpy.typing.NDArray
:canonical: corrct.alignment.fitting.refine_max_position_2d

```{autodoc2-docstring} corrct.alignment.fitting.refine_max_position_2d
```
````
