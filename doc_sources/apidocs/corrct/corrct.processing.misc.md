# {py:mod}`corrct.processing.misc`

```{py:module} corrct.processing.misc
```

```{autodoc2-docstring} corrct.processing.misc
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`circular_mask <corrct.processing.misc.circular_mask>`
  - ```{autodoc2-docstring} corrct.processing.misc.circular_mask
    :summary:
    ```
* - {py:obj}`ball <corrct.processing.misc.ball>`
  - ```{autodoc2-docstring} corrct.processing.misc.ball
    :summary:
    ```
* - {py:obj}`azimuthal_integration <corrct.processing.misc.azimuthal_integration>`
  - ```{autodoc2-docstring} corrct.processing.misc.azimuthal_integration
    :summary:
    ```
* - {py:obj}`lines_intersection <corrct.processing.misc.lines_intersection>`
  - ```{autodoc2-docstring} corrct.processing.misc.lines_intersection
    :summary:
    ```
* - {py:obj}`norm_cross_corr <corrct.processing.misc.norm_cross_corr>`
  - ```{autodoc2-docstring} corrct.processing.misc.norm_cross_corr
    :summary:
    ```
* - {py:obj}`inspect_fourier_img <corrct.processing.misc.inspect_fourier_img>`
  - ```{autodoc2-docstring} corrct.processing.misc.inspect_fourier_img
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`eps <corrct.processing.misc.eps>`
  - ```{autodoc2-docstring} corrct.processing.misc.eps
    :summary:
    ```
* - {py:obj}`NDArrayInt <corrct.processing.misc.NDArrayInt>`
  - ```{autodoc2-docstring} corrct.processing.misc.NDArrayInt
    :summary:
    ```
````

### API

````{py:data} eps
:canonical: corrct.processing.misc.eps
:value: >
   None

```{autodoc2-docstring} corrct.processing.misc.eps
```

````

````{py:data} NDArrayInt
:canonical: corrct.processing.misc.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.processing.misc.NDArrayInt
```

````

````{py:function} circular_mask(vol_shape_zxy: typing.Union[collections.abc.Sequence[int], corrct.processing.misc.NDArrayInt], radius_offset: float = 0, coords_ball: typing.Union[collections.abc.Sequence[int], corrct.processing.misc.NDArrayInt, None] = None, ball_norm: float = 2, vol_origin_zxy: typing.Union[collections.abc.Sequence[float], numpy.typing.NDArray, None] = None, taper_func: typing.Optional[str] = None, taper_target: typing.Union[str, float] = 'edge', super_sampling: int = 1, squeeze: bool = True, dtype: numpy.typing.DTypeLike = np.float32) -> numpy.typing.NDArray
:canonical: corrct.processing.misc.circular_mask

```{autodoc2-docstring} corrct.processing.misc.circular_mask
```
````

````{py:function} ball(data_shape_vu: numpy.typing.ArrayLike, radius: typing.Union[int, float], super_sampling: int = 5, dtype: numpy.typing.DTypeLike = np.float32, func: typing.Optional[typing.Callable] = None) -> numpy.typing.ArrayLike
:canonical: corrct.processing.misc.ball

```{autodoc2-docstring} corrct.processing.misc.ball
```
````

````{py:function} azimuthal_integration(img: numpy.typing.NDArray, axes: collections.abc.Sequence[int] = (-2, -1), domain: str = 'direct') -> numpy.typing.NDArray
:canonical: corrct.processing.misc.azimuthal_integration

```{autodoc2-docstring} corrct.processing.misc.azimuthal_integration
```
````

````{py:function} lines_intersection(line_1: numpy.typing.NDArray, line_2: typing.Union[float, numpy.typing.NDArray], position: str = 'first', x_lims: typing.Optional[tuple[typing.Optional[float], typing.Optional[float]]] = None) -> typing.Optional[tuple[float, float]]
:canonical: corrct.processing.misc.lines_intersection

```{autodoc2-docstring} corrct.processing.misc.lines_intersection
```
````

````{py:function} norm_cross_corr(img1: numpy.typing.NDArray, img2: typing.Optional[numpy.typing.NDArray] = None, axes: collections.abc.Sequence[int] = (-2, -1), t_match: bool = False, mode_full: bool = True, compute_profile: bool = True, plot: bool = True) -> typing.Union[numpy.typing.NDArray, tuple[numpy.typing.NDArray, numpy.typing.NDArray]]
:canonical: corrct.processing.misc.norm_cross_corr

```{autodoc2-docstring} corrct.processing.misc.norm_cross_corr
```
````

````{py:function} inspect_fourier_img(img: numpy.typing.NDArray, remove_zero: bool = False) -> None
:canonical: corrct.processing.misc.inspect_fourier_img

```{autodoc2-docstring} corrct.processing.misc.inspect_fourier_img
```
````
