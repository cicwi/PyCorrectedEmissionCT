# {py:mod}`corrct.processing.post`

```{py:module} corrct.processing.post
```

```{autodoc2-docstring} corrct.processing.post
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`com <corrct.processing.post.com>`
  - ```{autodoc2-docstring} corrct.processing.post.com
    :summary:
    ```
* - {py:obj}`power_spectrum <corrct.processing.post.power_spectrum>`
  - ```{autodoc2-docstring} corrct.processing.post.power_spectrum
    :summary:
    ```
* - {py:obj}`compute_frc <corrct.processing.post.compute_frc>`
  - ```{autodoc2-docstring} corrct.processing.post.compute_frc
    :summary:
    ```
* - {py:obj}`estimate_resolution <corrct.processing.post.estimate_resolution>`
  - ```{autodoc2-docstring} corrct.processing.post.estimate_resolution
    :summary:
    ```
* - {py:obj}`plot_frcs <corrct.processing.post.plot_frcs>`
  - ```{autodoc2-docstring} corrct.processing.post.plot_frcs
    :summary:
    ```
* - {py:obj}`fit_scale_bias <corrct.processing.post.fit_scale_bias>`
  - ```{autodoc2-docstring} corrct.processing.post.fit_scale_bias
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`eps <corrct.processing.post.eps>`
  - ```{autodoc2-docstring} corrct.processing.post.eps
    :summary:
    ```
````

### API

````{py:data} eps
:canonical: corrct.processing.post.eps
:value: >
   None

```{autodoc2-docstring} corrct.processing.post.eps
```

````

````{py:function} com(vol: numpy.typing.NDArray, axes: typing.Optional[numpy.typing.ArrayLike] = None) -> numpy.typing.NDArray
:canonical: corrct.processing.post.com

```{autodoc2-docstring} corrct.processing.post.com
```
````

````{py:function} power_spectrum(img: numpy.typing.NDArray, axes: typing.Optional[collections.abc.Sequence[int]] = None, smooth: typing.Optional[int] = 5, taper_ratio: typing.Optional[float] = 0.05, power: int = 2) -> numpy.typing.NDArray
:canonical: corrct.processing.post.power_spectrum

```{autodoc2-docstring} corrct.processing.post.power_spectrum
```
````

````{py:function} compute_frc(img1: numpy.typing.NDArray, img2: typing.Optional[numpy.typing.NDArray], snrt: float = 0.2071, axes: typing.Optional[collections.abc.Sequence[int]] = None, smooth: typing.Optional[int] = 5, taper_ratio: typing.Optional[float] = 0.05, supersampling: int = 1, theo_threshold: bool = True) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
:canonical: corrct.processing.post.compute_frc

```{autodoc2-docstring} corrct.processing.post.compute_frc
```
````

````{py:function} estimate_resolution(frc: numpy.typing.NDArray, t_hb: numpy.typing.NDArray) -> typing.Optional[tuple[float, float]]
:canonical: corrct.processing.post.estimate_resolution

```{autodoc2-docstring} corrct.processing.post.estimate_resolution
```
````

````{py:function} plot_frcs(volume_pairs: collections.abc.Sequence[tuple[numpy.typing.NDArray, numpy.typing.NDArray]], labels: collections.abc.Sequence[str], title: typing.Optional[str] = None, smooth: typing.Optional[int] = 5, snrt: float = 0.2071, axes: typing.Optional[collections.abc.Sequence[int]] = None, supersampling: int = 1, verbose: bool = False) -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
:canonical: corrct.processing.post.plot_frcs

```{autodoc2-docstring} corrct.processing.post.plot_frcs
```
````

````{py:function} fit_scale_bias(img_data: numpy.typing.NDArray, prj_data: numpy.typing.NDArray, prj: typing.Optional[corrct.operators.BaseTransform] = None) -> tuple[float, float]
:canonical: corrct.processing.post.fit_scale_bias

```{autodoc2-docstring} corrct.processing.post.fit_scale_bias
```
````
