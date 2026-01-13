# {py:mod}`corrct.physics.attenuation`

```{py:module} corrct.physics.attenuation
```

```{autodoc2-docstring} corrct.physics.attenuation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttenuationVolume <corrct.physics.attenuation.AttenuationVolume>`
  - ```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_linear_attenuation_coefficient <corrct.physics.attenuation.get_linear_attenuation_coefficient>`
  - ```{autodoc2-docstring} corrct.physics.attenuation.get_linear_attenuation_coefficient
    :summary:
    ```
* - {py:obj}`plot_emission_line_attenuation <corrct.physics.attenuation.plot_emission_line_attenuation>`
  - ```{autodoc2-docstring} corrct.physics.attenuation.plot_emission_line_attenuation
    :summary:
    ```
* - {py:obj}`plot_transmittance_decay <corrct.physics.attenuation.plot_transmittance_decay>`
  - ```{autodoc2-docstring} corrct.physics.attenuation.plot_transmittance_decay
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`num_threads <corrct.physics.attenuation.num_threads>`
  - ```{autodoc2-docstring} corrct.physics.attenuation.num_threads
    :summary:
    ```
* - {py:obj}`NDArrayFloat <corrct.physics.attenuation.NDArrayFloat>`
  - ```{autodoc2-docstring} corrct.physics.attenuation.NDArrayFloat
    :summary:
    ```
* - {py:obj}`NDArrayInt <corrct.physics.attenuation.NDArrayInt>`
  - ```{autodoc2-docstring} corrct.physics.attenuation.NDArrayInt
    :summary:
    ```
* - {py:obj}`CONVERT_UM_TO_CM <corrct.physics.attenuation.CONVERT_UM_TO_CM>`
  - ```{autodoc2-docstring} corrct.physics.attenuation.CONVERT_UM_TO_CM
    :summary:
    ```
````

### API

````{py:data} num_threads
:canonical: corrct.physics.attenuation.num_threads
:value: >
   'round(...)'

```{autodoc2-docstring} corrct.physics.attenuation.num_threads
```

````

````{py:data} NDArrayFloat
:canonical: corrct.physics.attenuation.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.NDArrayFloat
```

````

````{py:data} NDArrayInt
:canonical: corrct.physics.attenuation.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.NDArrayInt
```

````

````{py:data} CONVERT_UM_TO_CM
:canonical: corrct.physics.attenuation.CONVERT_UM_TO_CM
:value: >
   0.0001

```{autodoc2-docstring} corrct.physics.attenuation.CONVERT_UM_TO_CM
```

````

`````{py:class} AttenuationVolume(incident_local: corrct.physics.attenuation.NDArrayFloat | None, emitted_local: corrct.physics.attenuation.NDArrayFloat | None, angles_rot_rad: corrct.physics.attenuation.NDArrayFloat | collections.abc.Sequence[float], angles_det_rad: corrct.physics.attenuation.NDArrayFloat | collections.abc.Sequence[float | corrct.physics.xrf.DetectorXRF] | float | corrct.physics.xrf.DetectorXRF = np.pi / 2, emitted_sub_sampling: int = 1, dtype: numpy.typing.DTypeLike = np.float32)
:canonical: corrct.physics.attenuation.AttenuationVolume

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.__init__
```

````{py:attribute} incident_local
:canonical: corrct.physics.attenuation.AttenuationVolume.incident_local
:type: corrct.physics.attenuation.NDArrayFloat | None
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.incident_local
```

````

````{py:attribute} emitted_local
:canonical: corrct.physics.attenuation.AttenuationVolume.emitted_local
:type: corrct.physics.attenuation.NDArrayFloat | None
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.emitted_local
```

````

````{py:attribute} angles_rot_rad
:canonical: corrct.physics.attenuation.AttenuationVolume.angles_rot_rad
:type: corrct.physics.attenuation.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.angles_rot_rad
```

````

````{py:attribute} detectors
:canonical: corrct.physics.attenuation.AttenuationVolume.detectors
:type: collections.abc.Sequence[corrct.physics.xrf.DetectorXRF]
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.detectors
```

````

````{py:attribute} emitted_sub_sampling
:canonical: corrct.physics.attenuation.AttenuationVolume.emitted_sub_sampling
:type: int
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.emitted_sub_sampling
```

````

````{py:attribute} dtype
:canonical: corrct.physics.attenuation.AttenuationVolume.dtype
:type: numpy.typing.DTypeLike
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.dtype
```

````

````{py:attribute} vol_shape_zyx
:canonical: corrct.physics.attenuation.AttenuationVolume.vol_shape_zyx
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.vol_shape_zyx
```

````

````{py:attribute} maps
:canonical: corrct.physics.attenuation.AttenuationVolume.maps
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.maps
```

````

````{py:method} _get_detector_angles() -> numpy.typing.NDArray
:canonical: corrct.physics.attenuation.AttenuationVolume._get_detector_angles

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume._get_detector_angles
```

````

````{py:method} _compute_attenuation_angle_in(local_att: corrct.physics.attenuation.NDArrayFloat, angle_rad: float) -> numpy.typing.NDArray
:canonical: corrct.physics.attenuation.AttenuationVolume._compute_attenuation_angle_in

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume._compute_attenuation_angle_in
```

````

````{py:method} _compute_attenuation_angle_out(local_att: corrct.physics.attenuation.NDArrayFloat, angle_rad: float) -> numpy.typing.NDArray
:canonical: corrct.physics.attenuation.AttenuationVolume._compute_attenuation_angle_out

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume._compute_attenuation_angle_out
```

````

````{py:method} compute_maps(use_multithreading: bool = True, verbose: bool = True) -> None
:canonical: corrct.physics.attenuation.AttenuationVolume.compute_maps

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.compute_maps
```

````

````{py:method} plot_map(ax: matplotlib.axes._axes.Axes, rot_ind: int, det_ind: int = 0, slice_ind: int | None = None, axes: collections.abc.Sequence[int] | corrct.physics.attenuation.NDArrayInt = (-2, -1)) -> collections.abc.Sequence[float]
:canonical: corrct.physics.attenuation.AttenuationVolume.plot_map

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.plot_map
```

````

````{py:method} get_maps(roi: numpy.typing.ArrayLike | None = None, rot_ind: int | slice | collections.abc.Sequence[int] | corrct.physics.attenuation.NDArrayInt | None = None, det_ind: int | slice | collections.abc.Sequence[int] | corrct.physics.attenuation.NDArrayInt | None = None, binning: int = 1) -> numpy.typing.NDArray
:canonical: corrct.physics.attenuation.AttenuationVolume.get_maps

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.get_maps
```

````

````{py:method} get_projector_args(roi: numpy.typing.ArrayLike | None = None, rot_ind: int | slice | collections.abc.Sequence[int] | corrct.physics.attenuation.NDArrayInt | None = None, det_ind: int | slice | collections.abc.Sequence[int] | corrct.physics.attenuation.NDArrayInt | None = None, binning: int = 1) -> dict[str, numpy.typing.NDArray]
:canonical: corrct.physics.attenuation.AttenuationVolume.get_projector_args

```{autodoc2-docstring} corrct.physics.attenuation.AttenuationVolume.get_projector_args
```

````

`````

````{py:function} get_linear_attenuation_coefficient(compound: str | dict, energy_keV: float, pixel_size_um: float, density: float | None = None) -> float
:canonical: corrct.physics.attenuation.get_linear_attenuation_coefficient

```{autodoc2-docstring} corrct.physics.attenuation.get_linear_attenuation_coefficient
```
````

````{py:function} plot_emission_line_attenuation(compound: str | dict, thickness_um: float, mean_energy_keV: float, fwhm_keV: float, line_shape: str = 'lorentzian', num_points: int = 201, plot_lines_mean: bool = True) -> None
:canonical: corrct.physics.attenuation.plot_emission_line_attenuation

```{autodoc2-docstring} corrct.physics.attenuation.plot_emission_line_attenuation
```
````

````{py:function} plot_transmittance_decay(compounds: str | dict | collections.abc.Sequence[str | dict], mean_energy_keV: float, thickness_range_um: tuple[float, float, int] = (0.0, 10.0, 101)) -> None
:canonical: corrct.physics.attenuation.plot_transmittance_decay

```{autodoc2-docstring} corrct.physics.attenuation.plot_transmittance_decay
```
````
