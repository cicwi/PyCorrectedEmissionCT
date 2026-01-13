# {py:mod}`corrct.physics.materials`

```{py:module} corrct.physics.materials
```

```{autodoc2-docstring} corrct.physics.materials
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VolumeMaterial <corrct.physics.materials.VolumeMaterial>`
  - ```{autodoc2-docstring} corrct.physics.materials.VolumeMaterial
    :summary:
    ```
````

### API

`````{py:class} VolumeMaterial(materials_fractions: collections.abc.Sequence[numpy.typing.NDArray], materials_composition: collections.abc.Sequence, voxel_size_cm: float, dtype: numpy.typing.DTypeLike = None, verbose: bool = False)
:canonical: corrct.physics.materials.VolumeMaterial

```{autodoc2-docstring} corrct.physics.materials.VolumeMaterial
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.physics.materials.VolumeMaterial.__init__
```

````{py:method} get_attenuation(energy_keV: float) -> numpy.typing.NDArray
:canonical: corrct.physics.materials.VolumeMaterial.get_attenuation

```{autodoc2-docstring} corrct.physics.materials.VolumeMaterial.get_attenuation
```

````

````{py:method} get_phase_shift(energy_keV: float) -> numpy.typing.NDArray
:canonical: corrct.physics.materials.VolumeMaterial.get_phase_shift

```{autodoc2-docstring} corrct.physics.materials.VolumeMaterial.get_phase_shift
```

````

````{py:method} get_element_mass_fraction(element: str | int) -> numpy.typing.NDArray
:canonical: corrct.physics.materials.VolumeMaterial.get_element_mass_fraction

```{autodoc2-docstring} corrct.physics.materials.VolumeMaterial.get_element_mass_fraction
```

````

````{py:method} _check_parallax_detector(detector: corrct.physics.xrf.DetectorXRF, tolerance: float = 0.01) -> bool
:canonical: corrct.physics.materials.VolumeMaterial._check_parallax_detector

```{autodoc2-docstring} corrct.physics.materials.VolumeMaterial._check_parallax_detector
```

````

````{py:method} get_compton_scattering(energy_in_keV: float, angle_rad: float | None = None, detector: corrct.physics.xrf.DetectorXRF | None = None) -> tuple[float, numpy.typing.NDArray]
:canonical: corrct.physics.materials.VolumeMaterial.get_compton_scattering

```{autodoc2-docstring} corrct.physics.materials.VolumeMaterial.get_compton_scattering
```

````

````{py:method} get_fluo_production(element: str | int, energy_in_keV: float, fluo_lines: str | corrct.physics.xrf.FluoLine | collections.abc.Sequence[corrct.physics.xrf.FluoLine], detector: corrct.physics.xrf.DetectorXRF | None = None) -> tuple[float, numpy.typing.NDArray]
:canonical: corrct.physics.materials.VolumeMaterial.get_fluo_production

```{autodoc2-docstring} corrct.physics.materials.VolumeMaterial.get_fluo_production
```

````

`````
