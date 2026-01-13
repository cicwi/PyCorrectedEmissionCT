# {py:mod}`corrct.alignment.centering`

```{py:module} corrct.alignment.centering
```

```{autodoc2-docstring} corrct.alignment.centering
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RecenterVolume <corrct.alignment.centering.RecenterVolume>`
  - ```{autodoc2-docstring} corrct.alignment.centering.RecenterVolume
    :summary:
    ```
````

### API

`````{py:class} RecenterVolume(proj_geom: corrct.models.ProjectionGeometry, angles_rad: numpy.typing.NDArray | numpy.typing.ArrayLike, precision: int = 2)
:canonical: corrct.alignment.centering.RecenterVolume

```{autodoc2-docstring} corrct.alignment.centering.RecenterVolume
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.alignment.centering.RecenterVolume.__init__
```

````{py:method} _apply_displacement_vu(shifts_vu: numpy.typing.NDArray, displacemenet_zyx: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.alignment.centering.RecenterVolume._apply_displacement_vu

```{autodoc2-docstring} corrct.alignment.centering.RecenterVolume._apply_displacement_vu
```

````

````{py:method} to_com(shifts_vu: numpy.typing.ArrayLike | numpy.typing.NDArray, volume: numpy.typing.NDArray, com_ref_zyx: numpy.typing.ArrayLike | numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.alignment.centering.RecenterVolume.to_com

```{autodoc2-docstring} corrct.alignment.centering.RecenterVolume.to_com
```

````

````{py:method} as_reference(shifts_vu: numpy.typing.NDArray, volume: numpy.typing.NDArray, reference: numpy.typing.NDArray, method: str = 'com') -> numpy.typing.NDArray
:canonical: corrct.alignment.centering.RecenterVolume.as_reference

```{autodoc2-docstring} corrct.alignment.centering.RecenterVolume.as_reference
```

````

`````
