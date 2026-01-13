# {py:mod}`corrct.physics.xrf`

```{py:module} corrct.physics.xrf
```

```{autodoc2-docstring} corrct.physics.xrf
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FluoLine <corrct.physics.xrf.FluoLine>`
  - ```{autodoc2-docstring} corrct.physics.xrf.FluoLine
    :summary:
    ```
* - {py:obj}`LinesSiegbahn <corrct.physics.xrf.LinesSiegbahn>`
  - ```{autodoc2-docstring} corrct.physics.xrf.LinesSiegbahn
    :summary:
    ```
* - {py:obj}`DetectorXRF <corrct.physics.xrf.DetectorXRF>`
  - ```{autodoc2-docstring} corrct.physics.xrf.DetectorXRF
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_lines_list <corrct.physics.xrf._get_lines_list>`
  - ```{autodoc2-docstring} corrct.physics.xrf._get_lines_list
    :summary:
    ```
* - {py:obj}`get_radiation_rate <corrct.physics.xrf.get_radiation_rate>`
  - ```{autodoc2-docstring} corrct.physics.xrf.get_radiation_rate
    :summary:
    ```
* - {py:obj}`get_energy <corrct.physics.xrf.get_energy>`
  - ```{autodoc2-docstring} corrct.physics.xrf.get_energy
    :summary:
    ```
````

### API

`````{py:class} FluoLine
:canonical: corrct.physics.xrf.FluoLine

```{autodoc2-docstring} corrct.physics.xrf.FluoLine
```

````{py:attribute} name
:canonical: corrct.physics.xrf.FluoLine.name
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.physics.xrf.FluoLine.name
```

````

````{py:attribute} indx
:canonical: corrct.physics.xrf.FluoLine.indx
:type: int
:value: >
   None

```{autodoc2-docstring} corrct.physics.xrf.FluoLine.indx
```

````

`````

`````{py:class} LinesSiegbahn
:canonical: corrct.physics.xrf.LinesSiegbahn

```{autodoc2-docstring} corrct.physics.xrf.LinesSiegbahn
```

````{py:attribute} lines
:canonical: corrct.physics.xrf.LinesSiegbahn.lines
:value: >
   None

```{autodoc2-docstring} corrct.physics.xrf.LinesSiegbahn.lines
```

````

````{py:method} get_lines(line: str) -> collections.abc.Sequence[corrct.physics.xrf.FluoLine]
:canonical: corrct.physics.xrf.LinesSiegbahn.get_lines
:staticmethod:

```{autodoc2-docstring} corrct.physics.xrf.LinesSiegbahn.get_lines
```

````

`````

````{py:function} _get_lines_list(lines) -> collections.abc.Sequence[corrct.physics.xrf.FluoLine]
:canonical: corrct.physics.xrf._get_lines_list

```{autodoc2-docstring} corrct.physics.xrf._get_lines_list
```
````

````{py:function} get_radiation_rate(element: str | int, lines: str | corrct.physics.xrf.FluoLine | collections.abc.Sequence[corrct.physics.xrf.FluoLine], verbose: bool = False) -> numpy.typing.NDArray
:canonical: corrct.physics.xrf.get_radiation_rate

```{autodoc2-docstring} corrct.physics.xrf.get_radiation_rate
```
````

````{py:function} get_energy(element: str | int, lines: str | corrct.physics.xrf.FluoLine | collections.abc.Sequence[corrct.physics.xrf.FluoLine], *, compute_average: bool = False, verbose: bool = False) -> float | numpy.typing.NDArray
:canonical: corrct.physics.xrf.get_energy

```{autodoc2-docstring} corrct.physics.xrf.get_energy
```
````

`````{py:class} DetectorXRF
:canonical: corrct.physics.xrf.DetectorXRF

```{autodoc2-docstring} corrct.physics.xrf.DetectorXRF
```

````{py:attribute} surface_mm2
:canonical: corrct.physics.xrf.DetectorXRF.surface_mm2
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.physics.xrf.DetectorXRF.surface_mm2
```

````

````{py:attribute} distance_mm
:canonical: corrct.physics.xrf.DetectorXRF.distance_mm
:type: float | numpy.typing.NDArray
:value: >
   1.0

```{autodoc2-docstring} corrct.physics.xrf.DetectorXRF.distance_mm
```

````

````{py:attribute} angle_rad
:canonical: corrct.physics.xrf.DetectorXRF.angle_rad
:type: float
:value: >
   None

```{autodoc2-docstring} corrct.physics.xrf.DetectorXRF.angle_rad
```

````

````{py:method} __post_init__()
:canonical: corrct.physics.xrf.DetectorXRF.__post_init__

```{autodoc2-docstring} corrct.physics.xrf.DetectorXRF.__post_init__
```

````

````{py:property} solid_angle_sr
:canonical: corrct.physics.xrf.DetectorXRF.solid_angle_sr
:type: float | numpy.typing.NDArray

```{autodoc2-docstring} corrct.physics.xrf.DetectorXRF.solid_angle_sr
```

````

````{py:property} angle_range_rad
:canonical: corrct.physics.xrf.DetectorXRF.angle_range_rad
:type: tuple[float, float]

```{autodoc2-docstring} corrct.physics.xrf.DetectorXRF.angle_range_rad
```

````

`````
