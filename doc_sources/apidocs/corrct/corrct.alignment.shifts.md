# {py:mod}`corrct.alignment.shifts`

```{py:module} corrct.alignment.shifts
```

```{autodoc2-docstring} corrct.alignment.shifts
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DetectorShiftsBase <corrct.alignment.shifts.DetectorShiftsBase>`
  - ```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsBase
    :summary:
    ```
* - {py:obj}`DetectorShiftsPRE <corrct.alignment.shifts.DetectorShiftsPRE>`
  - ```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsPRE
    :summary:
    ```
* - {py:obj}`DetectorShiftsXC <corrct.alignment.shifts.DetectorShiftsXC>`
  - ```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsXC
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_filter_shifts <corrct.alignment.shifts._filter_shifts>`
  - ```{autodoc2-docstring} corrct.alignment.shifts._filter_shifts
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`eps <corrct.alignment.shifts.eps>`
  - ```{autodoc2-docstring} corrct.alignment.shifts.eps
    :summary:
    ```
* - {py:obj}`NDArrayFloat <corrct.alignment.shifts.NDArrayFloat>`
  - ```{autodoc2-docstring} corrct.alignment.shifts.NDArrayFloat
    :summary:
    ```
````

### API

````{py:data} eps
:canonical: corrct.alignment.shifts.eps
:value: >
   None

```{autodoc2-docstring} corrct.alignment.shifts.eps
```

````

````{py:data} NDArrayFloat
:canonical: corrct.alignment.shifts.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.alignment.shifts.NDArrayFloat
```

````

````{py:function} _filter_shifts(shifts_vu: corrct.alignment.shifts.NDArrayFloat, max_shifts: corrct.alignment.shifts.NDArrayFloat) -> corrct.alignment.shifts.NDArrayFloat
:canonical: corrct.alignment.shifts._filter_shifts

```{autodoc2-docstring} corrct.alignment.shifts._filter_shifts
```
````

`````{py:class} DetectorShiftsBase(data_dvwu: corrct.alignment.shifts.NDArrayFloat, rot_angle_rad: numpy.typing.ArrayLike | corrct.alignment.shifts.NDArrayFloat, *, data_format: str = 'dvwu', data_mask_dvwu: numpy.typing.NDArray | None = None, borders_dvwu: dict = {'d': None, 'v': None, 'w': None, 'u': None}, max_shifts: float | corrct.alignment.shifts.NDArrayFloat | None = None, precision_decimals: int = 2, verbose: bool = True)
:canonical: corrct.alignment.shifts.DetectorShiftsBase

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsBase.__init__
```

````{py:attribute} data_vwu
:canonical: corrct.alignment.shifts.DetectorShiftsBase.data_vwu
:type: corrct.alignment.shifts.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsBase.data_vwu
```

````

````{py:attribute} angles_rad
:canonical: corrct.alignment.shifts.DetectorShiftsBase.angles_rad
:type: corrct.alignment.shifts.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsBase.angles_rad
```

````

`````

`````{py:class} DetectorShiftsPRE(data_dvwu: corrct.alignment.shifts.NDArrayFloat, rot_angle_rad: numpy.typing.ArrayLike | corrct.alignment.shifts.NDArrayFloat, *, data_format: str = 'dvwu', data_mask_dvwu: numpy.typing.NDArray | None = None, borders_dvwu: dict = {'d': None, 'v': None, 'w': None, 'u': None}, max_shifts: float | corrct.alignment.shifts.NDArrayFloat | None = None, precision_decimals: int = 2, verbose: bool = True)
:canonical: corrct.alignment.shifts.DetectorShiftsPRE

Bases: {py:obj}`corrct.alignment.shifts.DetectorShiftsBase`

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsPRE
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsPRE.__init__
```

````{py:method} fit_v(use_derivative: bool = True, use_rfft: bool = True, normalize_fourier: bool = True) -> corrct.alignment.shifts.NDArrayFloat
:canonical: corrct.alignment.shifts.DetectorShiftsPRE.fit_v

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsPRE.fit_v
```

````

````{py:method} fit_u(fit_l1: bool = False, background: float | numpy.typing.NDArray | None = None, method: str = 'com') -> tuple[corrct.alignment.shifts.NDArrayFloat, float]
:canonical: corrct.alignment.shifts.DetectorShiftsPRE.fit_u

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsPRE.fit_u
```

````

`````

`````{py:class} DetectorShiftsXC(data_dvwu: corrct.alignment.shifts.NDArrayFloat, rot_angle_rad: numpy.typing.ArrayLike | corrct.alignment.shifts.NDArrayFloat, *, data_format: str = 'dvwu', data_mask_dvwu: numpy.typing.NDArray | None = None, borders_dvwu: dict = {'d': None, 'v': None, 'w': None, 'u': None}, max_shifts: float | corrct.alignment.shifts.NDArrayFloat | None = None, precision_decimals: int = 2, verbose: bool = True)
:canonical: corrct.alignment.shifts.DetectorShiftsXC

Bases: {py:obj}`corrct.alignment.shifts.DetectorShiftsBase`

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsXC
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsXC.__init__
```

````{py:method} fit_vu_accum_drifts(ref_data_dvwu: corrct.alignment.shifts.NDArrayFloat | None = None) -> numpy.typing.NDArray
:canonical: corrct.alignment.shifts.DetectorShiftsXC.fit_vu_accum_drifts

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsXC.fit_vu_accum_drifts
```

````

````{py:method} fit_vu(fit_l1: bool = False) -> numpy.typing.NDArray
:canonical: corrct.alignment.shifts.DetectorShiftsXC.fit_vu

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsXC.fit_vu
```

````

````{py:method} fit_u_180() -> float
:canonical: corrct.alignment.shifts.DetectorShiftsXC.fit_u_180

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsXC.fit_u_180
```

````

````{py:method} fit_u_360() -> float
:canonical: corrct.alignment.shifts.DetectorShiftsXC.fit_u_360

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsXC.fit_u_360
```

````

````{py:method} find_shifts_vu(data_dvwu: corrct.alignment.shifts.NDArrayFloat, proj_dvwu: corrct.alignment.shifts.NDArrayFloat, use_derivative: bool = False, xc_opts: collections.abc.Mapping = dict(normalize_fourier=False)) -> corrct.alignment.shifts.NDArrayFloat
:canonical: corrct.alignment.shifts.DetectorShiftsXC.find_shifts_vu

```{autodoc2-docstring} corrct.alignment.shifts.DetectorShiftsXC.find_shifts_vu
```

````

`````
