# {py:mod}`corrct.alignment.cone_beam`

```{py:module} corrct.alignment.cone_beam
```

```{autodoc2-docstring} corrct.alignment.cone_beam
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConeBeamGeometry <corrct.alignment.cone_beam.ConeBeamGeometry>`
  - ```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry
    :summary:
    ```
* - {py:obj}`FitConeBeamGeometry <corrct.alignment.cone_beam.FitConeBeamGeometry>`
  - ```{autodoc2-docstring} corrct.alignment.cone_beam.FitConeBeamGeometry
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_class_to_json <corrct.alignment.cone_beam._class_to_json>`
  - ```{autodoc2-docstring} corrct.alignment.cone_beam._class_to_json
    :summary:
    ```
* - {py:obj}`_get_rot_axis_angle_deg <corrct.alignment.cone_beam._get_rot_axis_angle_deg>`
  - ```{autodoc2-docstring} corrct.alignment.cone_beam._get_rot_axis_angle_deg
    :summary:
    ```
* - {py:obj}`tune_acquisition_geometry <corrct.alignment.cone_beam.tune_acquisition_geometry>`
  - ```{autodoc2-docstring} corrct.alignment.cone_beam.tune_acquisition_geometry
    :summary:
    ```
````

### API

````{py:function} _class_to_json(obj: object) -> str
:canonical: corrct.alignment.cone_beam._class_to_json

```{autodoc2-docstring} corrct.alignment.cone_beam._class_to_json
```
````

````{py:function} _get_rot_axis_angle_deg(center_1_vu: collections.abc.Sequence[float] | numpy.typing.NDArray, center_2_vu: collections.abc.Sequence[float] | numpy.typing.NDArray, decimals: int | None = 4, dtype: numpy.typing.DTypeLike = np.float32) -> float
:canonical: corrct.alignment.cone_beam._get_rot_axis_angle_deg

```{autodoc2-docstring} corrct.alignment.cone_beam._get_rot_axis_angle_deg
```
````

`````{py:class} ConeBeamGeometry
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry
```

````{py:attribute} theta_deg
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.theta_deg
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.theta_deg
```

````

````{py:attribute} phi_deg
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.phi_deg
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.phi_deg
```

````

````{py:attribute} eta_deg
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.eta_deg
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.eta_deg
```

````

````{py:attribute} D_pix
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.D_pix
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.D_pix
```

````

````{py:attribute} R_pix
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.R_pix
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.R_pix
```

````

````{py:attribute} v0_pix
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.v0_pix
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.v0_pix
```

````

````{py:attribute} u0_pix
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.u0_pix
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.u0_pix
```

````

````{py:attribute} det_size_v_pix
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.det_size_v_pix
:type: int
:value: >
   0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.det_size_v_pix
```

````

````{py:attribute} det_size_u_pix
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.det_size_u_pix
:type: int
:value: >
   0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.det_size_u_pix
```

````

````{py:attribute} pix_size_um
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.pix_size_um
:type: float
:value: >
   0.0

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.pix_size_um
```

````

````{py:method} __str__() -> str
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.__str__

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.__str__
```

````

````{py:method} get_prj_geom(translate_z_to_center: bool = True) -> corrct.models.ProjectionGeometry
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.get_prj_geom

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.get_prj_geom
```

````

````{py:method} get_vol_geom(up_sampling: int = 1) -> corrct.models.VolumeGeometry
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.get_vol_geom

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.get_vol_geom
```

````

````{py:method} update(field: str, val: float, is_relative: bool = True, decimals: int | None = 3) -> corrct.alignment.cone_beam.ConeBeamGeometry
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.update

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.update
```

````

````{py:method} get_tuning_params(field: str, val_range: collections.abc.Sequence[float] | numpy.typing.NDArray, is_relative: bool = True) -> collections.abc.Sequence[corrct.alignment.cone_beam.ConeBeamGeometry]
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.get_tuning_params

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.get_tuning_params
```

````

````{py:method} to_json() -> str
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.to_json

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.to_json
```

````

````{py:method} from_json(data_json: str) -> None
:canonical: corrct.alignment.cone_beam.ConeBeamGeometry.from_json

```{autodoc2-docstring} corrct.alignment.cone_beam.ConeBeamGeometry.from_json
```

````

`````

`````{py:class} FitConeBeamGeometry(prj_size_vu: collections.abc.Sequence[int] | numpy.typing.NDArray, points_ell1: collections.abc.Sequence[collections.abc.Sequence[float]] | numpy.typing.NDArray, points_ell2: collections.abc.Sequence[collections.abc.Sequence[float]] | numpy.typing.NDArray, points_axis: collections.abc.Sequence[collections.abc.Sequence[float]] | numpy.typing.NDArray | None = None, pix_size_um: float | None = None, use_l1_norm: bool = False, verbose: bool = True, plot_result: bool = False)
:canonical: corrct.alignment.cone_beam.FitConeBeamGeometry

```{autodoc2-docstring} corrct.alignment.cone_beam.FitConeBeamGeometry
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.alignment.cone_beam.FitConeBeamGeometry.__init__
```

````{py:attribute} acq_geom
:canonical: corrct.alignment.cone_beam.FitConeBeamGeometry.acq_geom
:type: corrct.alignment.cone_beam.ConeBeamGeometry
:value: >
   None

```{autodoc2-docstring} corrct.alignment.cone_beam.FitConeBeamGeometry.acq_geom
```

````

````{py:method} _initialize(use_l1_norm: bool) -> None
:canonical: corrct.alignment.cone_beam.FitConeBeamGeometry._initialize

```{autodoc2-docstring} corrct.alignment.cone_beam.FitConeBeamGeometry._initialize
```

````

````{py:method} fit(r: float, e: float = 1, meas_D_pix: float | None = None) -> corrct.alignment.cone_beam.ConeBeamGeometry
:canonical: corrct.alignment.cone_beam.FitConeBeamGeometry.fit

```{autodoc2-docstring} corrct.alignment.cone_beam.FitConeBeamGeometry.fit
```

````

````{py:method} _fit_distance_det2src(ellipse_1: corrct.alignment.fitting.Ellipse, ellipse_2: corrct.alignment.fitting.Ellipse, e: float = 1) -> float
:canonical: corrct.alignment.cone_beam.FitConeBeamGeometry._fit_distance_det2src
:staticmethod:

```{autodoc2-docstring} corrct.alignment.cone_beam.FitConeBeamGeometry._fit_distance_det2src
```

````

`````

````{py:function} tune_acquisition_geometry(acq_geom_init: corrct.alignment.cone_beam.ConeBeamGeometry, data: numpy.typing.NDArray, angles_rot_rad: collections.abc.Sequence[float] | numpy.typing.NDArray, params: dict[str, collections.abc.Sequence[float] | numpy.typing.NDArray], data_mask: numpy.typing.NDArray | None = None, verbose: bool = True) -> corrct.alignment.cone_beam.ConeBeamGeometry
:canonical: corrct.alignment.cone_beam.tune_acquisition_geometry

```{autodoc2-docstring} corrct.alignment.cone_beam.tune_acquisition_geometry
```
````
