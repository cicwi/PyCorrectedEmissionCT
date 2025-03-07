# {py:mod}`corrct._projector_backends`

```{py:module} corrct._projector_backends
```

```{autodoc2-docstring} corrct._projector_backends
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProjectorBackend <corrct._projector_backends.ProjectorBackend>`
  - ```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend
    :summary:
    ```
* - {py:obj}`ProjectorBackendSKimage <corrct._projector_backends.ProjectorBackendSKimage>`
  - ```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendSKimage
    :summary:
    ```
* - {py:obj}`ProjectorBackendASTRA <corrct._projector_backends.ProjectorBackendASTRA>`
  - ```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA
    :summary:
    ```
* - {py:obj}`ProjectorBackendDirectASTRA <corrct._projector_backends.ProjectorBackendDirectASTRA>`
  - ```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_attenuation <corrct._projector_backends.compute_attenuation>`
  - ```{autodoc2-docstring} corrct._projector_backends.compute_attenuation
    :summary:
    ```
````

### API

````{py:function} compute_attenuation(vol: numpy.typing.NDArray, angle_rad: float, invert: bool = False) -> numpy.typing.NDArray
:canonical: corrct._projector_backends.compute_attenuation

```{autodoc2-docstring} corrct._projector_backends.compute_attenuation
```
````

`````{py:class} ProjectorBackend()
:canonical: corrct._projector_backends.ProjectorBackend

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.__init__
```

````{py:attribute} vol_geom
:canonical: corrct._projector_backends.ProjectorBackend.vol_geom
:type: corrct.models.VolumeGeometry
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.vol_geom
```

````

````{py:attribute} vol_shape_zxy
:canonical: corrct._projector_backends.ProjectorBackend.vol_shape_zxy
:type: numpy.typing.NDArray[numpy.integer]
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.vol_shape_zxy
```

````

````{py:attribute} angles_w_rad
:canonical: corrct._projector_backends.ProjectorBackend.angles_w_rad
:type: numpy.typing.NDArray[numpy.floating]
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.angles_w_rad
```

````

````{py:attribute} prj_shape_vwu
:canonical: corrct._projector_backends.ProjectorBackend.prj_shape_vwu
:type: numpy.typing.NDArray[numpy.integer]
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.prj_shape_vwu
```

````

````{py:attribute} prj_shape_vu
:canonical: corrct._projector_backends.ProjectorBackend.prj_shape_vu
:type: numpy.typing.NDArray[numpy.integer]
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.prj_shape_vu
```

````

````{py:attribute} is_initialized
:canonical: corrct._projector_backends.ProjectorBackend.is_initialized
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.is_initialized
```

````

````{py:attribute} is_ready
:canonical: corrct._projector_backends.ProjectorBackend.is_ready
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.is_ready
```

````

````{py:method} initialize_geometry(vol_geom: corrct.models.VolumeGeometry, angles_rot_rad: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray], rot_axis_shift_pix: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None, prj_geom: typing.Optional[corrct.models.ProjectionGeometry] = None, create_single_projs: bool = False)
:canonical: corrct._projector_backends.ProjectorBackend.initialize_geometry

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.initialize_geometry
```

````

````{py:method} get_vol_shape() -> numpy.typing.NDArray
:canonical: corrct._projector_backends.ProjectorBackend.get_vol_shape

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.get_vol_shape
```

````

````{py:method} get_prj_shape() -> numpy.typing.NDArray
:canonical: corrct._projector_backends.ProjectorBackend.get_prj_shape

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.get_prj_shape
```

````

````{py:method} make_ready() -> None
:canonical: corrct._projector_backends.ProjectorBackend.make_ready

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.make_ready
```

````

````{py:method} dispose() -> None
:canonical: corrct._projector_backends.ProjectorBackend.dispose

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.dispose
```

````

````{py:method} __del__()
:canonical: corrct._projector_backends.ProjectorBackend.__del__

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.__del__
```

````

````{py:method} __repr__() -> str
:canonical: corrct._projector_backends.ProjectorBackend.__repr__

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.__repr__
```

````

````{py:method} fp(vol: numpy.typing.NDArray, angle_ind: typing.Optional[int] = None) -> numpy.typing.NDArray
:canonical: corrct._projector_backends.ProjectorBackend.fp
:abstractmethod:

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.fp
```

````

````{py:method} bp(prj: numpy.typing.NDArray, angle_ind: typing.Optional[int] = None) -> numpy.typing.NDArray
:canonical: corrct._projector_backends.ProjectorBackend.bp
:abstractmethod:

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackend.bp
```

````

`````

`````{py:class} ProjectorBackendSKimage()
:canonical: corrct._projector_backends.ProjectorBackendSKimage

Bases: {py:obj}`corrct._projector_backends.ProjectorBackend`

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendSKimage
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendSKimage.__init__
```

````{py:method} initialize_geometry(vol_geom: corrct.models.VolumeGeometry, angles_rot_rad: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray], rot_axis_shift_pix: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None, prj_geom: typing.Optional[corrct.models.ProjectionGeometry] = None, create_single_projs: bool = False)
:canonical: corrct._projector_backends.ProjectorBackendSKimage.initialize_geometry

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendSKimage.initialize_geometry
```

````

````{py:method} _set_filter_name(filt)
:canonical: corrct._projector_backends.ProjectorBackendSKimage._set_filter_name
:staticmethod:

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendSKimage._set_filter_name
```

````

````{py:method} _set_bpj_size(output_size)
:canonical: corrct._projector_backends.ProjectorBackendSKimage._set_bpj_size
:staticmethod:

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendSKimage._set_bpj_size
```

````

````{py:method} fp(vol: numpy.typing.NDArray, angle_ind: typing.Optional[int] = None) -> numpy.typing.NDArray
:canonical: corrct._projector_backends.ProjectorBackendSKimage.fp

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendSKimage.fp
```

````

````{py:method} bp(prj: numpy.typing.NDArray, angle_ind: typing.Optional[int] = None) -> numpy.typing.NDArray
:canonical: corrct._projector_backends.ProjectorBackendSKimage.bp

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendSKimage.bp
```

````

`````

`````{py:class} ProjectorBackendASTRA(super_sampling: int = 1)
:canonical: corrct._projector_backends.ProjectorBackendASTRA

Bases: {py:obj}`corrct._projector_backends.ProjectorBackend`

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.__init__
```

````{py:attribute} proj_id
:canonical: corrct._projector_backends.ProjectorBackendASTRA.proj_id
:type: typing.List
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.proj_id
```

````

````{py:attribute} astra_vol_geom
:canonical: corrct._projector_backends.ProjectorBackendASTRA.astra_vol_geom
:type: typing.Mapping
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.astra_vol_geom
```

````

````{py:attribute} proj_geom_ind
:canonical: corrct._projector_backends.ProjectorBackendASTRA.proj_geom_ind
:type: typing.Sequence[typing.Mapping]
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.proj_geom_ind
```

````

````{py:attribute} proj_geom_all
:canonical: corrct._projector_backends.ProjectorBackendASTRA.proj_geom_all
:type: typing.Mapping
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.proj_geom_all
```

````

````{py:method} initialize_geometry(vol_geom: corrct.models.VolumeGeometry, angles_rot_rad: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray], rot_axis_shift_pix: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None, prj_geom: typing.Optional[corrct.models.ProjectionGeometry] = None, create_single_projs: bool = False)
:canonical: corrct._projector_backends.ProjectorBackendASTRA.initialize_geometry

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.initialize_geometry
```

````

````{py:method} make_ready() -> None
:canonical: corrct._projector_backends.ProjectorBackendASTRA.make_ready

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.make_ready
```

````

````{py:method} _check_data(x: numpy.typing.NDArray, expected_shape: typing.Union[typing.Sequence[int], numpy.typing.NDArray[numpy.integer]]) -> numpy.typing.NDArray
:canonical: corrct._projector_backends.ProjectorBackendASTRA._check_data

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA._check_data
```

````

````{py:method} _check_prj_shape(prj: numpy.typing.NDArray) -> None
:canonical: corrct._projector_backends.ProjectorBackendASTRA._check_prj_shape

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA._check_prj_shape
```

````

````{py:method} dispose() -> None
:canonical: corrct._projector_backends.ProjectorBackendASTRA.dispose

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.dispose
```

````

````{py:method} fp(vol: numpy.typing.NDArray, angle_ind: typing.Optional[int] = None) -> numpy.typing.NDArray
:canonical: corrct._projector_backends.ProjectorBackendASTRA.fp

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.fp
```

````

````{py:method} bp(prj: numpy.typing.NDArray, angle_ind: typing.Optional[int] = None) -> numpy.typing.NDArray
:canonical: corrct._projector_backends.ProjectorBackendASTRA.bp

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendASTRA.bp
```

````

`````

`````{py:class} ProjectorBackendDirectASTRA(super_sampling: int = 1)
:canonical: corrct._projector_backends.ProjectorBackendDirectASTRA

Bases: {py:obj}`corrct._projector_backends.ProjectorBackendASTRA`

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA.__init__
```

````{py:attribute} astra_vol_shape
:canonical: corrct._projector_backends.ProjectorBackendDirectASTRA.astra_vol_shape
:type: typing.Sequence
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA.astra_vol_shape
```

````

````{py:attribute} astra_prj_shape
:canonical: corrct._projector_backends.ProjectorBackendDirectASTRA.astra_prj_shape
:type: typing.Sequence
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA.astra_prj_shape
```

````

````{py:attribute} astra_angle_prj_shape
:canonical: corrct._projector_backends.ProjectorBackendDirectASTRA.astra_angle_prj_shape
:type: typing.Sequence
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA.astra_angle_prj_shape
```

````

````{py:attribute} angle_prj_shape
:canonical: corrct._projector_backends.ProjectorBackendDirectASTRA.angle_prj_shape
:type: typing.Sequence
:value: >
   None

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA.angle_prj_shape
```

````

````{py:method} initialize_geometry(vol_geom: corrct.models.VolumeGeometry, angles_rot_rad: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray], rot_axis_shift_pix: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None, prj_geom: typing.Optional[corrct.models.ProjectionGeometry] = None, create_single_projs: bool = False)
:canonical: corrct._projector_backends.ProjectorBackendDirectASTRA.initialize_geometry

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA.initialize_geometry
```

````

````{py:method} make_ready()
:canonical: corrct._projector_backends.ProjectorBackendDirectASTRA.make_ready

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA.make_ready
```

````

````{py:method} fp(vol: numpy.typing.NDArray, angle_ind: typing.Optional[int] = None)
:canonical: corrct._projector_backends.ProjectorBackendDirectASTRA.fp

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA.fp
```

````

````{py:method} bp(prj: numpy.typing.NDArray, angle_ind: typing.Optional[int] = None)
:canonical: corrct._projector_backends.ProjectorBackendDirectASTRA.bp

```{autodoc2-docstring} corrct._projector_backends.ProjectorBackendDirectASTRA.bp
```

````

`````
