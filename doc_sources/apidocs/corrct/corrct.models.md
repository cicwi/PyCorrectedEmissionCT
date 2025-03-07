# {py:mod}`corrct.models`

```{py:module} corrct.models
```

```{autodoc2-docstring} corrct.models
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Geometry <corrct.models.Geometry>`
  - ```{autodoc2-docstring} corrct.models.Geometry
    :summary:
    ```
* - {py:obj}`ProjectionGeometry <corrct.models.ProjectionGeometry>`
  - ```{autodoc2-docstring} corrct.models.ProjectionGeometry
    :summary:
    ```
* - {py:obj}`VolumeGeometry <corrct.models.VolumeGeometry>`
  - ```{autodoc2-docstring} corrct.models.VolumeGeometry
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`combine_shifts_vu <corrct.models.combine_shifts_vu>`
  - ```{autodoc2-docstring} corrct.models.combine_shifts_vu
    :summary:
    ```
* - {py:obj}`get_rot_axis_dir <corrct.models.get_rot_axis_dir>`
  - ```{autodoc2-docstring} corrct.models.get_rot_axis_dir
    :summary:
    ```
* - {py:obj}`_get_data_dims <corrct.models._get_data_dims>`
  - ```{autodoc2-docstring} corrct.models._get_data_dims
    :summary:
    ```
* - {py:obj}`get_prj_geom_parallel <corrct.models.get_prj_geom_parallel>`
  - ```{autodoc2-docstring} corrct.models.get_prj_geom_parallel
    :summary:
    ```
* - {py:obj}`get_prj_geom_cone <corrct.models.get_prj_geom_cone>`
  - ```{autodoc2-docstring} corrct.models.get_prj_geom_cone
    :summary:
    ```
* - {py:obj}`get_vol_geom_from_data <corrct.models.get_vol_geom_from_data>`
  - ```{autodoc2-docstring} corrct.models.get_vol_geom_from_data
    :summary:
    ```
* - {py:obj}`get_vol_geom_from_volume <corrct.models.get_vol_geom_from_volume>`
  - ```{autodoc2-docstring} corrct.models.get_vol_geom_from_volume
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ROT_DIRS_VALID <corrct.models.ROT_DIRS_VALID>`
  - ```{autodoc2-docstring} corrct.models.ROT_DIRS_VALID
    :summary:
    ```
````

### API

````{py:data} ROT_DIRS_VALID
:canonical: corrct.models.ROT_DIRS_VALID
:value: >
   ('clockwise', 'counter-clockwise')

```{autodoc2-docstring} corrct.models.ROT_DIRS_VALID
```

````

`````{py:class} Geometry
:canonical: corrct.models.Geometry

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.models.Geometry
```

````{py:method} __str__() -> str
:canonical: corrct.models.Geometry.__str__

```{autodoc2-docstring} corrct.models.Geometry.__str__
```

````

`````

`````{py:class} ProjectionGeometry
:canonical: corrct.models.ProjectionGeometry

Bases: {py:obj}`corrct.models.Geometry`

```{autodoc2-docstring} corrct.models.ProjectionGeometry
```

````{py:attribute} geom_type
:canonical: corrct.models.ProjectionGeometry.geom_type
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.models.ProjectionGeometry.geom_type
```

````

````{py:attribute} src_pos_xyz
:canonical: corrct.models.ProjectionGeometry.src_pos_xyz
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.models.ProjectionGeometry.src_pos_xyz
```

````

````{py:attribute} det_pos_xyz
:canonical: corrct.models.ProjectionGeometry.det_pos_xyz
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.models.ProjectionGeometry.det_pos_xyz
```

````

````{py:attribute} det_u_xyz
:canonical: corrct.models.ProjectionGeometry.det_u_xyz
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.models.ProjectionGeometry.det_u_xyz
```

````

````{py:attribute} det_v_xyz
:canonical: corrct.models.ProjectionGeometry.det_v_xyz
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.models.ProjectionGeometry.det_v_xyz
```

````

````{py:attribute} rot_dir_xyz
:canonical: corrct.models.ProjectionGeometry.rot_dir_xyz
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.models.ProjectionGeometry.rot_dir_xyz
```

````

````{py:attribute} pix2vox_ratio
:canonical: corrct.models.ProjectionGeometry.pix2vox_ratio
:type: float
:value: >
   1

```{autodoc2-docstring} corrct.models.ProjectionGeometry.pix2vox_ratio
```

````

````{py:attribute} det_shape_vu
:canonical: corrct.models.ProjectionGeometry.det_shape_vu
:type: typing.Optional[numpy.typing.NDArray]
:value: >
   None

```{autodoc2-docstring} corrct.models.ProjectionGeometry.det_shape_vu
```

````

````{py:method} __post_init__() -> None
:canonical: corrct.models.ProjectionGeometry.__post_init__

```{autodoc2-docstring} corrct.models.ProjectionGeometry.__post_init__
```

````

````{py:method} __getitem__(indx: typing.Any)
:canonical: corrct.models.ProjectionGeometry.__getitem__

```{autodoc2-docstring} corrct.models.ProjectionGeometry.__getitem__
```

````

````{py:method} copy() -> corrct.models.ProjectionGeometry
:canonical: corrct.models.ProjectionGeometry.copy

```{autodoc2-docstring} corrct.models.ProjectionGeometry.copy
```

````

````{py:method} get_default_parallel(*, geom_type: str = '3d', rot_axis_shift_pix: typing.Optional[numpy.typing.ArrayLike] = None, rot_axis_dir: typing.Union[str, numpy.typing.ArrayLike] = 'clockwise') -> corrct.models.ProjectionGeometry
:canonical: corrct.models.ProjectionGeometry.get_default_parallel
:staticmethod:

```{autodoc2-docstring} corrct.models.ProjectionGeometry.get_default_parallel
```

````

````{py:property} ndim
:canonical: corrct.models.ProjectionGeometry.ndim
:type: int

```{autodoc2-docstring} corrct.models.ProjectionGeometry.ndim
```

````

````{py:method} get_3d() -> corrct.models.ProjectionGeometry
:canonical: corrct.models.ProjectionGeometry.get_3d

```{autodoc2-docstring} corrct.models.ProjectionGeometry.get_3d
```

````

````{py:method} set_detector_shape_vu(vu: typing.Union[int, typing.Sequence[int], numpy.typing.NDArray]) -> None
:canonical: corrct.models.ProjectionGeometry.set_detector_shape_vu

```{autodoc2-docstring} corrct.models.ProjectionGeometry.set_detector_shape_vu
```

````

````{py:method} set_detector_shifts_vu(det_pos_vu: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None, cor_pos_u: typing.Union[float, None] = None, det_dist_y: numpy.typing.ArrayLike = 0.0) -> None
:canonical: corrct.models.ProjectionGeometry.set_detector_shifts_vu

```{autodoc2-docstring} corrct.models.ProjectionGeometry.set_detector_shifts_vu
```

````

````{py:method} set_source_shifts_vu(src_pos_vu: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None) -> None
:canonical: corrct.models.ProjectionGeometry.set_source_shifts_vu

```{autodoc2-docstring} corrct.models.ProjectionGeometry.set_source_shifts_vu
```

````

````{py:method} set_detector_tilt(angles_t_rad: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray], tilt_axis: typing.Union[typing.Sequence[float], numpy.typing.NDArray] = (0, 1, 0), tilt_source: bool = False) -> None
:canonical: corrct.models.ProjectionGeometry.set_detector_tilt

```{autodoc2-docstring} corrct.models.ProjectionGeometry.set_detector_tilt
```

````

````{py:method} rotate(angles_w_rad: numpy.typing.ArrayLike, patch_astra_2d: bool = False) -> corrct.models.ProjectionGeometry
:canonical: corrct.models.ProjectionGeometry.rotate

```{autodoc2-docstring} corrct.models.ProjectionGeometry.rotate
```

````

````{py:method} get_field_scaled(field_name: str) -> numpy.typing.NDArray
:canonical: corrct.models.ProjectionGeometry.get_field_scaled

```{autodoc2-docstring} corrct.models.ProjectionGeometry.get_field_scaled
```

````

````{py:method} project_displacement_to_detector(disp_zyx: numpy.typing.ArrayLike) -> numpy.typing.NDArray
:canonical: corrct.models.ProjectionGeometry.project_displacement_to_detector

```{autodoc2-docstring} corrct.models.ProjectionGeometry.project_displacement_to_detector
```

````

````{py:method} get_pre_weights(det_shape_vu: typing.Union[typing.Sequence[int], numpy.typing.NDArray, None] = None) -> typing.Union[numpy.typing.NDArray, None]
:canonical: corrct.models.ProjectionGeometry.get_pre_weights

```{autodoc2-docstring} corrct.models.ProjectionGeometry.get_pre_weights
```

````

`````

`````{py:class} VolumeGeometry
:canonical: corrct.models.VolumeGeometry

Bases: {py:obj}`corrct.models.Geometry`

```{autodoc2-docstring} corrct.models.VolumeGeometry
```

````{py:attribute} _vol_shape_xyz
:canonical: corrct.models.VolumeGeometry._vol_shape_xyz
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.models.VolumeGeometry._vol_shape_xyz
```

````

````{py:attribute} vox_size
:canonical: corrct.models.VolumeGeometry.vox_size
:type: float
:value: >
   1.0

```{autodoc2-docstring} corrct.models.VolumeGeometry.vox_size
```

````

````{py:method} __post_init__()
:canonical: corrct.models.VolumeGeometry.__post_init__

```{autodoc2-docstring} corrct.models.VolumeGeometry.__post_init__
```

````

````{py:method} is_square() -> bool
:canonical: corrct.models.VolumeGeometry.is_square

```{autodoc2-docstring} corrct.models.VolumeGeometry.is_square
```

````

````{py:property} shape_xyz
:canonical: corrct.models.VolumeGeometry.shape_xyz
:type: numpy.typing.NDArray

```{autodoc2-docstring} corrct.models.VolumeGeometry.shape_xyz
```

````

````{py:property} shape_zxy
:canonical: corrct.models.VolumeGeometry.shape_zxy
:type: numpy.typing.NDArray

```{autodoc2-docstring} corrct.models.VolumeGeometry.shape_zxy
```

````

````{py:property} mask_shape
:canonical: corrct.models.VolumeGeometry.mask_shape
:type: numpy.typing.NDArray

```{autodoc2-docstring} corrct.models.VolumeGeometry.mask_shape
```

````

````{py:property} extent
:canonical: corrct.models.VolumeGeometry.extent
:type: typing.Sequence[float]

```{autodoc2-docstring} corrct.models.VolumeGeometry.extent
```

````

````{py:method} is_3D() -> bool
:canonical: corrct.models.VolumeGeometry.is_3D

```{autodoc2-docstring} corrct.models.VolumeGeometry.is_3D
```

````

````{py:method} get_3d() -> corrct.models.VolumeGeometry
:canonical: corrct.models.VolumeGeometry.get_3d

```{autodoc2-docstring} corrct.models.VolumeGeometry.get_3d
```

````

````{py:method} get_default_from_data(data: numpy.typing.NDArray, data_format: str = 'dvwu') -> corrct.models.VolumeGeometry
:canonical: corrct.models.VolumeGeometry.get_default_from_data
:staticmethod:

```{autodoc2-docstring} corrct.models.VolumeGeometry.get_default_from_data
```

````

````{py:method} get_default_from_volume(volume: numpy.typing.NDArray) -> corrct.models.VolumeGeometry
:canonical: corrct.models.VolumeGeometry.get_default_from_volume
:staticmethod:

```{autodoc2-docstring} corrct.models.VolumeGeometry.get_default_from_volume
```

````

`````

````{py:function} combine_shifts_vu(shifts_v: numpy.typing.NDArray, shifts_u: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.models.combine_shifts_vu

```{autodoc2-docstring} corrct.models.combine_shifts_vu
```
````

````{py:function} get_rot_axis_dir(rot_axis_dir: typing.Union[str, numpy.typing.ArrayLike, numpy.typing.NDArray] = 'clockwise') -> numpy.typing.NDArray
:canonical: corrct.models.get_rot_axis_dir

```{autodoc2-docstring} corrct.models.get_rot_axis_dir
```
````

````{py:function} _get_data_dims(data_shape: typing.Union[typing.Sequence[int], numpy.typing.NDArray], data_format: str = 'dvwu') -> dict[str, typing.Union[int, None]]
:canonical: corrct.models._get_data_dims

```{autodoc2-docstring} corrct.models._get_data_dims
```
````

````{py:function} get_prj_geom_parallel(*, geom_type: str = '3d', rot_axis_shift_pix: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None, rot_axis_dir: typing.Union[str, numpy.typing.ArrayLike, numpy.typing.NDArray] = 'clockwise', data_shape: typing.Union[typing.Sequence[int], numpy.typing.NDArray, None] = None, data_format: str = 'dvwu') -> corrct.models.ProjectionGeometry
:canonical: corrct.models.get_prj_geom_parallel

```{autodoc2-docstring} corrct.models.get_prj_geom_parallel
```
````

````{py:function} get_prj_geom_cone(*, src_to_sam_dist: float, rot_axis_shift_pix: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray, None] = None, rot_axis_dir: typing.Union[str, numpy.typing.ArrayLike, numpy.typing.NDArray] = 'clockwise', data_shape: typing.Union[typing.Sequence[int], numpy.typing.NDArray, None] = None, data_format: str = 'dvwu') -> corrct.models.ProjectionGeometry
:canonical: corrct.models.get_prj_geom_cone

```{autodoc2-docstring} corrct.models.get_prj_geom_cone
```
````

````{py:function} get_vol_geom_from_data(data: numpy.typing.NDArray, padding_u: typing.Union[int, typing.Sequence[int], numpy.typing.NDArray] = 0, data_format: str = 'dvwu', super_sampling: int = 1) -> corrct.models.VolumeGeometry
:canonical: corrct.models.get_vol_geom_from_data

```{autodoc2-docstring} corrct.models.get_vol_geom_from_data
```
````

````{py:function} get_vol_geom_from_volume(volume: numpy.typing.NDArray) -> corrct.models.VolumeGeometry
:canonical: corrct.models.get_vol_geom_from_volume

```{autodoc2-docstring} corrct.models.get_vol_geom_from_volume
```
````
