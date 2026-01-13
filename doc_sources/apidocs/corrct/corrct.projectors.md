# {py:mod}`corrct.projectors`

```{py:module} corrct.projectors
```

```{autodoc2-docstring} corrct.projectors
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProjectorMatrix <corrct.projectors.ProjectorMatrix>`
  - ```{autodoc2-docstring} corrct.projectors.ProjectorMatrix
    :summary:
    ```
* - {py:obj}`ProjectorUncorrected <corrct.projectors.ProjectorUncorrected>`
  - ```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected
    :summary:
    ```
* - {py:obj}`ProjectorAttenuationXRF <corrct.projectors.ProjectorAttenuationXRF>`
  - ```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`num_threads <corrct.projectors.num_threads>`
  - ```{autodoc2-docstring} corrct.projectors.num_threads
    :summary:
    ```
* - {py:obj}`astra_available <corrct.projectors.astra_available>`
  - ```{autodoc2-docstring} corrct.projectors.astra_available
    :summary:
    ```
````

### API

````{py:data} num_threads
:canonical: corrct.projectors.num_threads
:value: >
   'round(...)'

```{autodoc2-docstring} corrct.projectors.num_threads
```

````

````{py:data} astra_available
:canonical: corrct.projectors.astra_available
:value: >
   None

```{autodoc2-docstring} corrct.projectors.astra_available
```

````

`````{py:class} ProjectorMatrix(A: numpy.typing.NDArray | scipy.sparse.spmatrix, vol_shape: numpy.typing.ArrayLike, prj_shape: numpy.typing.ArrayLike)
:canonical: corrct.projectors.ProjectorMatrix

Bases: {py:obj}`corrct.operators.ProjectorOperator`

```{autodoc2-docstring} corrct.projectors.ProjectorMatrix
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.projectors.ProjectorMatrix.__init__
```

````{py:attribute} A
:canonical: corrct.projectors.ProjectorMatrix.A
:type: numpy.typing.NDArray | scipy.sparse.spmatrix
:value: >
   None

```{autodoc2-docstring} corrct.projectors.ProjectorMatrix.A
```

````

````{py:method} _transpose() -> corrct.operators.ProjectorOperator
:canonical: corrct.projectors.ProjectorMatrix._transpose

```{autodoc2-docstring} corrct.projectors.ProjectorMatrix._transpose
```

````

````{py:method} absolute() -> corrct.operators.ProjectorOperator
:canonical: corrct.projectors.ProjectorMatrix.absolute

```{autodoc2-docstring} corrct.projectors.ProjectorMatrix.absolute
```

````

````{py:method} fp(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorMatrix.fp

```{autodoc2-docstring} corrct.projectors.ProjectorMatrix.fp
```

````

````{py:method} bp(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorMatrix.bp

```{autodoc2-docstring} corrct.projectors.ProjectorMatrix.bp
```

````

`````

`````{py:class} ProjectorUncorrected(vol_geom: collections.abc.Sequence[int] | numpy.typing.NDArray[numpy.integer] | corrct.models.VolumeGeometry, angles_rot_rad: collections.abc.Sequence[float] | numpy.typing.NDArray, rot_axis_shift_pix: float | numpy.typing.ArrayLike | numpy.typing.NDArray | None = None, *, prj_geom: corrct.models.ProjectionGeometry | None = None, prj_intensities: numpy.typing.ArrayLike | None = None, psf: numpy.typing.ArrayLike | None = None, backend: str | corrct._projector_backends.ProjectorBackend = 'astra' if astra_available else 'skimage', create_single_projs: bool = False)
:canonical: corrct.projectors.ProjectorUncorrected

Bases: {py:obj}`corrct.operators.ProjectorOperator`

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.__init__
```

````{py:attribute} vol_geom
:canonical: corrct.projectors.ProjectorUncorrected.vol_geom
:type: corrct.models.VolumeGeometry
:value: >
   None

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.vol_geom
```

````

````{py:attribute} projector_backend
:canonical: corrct.projectors.ProjectorUncorrected.projector_backend
:type: corrct._projector_backends.ProjectorBackend
:value: >
   None

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.projector_backend
```

````

````{py:attribute} prj_intensities
:canonical: corrct.projectors.ProjectorUncorrected.prj_intensities
:type: numpy.typing.NDArray[numpy.floating] | None
:value: >
   None

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.prj_intensities
```

````

````{py:attribute} psf
:canonical: corrct.projectors.ProjectorUncorrected.psf
:type: numpy.typing.NDArray[numpy.floating] | float | None
:value: >
   None

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.psf
```

````

````{py:property} angles_rot_rad
:canonical: corrct.projectors.ProjectorUncorrected.angles_rot_rad
:type: numpy.typing.NDArray

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.angles_rot_rad
```

````

````{py:method} __enter__()
:canonical: corrct.projectors.ProjectorUncorrected.__enter__

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.__enter__
```

````

````{py:method} __exit__(*args)
:canonical: corrct.projectors.ProjectorUncorrected.__exit__

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.__exit__
```

````

````{py:method} _set_psf(psf: numpy.typing.ArrayLike | None, is_conv_symm: bool = False) -> None
:canonical: corrct.projectors.ProjectorUncorrected._set_psf

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected._set_psf
```

````

````{py:method} get_pre_weights() -> numpy.typing.NDArray | None
:canonical: corrct.projectors.ProjectorUncorrected.get_pre_weights

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.get_pre_weights
```

````

````{py:method} fp_angle(vol: numpy.typing.NDArray, angle_ind: int) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorUncorrected.fp_angle

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.fp_angle
```

````

````{py:method} bp_angle(prj_vu: numpy.typing.NDArray, angle_ind: int) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorUncorrected.bp_angle

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.bp_angle
```

````

````{py:method} fp(vol: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorUncorrected.fp

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.fp
```

````

````{py:method} bp(prj_vwu: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorUncorrected.bp

```{autodoc2-docstring} corrct.projectors.ProjectorUncorrected.bp
```

````

`````

`````{py:class} ProjectorAttenuationXRF(vol_geom: collections.abc.Sequence[int] | numpy.typing.NDArray[numpy.integer] | corrct.models.VolumeGeometry, angles_rot_rad: collections.abc.Sequence[float] | numpy.typing.NDArray, rot_axis_shift_pix: float | numpy.typing.ArrayLike | numpy.typing.NDArray | None = None, *, prj_geom: corrct.models.ProjectionGeometry | None = None, prj_intensities: numpy.typing.ArrayLike | None = None, backend: str | corrct._projector_backends.ProjectorBackend = 'astra' if astra_available else 'skimage', att_maps: numpy.typing.NDArray[numpy.floating] | None = None, att_in: numpy.typing.NDArray[numpy.floating] | None = None, att_out: numpy.typing.NDArray[numpy.floating] | None = None, angles_detectors_rad: float | numpy.typing.ArrayLike = np.pi / 2, weights_detectors: numpy.typing.ArrayLike | None = None, psf: numpy.typing.ArrayLike | None = None, is_symmetric: bool = False, weights_angles: numpy.typing.ArrayLike | None = None, use_multithreading: bool = True, data_type: numpy.typing.DTypeLike = np.float32, verbose: bool = True)
:canonical: corrct.projectors.ProjectorAttenuationXRF

Bases: {py:obj}`corrct.projectors.ProjectorUncorrected`

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.__init__
```

````{py:attribute} att_vol_angles
:canonical: corrct.projectors.ProjectorAttenuationXRF.att_vol_angles
:type: numpy.typing.NDArray[numpy.floating]
:value: >
   None

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.att_vol_angles
```

````

````{py:attribute} executor
:canonical: corrct.projectors.ProjectorAttenuationXRF.executor
:type: concurrent.futures.ThreadPoolExecutor | None
:value: >
   None

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.executor
```

````

````{py:method} __enter__()
:canonical: corrct.projectors.ProjectorAttenuationXRF.__enter__

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.__enter__
```

````

````{py:method} __exit__(*args)
:canonical: corrct.projectors.ProjectorAttenuationXRF.__exit__

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.__exit__
```

````

````{py:method} collapse_detectors() -> None
:canonical: corrct.projectors.ProjectorAttenuationXRF.collapse_detectors

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.collapse_detectors
```

````

````{py:method} fp_angle(vol: numpy.typing.NDArray, angle_ind: int) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorAttenuationXRF.fp_angle

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.fp_angle
```

````

````{py:method} bp_angle(sino: numpy.typing.NDArray, angle_ind: int, single_line: bool = False) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorAttenuationXRF.bp_angle

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.bp_angle
```

````

````{py:method} fp(vol: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorAttenuationXRF.fp

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.fp
```

````

````{py:method} bp(sino: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.projectors.ProjectorAttenuationXRF.bp

```{autodoc2-docstring} corrct.projectors.ProjectorAttenuationXRF.bp
```

````

`````
