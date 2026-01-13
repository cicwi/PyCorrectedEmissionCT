# {py:mod}`corrct.struct_illum`

```{py:module} corrct.struct_illum
```

```{autodoc2-docstring} corrct.struct_illum
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MaskCollection <corrct.struct_illum.MaskCollection>`
  - ```{autodoc2-docstring} corrct.struct_illum.MaskCollection
    :summary:
    ```
* - {py:obj}`MaskGenerator <corrct.struct_illum.MaskGenerator>`
  - ```{autodoc2-docstring} corrct.struct_illum.MaskGenerator
    :summary:
    ```
* - {py:obj}`MaskGeneratorPoint <corrct.struct_illum.MaskGeneratorPoint>`
  - ```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorPoint
    :summary:
    ```
* - {py:obj}`MaskGeneratorBernoulli <corrct.struct_illum.MaskGeneratorBernoulli>`
  - ```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorBernoulli
    :summary:
    ```
* - {py:obj}`MaskGeneratorHalfGaussian <corrct.struct_illum.MaskGeneratorHalfGaussian>`
  - ```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorHalfGaussian
    :summary:
    ```
* - {py:obj}`MaskGeneratorMURA <corrct.struct_illum.MaskGeneratorMURA>`
  - ```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorMURA
    :summary:
    ```
* - {py:obj}`ProjectorGhostImaging <corrct.struct_illum.ProjectorGhostImaging>`
  - ```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging
    :summary:
    ```
* - {py:obj}`ProjectorGhostTomography <corrct.struct_illum.ProjectorGhostTomography>`
  - ```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostTomography
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`reorder_masks <corrct.struct_illum.reorder_masks>`
  - ```{autodoc2-docstring} corrct.struct_illum.reorder_masks
    :summary:
    ```
* - {py:obj}`decompose_qr_masks <corrct.struct_illum.decompose_qr_masks>`
  - ```{autodoc2-docstring} corrct.struct_illum.decompose_qr_masks
    :summary:
    ```
* - {py:obj}`estimate_resolution <corrct.struct_illum.estimate_resolution>`
  - ```{autodoc2-docstring} corrct.struct_illum.estimate_resolution
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NDArrayInt <corrct.struct_illum.NDArrayInt>`
  - ```{autodoc2-docstring} corrct.struct_illum.NDArrayInt
    :summary:
    ```
````

### API

````{py:data} NDArrayInt
:canonical: corrct.struct_illum.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.NDArrayInt
```

````

````{py:function} reorder_masks(masks: numpy.typing.NDArray, buckets: numpy.typing.NDArray, shift: int) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
:canonical: corrct.struct_illum.reorder_masks

```{autodoc2-docstring} corrct.struct_illum.reorder_masks
```
````

````{py:function} decompose_qr_masks(masks: numpy.typing.NDArray, verbose: bool = False) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
:canonical: corrct.struct_illum.decompose_qr_masks

```{autodoc2-docstring} corrct.struct_illum.decompose_qr_masks
```
````

````{py:function} estimate_resolution(masks: numpy.typing.NDArray, verbose: bool = True, plot_result: bool = True) -> tuple[float, float]
:canonical: corrct.struct_illum.estimate_resolution

```{autodoc2-docstring} corrct.struct_illum.estimate_resolution
```
````

`````{py:class} MaskCollection(masks_enc: numpy.typing.NDArray, masks_dec: numpy.typing.NDArray | None = None, mask_dims: int = 2, mask_type: str = 'measured', mask_support: None | collections.abc.Sequence[int] | corrct.struct_illum.NDArrayInt = None)
:canonical: corrct.struct_illum.MaskCollection

```{autodoc2-docstring} corrct.struct_illum.MaskCollection
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.__init__
```

````{py:attribute} masks_enc
:canonical: corrct.struct_illum.MaskCollection.masks_enc
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.masks_enc
```

````

````{py:attribute} masks_dec
:canonical: corrct.struct_illum.MaskCollection.masks_dec
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.masks_dec
```

````

````{py:attribute} mask_dims
:canonical: corrct.struct_illum.MaskCollection.mask_dims
:type: int
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.mask_dims
```

````

````{py:attribute} mask_support
:canonical: corrct.struct_illum.MaskCollection.mask_support
:type: corrct.struct_illum.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.mask_support
```

````

````{py:attribute} mask_type
:canonical: corrct.struct_illum.MaskCollection.mask_type
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.mask_type
```

````

````{py:property} shape_fov
:canonical: corrct.struct_illum.MaskCollection.shape_fov
:type: collections.abc.Sequence[int]

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.shape_fov
```

````

````{py:property} shape_shifts
:canonical: corrct.struct_illum.MaskCollection.shape_shifts
:type: collections.abc.Sequence[int]

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.shape_shifts
```

````

````{py:property} num_buckets
:canonical: corrct.struct_illum.MaskCollection.num_buckets
:type: int

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.num_buckets
```

````

````{py:property} num_pixels
:canonical: corrct.struct_illum.MaskCollection.num_pixels
:type: int

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.num_pixels
```

````

````{py:method} info() -> str
:canonical: corrct.struct_illum.MaskCollection.info

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.info
```

````

````{py:method} upper() -> str
:canonical: corrct.struct_illum.MaskCollection.upper

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.upper
```

````

````{py:method} lower() -> str
:canonical: corrct.struct_illum.MaskCollection.lower

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.lower
```

````

````{py:method} get_mask(mask_inds_vu: collections.abc.Sequence | numpy.typing.NDArray, mask_encoding: bool = True) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.MaskCollection.get_mask

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.get_mask
```

````

````{py:method} get_qr_decomposition(buckets: numpy.typing.NDArray, shift: int = 0) -> tuple[corrct.struct_illum.MaskCollection, numpy.typing.NDArray]
:canonical: corrct.struct_illum.MaskCollection.get_qr_decomposition

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.get_qr_decomposition
```

````

````{py:method} bin_masks(binning: float) -> corrct.struct_illum.MaskCollection
:canonical: corrct.struct_illum.MaskCollection.bin_masks

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.bin_masks
```

````

````{py:method} inspect_masks(mask_inds_vu: None | collections.abc.Sequence[int] | corrct.struct_illum.NDArrayInt = None)
:canonical: corrct.struct_illum.MaskCollection.inspect_masks

```{autodoc2-docstring} corrct.struct_illum.MaskCollection.inspect_masks
```

````

`````

`````{py:class} MaskGenerator(shape_fov: collections.abc.Sequence[int] | corrct.struct_illum.NDArrayInt, shape_mask: collections.abc.Sequence[int] | corrct.struct_illum.NDArrayInt, shape_shifts: collections.abc.Sequence[int] | corrct.struct_illum.NDArrayInt, transmittance: float = 1.0, dtype: numpy.typing.DTypeLike = np.float32)
:canonical: corrct.struct_illum.MaskGenerator

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.__init__
```

````{py:attribute} shape_fov
:canonical: corrct.struct_illum.MaskGenerator.shape_fov
:type: corrct.struct_illum.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.shape_fov
```

````

````{py:attribute} shape_mask
:canonical: corrct.struct_illum.MaskGenerator.shape_mask
:type: corrct.struct_illum.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.shape_mask
```

````

````{py:attribute} shape_shifts
:canonical: corrct.struct_illum.MaskGenerator.shape_shifts
:type: corrct.struct_illum.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.shape_shifts
```

````

````{py:attribute} transmittance
:canonical: corrct.struct_illum.MaskGenerator.transmittance
:type: float
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.transmittance
```

````

````{py:attribute} dtype
:canonical: corrct.struct_illum.MaskGenerator.dtype
:type: numpy.typing.DTypeLike
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.dtype
```

````

````{py:attribute} _enc_dec_mismatch
:canonical: corrct.struct_illum.MaskGenerator._enc_dec_mismatch
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator._enc_dec_mismatch
```

````

````{py:attribute} __mask_name__
:canonical: corrct.struct_illum.MaskGenerator.__mask_name__
:value: >
   'generated'

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.__mask_name__
```

````

````{py:method} info() -> str
:canonical: corrct.struct_illum.MaskGenerator.info

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.info
```

````

````{py:method} __repr__() -> str
:canonical: corrct.struct_illum.MaskGenerator.__repr__

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.__repr__
```

````

````{py:property} max_buckets
:canonical: corrct.struct_illum.MaskGenerator.max_buckets
:type: int

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.max_buckets
```

````

````{py:property} num_pixels
:canonical: corrct.struct_illum.MaskGenerator.num_pixels
:type: int

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.num_pixels
```

````

````{py:method} _init_fov_mm(fov_size_mm: float | collections.abc.Sequence[float] | numpy.typing.NDArray, req_res_mm: float) -> corrct.struct_illum.NDArrayInt
:canonical: corrct.struct_illum.MaskGenerator._init_fov_mm

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator._init_fov_mm
```

````

````{py:method} generate_collection(buckets_fraction: float = 1, shift_type: str = 'sequential') -> corrct.struct_illum.MaskCollection
:canonical: corrct.struct_illum.MaskGenerator.generate_collection

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.generate_collection
```

````

````{py:method} _apply_transmission(masks: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.MaskGenerator._apply_transmission

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator._apply_transmission
```

````

````{py:method} generate_shifted_mask(mask_inds_vu: collections.abc.Sequence | numpy.typing.NDArray, mask_encoding: bool = True) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.MaskGenerator.generate_shifted_mask
:abstractmethod:

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.generate_shifted_mask
```

````

````{py:method} _generate_mask_shifts(shifts_v: collections.abc.Sequence | numpy.typing.NDArray, shifts_u: collections.abc.Sequence | numpy.typing.NDArray, mask_encoding: bool = True) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.MaskGenerator._generate_mask_shifts

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator._generate_mask_shifts
```

````

````{py:method} get_interval_shifts(interval: int | collections.abc.Sequence[int] | numpy.typing.NDArray, axes_order: collections.abc.Sequence[int] = (-2, -1)) -> collections.abc.Sequence[numpy.typing.NDArray]
:canonical: corrct.struct_illum.MaskGenerator.get_interval_shifts

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.get_interval_shifts
```

````

````{py:method} get_random_shifts(num_shifts: int, axes_order: collections.abc.Sequence[int] = (-2, -1)) -> collections.abc.Sequence[numpy.typing.NDArray]
:canonical: corrct.struct_illum.MaskGenerator.get_random_shifts

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.get_random_shifts
```

````

````{py:method} get_sequential_shifts(num_shifts: int | None = None, axes_order: collections.abc.Sequence[int] = (-2, -1)) -> collections.abc.Sequence[numpy.typing.NDArray]
:canonical: corrct.struct_illum.MaskGenerator.get_sequential_shifts

```{autodoc2-docstring} corrct.struct_illum.MaskGenerator.get_sequential_shifts
```

````

`````

`````{py:class} MaskGeneratorPoint(fov_size_mm: float | collections.abc.Sequence[float] | numpy.typing.NDArray, req_res_mm: float = 1.0)
:canonical: corrct.struct_illum.MaskGeneratorPoint

Bases: {py:obj}`corrct.struct_illum.MaskGenerator`

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorPoint
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorPoint.__init__
```

````{py:attribute} __mask_name__
:canonical: corrct.struct_illum.MaskGeneratorPoint.__mask_name__
:value: >
   'pencil'

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorPoint.__mask_name__
```

````

````{py:method} generate_shifted_mask(mask_inds_vu: collections.abc.Sequence | numpy.typing.NDArray, mask_encoding: bool = True) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.MaskGeneratorPoint.generate_shifted_mask

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorPoint.generate_shifted_mask
```

````

`````

`````{py:class} MaskGeneratorBernoulli(fov_size_mm: float | collections.abc.Sequence[float] | numpy.typing.NDArray, req_res_mm: float = 1.0, max_masks_ratio: float = 1.2)
:canonical: corrct.struct_illum.MaskGeneratorBernoulli

Bases: {py:obj}`corrct.struct_illum.MaskGenerator`

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorBernoulli
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorBernoulli.__init__
```

````{py:attribute} __mask_name__
:canonical: corrct.struct_illum.MaskGeneratorBernoulli.__mask_name__
:value: >
   'bernoulli'

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorBernoulli.__mask_name__
```

````

````{py:method} generate_shifted_mask(mask_inds_vu: collections.abc.Sequence | numpy.typing.NDArray, mask_encoding: bool = True) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.MaskGeneratorBernoulli.generate_shifted_mask

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorBernoulli.generate_shifted_mask
```

````

`````

`````{py:class} MaskGeneratorHalfGaussian(fov_size_mm: float | collections.abc.Sequence[float] | numpy.typing.NDArray, req_res_mm: float = 1.0, max_masks_ratio: float = 1.2)
:canonical: corrct.struct_illum.MaskGeneratorHalfGaussian

Bases: {py:obj}`corrct.struct_illum.MaskGenerator`

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorHalfGaussian
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorHalfGaussian.__init__
```

````{py:attribute} __mask_name__
:canonical: corrct.struct_illum.MaskGeneratorHalfGaussian.__mask_name__
:value: >
   'half-gaussian'

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorHalfGaussian.__mask_name__
```

````

````{py:method} generate_shifted_mask(mask_inds_vu: collections.abc.Sequence | numpy.typing.NDArray, mask_encoding: bool = True) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.MaskGeneratorHalfGaussian.generate_shifted_mask

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorHalfGaussian.generate_shifted_mask
```

````

`````

`````{py:class} MaskGeneratorMURA(fov_size_mm: float, req_res_mm: float = 1.0)
:canonical: corrct.struct_illum.MaskGeneratorMURA

Bases: {py:obj}`corrct.struct_illum.MaskGenerator`

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorMURA
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorMURA.__init__
```

````{py:attribute} __mask_name__
:canonical: corrct.struct_illum.MaskGeneratorMURA.__mask_name__
:value: >
   'mura'

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorMURA.__mask_name__
```

````

````{py:method} generate_shifted_mask(mask_inds_vu: collections.abc.Sequence | numpy.typing.NDArray, mask_encoding: bool = True) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.MaskGeneratorMURA.generate_shifted_mask

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorMURA.generate_shifted_mask
```

````

````{py:method} compute_possible_mask_sizes(fov_size: int) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.MaskGeneratorMURA.compute_possible_mask_sizes
:staticmethod:

```{autodoc2-docstring} corrct.struct_illum.MaskGeneratorMURA.compute_possible_mask_sizes
```

````

`````

`````{py:class} ProjectorGhostImaging(mask_collection: corrct.struct_illum.MaskCollection | numpy.typing.NDArray, backend: str = 'torch')
:canonical: corrct.struct_illum.ProjectorGhostImaging

Bases: {py:obj}`corrct.operators.ProjectorOperator`

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging.__init__
```

````{py:attribute} mc
:canonical: corrct.struct_illum.ProjectorGhostImaging.mc
:type: corrct.struct_illum.MaskCollection
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging.mc
```

````

````{py:method} _get_backend_device() -> tuple[str, str | None]
:canonical: corrct.struct_illum.ProjectorGhostImaging._get_backend_device

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging._get_backend_device
```

````

````{py:method} _init_backend()
:canonical: corrct.struct_illum.ProjectorGhostImaging._init_backend

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging._init_backend
```

````

````{py:method} fp(image: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.ProjectorGhostImaging.fp

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging.fp
```

````

````{py:method} bp(bucket_vals: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.ProjectorGhostImaging.bp

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging.bp
```

````

````{py:method} adjust_sampling_scaling(image: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.ProjectorGhostImaging.adjust_sampling_scaling

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging.adjust_sampling_scaling
```

````

````{py:method} fbp(bucket_vals: numpy.typing.NDArray, use_lstsq: bool = True, adjust_scaling: bool = False) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.ProjectorGhostImaging.fbp

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging.fbp
```

````

````{py:method} absolute()
:canonical: corrct.struct_illum.ProjectorGhostImaging.absolute

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostImaging.absolute
```

````

`````

`````{py:class} ProjectorGhostTomography(mask_collection: corrct.struct_illum.MaskCollection | numpy.typing.NDArray, tomo_proj: corrct.operators.ProjectorOperator, backend: str = 'torch')
:canonical: corrct.struct_illum.ProjectorGhostTomography

Bases: {py:obj}`corrct.struct_illum.ProjectorGhostImaging`

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostTomography
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostTomography.__init__
```

````{py:attribute} tomo_proj
:canonical: corrct.struct_illum.ProjectorGhostTomography.tomo_proj
:type: corrct.operators.ProjectorOperator
:value: >
   None

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostTomography.tomo_proj
```

````

````{py:method} fp(vol_zyx: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.ProjectorGhostTomography.fp

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostTomography.fp
```

````

````{py:method} bp(bucket_vals: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.struct_illum.ProjectorGhostTomography.bp

```{autodoc2-docstring} corrct.struct_illum.ProjectorGhostTomography.bp
```

````

`````
