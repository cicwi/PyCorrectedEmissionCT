# {py:mod}`corrct.filters`

```{py:module} corrct.filters
```

```{autodoc2-docstring} corrct.filters
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BasisOptions <corrct.filters.BasisOptions>`
  - ```{autodoc2-docstring} corrct.filters.BasisOptions
    :summary:
    ```
* - {py:obj}`BasisOptionsBlocks <corrct.filters.BasisOptionsBlocks>`
  - ```{autodoc2-docstring} corrct.filters.BasisOptionsBlocks
    :summary:
    ```
* - {py:obj}`BasisOptionsWavelets <corrct.filters.BasisOptionsWavelets>`
  - ```{autodoc2-docstring} corrct.filters.BasisOptionsWavelets
    :summary:
    ```
* - {py:obj}`Filter <corrct.filters.Filter>`
  - ```{autodoc2-docstring} corrct.filters.Filter
    :summary:
    ```
* - {py:obj}`FilterCustom <corrct.filters.FilterCustom>`
  - ```{autodoc2-docstring} corrct.filters.FilterCustom
    :summary:
    ```
* - {py:obj}`FilterFBP <corrct.filters.FilterFBP>`
  - ```{autodoc2-docstring} corrct.filters.FilterFBP
    :summary:
    ```
* - {py:obj}`FilterMR <corrct.filters.FilterMR>`
  - ```{autodoc2-docstring} corrct.filters.FilterMR
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_basis <corrct.filters.create_basis>`
  - ```{autodoc2-docstring} corrct.filters.create_basis
    :summary:
    ```
* - {py:obj}`create_basis_wavelet <corrct.filters.create_basis_wavelet>`
  - ```{autodoc2-docstring} corrct.filters.create_basis_wavelet
    :summary:
    ```
````

### API

`````{py:class} BasisOptions
:canonical: corrct.filters.BasisOptions

Bases: {py:obj}`abc.ABC`, {py:obj}`collections.abc.Mapping`

```{autodoc2-docstring} corrct.filters.BasisOptions
```

````{py:method} __len__() -> int
:canonical: corrct.filters.BasisOptions.__len__

```{autodoc2-docstring} corrct.filters.BasisOptions.__len__
```

````

````{py:method} __getitem__(k: typing.Any) -> typing.Any
:canonical: corrct.filters.BasisOptions.__getitem__

```{autodoc2-docstring} corrct.filters.BasisOptions.__getitem__
```

````

````{py:method} __iter__() -> typing.Any
:canonical: corrct.filters.BasisOptions.__iter__

```{autodoc2-docstring} corrct.filters.BasisOptions.__iter__
```

````

`````

`````{py:class} BasisOptionsBlocks
:canonical: corrct.filters.BasisOptionsBlocks

Bases: {py:obj}`corrct.filters.BasisOptions`

```{autodoc2-docstring} corrct.filters.BasisOptionsBlocks
```

````{py:attribute} binning_start
:canonical: corrct.filters.BasisOptionsBlocks.binning_start
:type: int | None
:value: >
   2

```{autodoc2-docstring} corrct.filters.BasisOptionsBlocks.binning_start
```

````

````{py:attribute} binning_type
:canonical: corrct.filters.BasisOptionsBlocks.binning_type
:type: str
:value: >
   'exponential'

```{autodoc2-docstring} corrct.filters.BasisOptionsBlocks.binning_type
```

````

````{py:attribute} order
:canonical: corrct.filters.BasisOptionsBlocks.order
:type: int
:value: >
   1

```{autodoc2-docstring} corrct.filters.BasisOptionsBlocks.order
```

````

````{py:attribute} normalized
:canonical: corrct.filters.BasisOptionsBlocks.normalized
:type: bool
:value: >
   True

```{autodoc2-docstring} corrct.filters.BasisOptionsBlocks.normalized
```

````

`````

`````{py:class} BasisOptionsWavelets
:canonical: corrct.filters.BasisOptionsWavelets

Bases: {py:obj}`corrct.filters.BasisOptions`

```{autodoc2-docstring} corrct.filters.BasisOptionsWavelets
```

````{py:attribute} wavelet
:canonical: corrct.filters.BasisOptionsWavelets.wavelet
:type: str
:value: >
   'bior2.2'

```{autodoc2-docstring} corrct.filters.BasisOptionsWavelets.wavelet
```

````

````{py:attribute} level
:canonical: corrct.filters.BasisOptionsWavelets.level
:type: int
:value: >
   5

```{autodoc2-docstring} corrct.filters.BasisOptionsWavelets.level
```

````

````{py:attribute} norm
:canonical: corrct.filters.BasisOptionsWavelets.norm
:type: float
:value: >
   1.0

```{autodoc2-docstring} corrct.filters.BasisOptionsWavelets.norm
```

````

`````

````{py:function} create_basis(num_pixels: int, binning_start: int | None = 2, binning_type: str = 'exponential', normalized: bool = False, order: int = 1, dtype: numpy.typing.DTypeLike = np.float32) -> numpy.typing.NDArray
:canonical: corrct.filters.create_basis

```{autodoc2-docstring} corrct.filters.create_basis
```
````

````{py:function} create_basis_wavelet(num_pixels: int, wavelet: str = 'bior2.2', level: int = 5, norm: float = 1.0, dtype: numpy.typing.DTypeLike = np.float32) -> numpy.typing.NDArray
:canonical: corrct.filters.create_basis_wavelet

```{autodoc2-docstring} corrct.filters.create_basis_wavelet
```
````

`````{py:class} Filter(fbp_filter: numpy.typing.ArrayLike | numpy.typing.NDArray[numpy.floating] | None, pad_mode: str, use_rfft: bool, dtype: numpy.typing.DTypeLike)
:canonical: corrct.filters.Filter

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.filters.Filter
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.filters.Filter.__init__
```

````{py:attribute} fbp_filter
:canonical: corrct.filters.Filter.fbp_filter
:type: numpy.typing.NDArray[numpy.floating]
:value: >
   None

```{autodoc2-docstring} corrct.filters.Filter.fbp_filter
```

````

````{py:attribute} pad_mode
:canonical: corrct.filters.Filter.pad_mode
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.filters.Filter.pad_mode
```

````

````{py:attribute} use_rfft
:canonical: corrct.filters.Filter.use_rfft
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.filters.Filter.use_rfft
```

````

````{py:attribute} dtype
:canonical: corrct.filters.Filter.dtype
:type: numpy.typing.DTypeLike
:value: >
   None

```{autodoc2-docstring} corrct.filters.Filter.dtype
```

````

````{py:method} get_padding_size(data_wu_shape: collections.abc.Sequence[int]) -> int
:canonical: corrct.filters.Filter.get_padding_size

```{autodoc2-docstring} corrct.filters.Filter.get_padding_size
```

````

````{py:method} to_fourier(data_wu: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.filters.Filter.to_fourier

```{autodoc2-docstring} corrct.filters.Filter.to_fourier
```

````

````{py:method} to_real(data_wu: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.filters.Filter.to_real

```{autodoc2-docstring} corrct.filters.Filter.to_real
```

````

````{py:property} filter_fourier
:canonical: corrct.filters.Filter.filter_fourier
:type: numpy.typing.NDArray[numpy.floating]

```{autodoc2-docstring} corrct.filters.Filter.filter_fourier
```

````

````{py:property} filter_real
:canonical: corrct.filters.Filter.filter_real
:type: numpy.typing.NDArray[numpy.floating]

```{autodoc2-docstring} corrct.filters.Filter.filter_real
```

````

````{py:property} num_filters
:canonical: corrct.filters.Filter.num_filters
:type: int

```{autodoc2-docstring} corrct.filters.Filter.num_filters
```

````

````{py:method} apply_filter(data_wu: numpy.typing.NDArray, fbp_filter: numpy.typing.NDArray | None = None) -> numpy.typing.NDArray
:canonical: corrct.filters.Filter.apply_filter

```{autodoc2-docstring} corrct.filters.Filter.apply_filter
```

````

````{py:method} compute_filter(data_wu: numpy.typing.NDArray) -> None
:canonical: corrct.filters.Filter.compute_filter
:abstractmethod:

```{autodoc2-docstring} corrct.filters.Filter.compute_filter
```

````

````{py:method} __call__(data_wu: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.filters.Filter.__call__

```{autodoc2-docstring} corrct.filters.Filter.__call__
```

````

````{py:method} plot_filters(fourier_abs: bool = False)
:canonical: corrct.filters.Filter.plot_filters

```{autodoc2-docstring} corrct.filters.Filter.plot_filters
```

````

`````

`````{py:class} FilterCustom(fbp_filter: numpy.typing.ArrayLike | numpy.typing.NDArray[numpy.floating] | None, pad_mode: str = 'constant', use_rfft: bool = True, dtype: numpy.typing.DTypeLike = np.float32)
:canonical: corrct.filters.FilterCustom

Bases: {py:obj}`corrct.filters.Filter`

```{autodoc2-docstring} corrct.filters.FilterCustom
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.filters.FilterCustom.__init__
```

````{py:method} compute_filter(data_wu: numpy.typing.NDArray) -> None
:canonical: corrct.filters.FilterCustom.compute_filter

```{autodoc2-docstring} corrct.filters.FilterCustom.compute_filter
```

````

`````

`````{py:class} FilterFBP(filter_name: str = 'ramp', pad_mode: str = 'constant', use_rfft: bool = True, dtype: numpy.typing.DTypeLike = np.float32)
:canonical: corrct.filters.FilterFBP

Bases: {py:obj}`corrct.filters.Filter`

```{autodoc2-docstring} corrct.filters.FilterFBP
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.filters.FilterFBP.__init__
```

````{py:attribute} filter_name
:canonical: corrct.filters.FilterFBP.filter_name
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.filters.FilterFBP.filter_name
```

````

````{py:attribute} FILTERS
:canonical: corrct.filters.FilterFBP.FILTERS
:value: >
   ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann')

```{autodoc2-docstring} corrct.filters.FilterFBP.FILTERS
```

````

````{py:method} compute_filter(data_wu: numpy.typing.NDArray) -> None
:canonical: corrct.filters.FilterFBP.compute_filter

```{autodoc2-docstring} corrct.filters.FilterFBP.compute_filter
```

````

````{py:method} get_available_filters() -> collections.abc.Sequence[str]
:canonical: corrct.filters.FilterFBP.get_available_filters

```{autodoc2-docstring} corrct.filters.FilterFBP.get_available_filters
```

````

`````

`````{py:class} FilterMR(projector: corrct.operators.BaseTransform, binning_type: str = 'exponential', binning_start: int | None = 2, lambda_smooth: float | None = None, pad_mode: str = 'constant', use_rfft: bool = True, dtype: numpy.typing.DTypeLike = np.float32)
:canonical: corrct.filters.FilterMR

Bases: {py:obj}`corrct.filters.Filter`

```{autodoc2-docstring} corrct.filters.FilterMR
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.filters.FilterMR.__init__
```

````{py:attribute} projector
:canonical: corrct.filters.FilterMR.projector
:type: corrct.operators.BaseTransform
:value: >
   None

```{autodoc2-docstring} corrct.filters.FilterMR.projector
```

````

````{py:attribute} binning_type
:canonical: corrct.filters.FilterMR.binning_type
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.filters.FilterMR.binning_type
```

````

````{py:attribute} binning_start
:canonical: corrct.filters.FilterMR.binning_start
:type: int | None
:value: >
   None

```{autodoc2-docstring} corrct.filters.FilterMR.binning_start
```

````

````{py:attribute} lambda_smooth
:canonical: corrct.filters.FilterMR.lambda_smooth
:type: float | None
:value: >
   None

```{autodoc2-docstring} corrct.filters.FilterMR.lambda_smooth
```

````

````{py:attribute} is_initialized
:canonical: corrct.filters.FilterMR.is_initialized
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.filters.FilterMR.is_initialized
```

````

````{py:method} initialize(data_wu_shape: collections.abc.Sequence[int]) -> None
:canonical: corrct.filters.FilterMR.initialize

```{autodoc2-docstring} corrct.filters.FilterMR.initialize
```

````

````{py:method} compute_filter(data_wu: numpy.typing.NDArray) -> None
:canonical: corrct.filters.FilterMR.compute_filter

```{autodoc2-docstring} corrct.filters.FilterMR.compute_filter
```

````

`````
