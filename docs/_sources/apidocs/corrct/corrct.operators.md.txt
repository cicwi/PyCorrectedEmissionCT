# {py:mod}`corrct.operators`

```{py:module} corrct.operators
```

```{autodoc2-docstring} corrct.operators
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseTransform <corrct.operators.BaseTransform>`
  - ```{autodoc2-docstring} corrct.operators.BaseTransform
    :summary:
    ```
* - {py:obj}`TransformFunctions <corrct.operators.TransformFunctions>`
  - ```{autodoc2-docstring} corrct.operators.TransformFunctions
    :summary:
    ```
* - {py:obj}`ProjectorOperator <corrct.operators.ProjectorOperator>`
  - ```{autodoc2-docstring} corrct.operators.ProjectorOperator
    :summary:
    ```
* - {py:obj}`TransformIdentity <corrct.operators.TransformIdentity>`
  - ```{autodoc2-docstring} corrct.operators.TransformIdentity
    :summary:
    ```
* - {py:obj}`TransformDiagonalScaling <corrct.operators.TransformDiagonalScaling>`
  - ```{autodoc2-docstring} corrct.operators.TransformDiagonalScaling
    :summary:
    ```
* - {py:obj}`TransformConvolution <corrct.operators.TransformConvolution>`
  - ```{autodoc2-docstring} corrct.operators.TransformConvolution
    :summary:
    ```
* - {py:obj}`BaseWaveletTransform <corrct.operators.BaseWaveletTransform>`
  - ```{autodoc2-docstring} corrct.operators.BaseWaveletTransform
    :summary:
    ```
* - {py:obj}`TransformDecimatedWavelet <corrct.operators.TransformDecimatedWavelet>`
  - ```{autodoc2-docstring} corrct.operators.TransformDecimatedWavelet
    :summary:
    ```
* - {py:obj}`TransformStationaryWavelet <corrct.operators.TransformStationaryWavelet>`
  - ```{autodoc2-docstring} corrct.operators.TransformStationaryWavelet
    :summary:
    ```
* - {py:obj}`TransformGradient <corrct.operators.TransformGradient>`
  - ```{autodoc2-docstring} corrct.operators.TransformGradient
    :summary:
    ```
* - {py:obj}`TransformFourier <corrct.operators.TransformFourier>`
  - ```{autodoc2-docstring} corrct.operators.TransformFourier
    :summary:
    ```
* - {py:obj}`TransformLaplacian <corrct.operators.TransformLaplacian>`
  - ```{autodoc2-docstring} corrct.operators.TransformLaplacian
    :summary:
    ```
* - {py:obj}`TransformSVD <corrct.operators.TransformSVD>`
  - ```{autodoc2-docstring} corrct.operators.TransformSVD
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NDArrayInt <corrct.operators.NDArrayInt>`
  - ```{autodoc2-docstring} corrct.operators.NDArrayInt
    :summary:
    ```
````

### API

````{py:data} NDArrayInt
:canonical: corrct.operators.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.operators.NDArrayInt
```

````

`````{py:class} BaseTransform()
:canonical: corrct.operators.BaseTransform

Bases: {py:obj}`scipy.sparse.linalg.LinearOperator`, {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.operators.BaseTransform
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.BaseTransform.__init__
```

````{py:attribute} dir_shape
:canonical: corrct.operators.BaseTransform.dir_shape
:type: corrct.operators.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.operators.BaseTransform.dir_shape
```

````

````{py:attribute} adj_shape
:canonical: corrct.operators.BaseTransform.adj_shape
:type: corrct.operators.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.operators.BaseTransform.adj_shape
```

````

````{py:method} _matvec(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.BaseTransform._matvec

```{autodoc2-docstring} corrct.operators.BaseTransform._matvec
```

````

````{py:method} rmatvec(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.BaseTransform.rmatvec

```{autodoc2-docstring} corrct.operators.BaseTransform.rmatvec
```

````

````{py:method} _transpose() -> corrct.operators.BaseTransform
:canonical: corrct.operators.BaseTransform._transpose

```{autodoc2-docstring} corrct.operators.BaseTransform._transpose
```

````

````{py:method} _adjoint() -> corrct.operators.BaseTransform
:canonical: corrct.operators.BaseTransform._adjoint

```{autodoc2-docstring} corrct.operators.BaseTransform._adjoint
```

````

````{py:method} absolute() -> corrct.operators.BaseTransform
:canonical: corrct.operators.BaseTransform.absolute

```{autodoc2-docstring} corrct.operators.BaseTransform.absolute
```

````

````{py:method} explicit() -> numpy.typing.NDArray
:canonical: corrct.operators.BaseTransform.explicit

```{autodoc2-docstring} corrct.operators.BaseTransform.explicit
```

````

````{py:method} __call__(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.BaseTransform.__call__

```{autodoc2-docstring} corrct.operators.BaseTransform.__call__
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.BaseTransform._op_direct
:abstractmethod:

```{autodoc2-docstring} corrct.operators.BaseTransform._op_direct
```

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.BaseTransform._op_adjoint
:abstractmethod:

```{autodoc2-docstring} corrct.operators.BaseTransform._op_adjoint
```

````

`````

`````{py:class} TransformFunctions(dir_shape: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray], adj_shape: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray], A: typing.Callable[[numpy.typing.NDArray], numpy.typing.NDArray], At: typing.Optional[typing.Callable[[numpy.typing.NDArray], numpy.typing.NDArray]] = None)
:canonical: corrct.operators.TransformFunctions

Bases: {py:obj}`corrct.operators.BaseTransform`

```{autodoc2-docstring} corrct.operators.TransformFunctions
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformFunctions.__init__
```

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformFunctions._op_direct

```{autodoc2-docstring} corrct.operators.TransformFunctions._op_direct
```

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformFunctions._op_adjoint

```{autodoc2-docstring} corrct.operators.TransformFunctions._op_adjoint
```

````

````{py:method} absolute() -> corrct.operators.TransformFunctions
:canonical: corrct.operators.TransformFunctions.absolute

```{autodoc2-docstring} corrct.operators.TransformFunctions.absolute
```

````

`````

`````{py:class} ProjectorOperator()
:canonical: corrct.operators.ProjectorOperator

Bases: {py:obj}`corrct.operators.BaseTransform`

```{autodoc2-docstring} corrct.operators.ProjectorOperator
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.ProjectorOperator.__init__
```

````{py:property} vol_shape
:canonical: corrct.operators.ProjectorOperator.vol_shape
:type: corrct.operators.NDArrayInt

```{autodoc2-docstring} corrct.operators.ProjectorOperator.vol_shape
```

````

````{py:property} prj_shape
:canonical: corrct.operators.ProjectorOperator.prj_shape
:type: corrct.operators.NDArrayInt

```{autodoc2-docstring} corrct.operators.ProjectorOperator.prj_shape
```

````

````{py:method} fp(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.ProjectorOperator.fp
:abstractmethod:

```{autodoc2-docstring} corrct.operators.ProjectorOperator.fp
```

````

````{py:method} bp(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.ProjectorOperator.bp
:abstractmethod:

```{autodoc2-docstring} corrct.operators.ProjectorOperator.bp
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.ProjectorOperator._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.ProjectorOperator._op_adjoint

````

````{py:method} get_pre_weights() -> typing.Union[numpy.typing.NDArray, None]
:canonical: corrct.operators.ProjectorOperator.get_pre_weights

```{autodoc2-docstring} corrct.operators.ProjectorOperator.get_pre_weights
```

````

`````

`````{py:class} TransformIdentity(x_shape: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray])
:canonical: corrct.operators.TransformIdentity

Bases: {py:obj}`corrct.operators.BaseTransform`

```{autodoc2-docstring} corrct.operators.TransformIdentity
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformIdentity.__init__
```

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformIdentity._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformIdentity._op_adjoint

````

`````

`````{py:class} TransformDiagonalScaling(x_shape: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray], scale: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray])
:canonical: corrct.operators.TransformDiagonalScaling

Bases: {py:obj}`corrct.operators.BaseTransform`

```{autodoc2-docstring} corrct.operators.TransformDiagonalScaling
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformDiagonalScaling.__init__
```

````{py:attribute} scale
:canonical: corrct.operators.TransformDiagonalScaling.scale
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.operators.TransformDiagonalScaling.scale
```

````

````{py:method} absolute() -> corrct.operators.TransformDiagonalScaling
:canonical: corrct.operators.TransformDiagonalScaling.absolute

```{autodoc2-docstring} corrct.operators.TransformDiagonalScaling.absolute
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformDiagonalScaling._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformDiagonalScaling._op_adjoint

````

`````

`````{py:class} TransformConvolution(x_shape: numpy.typing.ArrayLike, kernel: numpy.typing.ArrayLike, pad_mode: str = 'edge', is_symm: bool = True, flip_adjoint: bool = False)
:canonical: corrct.operators.TransformConvolution

Bases: {py:obj}`corrct.operators.BaseTransform`

```{autodoc2-docstring} corrct.operators.TransformConvolution
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformConvolution.__init__
```

````{py:attribute} kernel
:canonical: corrct.operators.TransformConvolution.kernel
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.operators.TransformConvolution.kernel
```

````

````{py:attribute} pad_mode
:canonical: corrct.operators.TransformConvolution.pad_mode
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.operators.TransformConvolution.pad_mode
```

````

````{py:attribute} is_symm
:canonical: corrct.operators.TransformConvolution.is_symm
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.operators.TransformConvolution.is_symm
```

````

````{py:attribute} flip_adjoint
:canonical: corrct.operators.TransformConvolution.flip_adjoint
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.operators.TransformConvolution.flip_adjoint
```

````

````{py:method} absolute() -> corrct.operators.TransformConvolution
:canonical: corrct.operators.TransformConvolution.absolute

```{autodoc2-docstring} corrct.operators.TransformConvolution.absolute
```

````

````{py:method} _pad_valid(x: numpy.typing.NDArray) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
:canonical: corrct.operators.TransformConvolution._pad_valid

```{autodoc2-docstring} corrct.operators.TransformConvolution._pad_valid
```

````

````{py:method} _crop_valid(x: numpy.typing.NDArray, pad_width: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformConvolution._crop_valid

```{autodoc2-docstring} corrct.operators.TransformConvolution._crop_valid
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformConvolution._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformConvolution._op_adjoint

````

`````

`````{py:class} BaseWaveletTransform()
:canonical: corrct.operators.BaseWaveletTransform

Bases: {py:obj}`corrct.operators.BaseTransform`, {py:obj}`abc.ABC`

```{autodoc2-docstring} corrct.operators.BaseWaveletTransform
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.BaseWaveletTransform.__init__
```

````{py:attribute} axes
:canonical: corrct.operators.BaseWaveletTransform.axes
:type: corrct.operators.NDArrayInt
:value: >
   None

```{autodoc2-docstring} corrct.operators.BaseWaveletTransform.axes
```

````

````{py:attribute} wavelet
:canonical: corrct.operators.BaseWaveletTransform.wavelet
:type: str
:value: >
   None

```{autodoc2-docstring} corrct.operators.BaseWaveletTransform.wavelet
```

````

````{py:attribute} labels
:canonical: corrct.operators.BaseWaveletTransform.labels
:type: list[str]
:value: >
   None

```{autodoc2-docstring} corrct.operators.BaseWaveletTransform.labels
```

````

````{py:attribute} wlet_dec_filter_mult
:canonical: corrct.operators.BaseWaveletTransform.wlet_dec_filter_mult
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.operators.BaseWaveletTransform.wlet_dec_filter_mult
```

````

````{py:attribute} wlet_rec_filter_mult
:canonical: corrct.operators.BaseWaveletTransform.wlet_rec_filter_mult
:type: numpy.typing.NDArray
:value: >
   None

```{autodoc2-docstring} corrct.operators.BaseWaveletTransform.wlet_rec_filter_mult
```

````

````{py:method} _initialize_filter_bank() -> None
:canonical: corrct.operators.BaseWaveletTransform._initialize_filter_bank

```{autodoc2-docstring} corrct.operators.BaseWaveletTransform._initialize_filter_bank
```

````

`````

`````{py:class} TransformDecimatedWavelet(x_shape: typing.Union[numpy.typing.ArrayLike, numpy.typing.NDArray], wavelet: str, level: int, axes: typing.Optional[numpy.typing.ArrayLike] = None, pad_on_demand: str = 'edge')
:canonical: corrct.operators.TransformDecimatedWavelet

Bases: {py:obj}`corrct.operators.BaseWaveletTransform`

```{autodoc2-docstring} corrct.operators.TransformDecimatedWavelet
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformDecimatedWavelet.__init__
```

````{py:method} direct_dwt(x: numpy.typing.NDArray) -> list
:canonical: corrct.operators.TransformDecimatedWavelet.direct_dwt

```{autodoc2-docstring} corrct.operators.TransformDecimatedWavelet.direct_dwt
```

````

````{py:method} inverse_dwt(y: list) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformDecimatedWavelet.inverse_dwt

```{autodoc2-docstring} corrct.operators.TransformDecimatedWavelet.inverse_dwt
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformDecimatedWavelet._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformDecimatedWavelet._op_adjoint

````

`````

`````{py:class} TransformStationaryWavelet(x_shape: numpy.typing.ArrayLike, wavelet: str, level: int, axes: typing.Optional[numpy.typing.ArrayLike] = None, pad_on_demand: str = 'edge', normalized: bool = True)
:canonical: corrct.operators.TransformStationaryWavelet

Bases: {py:obj}`corrct.operators.BaseWaveletTransform`

```{autodoc2-docstring} corrct.operators.TransformStationaryWavelet
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformStationaryWavelet.__init__
```

````{py:method} direct_swt(x: numpy.typing.NDArray) -> list
:canonical: corrct.operators.TransformStationaryWavelet.direct_swt

```{autodoc2-docstring} corrct.operators.TransformStationaryWavelet.direct_swt
```

````

````{py:method} inverse_swt(y: list) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformStationaryWavelet.inverse_swt

```{autodoc2-docstring} corrct.operators.TransformStationaryWavelet.inverse_swt
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformStationaryWavelet._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformStationaryWavelet._op_adjoint

````

`````

`````{py:class} TransformGradient(x_shape: numpy.typing.ArrayLike, axes: typing.Optional[numpy.typing.ArrayLike] = None, pad_mode: str = 'edge')
:canonical: corrct.operators.TransformGradient

Bases: {py:obj}`corrct.operators.BaseTransform`

```{autodoc2-docstring} corrct.operators.TransformGradient
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformGradient.__init__
```

````{py:method} gradient(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformGradient.gradient

```{autodoc2-docstring} corrct.operators.TransformGradient.gradient
```

````

````{py:method} divergence(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformGradient.divergence

```{autodoc2-docstring} corrct.operators.TransformGradient.divergence
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformGradient._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformGradient._op_adjoint

````

`````

`````{py:class} TransformFourier(x_shape: numpy.typing.ArrayLike, axes: typing.Optional[numpy.typing.ArrayLike] = None)
:canonical: corrct.operators.TransformFourier

Bases: {py:obj}`corrct.operators.BaseTransform`

```{autodoc2-docstring} corrct.operators.TransformFourier
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformFourier.__init__
```

````{py:method} fft(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformFourier.fft

```{autodoc2-docstring} corrct.operators.TransformFourier.fft
```

````

````{py:method} ifft(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformFourier.ifft

```{autodoc2-docstring} corrct.operators.TransformFourier.ifft
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformFourier._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformFourier._op_adjoint

````

`````

`````{py:class} TransformLaplacian(x_shape: numpy.typing.ArrayLike, axes: typing.Optional[numpy.typing.ArrayLike] = None, pad_mode: str = 'edge')
:canonical: corrct.operators.TransformLaplacian

Bases: {py:obj}`corrct.operators.BaseTransform`

```{autodoc2-docstring} corrct.operators.TransformLaplacian
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformLaplacian.__init__
```

````{py:method} laplacian(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformLaplacian.laplacian

```{autodoc2-docstring} corrct.operators.TransformLaplacian.laplacian
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformLaplacian._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformLaplacian._op_adjoint

````

`````

`````{py:class} TransformSVD(x_shape: numpy.typing.ArrayLike, axes_rows: typing.Union[collections.abc.Sequence[int], numpy.typing.NDArray] = (0, ), axes_cols: typing.Union[collections.abc.Sequence[int], numpy.typing.NDArray] = (-1, ), rescale: bool = False)
:canonical: corrct.operators.TransformSVD

Bases: {py:obj}`corrct.operators.BaseTransform`

```{autodoc2-docstring} corrct.operators.TransformSVD
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.operators.TransformSVD.__init__
```

````{py:attribute} U
:canonical: corrct.operators.TransformSVD.U
:type: typing.Optional[numpy.typing.NDArray]
:value: >
   None

```{autodoc2-docstring} corrct.operators.TransformSVD.U
```

````

````{py:attribute} Vt
:canonical: corrct.operators.TransformSVD.Vt
:type: typing.Optional[numpy.typing.NDArray]
:value: >
   None

```{autodoc2-docstring} corrct.operators.TransformSVD.Vt
```

````

````{py:attribute} rescale
:canonical: corrct.operators.TransformSVD.rescale
:type: bool
:value: >
   None

```{autodoc2-docstring} corrct.operators.TransformSVD.rescale
```

````

````{py:method} direct_svd(x: numpy.typing.NDArray)
:canonical: corrct.operators.TransformSVD.direct_svd

```{autodoc2-docstring} corrct.operators.TransformSVD.direct_svd
```

````

````{py:method} inverse_svd(U: numpy.typing.NDArray, s: numpy.typing.NDArray, Vt: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformSVD.inverse_svd

```{autodoc2-docstring} corrct.operators.TransformSVD.inverse_svd
```

````

````{py:method} _op_direct(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformSVD._op_direct

````

````{py:method} _op_adjoint(x: numpy.typing.NDArray) -> numpy.typing.NDArray
:canonical: corrct.operators.TransformSVD._op_adjoint

````

`````
