# Alignment

The [`alignment`](corrct.html#module-corrct.alignment) module includes support
for aligning projection data.
The provided tools are:
1. Pre-alignment routines for tomographic data
2. Image stack shift finding

## Pre-alignment 

The class `DetectorShiftsPRE` offers support for both finding the vertical and
horizontal shifts of tomographic projection data. They are exposed through the
methods `fit_v` and `fit_u`.

```python
class DetectorShiftsPRE(DetectorShiftsBase):
    """Compute the pre-alignment detector shifts for a given dataset."""

    def fit_v(
        self,
        use_derivative: bool = True,
        use_rfft: bool = True,
        normalize_fourier: bool = True,
    ) -> NDArrayFloat:
        ...

    def fit_u(
        self,
        fit_l1: bool = False,
        background: Union[float, NDArray, None] = None,
        method: str = "com",
    ) -> tuple[NDArrayFloat, float]:
        ...
```

The `fit_v` method computes the vertical shifts of the stack with 1D cross-correlations.
The cross-correlation function is computed per angle on the intensity profile resulting
from computing the integral of the projections along the U axis, and their derivative
along the V axis.

The `fit_u` method computes the horizontal shifts of the stack, by computing the
sinusoid that interpolates the chosen value of interest across all the rotation
angles. The value of interest can include the center-of-mass (CoM) or the position
of the highest intensity peak of the projections.
