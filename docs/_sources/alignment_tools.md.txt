# Alignment

The [](#alignment) module includes support for aligning projection data.
The provided tools are:
1. Pre-alignment routines for tomographic data
2. Image stack shift finding

## Pre-alignment 

The class [](#alignment.shifts.DetectorShiftsPRE) offers support for both finding the
vertical and horizontal shifts of tomographic projection data.
They are exposed through the methods `fit_v` and `fit_u`.

```Python
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

We present here an example of how to use the `fit_u` method to compute the horizontal shifts.
```Python
import corrct as cct

align_pre = cct.alignment.DetectorShiftsPRE(data_vwu, angles_rad)

diffs_u_pre, cor = align_pre.fit_u()
```
where the projection data is passed to the `DetectorShiftsPRE` class with the
following axes order: [V], W, U, which means that V is the slowest varying axis,
but also optional (in case of 2D data).

These shifts can be used to create a `ProjectionGeometry` object, which can be
used to correct the projection data, when passed to projection operators as follows:
```Python
prj_geom = cct.models.get_prj_geom_parallel(geom_type="2d")
prj_geom.set_detector_shifts_vu(diffs_u_pre, cor)
vol_geom = cct.models.get_vol_geom_from_data(data_vwu)

solver = cct.solvers.SIRT()
with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, prj_geom=prj_geom) as A:
    rec_pre, _ = solver(A, data_test, iterations=100)
```

## Image stack alignment

The [](#alignment.shifts.DetectorShiftsXC.fit_vu_accum_drifts) function calculates the
shifts in the vertical and possibly horizontal directions of each image in a stack
relative to a reference image or images.
It ensures that the number of reference images matches the number of data images,
and returns an array containing these shifts.