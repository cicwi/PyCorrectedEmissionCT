# Tutorial

In this tutorial, we will first learn the basics of how to reconstruct
projection data.
We will then introduce attenuation correction, data terms, and regularizers.
Finally, we will see the more advanced topics like guided regularization
parameter selection.

## Reconstructing projection data

### The data model

In `corrct`, the volumes are always organized with coordinates \[Z\]YX, where Z
is only present in 3D volumes.
The projection data is organized with coordinates \[V\]WU, where:

* V is the optional vertical coordinate of the detector images. It is only
present for 3D reconstructions.
* W is the rotation angle coordinate. W stands for omega, which is the rotation
angle.
* U is the horizontal coordinate of the detector images.

### The geometry

The geometry is supposed to be circular parallel-beam by default, at least in
the simplest settings.
Attenuation correction is usually meant in X-ray Fluorescence (XRF) settings.
Thus, the detector is supposed to be pointed at the sample with an angles, that
needs to be specified.
The rotation axis is supposed to be in the center of the reconstruction volume.
Shifts in the rotation axis position are either intended as or converted to
detector shifts with respect to the origin.

### The projectors

The projection geometry is specified through the creation of *projectors*.
The simplest projector is the one in `corrct.projectors.ProjectorUncorrected`.
The projector `corrct.projectors.ProjectorAttenuationXRF` derives from the
simpler projector, and it implements the XRF specific bits, with respect to
multi-detector / multi-element handling and attenuation correction.
Projectors are usually used through the `with` statement, because it takes care
of initializing their underlying resources.

To create a simple projector for a `10 x 10` volume and `16 x 10` sinogram (16
angles), we will do:
```python
import numpy as np
import corrct as cct

vol_shape_xy = [10, 10]
angles_rad = np.deg2rad(np.linspace(0, 180, 16, endpoint=False))

with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    prj = p(np.ones((10, 10)))
```
This code also uses the projector to create a sinogram, that is the projection
of a volume of all ones.

The back-projection can be done in a very similar manner:
```python
with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    vol = p.T(np.ones((16, 10)))
```

Creating attenuation correction projectors is a bit more involved, and we will
see it later.

### The solvers

Tomographic reconstructions can be achieved using either the included solvers,
or with `scipy`'s solvers.
Here, we will only see the included solvers, which are:

* Filtered Back-Projection (**FBP**).
* Simultaneous Algebraic Reconstruction Technique (**SART**).
* Simultaneous Iterative Reconstruction Technique (**SIRT**).
* Primal-Dual Hybrid Gradient (**PDHG**), from Chambolle and Pock.

#### FBP

The FBP is the only analytical (non iterative) algorithm in the group. It
exposes one parameter that is not available for the other methods: `fbp_filter`.
This parameter can either be:

* a filter name, as available from `scikit-image`.
* a custom filter, specified by the user.
* an MR data-driven filter, as per [1].

Reconstructing with FBP can be done like the following, assuming that the
`16 x 10` sinogram is contained in the variable called `sino`:
```python
import numpy as np
import corrct as cct

vol_shape_xy = [10, 10]
angles_rad = np.deg2rad(np.linspace(0, 180, 16, endpoint=False))

solver_fbp = cct.solvers.FBP()

with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    vol, _ = solver_fbp(p, sino)
```

By default, the `"ramp"` filter is selected. Another filter, like the Hann
filter can be selected, by passing the `fbp_filter` parameter at initialization:
```python
solver_fbp = cct.solvers.FBP(fbp_filter="shepp-logan")
```

#### SIRT and PDHG

The SIRT and PDHG algorithms, are algebraic (iterative) methods. They both
support regularization, and box constraints on the solution. The PDHG also
supports various data fidelity terms.

The interface of the iterative methods is the same as for the FBP, with the only
difference of requiring an iterations count:
```python
import numpy as np
import corrct as cct

vol_shape_xy = [10, 10]
angles_rad = np.deg2rad(np.linspace(0, 180, 16, endpoint=False))

solver_sirt = cct.solvers.SIRT()

with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    vol, _ = solver_sirt(p, sino, iterations=100)
```

It is possible to specify an intial solution or box limits on the solutions like
the following:
```python
x0 = np.ones(vol_shape_xy)  # Initial solution
lower_limit = 0.0  # Constraint

with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    vol, _ = solver_sirt(p, sino, iterations=100, x0=x0, lower_limit=lower_limit)
```
The same goes for the parameter `upper_limit`.

## Attenuation correction

Here, we describe how to do attenuation correction.

## Data terms and regularization

Iterative methods support regularizers, and data fidelity terms. The former can
be used to impose prior knowledge on the reconstructed solution, while the
latter impose prior knowledge on the weight given to the data points.

### Regularizers

Famous regularizers are the TV-min and wavelet l1-min. They can be found in the
`regularizers` module.

### Data fidelity terms

The PDHG algorithm supports various data fidelity terms. They can be found in
the `data_terms` module, and they include:
* l2 norm - least squares reconstruction - default: `DataFidelity_l2`
* weighted l2 norm - when the variance of the sinogram points is known: `DataFidelity_wl2`
* l1 norm - when the sinogram noise is mostly sparse: `DataFidelity_l1`
* Kullback-Leibler - when dealing with Poisson noise: `DataFidelity_KL`

## Guided regularization parameter selection

Regularizer parameter selection can be performed through either
cross-validation, or the elbow method.

## References

[1] Pelt, D. M., & Batenburg, K. J. (2014). Improving filtered backprojection
reconstruction by data-dependent filtering. Image Processing, IEEE
Transactions on, 23(11), 4750-4762.
