# Tutorial (basics)

In this tutorial, we will first learn the basics of how to reconstruct
projection data with `corrct`.
We will then describe how to use data terms and regularizers.
Finally, we will introduce more advanced topics like attenuation correction and
guided regularization hyper-parameter selection, which will be covered in
dedicated tutorials.

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

The projection geometry is specified through the creation of *projectors* from
the [](#projectors) module.
The simplest projector is called
[`ProjectorUncorrected`](#projectors.ProjectorUncorrected),
and it serves as basis for more complex projectors. The projector
[`ProjectorAttenuationXRF`](#projectors.ProjectorAttenuationXRF)
derives from the
[`ProjectorUncorrected`](#projectors.ProjectorUncorrected),
and it implements the XRF specific bits, with respect to multi-detector /
multi-element handling and attenuation correction.

Projectors are usually used through the `with` statement. This takes care of
initializing and de-initializing their underlying resources (e.g. GPU usage).

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

Creating attenuation correction projectors is a bit more involved, and it
requires knowledge about the sample, regarding composition and morphology.
It is briefly introduce in the [attenuation section](#attenuation-correction),
and described in the [attenuation correction tutorial](attenuation_tutorial.md).

Projectors can use different backends, depending on the available packages,
system resources, and user requests. The included projector backends are based
on the `scikit-image` and `astra-toolbox` packages.
They can be selected by passing the strings `"astra"` or `"skimage"` to the
parameter `backend`.
Advanced users can create custom backends, by deriving the base class
`ProjectorBackend` from the module [](#_projector_backends).

### The solvers

Tomographic reconstructions can be achieved using either the included solvers
from [](#solvers) module, or with `scipy`'s solvers.
The included solvers are:

* Filtered Back-Projection: [**FBP**](#solvers.FBP).
* Simultaneous Algebraic Reconstruction Technique [**SART**](#solvers.SART).
* Simultaneous Iterative Reconstruction Technique [**SIRT**](#solvers.SIRT).
* Primal-Dual Hybrid Gradient [**PDHG**](#solvers.PDHG), from Chambolle and Pock.

#### FBP

FBP is the only analytical (non iterative) algorithm in the group. It
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

#### SIRT, PDHG, and MLEM

The SIRT and PDHG algorithms, are algebraic (iterative) methods. They both
support regularization, and box constraints on the solution. The PDHG also
supports various data fidelity terms.
The MLEM algorithm is also an iterative algorithm to find the maximum likelihood estimation of the reconstructed signal. The MLEM does not currently support any regularization.

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
The MLEM algorithm assumes a `lower_limit` of 0.

## Data terms and regularization

Iterative methods support regularizers, and data fidelity terms. The former can
be used to impose prior knowledge on the reconstructed solution, while the
latter impose prior knowledge on the weight given to the data points.

### Regularizers

Famous regularizers are the TV-min and wavelet l1-min. They can be found in the
[](#regularizers) module.

### Data fidelity terms

The PDHG algorithm supports various data fidelity terms. They can be found in
the [](#data_terms) module, and they include:
* l2 norm - least squares reconstruction - default:
[`DataFidelity_l2`](#data_terms.DataFidelity_l2)
* weighted l2 norm - when the variance of the sinogram points is known:
[`DataFidelity_wl2`](#data_terms.DataFidelity_wl2)
* l1 norm - when the sinogram noise is mostly sparse:
[`DataFidelity_l1`](#data_terms.DataFidelity_l1)
* Kullback-Leibler - when dealing with Poisson noise:
[`DataFidelity_KL`](#data_terms.DataFidelity_KL)

## Attenuation correction

This package implements the attenuation correction method described in [2].
The correction of the attenuation effects is subject to the knowledge of an
attenuation map for the following experimental conditions:

* Acquisition geometry (i.e. sample rotation angles, beam size / resolution, detector position, etc)
* Excitation beam energy and emission photon energy
* Sample morphology and local average composition

This is usually achieved in two ways. The simplest way is to provide the projector
[](#projectors.ProjectorAttenuationXRF)
with the corresponding attenuation maps for the excitation beam and emitted photons.
The respective parameters are: `att_in` and `att_out`. This also requires to
provide the angle(s) of the detector(s) with respect to the incoming beam
direction, through the parameter `angles_detectors_rad`.
The values in `att_in` and `att_out` should be in "linear attenuation" per pixel
length. The values in `angles_detectors_rad` should be in radians, as suggested
by the name of the parameter.  
The drawback of the simple way is that the computed local attenuation per angle
cannot be re-used with other projectors, and the computation / scaling of the
maps is delegated entirely to the user.

The user can also choose to use the class [](#physics.attenuation.AttenuationVolume)
from the [](#physics.attenuation) sub-module of the [](#physics) module.
This class is used internally in the projector [](#projectors.ProjectorAttenuationXRF),
and it can be used particularly in conjunction with the class
[](#physics.materials.VolumeMaterial) from the [](#physics) module.

For a more in-depth walk-through on attenuation correction, we refer to the dedicated
[attenuation tutorial](attenuation_tutorial.md).
For a dedicated description of the projection and attenuation correction geometry,
the reader can have a look at the dedicated [geometry page](geometry.md).
For a in-depth description of the functionality available in the physics module,
instead, the reader can have a look at the dedicated [physics page](physics_model.md).

## Guided regularization parameter selection

Regularizer parameter selection can be performed through either
cross-validation, or the elbow method.

## References

[1] Pelt, D. M., & Batenburg, K. J. (2014). Improving filtered backprojection
reconstruction by data-dependent filtering. Image Processing, IEEE
Transactions on, 23(11), 4750-4762.  
[2] N. Viganò and V. A. Solé, "Physically corrected forward operators for
induced emission tomography: a simulation study," Meas. Sci. Technol., no.
Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.  
