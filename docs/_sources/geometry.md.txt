# Reconstruction geometry

Here we provide visual ways to assess the correctness of the geometry. In particular, we look at:

1. Consistency of the projectors in `corrct`
2. Flip of the sinogram along the U coordinate
3. Rotation direction: `clockwise` vs `counter-clockwise`
4. The excitation beam direction: `bottom-up`, `top-down`, `left-rightwards`, `right-leftwards`
5. The position of the XRF detector with respect to the excitation beam: `right` vs `left`

To produce the relevant figures, we use the following code:
```python
import numpy as np
from matplotlib import pyplot as plt
import corrct as cct


vol_shape = [256, 256]
sino_wu = np.zeros((4, vol_shape[0]))
sino_wu[:, 10] = 1

test_angles = np.deg2rad([0, 45, 90, 180])

with cct.projectors.ProjectorUncorrected(vol_shape, test_angles, backend="skimage") as A:
    bp_angles_s = A.bp(sino_wu)

with cct.projectors.ProjectorUncorrected(vol_shape, test_angles, backend="astra") as A:
    bp_angles_a = A.bp(sino_wu)

vol_shape = [256, 256, 2]
sino_wu = np.zeros((2, 4, vol_shape[0]))
sino_wu[..., 10] = 1

with cct.projectors.ProjectorUncorrected(vol_shape, test_angles) as A:
    bp_angles_3 = A.bp(sino_wu)

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=[9, 3.5])
fig.suptitle("Projector consistency, and sinogram horizontal (U) flip", fontsize=16)
axs[0].imshow(bp_angles_s, vmin=0.0, vmax=1)
axs[0].set_title("Scikit-image")
axs[1].imshow(bp_angles_a, vmin=0.0, vmax=1)
axs[1].set_title("Astra 2D")
axs[2].imshow(bp_angles_3[0], vmin=0.0, vmax=1)
axs[2].set_title("Astra 3D")
plt.tight_layout()

vol_shape = [256, 256]
vol_att_test = cct.processing.circular_mask(vol_shape, radius_offset=-80).astype(np.float32)
det_angle_rad = -np.pi / 2

att_vol = cct.attenuation.AttenuationVolume(
    incident_local=vol_att_test, emitted_local=None, angles_rot_rad=test_angles, angles_det_rad=det_angle_rad
)
att_vol.compute_maps()

fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=[9, 2.5])
fig.suptitle("Attenuation IN", fontsize=16)
for ii_a, a in enumerate(test_angles):
    att_vol.plot_map(ax[ii_a], ii_a)
    ax[ii_a].set_title(f"{np.rad2deg(a)}")
fig.tight_layout()

att_vol = cct.attenuation.AttenuationVolume(
    incident_local=None, emitted_local=vol_att_test, angles_rot_rad=test_angles, angles_det_rad=det_angle_rad
)
att_vol.compute_maps()

fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=[9, 2.5])
fig.suptitle("Attenuation OUT", fontsize=16)
for ii_a, a in enumerate(test_angles):
    att_vol.plot_map(ax[ii_a], ii_a)
    ax[ii_a].set_title(f"{np.rad2deg(a)}")
fig.tight_layout()
```

## Rotation direction and sinogram flip

![geometry-projectors-coherence](images/geometry-projectors-comparison.png)

## Incoming beam direction

![geometry-attenuation-in](images/geometry-attenuation-incoming-beam.png)

## Detector position

![geometry-attenuation-out](images/geometry-attenuation-emitted-photons.png)

