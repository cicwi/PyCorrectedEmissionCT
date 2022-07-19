# -*- coding: utf-8 -*-
"""
This example shows the use and performance of the data dependent FBP filter, from:

Pelt, D. M., & Batenburg, K. J. (2014). Improving filtered backprojection
reconstruction by data-dependent filtering. Image Processing, IEEE
Transactions on, 23(11), 4750-4762.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple

import matplotlib.pyplot as plt

import corrct as cct
import corrct.utils_test

try:
    import phantom
except ImportError:
    cct.utils_test.download_phantom()
    import phantom


def cm2inch(x: ArrayLike) -> Tuple[float, float]:
    """Convert cm to inch.

    Parameters
    ----------
    x : ArrayLike
        Sizes in cm.

    Returns
    -------
    Tuple[float, float]
        Sizes in inch.
    """
    return tuple(np.array(x) / 2.54)


vol_shape_xy = [256, 256]
ph = np.squeeze(phantom.modified_shepp_logan([*vol_shape_xy, 3]).astype(np.float32))
ph = ph[:, :, 1]

photon_flux_clean = 1e6
photon_flux_delta = 1e2
photon_flux_noise = photon_flux_clean / photon_flux_delta

num_angles_clean = int(180 * np.pi / 2)
num_angles_noise = 60

sino_clean, angles_rad_clean, expected_ph, _ = cct.utils_test.create_sino(
    ph, num_angles_clean, add_poisson=True, photon_flux=photon_flux_clean
)
sino_noise, angles_rad_noise, _, _ = cct.utils_test.create_sino(
    ph, num_angles_noise, add_poisson=True, photon_flux=photon_flux_noise
)
sino_noise2, angles_rad_noise, _, _ = cct.utils_test.create_sino(
    ph, num_angles_noise, add_poisson=True, photon_flux=photon_flux_noise
)

expected_ph /= 1e4
sino_clean /= 1e4
sino_noise /= 1e4 / photon_flux_delta
sino_noise2 /= 1e4 / photon_flux_delta

print("High quality:")
with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad_clean) as p:
    solver_fbp_sl = cct.solvers.FBP()
    vol_hq, _ = solver_fbp_sl(p, sino_clean)
    print("- Phantom power: %g, noise power: %g" % cct.utils_test.compute_error_power(expected_ph, vol_hq))

vol_mask = cct.utils_proc.get_circular_mask(vol_hq.shape)

with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad_noise) as p:
    solver_fbp_sl = cct.solvers.FBP()
    vol_lq, _ = solver_fbp_sl(p, sino_noise)
    print("Training:")
    den = cct.denoisers.Denoiser_NNFBP(projector=p, num_pixels_train=512, num_fbps=7, hidden_layers=[5, 3])
    den.compute_filter(sino_noise, vol_hq)

    # vol_den_hq = den.apply_filter(sino_noise)
    vol_den_hq2 = den.apply_filter(sino_noise2)

    filt_learned_f = den.filters.filter_fourier
    filt_learned_r = den.filters.filter_real


f, axes = plt.subplots(1, 2, figsize=cm2inch([24, 10]))
for ii in range(den.filters.num_filters):
    axes[0].plot(filt_learned_r[ii, ...], label=f"Learned-{ii}")
axes[0].set_title("Real-space")
axes[0].set_xlabel("Pixel")

for ii in range(den.filters.num_filters):
    axes[1].plot(filt_learned_f[ii, ...], label=f"Learned-{ii}")
axes[1].set_title("Fourier-space")
axes[1].set_xlabel("Frequency")

axes[0].grid()
axes[1].grid()
axes[1].legend()
f.tight_layout()

learned_rec_label = f"Learned ({den.num_fbps}" + "".join([f"=>{l}" for l in den.hidden_layers]) + "=>1)"

f, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=cm2inch([24, 24]))
axes[0, 0].imshow(expected_ph)
axes[0, 0].set_title("Phantom")
axes[0, 1].imshow(vol_hq * vol_mask)
axes[0, 1].set_title(f"High quality")
axes[1, 0].imshow(vol_lq * vol_mask)
axes[1, 0].set_title("Low quality")
axes[1, 1].imshow(vol_den_hq2 * vol_mask)
axes[1, 1].set_title(learned_rec_label)
f.tight_layout()


f, ax = plt.subplots(figsize=cm2inch([24, 10]))
ax.plot(np.squeeze((vol_hq * vol_mask)[..., 172]), label="High quality")
ax.plot(np.squeeze((vol_lq * vol_mask)[..., 172]), label="Low quality")
ax.plot(np.squeeze((vol_den_hq2 * vol_mask)[..., 172]), label=learned_rec_label)
ax.plot(np.squeeze(expected_ph[..., 172]), label="Phantom")
ax.legend()
ax.grid()
f.tight_layout()

vols = [vol_hq * vol_mask, vol_lq * vol_mask, vol_den_hq2 * vol_mask]
frcs = [None] * len(vols)
for ii, rec in enumerate(vols):
    frcs[ii], T = cct.utils_proc.compute_frc(expected_ph, rec, snrt=0.4142, supersampling=2)

f, axs = plt.subplots(1, 1, sharex=True, sharey=True)
axs.plot(np.squeeze(frcs[0]), label="High quality")
axs.plot(np.squeeze(frcs[1]), label="Low quality")
axs.plot(np.squeeze(frcs[2]), label=learned_rec_label)
axs.plot(np.squeeze(T), label="T 1/2 bit")
axs.legend()
axs.grid()
f.tight_layout()

plt.show(block=False)
