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

num_angles_clean = int(256 * np.pi / 2)
num_angles_noise = 256

sino_clean, angles_rad_clean, expected_ph, _ = cct.utils_test.create_sino(
    ph, num_angles_clean, add_poisson=True, photon_flux=photon_flux_clean
)
sino_noise_1, angles_rad_noise_1, _, _ = cct.utils_test.create_sino(
    ph, num_angles_noise, add_poisson=True, photon_flux=photon_flux_noise
)
sino_noise_2, angles_rad_noise_2, _, _ = cct.utils_test.create_sino(
    ph, num_angles_noise, add_poisson=True, photon_flux=photon_flux_noise
)

expected_ph /= 1e4
sino_clean /= 1e4
sino_noise_1 /= 1e4 / photon_flux_delta
sino_noise_2 /= 1e4 / photon_flux_delta

with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad_clean) as p:
    solver_fbp_sl = cct.solvers.FBP()
    vol_hq, _ = solver_fbp_sl(p, sino_clean)
    print("High quality:")
    print("- Phantom power: %g, noise power: %g" % cct.utils_test.compute_error_power(expected_ph, vol_hq))

with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad_noise_2) as p:
    solver_fbp_sl = cct.solvers.FBP()
    vol_lq_1, _ = solver_fbp_sl(p, sino_noise_1)
    vol_lq_2, _ = solver_fbp_sl(p, sino_noise_2)
    print("Low quality:")
    print("- Phantom power: %g, noise power: %g" % cct.utils_test.compute_error_power(expected_ph, vol_lq_1))

    num_pix_trn = 2**13
    num_pix_tst = 2**11

    # denoiser = cct.denoisers.Denoiser_NNFBP(
    #     projector=p, num_pixels_trn=num_pix_trn, num_pixels_tst=num_pix_tst, num_fbps=4, hidden_layers=[4]
    # )
    # denoiser.fit(sino_noise_1, vol_hq, train_epochs=30)

    # denoiser = cct.denoisers.Denoiser_N2F(
    #     projector=p, num_pixels_trn=num_pix_trn, num_pixels_tst=num_pix_tst, num_fbps=4, hidden_layers=[4]
    # )
    # denoiser.fit(np.array([sino_noise_1, sino_noise_2]), np.array([vol_lq_2, vol_lq_1]), train_epochs=30)

    angles_partial_0 = angles_rad_noise_1[0::2]
    angles_partial_1 = angles_rad_noise_1[1::2]

    sino_partial_0 = sino_noise_1[..., 0::2, :]
    sino_partial_1 = sino_noise_1[..., 1::2, :]

    with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_partial_0) as p0:
        with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_partial_1) as p1:
            vol_partial_0, _ = solver_fbp_sl(p0, sino_partial_0)
            vol_partial_1, _ = solver_fbp_sl(p1, sino_partial_1)

            denoiser = cct.denoisers.Denoiser_N2F(
                projector=p, num_pixels_trn=num_pix_trn, num_pixels_tst=num_pix_tst, num_fbps=4, hidden_layers=[4]
            )
            denoiser.fit(
                np.array([sino_partial_0, sino_partial_1]),
                np.array([vol_partial_1, vol_partial_0]),
                train_epochs=30,
                projectors=[p0, p1],
            )

    # vol_den_1 = denoiser.predict(sino_noise_1)
    vol_den_2 = denoiser.predict(sino_noise_2)
    print("NNFBP quality:")
    print("- Phantom power: %g, noise power: %g" % cct.utils_test.compute_error_power(expected_ph, vol_den_2))

    filt_learned_f = denoiser.filters.filter_fourier
    filt_learned_r = denoiser.filters.filter_real

    sino_noise_mean = np.mean([sino_noise_1, sino_noise_2], axis=0)
    solver_fbp_sl = cct.solvers.FBP(fbp_filter="hann")
    vol_lq_m, _ = solver_fbp_sl(p, sino_noise_mean)
    vol_den_m = denoiser.predict(sino_noise_mean)


denoiser.filters.plot_filters()

learned_rec_label = f"Learned ({denoiser.num_fbps}" + "".join([f"=>{l}" for l in denoiser.hidden_layers]) + "=>1)"

# f, ax = plt.subplots(figsize=cm2inch([24, 10]))
# ax.plot(np.squeeze((vol_hq * vol_mask)[..., 172]), label="High quality")
# ax.plot(np.squeeze((vol_lq_2 * vol_mask)[..., 172]), label="Low quality")
# ax.plot(np.squeeze((vol_den_2 * vol_mask)[..., 172]), label=learned_rec_label)
# ax.plot(np.squeeze(expected_ph[..., 172]), label="Phantom")
# ax.legend()
# ax.grid()
# f.tight_layout()

vol_mask = cct.utils_proc.get_circular_mask(vol_hq.shape)

f, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=cm2inch([36, 24]))
axes[0, 0].imshow(expected_ph)
axes[0, 0].set_title("Phantom")
axes[1, 0].imshow(vol_hq * vol_mask)
axes[1, 0].set_title(f"High quality (projs n.: {num_angles_clean} - max photons: {photon_flux_clean:.2e})")
axes[0, 1].imshow(vol_lq_2 * vol_mask)
axes[0, 1].set_title(f"Low quality (projs n.: {num_angles_noise} - max photons {photon_flux_noise:.2e})")
axes[1, 1].imshow(vol_den_2 * vol_mask)
axes[1, 1].set_title(learned_rec_label)
axes[0, 2].imshow(vol_lq_m * vol_mask)
axes[0, 2].set_title("Low quality - x2 dose")
axes[1, 2].imshow(vol_den_m * vol_mask)
axes[1, 2].set_title(learned_rec_label + " - x2 dose")
f.tight_layout()

vols = [vol_hq * vol_mask, vol_lq_2 * vol_mask, vol_den_2 * vol_mask, vol_lq_m * vol_mask, vol_den_m * vol_mask]
frcs = [np.array([])] * len(vols)
for ii, rec in enumerate(vols):
    frcs[ii], T = cct.utils_proc.compute_frc(expected_ph, rec, snrt=0.4142)

f, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=cm2inch((24, 12)))
axs.plot(np.squeeze(frcs[0]), label="High quality")
axs.plot(np.squeeze(frcs[1]), label="Low quality")
axs.plot(np.squeeze(frcs[2]), label=learned_rec_label)
axs.plot(np.squeeze(frcs[3]), label="Low quality - x2 dose")
axs.plot(np.squeeze(frcs[4]), label=learned_rec_label + " - x2 dose")
axs.plot(np.squeeze(T), label="T 1/2 bit")
axs.set_ylim(0, None)
axs.legend()
axs.grid()
f.tight_layout()

plt.show(block=False)
