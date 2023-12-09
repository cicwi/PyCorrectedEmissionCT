"""
This example reproduces the phantom used in:

- N. Viganò and V. A. Solé, “Physically corrected forward operators for
induced emission tomography: a simulation study,” Meas. Sci. Technol., no.
Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands
"""

import matplotlib.pyplot as plt
import numpy as np
import corrct as cct


try:
    import phantom
except ImportError:
    cct.testing.download_phantom()
    import phantom


vol_shape = [256, 256, 3]

ph_or = np.squeeze(phantom.modified_shepp_logan(vol_shape).astype(np.float32))
ph_or = ph_or[:, :, 1]

# PSF = spsig.gaussian(11, 1)
PSF = None
LOWER_LIMIT = 0.0

# We first create the data (with attenuation)
(ph, vol_att_in, vol_att_out) = cct.testing.phantom_assign_concentration(ph_or)
(sino, angles, expected_ph, _) = cct.testing.create_sino(ph, 120, vol_att_in=vol_att_in, vol_att_out=vol_att_out, psf=PSF)

# We create the two solvers that we will use to reconstruct the data
solver_sart = cct.solvers.SART(verbose=True)
solver_sirt = cct.solvers.SIRT(verbose=True)

print("Reconstructing w/o corrections:")
with cct.projectors.ProjectorUncorrected(ph.shape, angles, psf=PSF, create_single_projs=True) as p:
    rec_sart_uncorr, _ = solver_sart(p, sino, iterations=5, lower_limit=LOWER_LIMIT)

    expected_power, noise_power = cct.testing.compute_error_power(expected_ph, rec_sart_uncorr)
    print(f"- Phantom power: {expected_power:g}, noise power: {noise_power:g}")

    rec_sirt_uncorr, _ = solver_sirt(p, sino, iterations=250, lower_limit=LOWER_LIMIT)

    expected_power, noise_power = cct.testing.compute_error_power(expected_ph, rec_sirt_uncorr)
    print(f"- Phantom power: {expected_power:g}, noise power: {noise_power:g}")

print("Reconstructing w/ corrections:")
with cct.projectors.ProjectorAttenuationXRF(ph.shape, angles, psf=PSF, att_in=vol_att_in, att_out=vol_att_out) as p:
    rec_sart_corr, _ = solver_sart(p, sino, iterations=5, lower_limit=LOWER_LIMIT)

    expected_power, noise_power = cct.testing.compute_error_power(expected_ph, rec_sart_corr)
    print(f"- Phantom power: {expected_power:g}, noise power: {noise_power:g}")

    rec_sirt_corr, _ = solver_sirt(p, sino, iterations=250, lower_limit=LOWER_LIMIT)

    expected_power, noise_power = cct.testing.compute_error_power(expected_ph, rec_sirt_corr)
    print(f"- Phantom power: {expected_power:g}, noise power: {noise_power:g}")


(fig, axes) = plt.subplots(2, 3)

axes[0, 0].imshow(expected_ph)
axes[0, 0].set_title("Phantom")
axes[1, 0].imshow(sino)
axes[1, 0].set_title("Sinogram")

axes[0, 1].imshow(rec_sart_uncorr)
axes[0, 1].set_title("SART - Uncorrected")
axes[0, 2].imshow(rec_sirt_uncorr)
axes[0, 2].set_title("SIRT - Uncorrected")
axes[1, 1].imshow(rec_sart_corr)
axes[1, 1].set_title("SART - Corrected")
axes[1, 2].imshow(rec_sirt_corr)
axes[1, 2].set_title("SIRT - Corrected")

fig.tight_layout()

plt.show(block=False)
