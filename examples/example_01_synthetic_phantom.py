# -*- coding: utf-8 -*-
"""
This example reproduces the phantom used in:

- N. Viganò and V. A. Solé, “Physically corrected forward operators for
induced emission tomography: a simulation study,” Meas. Sci. Technol., no.
Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands
"""

import numpy as np

import matplotlib.pyplot as plt

import corrct as cct
import corrct.utils_test

try:
    import phantom
except ImportError:
    cct.utils_test.download_phantom()
    import phantom


vol_shape = [256, 256, 3]
data_type = np.float32

ph_or = np.squeeze(phantom.modified_shepp_logan(vol_shape).astype(data_type))
ph_or = ph_or[:, :, 1]

# psf = spsig.gaussian(11, 1)
psf = None

(ph, vol_att_in, vol_att_out) = cct.utils_test.phantom_assign_concentration(ph_or)
(sino, angles, expected_ph) = cct.utils_test.create_sino(
    ph, 120, vol_att_in=vol_att_in, vol_att_out=vol_att_out, psf=psf)

apply_corrections = True
short_rec = False

if short_rec is True:
    num_sart_iterations = 1
    num_sirt_iterations = 25
    num_cp_iterations = 10
    num_cptv_iterations = 10
else:
    num_sart_iterations = 5
    num_sirt_iterations = 250
    num_cp_iterations = 100
    num_cptv_iterations = 250

renorm_factor = np.max(np.sum(sino, axis=-1)) / np.sqrt(np.sum(np.array(ph.shape) ** 2)) / 50

if apply_corrections:
    print('Reconstructing with SART w/ corrections')
    rec_sart = cct.reconstruct(
        'SART', sino, angles, vol_att_in=vol_att_in, vol_att_out=vol_att_out,
        psf=psf, lower_limit=0, data_term='l2', iterations=num_sart_iterations)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_sart))

    print('Reconstructing with SIRT w/ corrections')
    rec_sirt = cct.reconstruct(
        'SIRT', sino, angles, vol_att_in=vol_att_in, vol_att_out=vol_att_out,
        psf=psf, lower_limit=0, data_term='l2', iterations=num_sirt_iterations)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_sirt))

    print('Reconstructing with CP - using KL w/ corrections')
    rec_cpkl = cct.reconstruct(
        'CP', sino / renorm_factor, angles, iterations=num_cp_iterations,
        data_term='kl', vol_att_in=vol_att_in, vol_att_out=vol_att_out,
        psf=psf, lower_limit=0) * renorm_factor
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_cpkl))

    print('Reconstructing with CPTV - using KL w/ corrections')
    rec_cptvkl = cct.reconstruct(
        'CPTV', sino / renorm_factor, angles, iterations=num_cptv_iterations,
        data_term='kl', vol_att_in=vol_att_in, vol_att_out=vol_att_out,
        psf=psf, lower_limit=0, lambda_reg=2e-1) * renorm_factor
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_cptvkl))

else:
    print('Reconstructing with SART w/o corrections')
    rec_sart = cct.reconstruct(
        'SART', sino, angles, lower_limit=0, data_term='l2', iterations=num_sart_iterations)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_sart))

    print('Reconstructing with SIRT w/o corrections')
    rec_sirt = cct.reconstruct(
        'SIRT', sino, angles, lower_limit=0, data_term='l2', iterations=num_sirt_iterations)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_sirt))

    print('Reconstructing with CP - using KL w/o corrections')
    rec_cpkl = cct.reconstruct(
        'CP', sino / renorm_factor, angles, iterations=num_cp_iterations,
        data_term='kl', lower_limit=0) * renorm_factor
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_cpkl))

    print('Reconstructing with CPTV - using KL w/o corrections')
    rec_cptvkl = cct.reconstruct(
        'CPTV', sino / renorm_factor, angles, iterations=num_cptv_iterations,
        data_term='kl', lower_limit=0, lambda_reg=2e-1) * renorm_factor
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_cptvkl))

(f, axes) = plt.subplots(2, 3)
axes[0, 0].imshow(expected_ph)
axes[0, 0].set_title('Phantom')
axes[1, 0].imshow(sino)
axes[1, 0].set_title('Sinogram')

axes[0, 1].imshow(rec_sart)
axes[0, 1].set_title('SART')
axes[0, 2].imshow(rec_sirt)
axes[0, 2].set_title('SIRT')
axes[1, 1].imshow(rec_cpkl)
axes[1, 1].set_title('CP-KL')
axes[1, 2].imshow(rec_cptvkl)
axes[1, 2].set_title('CP-KL-TV')

plt.show()
