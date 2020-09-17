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

import matplotlib.pyplot as plt

import corrct as cct
import corrct.utils_test

try:
    import phantom
except ImportError:
    cct.utils_test.download_phantom()
    import phantom


vol_shape_xy = [256, 256]
ph = np.squeeze(phantom.modified_shepp_logan([*vol_shape_xy, 3]).astype(np.float32))
ph = ph[:, :, 1]

(sino, angles_rad, expected_ph, _) = cct.utils_test.create_sino(ph, 30, add_poisson=True, photon_flux=1e4)

print('Reconstructing sino:')
with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    print('- Filter: "Shepp-Logan"')
    vol_sl = p.fbp(sino)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, vol_sl))
    print('- Filter: "MR"')
    filter_mr = cct.projectors.FilterMR()
    vol_mr = p.fbp(sino, filter_mr)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, vol_mr))
    print('- Filter: "MR-smooth"')
    filter_mr_reg = cct.projectors.FilterMR(lambda_smooth=5e0)
    vol_mr_reg = p.fbp(sino, filter_mr_reg)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, vol_mr_reg))

    filt = filter_mr.compute_filter(sino, p)
    filt_reg = filter_mr_reg.compute_filter(sino, p)

(f, axes) = plt.subplots(2, 2, sharex=True, sharey=True)
axes[0, 0].imshow(expected_ph)
axes[0, 0].set_title('Phantom')
axes[0, 1].imshow(vol_sl)
axes[0, 1].set_title('FBP')
axes[1, 0].imshow(vol_mr)
axes[1, 0].set_title('FBP-MR')
axes[1, 1].imshow(vol_mr_reg)
axes[1, 1].set_title('FBP-MR-smooth')
plt.show()

(f, axes) = plt.subplots(1, 2)
axes[0].plot(filt)
axes[0].plot(filt_reg)
axes[1].plot(np.abs(np.fft.fft(np.fft.ifftshift(filt))))
axes[1].plot(np.abs(np.fft.fft(np.fft.ifftshift(filt_reg))))
plt.show()
