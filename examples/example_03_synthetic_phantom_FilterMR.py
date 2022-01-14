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

try:
    import skimage.transform
    __have_skimage__ = True
except ImportError:
    print('Scikit image not available, no comparison against other filters will be possible')
    __have_skimage__ = False


vol_shape_xy = [256, 256]
ph = np.squeeze(phantom.modified_shepp_logan([*vol_shape_xy, 3]).astype(np.float32))
ph = ph[:, :, 1]
fbp_filter = 'Shepp-Logan'

(sino, angles_rad, expected_ph, _) = cct.utils_test.create_sino(ph, 30, add_poisson=True, photon_flux=1e4)

print('Reconstructing sino:')
with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    print('- Filter: "%s"' % fbp_filter)
    vol_sl = p.fbp(sino, fbp_filter=fbp_filter)
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
axes[0, 1].set_title('FBP (%s)' % fbp_filter)
axes[1, 0].imshow(vol_mr)
axes[1, 0].set_title('FBP-MR')
axes[1, 1].imshow(vol_mr_reg)
axes[1, 1].set_title('FBP-MR-smooth')
plt.show(block=False)

filt_f = np.abs(np.fft.fft(np.fft.ifftshift(filt)))[:(len(filt) // 2)]
filt_reg_f = np.abs(np.fft.fft(np.fft.ifftshift(filt_reg)))[:(len(filt) // 2)]

(f, axes) = plt.subplots(1, 2)
axes[0].plot(filt, label='FBP-MR')
axes[0].plot(filt_reg, label='FBP-MR-smooth')
axes[0].set_title('Real-space')
axes[0].set_xlabel('Pixel')
axes[1].plot(filt_f, label='FBP-MR')
axes[1].plot(filt_reg_f, label='FBP-MR-smooth')
axes[1].set_title('Fourier-space')
axes[1].set_xlabel('Frequency')

if __have_skimage__:
    filt_sl_f = skimage.transform.radon_transform._get_fourier_filter(vol_shape_xy[0], fbp_filter.lower())
    filt_sl_f = np.squeeze(filt_sl_f)
    filt_sl = np.real(np.fft.fftshift(np.fft.ifft(filt_sl_f)))
    filt_sl_f = filt_sl_f[:(len(filt) // 2)]
    axes[0].plot(filt_sl, label=fbp_filter)
    axes[1].plot(filt_sl_f, label=fbp_filter)

axes[0].grid()
axes[1].grid()
axes[1].legend()

(f_prof, ax) = plt.subplots()
ax.plot(np.squeeze(expected_ph[..., 172]), label='Phantom')
ax.plot(np.squeeze(vol_sl[..., 172]), label='FBP (%s)' % fbp_filter)
ax.plot(np.squeeze(vol_mr[..., 172]), label='FBP-MR')
ax.plot(np.squeeze(vol_mr_reg[..., 172]), label='FBP-MR-smooth')
ax.legend()
ax.grid()

plt.show(block=False)
