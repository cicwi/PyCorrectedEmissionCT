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

filter_name = 'Shepp-Logan'

(sino, angles_rad, expected_ph, _) = cct.utils_test.create_sino(ph, 30, add_poisson=True, photon_flux=1e4)

print('Reconstructing sino:')
with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    print('- Filter: "%s"' % filter_name)
    filter_fbp = cct.filters.FilterFBP(filter_name=filter_name)
    solver_fbp = cct.solvers.FBP(fbp_filter=filter_fbp)
    # solver_fbp = cct.solvers.FBP(fbp_filter=filter_name)  # Alternate API
    vol_sl, _ = solver_fbp(p, sino)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, vol_sl))

    print('- Filter: "MR"')
    filter_mr = cct.filters.FilterMR(projector=p, binning_start=10, binning_type="incremental")
    solver_fbp = cct.solvers.FBP(fbp_filter=filter_mr)
    # solver_fbp = cct.solvers.FBP(fbp_filter="data")  # Alternate API
    vol_mr, _ = solver_fbp(p, sino)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, vol_mr))

    print('- Filter: "MR-smooth"')
    filter_mr_reg = cct.filters.FilterMR(projector=p, lambda_smooth=1e1*0)
    solver_fbp = cct.solvers.FBP(fbp_filter=filter_mr_reg)
    vol_mr_reg, _ = solver_fbp(p, sino)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, vol_mr_reg))

    filt_fbp_f = filter_fbp.filter_fourier
    filt_mr_f = filter_mr.filter_fourier
    filt_reg_f = filter_mr_reg.filter_fourier

    filt_fbp_r = filter_fbp.filter_real
    filt_mr_r = filter_mr.filter_real
    filt_reg_r = filter_mr_reg.filter_real

f, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=cm2inch([24, 24]))
axes[0, 0].imshow(expected_ph)
axes[0, 0].set_title('Phantom')
axes[0, 1].imshow(vol_sl)
axes[0, 1].set_title('FBP (%s)' % filter_name)
axes[1, 0].imshow(vol_mr)
axes[1, 0].set_title('FBP-MR')
axes[1, 1].imshow(vol_mr_reg)
axes[1, 1].set_title('FBP-MR-smooth')
f.tight_layout()

f, axes = plt.subplots(1, 2, figsize=cm2inch([24, 10]))
axes[0].plot(filt_fbp_r, label=filter_name)
axes[0].plot(filt_mr_r, label='FBP-MR')
axes[0].plot(filt_reg_r, label='FBP-MR-smooth')
axes[0].set_title('Real-space')
axes[0].set_xlabel('Pixel')
axes[1].plot(filt_fbp_f, label=filter_name)
axes[1].plot(filt_mr_f, label='FBP-MR')
axes[1].plot(filt_reg_f, label='FBP-MR-smooth')
axes[1].set_title('Fourier-space')
axes[1].set_xlabel('Frequency')

axes[0].grid()
axes[1].grid()
axes[1].legend()
f.tight_layout()

f, ax = plt.subplots(figsize=cm2inch([24, 10]))
ax.plot(np.squeeze(vol_sl[..., 172]), label='FBP (%s)' % filter_name)
ax.plot(np.squeeze(vol_mr[..., 172]), label='FBP-MR')
ax.plot(np.squeeze(vol_mr_reg[..., 172]), label='FBP-MR-smooth')
ax.plot(np.squeeze(expected_ph[..., 172]), label='Phantom')
ax.legend()
ax.grid()
f.tight_layout()

plt.show(block=False)
