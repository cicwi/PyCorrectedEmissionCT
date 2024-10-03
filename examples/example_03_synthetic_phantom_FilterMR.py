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

import matplotlib.pyplot as plt

import corrct as cct

try:
    import phantom
except ImportError:
    cct.testing.download_phantom()
    import phantom


def cm2inch(x: ArrayLike) -> tuple[float, float]:
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

filter_name = "Shepp-Logan"
filter_mr_lambda = 5e0

(sino, angles_rad, expected_ph, _) = cct.testing.create_sino(ph, 30, add_poisson=True, photon_flux=1e4)

print("Reconstructing sino:")
with cct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    print('- Filter: "%s"' % filter_name)
    filter_fbp = cct.filters.FilterFBP(filter_name=filter_name)
    solver_fbp_sl = cct.solvers.FBP(fbp_filter=filter_fbp)
    # solver_sl = cct.solvers.FBP(fbp_filter=filter_name)  # Alternate API
    vol_sl, _ = solver_fbp_sl(p, sino)
    print("- Phantom power: %g, noise power: %g" % cct.testing.compute_error_power(expected_ph, vol_sl))

    print('- Filter: "MR"')
    filter_mr = cct.filters.FilterMR(projector=p)
    solver_fbp_mr = cct.solvers.FBP(fbp_filter=filter_mr)
    # solver_fbp_mr = cct.solvers.FBP(fbp_filter="data")  # Alternate API
    vol_mr, _ = solver_fbp_mr(p, sino)
    print("- Phantom power: %g, noise power: %g" % cct.testing.compute_error_power(expected_ph, vol_mr))

    print('- Filter: "MR-smooth"')
    filter_mr_reg = cct.filters.FilterMR(projector=p, lambda_smooth=filter_mr_lambda)
    solver_fbp_mr_reg = cct.solvers.FBP(fbp_filter=filter_mr_reg)
    vol_mr_reg, _ = solver_fbp_mr_reg(p, sino)
    print("- Phantom power: %g, noise power: %g" % cct.testing.compute_error_power(expected_ph, vol_mr_reg))

    filt_fbp_f = filter_fbp.filter_fourier
    filt_mr_f = filter_mr.filter_fourier
    filt_reg_f = filter_mr_reg.filter_fourier

    filt_fbp_r = filter_fbp.filter_real
    filt_mr_r = filter_mr.filter_real
    filt_reg_r = filter_mr_reg.filter_real

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=cm2inch([24, 24]))
axes[0, 0].imshow(expected_ph)
axes[0, 0].set_title("Phantom")
axes[0, 1].imshow(vol_sl)
axes[0, 1].set_title(f"FBP-{filter_name}")
axes[1, 0].imshow(vol_mr)
axes[1, 0].set_title("FBP-MR")
axes[1, 1].imshow(vol_mr_reg)
axes[1, 1].set_title(f"FBP-MR-Smooth(W:{filter_mr_lambda})")
fig.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=cm2inch([24, 10]))
axes[0].plot(filt_fbp_r, label=f"FBP-{filter_name}")
axes[0].plot(filt_mr_r, label="FBP-MR")
axes[0].plot(filt_reg_r, label=f"FBP-MR-Smooth(W:{filter_mr_lambda})")
axes[0].set_title("Real-space")
axes[0].set_xlabel("Pixel")
axes[1].plot(filt_fbp_f, label=f"FBP-{filter_name}")
axes[1].plot(filt_mr_f, label="FBP-MR")
axes[1].plot(filt_reg_f, label=f"FBP-MR-Smooth(W:{filter_mr_lambda})")
axes[1].set_title("Fourier-space")
axes[1].set_xlabel("Frequency")

axes[0].grid()
axes[1].grid()
axes[1].legend()
fig.tight_layout()

fig, ax = plt.subplots(figsize=cm2inch([24, 10]))
ax.plot(np.squeeze(vol_sl[..., 172]), label=f"FBP-{filter_name}")
ax.plot(np.squeeze(vol_mr[..., 172]), label="FBP-MR")
ax.plot(np.squeeze(vol_mr_reg[..., 172]), label=f"FBP-MR-Smooth(W:{filter_mr_lambda})")
ax.plot(np.squeeze(expected_ph[..., 172]), label="Phantom")
ax.legend()
ax.grid()
fig.tight_layout()

plt.show(block=False)
