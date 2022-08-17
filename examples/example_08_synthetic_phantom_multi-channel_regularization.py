#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-channel tomography TV regularization example.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import numpy.random

import matplotlib.pyplot as plt

import corrct as cct
import corrct.utils_test as cct_test

from typing import Tuple
from numpy.typing import ArrayLike

try:
    import phantom
except ImportError:
    cct_test.download_phantom()
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


data_type = np.float32

# Create the phantom shape
ph_or = np.squeeze(phantom.modified_shepp_logan([256, 256, 3]).astype(data_type))
ph_or = ph_or[:, :, 1]

dwell_time_s = 1e-2
bckgnd_avg = 1e-4  # Background concentrations averages
beam_energy_keV = 20.0

# Simulate the XRF-CT acquisition data (sinogram)
(phantoms, vol_att_in, vols_att_out) = cct_test.phantom_assign_concentration_multi(ph_or, in_energy_keV=beam_energy_keV)

num_vols = len(phantoms) + 1
sinogram = [np.array([])] * num_vols
expected_ph = [np.array([])] * num_vols
background_avg = np.zeros((num_vols,))

# Compute Fluorescence and Compton projections
for ii, (ph, vol_att_out) in enumerate(zip(phantoms, vols_att_out)):
    (sinogram[ii], angles, expected_ph[ii], background_avg[ii]) = cct_test.create_sino(
        ph, 30, add_poisson=True, dwell_time_s=dwell_time_s, background_avg=bckgnd_avg
    )

# Compute attenuation projections
(sinogram_t, flat, angles, expected_ph[-1]) = cct_test.create_sino_transmission(
    vol_att_in, 30, add_poisson=True, dwell_time_s=dwell_time_s
)
sinogram[-1] = cct.utils_proc.apply_minus_log(cct.utils_proc.apply_flat_field(sinogram_t, flat))

sinogram = np.array(sinogram)
expected_ph = np.array(expected_ph)

iterations = 200
lower_limit = 0.0
lambda_tv = 10.0
vol_mask = cct.utils_proc.get_circular_mask(ph_or.shape)

# Subtract background from sinogram
sino_substr = sinogram - background_avg[:, None, None]

# f, axs = plt.subplots(sino_substr.shape[0], 1, sharex=True, sharey=True)
# for ax, sino in zip(axs, sino_substr):
#     ax.imshow(sino)
# f.tight_layout()
# plt.show(block=False)

sino_substr = np.fmax(sino_substr, 0)

norm_signals = np.mean(sino_substr, axis=(-2, -1), keepdims=True)
renorm_transm = norm_signals[-1] / np.mean(norm_signals[:2])

sino_substr[-1] /= renorm_transm

sino_variances_poisson = cct.utils_proc.compute_variance_poisson(sinogram[:-1, ...])
sino_variances_gauss = cct.utils_proc.compute_variance_transmission(sinogram_t, flat)

sino_weights_poisson = np.stack(
    [cct.utils_proc.compute_variance_weigth(v, normalized=True, semilog=True) for v in sino_variances_poisson], axis=0
)
sino_weights_transmission = cct.utils_proc.compute_variance_weigth(sino_variances_gauss[None, ...], normalized=True)
sino_weights = np.concatenate((sino_weights_poisson, sino_weights_transmission), axis=0)

# f, axs = plt.subplots(sino_weights.shape[0], 1, sharex=True, sharey=True)
# for ax, sino in zip(axs, sino_weights):
#     ax.imshow(sino)
# f.tight_layout()
# plt.show(block=False)

# Data fitting term: weighted least-squares, based on the standard deviation of the noise.
data_term_lsw = cct.data_terms.DataFidelity_wl2(sino_weights)

with cct.projectors.ProjectorUncorrected([*ph_or.shape, num_vols], angles) as A:
    # Weighted least squares
    solver_wls = cct.solvers.PDHG(verbose=True, data_term=data_term_lsw)
    rec_wls, _ = solver_wls(A, sino_substr, iterations, x_mask=vol_mask, lower_limit=lower_limit)

    # Single channel TV regularizer
    reg_tv = cct.regularizers.Regularizer_TV2D(lambda_tv)
    # Solver, which in this case is the PDHG method from Chambolle and Pock
    solver_tv = cct.solvers.PDHG(data_term=data_term_lsw, verbose=True, regularizer=reg_tv)
    # We now run the solver on the noisy image
    (rec_tvs, _) = solver_tv(A, sino_substr, iterations, x_mask=vol_mask, lower_limit=lower_limit)

    # Multi channel TV regularizer - aka TNV
    reg_vtv = cct.regularizers.Regularizer_TNV(lambda_tv)
    # Solver, which in this case is the PDHG method from Chambolle and Pock
    solver_tnv = cct.solvers.PDHG(data_term=data_term_lsw, verbose=True, regularizer=reg_vtv)
    # We now run the solver on the noisy image
    (rec_tvm, _) = solver_tnv(A, sino_substr, iterations, x_mask=vol_mask, lower_limit=lower_limit)

rec_wls[-1] *= renorm_transm
rec_tvs[-1] *= renorm_transm
rec_tvm[-1] *= renorm_transm

f, axs = plt.subplots(3, num_vols + 1, sharex=True, sharey=True, figsize=cm2inch([27, 20]))
for ax, ph, vol_wls, vol_tvs, vol_tvm in zip(axs, expected_ph, rec_wls, rec_tvs, rec_tvm):
    ax[0].imshow(ph)
    ax[1].imshow(vol_wls)
    ax[2].imshow(vol_tvs)
    ax[3].imshow(vol_tvm)
axs[0, 0].set_title("Phantom")
axs[0, 1].set_title(solver_wls.info())
axs[0, 2].set_title(solver_tv.info())
axs[0, 3].set_title(solver_tnv.info())
axs[0, 0].set_ylabel("Ca")
axs[1, 0].set_ylabel("Fe")
axs[2, 0].set_ylabel("Attenuation")
f.tight_layout()

# Comparing FRCs for each reconstruction
frcs = [np.array([])] * 3
for ii, rec in enumerate([rec_wls[0], rec_tvs[0], rec_tvm[0]]):
    frcs[ii], T = cct.utils_proc.compute_frc(expected_ph[0], rec, snrt=0.4142)

f, ax = plt.subplots(1, 1, sharex=True, sharey=True)
ax.plot(np.squeeze(frcs[0]), label=solver_wls.info().upper())
ax.plot(np.squeeze(frcs[1]), label=solver_tv.info().upper())
ax.plot(np.squeeze(frcs[2]), label=solver_tnv.info().upper())
ax.plot(np.squeeze(T), label="T 1/2 bit")
ax.legend()
ax.grid()
ax.set_title("FRCs for Ca-Ka")
f.tight_layout()

plt.show(block=False)
