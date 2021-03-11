# -*- coding: utf-8 -*-
"""
This example reproduces the l2 dead-zone reconstruction of the phantom used in:

- N. Viganò and V. A. Solé, “Physically corrected forward operators for
induced emission tomography: a simulation study,” Meas. Sci. Technol., no.
Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.

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


def cm2inch(x):
    return np.array(x) / 2.54


vol_shape = [256, 256, 3]
data_type = np.float32

ph_or = np.squeeze(phantom.modified_shepp_logan(vol_shape).astype(data_type))
ph_or = ph_or[:, :, 1]

(ph, vol_att_in, vol_att_out) = cct.utils_test.phantom_assign_concentration(ph_or)
(sino, angles, expected_ph, background_avg) = cct.utils_test.create_sino(
    ph, 30, psf=None, add_poisson=True, dwell_time_s=1e-2, background_avg=1e-2, background_std=1e-4)

bckgnd_weight = np.sqrt(background_avg / (vol_shape[0] * np.sqrt(2)))

precond = True
num_iterations = 200
lower_limit = None
vol_mask = corrct.utils_proc.get_circular_mask(ph_or.shape)

sino_substract = sino - background_avg

sino_stddev = np.sqrt(np.abs(sino))
min_nonzero_stddev = np.min(sino_stddev[sino > 0])
sino_tolerances = np.fmax(sino_stddev, min_nonzero_stddev)
sino_weights = 1 / sino_tolerances

lowlim_l2 = corrct.solvers.Constraint_LowerLimit(0, norm=corrct.solvers.DataFidelity_l2())
lowlim_l2w = corrct.solvers.Constraint_LowerLimit(0, norm=corrct.solvers.DataFidelity_wl2(1 / bckgnd_weight))

data_term_ls = corrct.solvers.DataFidelity_l2()
data_term_lsw = corrct.solvers.DataFidelity_wl2(sino_weights)
data_term_lsb = corrct.solvers.DataFidelity_l2b(sino_tolerances)

with corrct.projectors.ProjectorUncorrected(ph.shape, angles) as A:
    solver_ls = corrct.solvers.CP(verbose=True, data_term=data_term_ls)
    rec_ls, _ = solver_ls(A, sino_substract, num_iterations, x_mask=vol_mask, precondition=precond, lower_limit=0)

    solver_wls = corrct.solvers.CP(verbose=True, data_term=data_term_lsw)
    rec_wls, _ = solver_wls(A, sino_substract, num_iterations, x_mask=vol_mask, precondition=precond, lower_limit=0)

    solver_lsb = corrct.solvers.CP(verbose=True, data_term=data_term_lsb)
    rec_lsb, _ = solver_lsb(A, sino_substract, num_iterations, x_mask=vol_mask, precondition=precond, lower_limit=0)

    solver_ls_l = corrct.solvers.CP(verbose=True, data_term=data_term_ls, regularizer=[lowlim_l2w])
    rec_ls_l, _ = solver_ls_l(A, sino_substract, num_iterations, x_mask=vol_mask, precondition=precond)

    solver_wls_l = corrct.solvers.CP(verbose=True, data_term=data_term_lsw, regularizer=[lowlim_l2w])
    rec_wls_l, _ = solver_wls_l(A, sino_substract, num_iterations, x_mask=vol_mask, precondition=precond)

    solver_lsb_l = corrct.solvers.CP(verbose=True, data_term=data_term_lsb, regularizer=[lowlim_l2w])
    rec_lsb_l, _ = solver_lsb_l(A, sino_substract, num_iterations, x_mask=vol_mask, precondition=precond)

# Reconstructions
f = plt.figure(figsize=cm2inch([48, 24]))
gs = f.add_gridspec(8, 4)
ax_ph = f.add_subplot(gs[:4, 0])
im_ph = ax_ph.imshow(expected_ph)
ax_ph.set_title('Phantom')
f.colorbar(im_ph, ax=ax_ph)

ax_sino_clean = f.add_subplot(gs[4, 0])
with corrct.projectors.ProjectorUncorrected(ph_or.shape, angles) as p:
    sino_clean = p.fp(expected_ph)
im_sino_clean = ax_sino_clean.imshow(sino_clean)
ax_sino_clean.set_title('Clean sinogram')

ax_sino_noise = f.add_subplot(gs[5, 0])
im_sino_noise = ax_sino_noise.imshow(sino_substract)
ax_sino_noise.set_title('Noisy sinogram')

ax_sino_lines = f.add_subplot(gs[6:, 0])
im_sino_lines = ax_sino_lines.plot(sino_substract[9, :], label='Noisy')
im_sino_lines = ax_sino_lines.plot(sino_clean[9, :], label='Clean')
ax_sino_lines.set_title('Sinograms - angle: 10')
ax_sino_lines.legend()

ax_ls = f.add_subplot(gs[:4, 1], sharex=ax_ph, sharey=ax_ph)
im_ls = ax_ls.imshow(np.squeeze(rec_ls))
ax_ls.set_title(solver_ls.info().upper())
f.colorbar(im_ls, ax=ax_ls)

ax_reg = f.add_subplot(gs[:4, 2], sharex=ax_ph, sharey=ax_ph)
im_reg = ax_reg.imshow(np.squeeze(rec_wls))
ax_reg.set_title(solver_wls.info().upper())
f.colorbar(im_reg, ax=ax_reg)

ax_reg = f.add_subplot(gs[:4, 3], sharex=ax_ph, sharey=ax_ph)
im_reg = ax_reg.imshow(np.squeeze(rec_lsb))
ax_reg.set_title(solver_lsb.info().upper())
f.colorbar(im_reg, ax=ax_reg)

ax_ls = f.add_subplot(gs[4:, 1], sharex=ax_ph, sharey=ax_ph)
im_ls = ax_ls.imshow(np.squeeze(rec_ls_l))
ax_ls.set_title(solver_ls_l.info().upper())
f.colorbar(im_ls, ax=ax_ls)

ax_reg = f.add_subplot(gs[4:, 2], sharex=ax_ph, sharey=ax_ph)
im_reg = ax_reg.imshow(np.squeeze(rec_wls_l))
ax_reg.set_title(solver_wls_l.info().upper())
f.colorbar(im_reg, ax=ax_reg)

ax_reg = f.add_subplot(gs[4:, 3], sharex=ax_ph, sharey=ax_ph)
im_reg = ax_reg.imshow(np.squeeze(rec_lsb_l))
ax_reg.set_title(solver_lsb_l.info().upper())
f.colorbar(im_reg, ax=ax_reg)

f.tight_layout()

(f_prof, axs) = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].plot(np.squeeze(expected_ph[..., 172]), label='Phantom')
axs[0].plot(np.squeeze(rec_ls[..., 172]), label=solver_ls.info().upper())
axs[0].plot(np.squeeze(rec_wls[..., 172]), label=solver_wls.info().upper())
axs[0].plot(np.squeeze(rec_lsb[..., 172]), label=solver_lsb.info().upper())
axs[0].legend()
axs[0].grid()
axs[1].plot(np.squeeze(expected_ph[..., 172]), label='Phantom')
axs[1].plot(np.squeeze(rec_ls_l[..., 172]), label=solver_ls_l.info().upper())
axs[1].plot(np.squeeze(rec_wls_l[..., 172]), label=solver_wls_l.info().upper())
axs[1].plot(np.squeeze(rec_lsb_l[..., 172]), label=solver_lsb_l.info().upper())
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show(block=False)
