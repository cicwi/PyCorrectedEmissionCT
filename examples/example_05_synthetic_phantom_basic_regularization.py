# -*- coding: utf-8 -*-
"""
This example reproduces the l2-ball dead-zone reconstruction of the phantom used in:

- N. Viganò and V. A. Solé, “Physically corrected forward operators for
induced emission tomography: a simulation study,” Meas. Sci. Technol., no.
Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.

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


vol_shape = [256, 256, 3]
data_type = np.float32

ph_or = np.squeeze(phantom.modified_shepp_logan(vol_shape).astype(data_type))
ph_or = ph_or[:, :, 1]

(ph, vol_att_in, vol_att_out) = cct.testing.phantom_assign_concentration(ph_or)
(sino, angles, expected_ph, background_avg) = cct.testing.create_sino(
    ph, 30, psf=None, add_poisson=True, dwell_time_s=1e-2, background_avg=1e-2, background_std=1e-4
)

bckgnd_weight = np.sqrt(background_avg / (vol_shape[0] * np.sqrt(2)))

num_iterations = 200
reg_weight = 1 / 5
lower_limit = None
vol_mask = cct.processing.circular_mask(ph_or.shape)

sino_substract = sino - background_avg

sino_variance = cct.processing.compute_variance_poisson(sino)
sino_weights = cct.processing.compute_variance_weight(sino_variance)

reg_tv = cct.regularizers.Regularizer_TV2D(reg_weight)
reg_tv_hub = cct.regularizers.Regularizer_HubTV2D(reg_weight, huber_size=0.05)
reg_lap = cct.regularizers.Regularizer_lap2D(reg_weight)
reg_smooth = cct.regularizers.Regularizer_smooth2D(reg_weight)
reg_l1med = cct.regularizers.Regularizer_l1med(reg_weight)
reg_l2med = cct.regularizers.Regularizer_l2med(reg_weight, filt_size=5)
reg_dwl = cct.regularizers.Regularizer_l1dwl(reg_weight, "haar", 4)
reg_swl = cct.regularizers.Regularizer_l1swl(reg_weight, "bior4.4", 3)

lowlim_l2 = cct.regularizers.Constraint_LowerLimit(0, norm=cct.data_terms.DataFidelity_l2())
lowlim_l2w = cct.regularizers.Constraint_LowerLimit(0, norm=cct.data_terms.DataFidelity_wl2(1 / bckgnd_weight))
lowlim_l2b = cct.regularizers.Constraint_LowerLimit(0, norm=cct.data_terms.DataFidelity_l2b(bckgnd_weight))
lowlim_l1 = cct.regularizers.Constraint_LowerLimit(0, norm=cct.data_terms.DataFidelity_l1())
lowlim_l1b = cct.regularizers.Constraint_LowerLimit(0, norm=cct.data_terms.DataFidelity_l1b(bckgnd_weight))
lowlim_hub = cct.regularizers.Constraint_LowerLimit(0, norm=cct.data_terms.DataFidelity_Huber(bckgnd_weight))

reg_1 = reg_2 = reg_3 = reg_4 = [lowlim_l2w, reg_tv]

data_term_ls = cct.data_terms.DataFidelity_l2()
data_term_l1 = cct.data_terms.DataFidelity_l1()
data_term_kl = cct.data_terms.DataFidelity_KL()
data_term_kl_bck = cct.data_terms.DataFidelity_KL(background=background_avg)

data_term_lsw = cct.data_terms.DataFidelity_wl2(sino_weights)
data_term_lsb = cct.data_terms.DataFidelity_l2b(sino_variance)
data_term_l1b = cct.data_terms.DataFidelity_l1b(sino_variance)
data_term_hub = cct.data_terms.DataFidelity_Huber(sino_variance)

solver_1 = cct.solvers.PDHG(verbose=True, data_term=data_term_ls, regularizer=reg_1, tolerance=0, data_term_test=data_term_lsw)
solver_2 = cct.solvers.PDHG(
    verbose=True, data_term=data_term_lsb, regularizer=reg_2, tolerance=0, data_term_test=data_term_lsw
)

solver_3 = cct.solvers.PDHG(
    verbose=True, data_term=data_term_lsw, regularizer=reg_3, tolerance=0, data_term_test=data_term_lsw
)
solver_4 = cct.solvers.PDHG(
    verbose=True, data_term=data_term_kl_bck, regularizer=reg_4, tolerance=0, data_term_test=data_term_lsw
)

b_test_mask = np.zeros_like(sino)
num_test_pixels = int(np.ceil(sino.size * 0.05))
test_pixels = np.random.permutation(sino.size)
test_pixels = np.unravel_index(test_pixels[:num_test_pixels], sino.shape)
b_test_mask[test_pixels] = 1

with cct.projectors.ProjectorUncorrected(ph.shape, angles) as A:
    print("Reconstructing:")
    (rec_1, info_1) = solver_1(
        A, sino_substract, num_iterations, lower_limit=lower_limit, x_mask=vol_mask, b_test_mask=b_test_mask
    )
    print("- Phantom power: %g, noise power: %g" % cct.testing.compute_error_power(expected_ph, rec_1))
    (rec_2, info_2) = solver_2(
        A, sino_substract, num_iterations, lower_limit=lower_limit, x_mask=vol_mask, b_test_mask=b_test_mask
    )
    print("- Phantom power: %g, noise power: %g" % cct.testing.compute_error_power(expected_ph, rec_2))

    (rec_3, info_3) = solver_3(
        A, sino_substract, num_iterations, lower_limit=lower_limit, x_mask=vol_mask, b_test_mask=b_test_mask
    )
    print("- Phantom power: %g, noise power: %g" % cct.testing.compute_error_power(expected_ph, rec_3))
    (rec_4, info_4) = solver_4(A, sino, num_iterations, lower_limit=lower_limit, x_mask=vol_mask, b_test_mask=b_test_mask)
    print("- Phantom power: %g, noise power: %g" % cct.testing.compute_error_power(expected_ph, rec_4))

label_1 = solver_1.info().upper()
label_2 = solver_2.info().upper()
label_3 = solver_3.info().upper()
label_4 = solver_4.info().upper()

# Reconstructions
fig = plt.figure(figsize=cm2inch([36, 24]))
gs = fig.add_gridspec(8, 3)
ax_ph = fig.add_subplot(gs[:4, 0])
im_ph = ax_ph.imshow(expected_ph)
ax_ph.set_title("Phantom")
fig.colorbar(im_ph, ax=ax_ph)

ax_sino_clean = fig.add_subplot(gs[4, 0])
with cct.projectors.ProjectorUncorrected(ph_or.shape, angles) as p:
    sino_clean = p.fp(expected_ph)
im_sino_clean = ax_sino_clean.imshow(sino_clean)
ax_sino_clean.set_title("Clean sinogram")

ax_sino_noise = fig.add_subplot(gs[5, 0])
im_sino_noise = ax_sino_noise.imshow(sino_substract)
ax_sino_noise.set_title("Noisy sinogram")

ax_sino_lines = fig.add_subplot(gs[6:, 0])
im_sino_lines = ax_sino_lines.plot(sino_substract[9, :], label="Noisy")
im_sino_lines = ax_sino_lines.plot(sino_clean[9, :], label="Clean")
ax_sino_lines.set_title("Sinograms - angle: 10")
ax_sino_lines.grid()
ax_sino_lines.legend()

ax_ls = fig.add_subplot(gs[:4, 1], sharex=ax_ph, sharey=ax_ph)
im_ls = ax_ls.imshow(np.squeeze(rec_1))
ax_ls.set_title(label_1)
fig.colorbar(im_ls, ax=ax_ls)

ax_reg = fig.add_subplot(gs[:4, 2], sharex=ax_ph, sharey=ax_ph)
im_reg = ax_reg.imshow(np.squeeze(rec_2))
ax_reg.set_title(label_2)
fig.colorbar(im_reg, ax=ax_reg)

ax_ls = fig.add_subplot(gs[4:, 1], sharex=ax_ph, sharey=ax_ph)
im_ls = ax_ls.imshow(np.squeeze(rec_3))
ax_ls.set_title(label_3)
fig.colorbar(im_ls, ax=ax_ls)

ax_reg = fig.add_subplot(gs[4:, 2], sharex=ax_ph, sharey=ax_ph)
im_reg = ax_reg.imshow(np.squeeze(rec_4))
ax_reg.set_title(label_4)
fig.colorbar(im_reg, ax=ax_reg)

fig.tight_layout()

fig, ax = plt.subplots()
ax.plot(np.squeeze(expected_ph[..., 172]), label="Phantom")
ax.plot(np.squeeze(rec_1[..., 172]), label=label_1)
ax.plot(np.squeeze(rec_2[..., 172]), label=label_2)
ax.plot(np.squeeze(rec_3[..., 172]), label=label_3)
ax.plot(np.squeeze(rec_4[..., 172]), label=label_4)
ax.legend()
ax.grid()
fig.tight_layout()

fig, ax = plt.subplots()
ax.semilogy(np.squeeze(info_1.residuals_rel), "C0", label=label_1)
ax.semilogy(np.squeeze(info_2.residuals_rel), "C1", label=label_2)
ax.semilogy(np.squeeze(info_3.residuals_rel), "C2", label=label_3)
ax.semilogy(np.squeeze(info_4.residuals_rel), "C3", label=label_4)

ax.semilogy(np.squeeze(info_1.residuals_cv_rel), "C0-.", label=(label_1 + " - Cross-Validation"))
ax.semilogy(np.squeeze(info_2.residuals_cv_rel), "C1-.", label=(label_2 + " - Cross-Validation"))
ax.semilogy(np.squeeze(info_3.residuals_cv_rel), "C2-.", label=(label_3 + " - Cross-Validation"))
ax.semilogy(np.squeeze(info_4.residuals_cv_rel), "C3-.", label=(label_4 + " - Cross-Validation"))

ax.legend()
ax.grid()
fig.tight_layout()

print(np.std((expected_ph - rec_1) / (expected_ph + (expected_ph == 0))))
print(np.std((expected_ph - rec_2) / (expected_ph + (expected_ph == 0))))
print(np.std((expected_ph - rec_3) / (expected_ph + (expected_ph == 0))))
print(np.std((expected_ph - rec_4) / (expected_ph + (expected_ph == 0))))

# Comparing FRCs for each reconstruction
labels = [solver_1.info().upper(), solver_2.info().upper(), solver_3.info().upper(), solver_3.info().upper()]
vols = [rec_1, rec_2, rec_3, rec_4]

cct.processing.post.plot_frcs([(expected_ph, rec) for rec in vols], labels=labels, snrt=0.4142)

plt.show(block=False)
