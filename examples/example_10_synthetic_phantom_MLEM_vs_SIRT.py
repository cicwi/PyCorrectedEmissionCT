"""
This example compares the MLEM solver against SIRT and weighted least-squares
reconstructions (implemented with the PDHG algorithm) of the Shepp-Logan phantom.

@author: Jérome Lesaint, ESRF - The European Synchrotron, Grenoble, France
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
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
# Create sino with no background noise.
(sino, angles, expected_ph, background_avg) = cct.testing.create_sino(
    ph, 30, psf=None, add_poisson=True, dwell_time_s=1e-2, background_avg=1e-2, background_std=1e-4
)

num_iterations = 100
lower_limit = 0
vol_mask = cct.processing.circular_mask(ph_or.shape)

sino_variance = cct.processing.compute_variance_poisson(sino)
sino_weights = cct.processing.compute_variance_weight(sino_variance)

data_term_ls = cct.data_terms.DataFidelity_l2(background=background_avg)
data_term_lsw = cct.data_terms.DataFidelity_wl2(sino_weights, background=background_avg)
data_term_kl = cct.data_terms.DataFidelity_KL(background=background_avg)

with cct.projectors.ProjectorUncorrected(ph.shape, angles) as A:
    solver_pdhg = cct.solvers.PDHG(verbose=True, data_term=data_term_kl)
    rec_pdhg, _ = solver_pdhg(A, sino, num_iterations, x_mask=vol_mask)

    solver_sirt = cct.solvers.SIRT(verbose=True, data_term=data_term_ls)
    rec_sirt, _ = solver_sirt(A, sino, num_iterations, x_mask=vol_mask)

    solver_mlem = cct.solvers.MLEM(verbose=True, data_term=data_term_kl)
    rec_mlem, _ = solver_mlem(A, sino, num_iterations, x_mask=vol_mask)


# Reconstructions
fig = plt.figure(figsize=cm2inch([36, 24]))
gs = fig.add_gridspec(8, 3)
ax_ph = fig.add_subplot(gs[:4, 0])
im_ph = ax_ph.imshow(expected_ph, vmin=0.0, vmax=3.0)
ax_ph.set_title("Phantom")
fig.colorbar(im_ph, ax=ax_ph)

ax_sino_clean = fig.add_subplot(gs[4, 0])
with cct.projectors.ProjectorUncorrected(ph_or.shape, angles) as p:
    sino_clean = p.fp(expected_ph)
im_sino_clean = ax_sino_clean.imshow(sino_clean)
ax_sino_clean.set_title("Clean sinogram")

ax_sino_noise = fig.add_subplot(gs[5, 0])
im_sino_noise = ax_sino_noise.imshow(sino - background_avg)
ax_sino_noise.set_title("Noisy sinogram")

ax_sino_lines = fig.add_subplot(gs[6:, 0])
im_sino_lines = ax_sino_lines.plot(sino[9, :] - background_avg, label="Noisy")
im_sino_lines = ax_sino_lines.plot(sino_clean[9, :], label="Clean")
ax_sino_lines.set_title("Sinograms - angle: 10")
ax_sino_lines.legend()
ax_sino_lines.grid()

ax_wls_l = fig.add_subplot(gs[:4, 1], sharex=ax_ph, sharey=ax_ph)
im_wls_l = ax_wls_l.imshow(np.squeeze(rec_pdhg), vmin=0.0, vmax=3.0)
ax_wls_l.set_title(solver_pdhg.info().upper())
fig.colorbar(im_wls_l, ax=ax_wls_l)

ax_sirt = fig.add_subplot(gs[4:, 1], sharex=ax_ph, sharey=ax_ph)
im_sirt = ax_sirt.imshow(np.squeeze(rec_sirt), vmin=0.0, vmax=3.0)
ax_sirt.set_title(solver_sirt.info().upper())
fig.colorbar(im_sirt, ax=ax_sirt)

ax_mlem = fig.add_subplot(gs[:4, 2], sharex=ax_ph, sharey=ax_ph)
im_mlem = ax_mlem.imshow(np.squeeze(rec_mlem), vmin=0.0, vmax=3.0)
ax_mlem.set_title(solver_mlem.info().upper())
fig.colorbar(im_mlem, ax=ax_mlem)

axs = fig.add_subplot(gs[4:, 2])
axs.plot(np.squeeze(expected_ph[..., 172]), label="Phantom")
axs.plot(np.squeeze(rec_pdhg[..., 172]), label=solver_pdhg.info().upper())
axs.plot(np.squeeze(rec_sirt[..., 172]), label=solver_sirt.info().upper())
axs.plot(np.squeeze(rec_mlem[..., 172]), label=solver_mlem.info().upper())
axs.grid()
axs.legend()
fig.tight_layout()

# Comparing FRCs for each reconstruction
labels = [solver_pdhg.info().upper(), solver_sirt.info().upper(), solver_mlem.info().upper()]
vols = [rec_pdhg, rec_sirt, rec_mlem]
cct.processing.post.plot_frcs([(expected_ph, rec) for rec in vols], labels=labels, snrt=0.4142)

plt.show(block=False)
