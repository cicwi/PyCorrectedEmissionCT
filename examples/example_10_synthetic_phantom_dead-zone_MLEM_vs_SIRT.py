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
from numpy.typing import ArrayLike
from typing import Tuple

import matplotlib.pyplot as plt

import corrct as cct

try:
    import phantom
except ImportError:
    cct.testing.download_phantom()
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
lower_limit = 0
vol_mask = cct.processing.circular_mask(ph_or.shape)

sino_substract = sino - background_avg

# Extend detector to prevent numerical issues
fov_extension_factor = 0.05
det_shape_from_vol = sino_substract.shape[1]
nb_additional_pixels = int(det_shape_from_vol * fov_extension_factor)
new_det_shape = det_shape_from_vol + 2* nb_additional_pixels
extended_sino_substract = np.zeros([sino_substract.shape[0],new_det_shape],dtype=np.float32)
extended_sino_substract[:,nb_additional_pixels:nb_additional_pixels + det_shape_from_vol] = sino_substract




sino_variance = cct.processing.compute_variance_poisson(sino)
sino_weights = cct.processing.compute_variance_weight(sino_variance)

lowlim_l2 = cct.solvers.Constraint_LowerLimit(0, norm=cct.solvers.DataFidelity_l2())
lowlim_l2w = cct.solvers.Constraint_LowerLimit(0, norm=cct.solvers.DataFidelity_wl2(1 / bckgnd_weight))

data_term_ls = cct.solvers.DataFidelity_l2()
data_term_lsw = cct.solvers.DataFidelity_wl2(sino_weights)
data_term_lsb = cct.solvers.DataFidelity_l2b(sino_variance)


with cct.projectors.ProjectorUncorrected(ph.shape, angles) as A:
    
    solver_sirt = cct.solvers.SIRT(verbose=True)
    rec_sirt, _ = solver_sirt(A, sino_substract, num_iterations, x_mask=vol_mask)

    solver_mlem = cct.solvers.MLEM(verbose=True, data_term=data_term_ls)
    prj_geom = cct.models.ProjectionGeometry.get_default_parallel(geom_type="2d")
    det_shape_from_vol = A.projector_backend.vol_geom.shape_xyz[0]
    prj_geom.det_shape_vu = np.array(list([new_det_shape]), dtype=int)
    A.projector_backend.initialize_geometry(A.projector_backend.vol_geom,angles,prj_geom=prj_geom)
    rec_mlem, _ = solver_mlem(A, extended_sino_substract, num_iterations, x_mask=vol_mask)

# Reconstructions
f = plt.figure(figsize=cm2inch([24, 24]))
gs = f.add_gridspec(8, 2)
ax_ph = f.add_subplot(gs[:4, 0])
im_ph = ax_ph.imshow(expected_ph,vmin=0.,vmax=3.)
ax_ph.set_title("Phantom")
f.colorbar(im_ph, ax=ax_ph)

ax_sino_clean = f.add_subplot(gs[4, 0])
with cct.projectors.ProjectorUncorrected(ph_or.shape, angles) as p:
    sino_clean = p.fp(expected_ph)
im_sino_clean = ax_sino_clean.imshow(sino_clean)
ax_sino_clean.set_title("Clean sinogram")

ax_sino_noise = f.add_subplot(gs[5, 0])
im_sino_noise = ax_sino_noise.imshow(sino_substract)
ax_sino_noise.set_title("Noisy sinogram")

ax_sino_lines = f.add_subplot(gs[6:, 0])
im_sino_lines = ax_sino_lines.plot(sino_substract[9, :], label="Noisy")
im_sino_lines = ax_sino_lines.plot(sino_clean[9, :], label="Clean")
ax_sino_lines.set_title("Sinograms - angle: 10")
ax_sino_lines.legend()

ax_sirt = f.add_subplot(gs[:4, 1], sharex=ax_ph, sharey=ax_ph)
im_sirt = ax_sirt.imshow(np.squeeze(rec_sirt),vmin=0.,vmax=3.)
ax_sirt.set_title(solver_sirt.info().upper())
f.colorbar(im_sirt, ax=ax_sirt)


ax_mlem = f.add_subplot(gs[4:, 1], sharex=ax_ph, sharey=ax_ph)
im_mlem = ax_mlem.imshow(np.squeeze(rec_mlem),vmin=0.,vmax=3.)
ax_mlem.set_title(solver_mlem.info().upper())
f.colorbar(im_mlem, ax=ax_mlem)

f.tight_layout()




f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].plot(np.squeeze(expected_ph[..., 172]), label="Phantom")
axs[0].plot(np.squeeze(rec_ls[..., 172]), label=solver_ls.info().upper())
axs[0].plot(np.squeeze(rec_wls[..., 172]), label=solver_wls.info().upper())
axs[0].plot(np.squeeze(rec_lsb[..., 172]), label=solver_lsb.info().upper())
axs[0].legend()
axs[0].grid()
axs[1].plot(np.squeeze(expected_ph[..., 172]), label="Phantom")
axs[1].plot(np.squeeze(rec_ls_l[..., 172]), label=solver_ls_l.info().upper())
axs[1].plot(np.squeeze(rec_wls_l[..., 172]), label=solver_wls_l.info().upper())
axs[1].plot(np.squeeze(rec_lsb_l[..., 172]), label=solver_lsb_l.info().upper())
axs[1].legend()
axs[1].grid()
f.tight_layout()

# Comparing FRCs for each reconstruction
labels = [solver_ls.info().upper(), solver_wls.info().upper(), solver_lsb.info().upper()]
vols = [rec_ls, rec_wls, rec_lsb]
cct.processing.post.plot_frcs([(expected_ph, rec) for rec in vols], labels=labels, snrt=0.4142)

plt.show(block=False)
