# -*- coding: utf-8 -*-
"""
Example demonstrating the use of pre-alignment routines.

@author: Nicola VIGANÒ, CEA-IRIG, Grenoble, France
"""

import numpy as np

import matplotlib.pyplot as plt

import corrct as cct

from typing import Sequence, Union
from numpy.typing import NDArray


try:
    import phantom
except ImportError:
    cct.testing.download_phantom()
    import phantom


def cm2inch(x: Union[float, Sequence[float], NDArray]) -> Sequence[float]:
    """Convert centimeters to inches.

    Parameters
    ----------
    x : float
        length in cm.

    Returns
    -------
    float
        length in inch.
    """
    return list(np.array(x) / 2.54)


data_type = np.float32

# Basic geometry parameters
angles_start = 0
angles_range = 180
angles_num = 41
angles_deg = np.linspace(angles_start, angles_start + angles_range, angles_num, endpoint=True)
angles_rad = np.deg2rad(angles_deg)

theo_rot_axis = -1.25

# Randomized shift errors
sigma_error = 0.25
linear_error = -0.05
exponential_error = 7.5

random_drifts = sigma_error * np.random.randn(len(angles_rad))
linear_drifts = linear_error * np.linspace(-(len(angles_rad) - 1) / 2, (len(angles_rad) - 1) / 2, len(angles_rad))
exponential_drifts = exponential_error * np.exp(-np.linspace(0, 5, len(angles_rad)))

theo_shifts = random_drifts + linear_drifts + exponential_drifts + theo_rot_axis
theo_shifts = np.around(theo_shifts, decimals=2)

# Create the phantom shape
ph = np.squeeze(phantom.modified_shepp_logan([256, 256, 3]).astype(data_type))
ph = ph[:, :, 1]

with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, theo_shifts) as A:
    data_theo = A(ph)

com_ph_yx = cct.processing.post.com(ph)

prj_geom = cct.models.ProjectionGeometry.get_default_parallel(geom_type="2d")
rot_prj_geom = prj_geom.rotate(angles_rad)


def recenter_rec(shifts_vu: Union[float, NDArray], rec: NDArray) -> NDArray:
    com_rec_yx = cct.processing.post.com(rec)
    shifts_vu_corrs = rot_prj_geom.project_displacement_to_detector(com_ph_yx - com_rec_yx)
    return np.around(shifts_vu + shifts_vu_corrs, decimals=2)


# Adding noise
num_photons = 1e1
background_avg = 2e0
add_poisson = True
data_noise, data_theo, background = cct.testing.add_noise(
    data_theo, num_photons=num_photons, add_poisson=add_poisson, background_avg=background_avg
)

data_test = data_noise - background

# Setting up the pre-alignment routine
align_pre = cct.alignment.DetectorShiftsPRE(data_test, angles_rad)

# Runnning pre-alignment
diffs_u_pre, cor = align_pre.fit_u()
shifts_u_pre = cor + diffs_u_pre

solver_opts = dict(lower_limit=0.0)
iterations = 100

solver = cct.solvers.SIRT()
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, theo_shifts) as A:
    rec_noise_truth, _ = solver(A, data_test, iterations=iterations, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, theo_rot_axis) as A:
    rec_noise_theocor, _ = solver(A, data_test, iterations=iterations, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, cor) as A:
    rec_noise_precor, _ = solver(A, data_test, iterations=iterations, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, shifts_u_pre) as A:
    rec_noise_pre, _ = solver(A, data_test, iterations=iterations, **solver_opts)

# Recentering the reconstructions on the phantom's center-of-mass -> moving the shifts accordingly
theo_rot_axis = recenter_rec(theo_rot_axis, rec_noise_theocor)
cor = recenter_rec(cor, rec_noise_precor)
shifts_u_pre = recenter_rec(shifts_u_pre, rec_noise_pre)

# Reconstructing again, with centered shifts
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, theo_rot_axis) as A:
    rec_noise_theocor, _ = solver(A, data_test, iterations=iterations, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, cor) as A:
    rec_noise_precor, _ = solver(A, data_test, iterations=iterations, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, shifts_u_pre) as A:
    rec_noise_pre, _ = solver(A, data_test, iterations=iterations, **solver_opts)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=[5, 2.5])
axs[0].imshow(data_theo)
axs[0].set_title("Theoretical data")
axs[1].imshow(data_test)
axs[1].set_title("Noisy data")
fig.tight_layout()

vmin = 0.0
vmax = rec_noise_truth.max()

vols = [rec_noise_truth, rec_noise_theocor, rec_noise_precor, rec_noise_pre]
labs = ["Ground truth", "Center-of-rotation (theoretical)", "Center-of-rotation (computed)", "Pre-alignment"]

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[6, 6.25])
for ax, (rec, lab) in zip(axs.flatten(), zip(vols, labs)):
    ax.imshow(rec)
    ax.set_title(lab)
fig.tight_layout()

fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=[5, 2.5])
axs.plot(theo_shifts, label="Ground truth")
axs.plot(np.ones_like(theo_shifts) * theo_rot_axis, label="Center-of-rotation (theoretical)")
axs.plot(np.ones_like(theo_shifts) * cor, label="Center-of-rotation (computed)")
axs.plot(shifts_u_pre, label="Pre-alignment shifts")
axs.grid()
axs.legend()
fig.tight_layout()

vol_pairs = [(ph, rec) for rec in vols]

cct.processing.post.plot_frcs(vol_pairs, labels=labs)

plt.show(block=False)
