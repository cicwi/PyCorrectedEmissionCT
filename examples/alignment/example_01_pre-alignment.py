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
ANGLES_START = 0
ANGLES_RANGE = 180
ANGLES_NUM = 41
angles_deg = np.linspace(ANGLES_START, ANGLES_START + ANGLES_RANGE, ANGLES_NUM, endpoint=True)
angles_rad = np.deg2rad(angles_deg)

THEO_ROT_AXIS = -1.25

# Randomized shift errors
SIGMA_ERROR = 0.25
LINEAR_ERROR = -0.05
EXPONENTIAL_ERROR = 7.5

random_drifts = SIGMA_ERROR * np.random.randn(ANGLES_NUM)
linear_drifts = LINEAR_ERROR * np.linspace(-(ANGLES_NUM - 1) / 2, (ANGLES_NUM - 1) / 2, ANGLES_NUM)
exponential_drifts = EXPONENTIAL_ERROR * np.exp(-np.linspace(0, 5, ANGLES_NUM))

theo_shifts = random_drifts + linear_drifts + exponential_drifts + THEO_ROT_AXIS
theo_shifts = np.around(theo_shifts, decimals=2)

# Create the phantom shape
ph = np.squeeze(phantom.modified_shepp_logan([256, 256, 3]).astype(data_type))
ph = ph[:, :, 1]

with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, theo_shifts) as A:
    data_theo = A(ph)

com_ph_yx = cct.processing.post.com(ph)
recenter = cct.alignment.RecenterVolume(cct.models.ProjectionGeometry.get_default_parallel(geom_type="2d"), angles_rad)

# Adding noise
NUM_PHOTONS = 1e1
BACKGROUND_AVG = 2e0
ADD_POISSON = True
data_noise, data_theo, background = cct.testing.add_noise(
    data_theo, num_photons=NUM_PHOTONS, add_poisson=ADD_POISSON, background_avg=BACKGROUND_AVG
)

data_test = data_noise - background

# Setting up the pre-alignment routine
align_pre = cct.alignment.DetectorShiftsPRE(data_test, angles_rad)

# Runnning pre-alignment
diffs_u_pre, cor = align_pre.fit_u()
shifts_u_pre = cor + diffs_u_pre

solver_opts = dict(lower_limit=0.0)
ITERATIONS = 100

solver = cct.solvers.SIRT()
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, theo_shifts) as A:
    rec_noise_truth, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, THEO_ROT_AXIS) as A:
    rec_noise_theocor, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, cor) as A:
    rec_noise_precor, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, shifts_u_pre) as A:
    rec_noise_pre, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)

# Recentering the reconstructions on the phantom's center-of-mass -> moving the shifts accordingly
theo_rot_axis = recenter.recenter_to(THEO_ROT_AXIS, rec_noise_theocor, com_ph_yx)
cor = recenter.recenter_to(cor, rec_noise_precor, com_ph_yx)
shifts_u_pre = recenter.recenter_to(shifts_u_pre, rec_noise_pre, com_ph_yx)

# Reconstructing again, with centered shifts
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, theo_rot_axis) as A:
    rec_noise_theocor, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, cor) as A:
    rec_noise_precor, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)
with cct.projectors.ProjectorUncorrected(ph.shape, angles_rad, shifts_u_pre) as A:
    rec_noise_pre, _ = solver(A, data_test, iterations=ITERATIONS, **solver_opts)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=[5, 2.5])
axs[0].imshow(data_theo)
axs[0].set_title("Theoretical data")
axs[1].imshow(data_test)
axs[1].set_title("Noisy data")
fig.tight_layout()

vmin = rec_noise_truth.min()
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
axs.plot(np.ones_like(theo_shifts) * THEO_ROT_AXIS, label="Center-of-rotation (theoretical)")
axs.plot(np.ones_like(theo_shifts) * cor, label="Center-of-rotation (computed)")
axs.plot(shifts_u_pre, label="Pre-alignment shifts")
axs.grid()
axs.legend()
fig.tight_layout()

vol_pairs = [(ph, rec) for rec in vols]

cct.processing.post.plot_frcs(vol_pairs, labels=labs)

plt.show(block=False)
