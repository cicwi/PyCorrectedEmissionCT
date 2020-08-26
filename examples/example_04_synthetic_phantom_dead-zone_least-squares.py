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

import matplotlib.pyplot as plt

import corrct as cct
import corrct.utils_test

try:
    import phantom
except ImportError:
    cct.utils_test.download_phantom()
    import phantom


vol_shape = [256, 256, 3]
data_type = np.float32

ph_or = np.squeeze(phantom.modified_shepp_logan(vol_shape).astype(data_type))
ph_or = ph_or[:, :, 1]

# psf = spsig.gaussian(11, 1)
psf = None

(ph, vol_att_in, vol_att_out) = cct.utils_test.phantom_assign_concentration(ph_or)
(sino, angles, expected_ph) = cct.utils_test.create_sino(
    ph, 120, psf=psf, add_poisson=True, dwell_time_s=1e-2)  # , vol_att_in=vol_att_in, vol_att_out=vol_att_out

num_iterations = 250
# reg_weight = 1e-2

data_term_ls = corrct.solvers.DataFidelity_l2()
# reg_tv = corrct.solvers.Regularizer_TV2D(reg_weight)

sino_stddev = np.sqrt(np.abs(sino))
min_nonzero_stddev = np.min(sino_stddev[sino > 0])
sino_weights = np.fmax(sino_stddev, min_nonzero_stddev)
data_term_lsb = corrct.solvers.DataFidelity_l2b(sino_weights)

solver_ls = corrct.solvers.CP(verbose=True, data_term=data_term_ls)  # , regularizer=reg_tv
solver_lsb = corrct.solvers.CP(verbose=True, data_term=data_term_lsb)  # , regularizer=reg_tv

# with corrct.projectors.AttenuationProjector(ph.shape, angles, att_in=vol_att_in, att_out=vol_att_out, psf=psf) as A:
with corrct.projectors.ProjectorUncorrected(ph.shape, angles) as A:
    print('Reconstructing:')
    (rec_cpls, _) = solver_ls(A, sino, num_iterations)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_cpls))
    (rec_cplsb, _) = solver_lsb(A, sino, num_iterations)
    print('- Phantom power: %g, noise power: %g' % cct.utils_test.compute_error_power(expected_ph, rec_cplsb))

(f, axes) = plt.subplots(2, 2)
im_ph = axes[0, 0].imshow(expected_ph)
axes[0, 0].set_title('Phantom')
f.colorbar(im_ph, ax=axes[0, 0])
axes[1, 0].imshow(sino)
axes[1, 0].set_title('Sinogram')

im_cpls = axes[0, 1].imshow(np.squeeze(rec_cpls))
axes[0, 1].set_title(solver_ls.upper() + '-' + data_term_ls.upper())
f.colorbar(im_cpls, ax=axes[0, 1])
im_cplsb = axes[1, 1].imshow(np.squeeze(rec_cplsb))
axes[1, 1].set_title(solver_lsb.upper() + '-' + data_term_lsb.upper())
f.colorbar(im_cplsb, ax=axes[1, 1])

plt.tight_layout()

(f_prof, ax) = plt.subplots()
ax.plot(np.squeeze(expected_ph[..., 172]), label='Phantom')
ax.plot(np.squeeze(rec_cpls[..., 172]), label=(solver_ls.upper() + '-' + data_term_ls.upper()))
ax.plot(np.squeeze(rec_cplsb[..., 172]), label=(solver_lsb.upper() + '-' + data_term_lsb.upper()))
ax.legend()
ax.grid()

plt.tight_layout()
plt.show(block=False)

print(np.std((expected_ph - rec_cpls) / (expected_ph + (expected_ph == 0))))
print(np.std((expected_ph - rec_cplsb) / (expected_ph + (expected_ph == 0))))
