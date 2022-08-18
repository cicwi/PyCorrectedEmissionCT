#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-channel TV regularization example.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import numpy.random

import matplotlib.pyplot as plt

import skimage as sk
import skimage.data

import time as tm

import corrct as cct

from numpy.typing import NDArray

# Noise parameters
gauss_stddev = None
photon_num = 1e2

# Reconstruction parameters
lambda_tv = 2.5e-3
iterations = 125


c_s = tm.time()
# Loading the image
img_orig: NDArray = sk.data.astronaut()
c_load = tm.time()
print("Loaded img shape: %s, dtype: %s, in %g seconds." % (str(img_orig.shape), str(img_orig.dtype), (c_load - c_s)))

# Transposing the image to have the multi-channel dimension as the slowest (axis=0 in Python)
img_orig = img_orig.transpose([2, 0, 1])
# Rescaling the image to be within the interval [0, 1]
img_orig = img_orig.astype(np.float32) / 255

img_noise = img_orig.copy()
img_variance = np.zeros_like(img_orig)
if photon_num is not None:
    # This adds Poisson noise
    img_noise = np.random.poisson(img_noise * photon_num) / photon_num
    img_variance = img_variance + img_noise / photon_num
if gauss_stddev is not None:
    # This adds gaussian noise
    img_noise += np.random.normal(size=img_orig.shape, scale=gauss_stddev)
    img_variance = img_variance + gauss_stddev**2

c_noise = tm.time()
print("Added noise in %g seconds." % (c_noise - c_load))

img_weights = cct.utils_proc.compute_variance_weigth(img_variance, normalized=True)

# Data fitting term: weighted least-squares, based on the standard deviation of the noise. This is optional.
data_term = cct.data_terms.DataFidelity_wl2(img_weights)

# "Projection" operator. In this case, it's just a place holder, because we are denoising an image.
A = cct.operators.TransformIdentity(img_orig.shape)

c_stddev = tm.time()
print("Prepared pixel confidence in %g seconds." % (c_stddev - c_noise))

# Single channel TV regularizer
reg_tv_s = cct.regularizers.Regularizer_TV2D(lambda_tv)
# Solver, which in this case is the PDHG method from Chambolle and Pock
solver_tv_s = cct.solvers.PDHG(data_term=data_term, verbose=True, regularizer=reg_tv_s)
# We now run the solver on the noisy image
(img_tvs, _) = solver_tv_s(A, img_noise, x0=img_noise, iterations=iterations, lower_limit=0.0)

# Multi channel TV regularizer - aka TNV
reg_tv_m = cct.regularizers.Regularizer_TNV(lambda_tv)
# Solver, which in this case is the PDHG method from Chambolle and Pock
solver_tv_m = cct.solvers.PDHG(data_term=data_term, verbose=True, regularizer=reg_tv_m)
# We now run the solver on the noisy image
(img_tvm, _) = solver_tv_m(A, img_noise, x0=img_noise, iterations=iterations, lower_limit=0.0)

imgs = np.array([img_orig, img_noise, img_tvs, img_tvm]).clip(0, 1).transpose([0, 2, 3, 1])
labs = ["Original", "Noisy", f"Single-channel TV, weight: {lambda_tv:.5}", f"Multi-channel TV, weight: {lambda_tv:.5}"]

# Plotting the result
f, axs = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False, figsize=[7, 7])
for ax, (im, lb) in zip(axs.flatten(), zip(imgs, labs)):
    ax.imshow(im)
    ax.set_title(lb)
f.tight_layout()

# Comparing FRCs for each reconstruction
frcs = [np.array([])] * (imgs.shape[0] - 1)
for ii, im in enumerate(imgs[1:]):
    fc = [np.array([])] * 3
    for ii_c in range(3):
        fc[ii_c], T = cct.utils_proc.compute_frc(imgs[0][..., ii_c], im[..., ii_c], snrt=0.4142)
    frcs[ii] = np.mean(fc, axis=0)

f, ax = plt.subplots(1, 1, sharex=True, sharey=True)
for fc, lb in zip(frcs, labs[1:]):
    ax.plot(np.squeeze(fc), label=lb)
ax.plot(np.squeeze(T), label="T 1/2 bit")
ax.legend()
ax.grid()
ax.set_title("Fourier Ring Correlation")
f.tight_layout()

plt.show(block=False)
