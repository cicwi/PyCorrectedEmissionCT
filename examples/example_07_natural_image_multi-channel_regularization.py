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
lambda_tv_sc = 2.5e-3
lambda_tv_mc = 1.7e-3
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
reg_tv_s = cct.regularizers.Regularizer_TV2D(lambda_tv_sc)
# Solver, which in this case is the PDHG method from Chambolle and Pock
solver_tv_s = cct.solvers.PDHG(data_term=data_term, verbose=True, regularizer=reg_tv_s)
# We now run the solver on the noisy image
(img_tvs, _) = solver_tv_s(A, img_noise, x0=img_noise, iterations=iterations, lower_limit=0.0)

# Multi channel TV regularizer - aka TNV
reg_tv_m = cct.regularizers.Regularizer_TNV(lambda_tv_mc)
# Solver, which in this case is the PDHG method from Chambolle and Pock
solver_tv_m = cct.solvers.PDHG(data_term=data_term, verbose=True, regularizer=reg_tv_m)
# We now run the solver on the noisy image
(img_tvm, _) = solver_tv_m(A, img_noise, x0=img_noise, iterations=iterations, lower_limit=0.0)

# Plotting the result
f, axs = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False, figsize=[7, 7])
axs[0, 0].imshow(img_orig.transpose([1, 2, 0]))
axs[0, 0].set_title("Original")
axs[1, 0].imshow(img_noise.clip(0, 1).transpose([1, 2, 0]))
axs[1, 0].set_title("Noise")

axs[0, 1].imshow(img_tvs.clip(0, 1).transpose([1, 2, 0]))
axs[0, 1].set_title("Single-channel TV, weight: %f" % (lambda_tv_sc))
axs[1, 1].imshow(img_tvm.clip(0, 1).transpose([1, 2, 0]))
axs[1, 1].set_title("Multi-channel TV, weight: %f" % (lambda_tv_mc))

f.tight_layout()
plt.show(block=False)
