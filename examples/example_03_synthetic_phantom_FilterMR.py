# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:01:10 2020

@author: VIGANO
"""


import numpy as np
from numpy import random as rnd

import matplotlib.pyplot as plt

import corrct

try:
    import phantom
except ImportError:
    phantom_url = 'https://raw.githubusercontent.com/nbarbey/TomograPy/master/tomograpy/phantom.py'
    phantom_path = './phantom.py'

    print("""This example uses the phantom definition from the package Tomograpy,
            developed by Nicolas Barbey. The needed module will be downloaded from: %s""" % phantom_url)

    import urllib
    urllib.request.urlretrieve(phantom_url, phantom_path)

    with open(phantom_path, 'r') as f:
        file_content = f.read()
    with open(phantom_path, 'w') as f:
        f.write(file_content.replace('xrange', 'range'))

    import phantom


def create_sino(
        ph, num_angles, start_angle=0, end_angle=180, add_poisson=False,
        background_std=None):
    print('Creating Sino with %d angles' % num_angles)
    angles_deg = np.linspace(start_angle, end_angle, num_angles, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)
    print(angles_deg)

    with corrct.projectors.ProjectorUncorrected(ph.shape, angles_rad) as p:
        sino = p.fp(ph)

    # Adding noise
    sino_noise = sino
    if background_std is not None:
        sino_noise = sino_noise + np.abs(rnd.normal(0, background_std, sino.shape))
    if add_poisson:
        sino_noise = rnd.poisson(sino_noise)

    return (sino_noise, angles_rad)

vol_shape_xy = [256, 256]

ph = np.squeeze(phantom.modified_shepp_logan([*vol_shape_xy, 3]).astype(np.float32))
ph = ph[:, :, 1]

(sino, angles_rad) = create_sino(ph, 30, add_poisson=True, background_std=None)

print('Reconstructing sino:')
with corrct.projectors.ProjectorUncorrected(vol_shape_xy, angles_rad) as p:
    print('- Filter: "Shepp-Logan"')
    vol_sl = p.fbp(sino)
    print('- Filter: "MR"')
    filter_mr = corrct.projectors.FilterMR()
    vol_mr = p.fbp(sino, filter_mr)
    print('- Filter: "MR-smooth"')
    filter_mr_reg = corrct.projectors.FilterMR(lambda_smooth=5e0)
    vol_mr_reg = p.fbp(sino, filter_mr_reg)

    filt = filter_mr.compute_filter(sino, p)
    filt_reg = filter_mr_reg.compute_filter(sino, p)

(f, axes) = plt.subplots(2, 2, sharex=True, sharey=True)
axes[0, 0].imshow(ph)
axes[0, 0].set_title('Phantom')
axes[0, 1].imshow(vol_sl)
axes[0, 1].set_title('FBP')
axes[1, 0].imshow(vol_mr)
axes[1, 0].set_title('FBP-MR')
axes[1, 1].imshow(vol_mr_reg)
axes[1, 1].set_title('FBP-MR-smooth')
plt.show()

(f, axes) = plt.subplots(1, 2)
axes[0].plot(filt)
axes[0].plot(filt_reg)
axes[1].plot(np.abs(np.fft.fft(np.fft.ifftshift(filt))))
axes[1].plot(np.abs(np.fft.fft(np.fft.ifftshift(filt_reg))))
plt.show()


