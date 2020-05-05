# -*- coding: utf-8 -*-
"""
Example tha reproduces a multi-detector version of the study case used in:
* N. Viganò and V. A. Solé, “Physically corrected forward operators for
induced emission tomography: a simulation study,” Meas. Sci. Technol., no.
Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands
"""

import numpy as np
from numpy import random as rnd

import matplotlib.pyplot as plt

import xraylib

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


def create_phantom(vol_shape, data_type=np.float32):
    print('Creating Phantom')
    ph_or = np.squeeze(phantom.modified_shepp_logan(vol_shape).astype(data_type))
    ph_or = ph_or[:, :, 1]

#    ph_air = ph_or < 0.1
    ph_FeO = 0.5 < ph_or
    ph_CaO = np.logical_and(0.25 < ph_or, ph_or < 0.5)
    ph_CaC = np.logical_and(0.1 < ph_or, ph_or < 0.25)

    conv_mm_to_cm = 1e-1
    conv_um_to_mm = 1e-3
    voxel_size_um = 0.5
    voxel_size_cm = voxel_size_um * conv_um_to_mm * conv_mm_to_cm  # cm to micron
    print('Sample size: [%g %g] um' % (vol_shape[0] * voxel_size_um, vol_shape[1] * voxel_size_um))

    xraylib.XRayInit()
    cp_fo = xraylib.GetCompoundDataNISTByName('Ferric Oxide')
    cp_co = xraylib.GetCompoundDataNISTByName('Calcium Oxide')
    cp_cc = xraylib.GetCompoundDataNISTByName('Calcium Carbonate')

    ca_an = xraylib.SymbolToAtomicNumber('Ca')
    ca_kal = xraylib.LineEnergy(ca_an, xraylib.KA_LINE)

    in_energy_keV = 20
    out_energy_keV = ca_kal

    ph_lin_att_in = ph_FeO * xraylib.CS_Total_CP('Ferric Oxide', in_energy_keV) * cp_fo['density'] \
        + ph_CaC * xraylib.CS_Total_CP('Calcium Carbonate', in_energy_keV) * cp_cc['density'] \
        + ph_CaO * xraylib.CS_Total_CP('Calcium Oxide', in_energy_keV) * cp_co['density']

    ph_lin_att_out = ph_FeO * xraylib.CS_Total_CP('Ferric Oxide', out_energy_keV) * cp_fo['density'] \
        + ph_CaC * xraylib.CS_Total_CP('Calcium Carbonate', out_energy_keV) * cp_cc['density'] \
        + ph_CaO * xraylib.CS_Total_CP('Calcium Oxide', out_energy_keV) * cp_co['density']

    vol_att_in = ph_lin_att_in * voxel_size_cm
    vol_att_out = ph_lin_att_out * voxel_size_cm

    ca_cs = xraylib.CS_FluorLine_Kissel(ca_an, xraylib.KA_LINE, in_energy_keV)  # fluo production for cm2/g
    ph_CaC_mass_fract = cp_cc['massFractions'][np.where(np.array(cp_cc['Elements']) == ca_an)[0][0]]
    ph_CaO_mass_fract = cp_co['massFractions'][np.where(np.array(cp_co['Elements']) == ca_an)[0][0]]

    ph = ph_CaC * ph_CaC_mass_fract * cp_cc['density'] + ph_CaO * ph_CaO_mass_fract * cp_co['density']
    ph = ph * ca_cs * voxel_size_cm

    return (ph, vol_att_in, vol_att_out)


def create_sino(
        ph, vol_att_in, vol_att_out, num_angles, det_angles, psf, add_poisson=False,
        background_std=None):
    print('Creating Sino with %d angles' % num_angles)
    angles = np.deg2rad(np.linspace(0, 180, num_angles, endpoint=False))
    print(np.rad2deg(angles))

    seconds = 1
    num_photons = 1e9 * seconds
    detector_angle_sr = (2.4 / 2 / 16 / 2) ** 2

    sino = num_photons * detector_angle_sr * corrct.create_sino(
            ph, angles, vol_att_in=vol_att_in, vol_att_out=vol_att_out,
            angles_detectors_rad=det_angles, psf=psf)

    # Adding noise
    sino_noise = sino
    if background_std is not None:
        sino_noise = sino_noise + np.abs(rnd.normal(0, background_std, sino.shape))
    if add_poisson:
        sino_noise = rnd.poisson(sino_noise)

    return (sino_noise, angles)


# psf = spsig.gaussian(11, 1)
psf = None

det_angles = [+np.pi/2, -np.pi/2]
(ph, vol_att_in, vol_att_out) = create_phantom([256, 256, 3])
(sino, angles) = create_sino(ph, vol_att_in, vol_att_out, 120, det_angles, psf)

apply_corrections = True
short_rec = False

if short_rec is True:
    num_sart_iterations = 1
    num_sirt_iterations = 25
    num_cp_iterations = 10
    num_cptv_iterations = 10
else:
    num_sart_iterations = 5
    num_sirt_iterations = 250
    num_cp_iterations = 100
    num_cptv_iterations = 250

renorm_factor = np.max(np.sum(sino, axis=-1)) / np.sqrt(np.sum(np.array(ph.shape) ** 2)) / 50

if apply_corrections:
    print('Reconstructing with SART w/ corrections')
    rec_sart = corrct.reconstruct(
        'SART', sino, angles, vol_att_in=vol_att_in, vol_att_out=vol_att_out,
        angles_detectors_rad=det_angles, psf=psf, lower_limit=0,
        data_term='l2', iterations=num_sart_iterations)
    print('- Phantom power: %g, noise power: %g' % (
        np.sqrt(np.sum((ph) ** 2) / ph.size), np.sqrt(np.sum((rec_sart - ph) ** 2) / ph.size)))

    print('Reconstructing with SIRT w/ corrections')
    rec_sirt = corrct.reconstruct(
        'SIRT', sino, angles, vol_att_in=vol_att_in, vol_att_out=vol_att_out,
        angles_detectors_rad=det_angles, psf=psf, lower_limit=0,
        data_term='l2', iterations=num_sirt_iterations)
    print('- Phantom power: %g, noise power: %g' % (
        np.sqrt(np.sum((ph) ** 2) / ph.size), np.sqrt(np.sum((rec_sirt - ph) ** 2) / ph.size)))

    print('Reconstructing with CP - using KL w/ corrections')
    rec_cpkl = corrct.reconstruct(
        'CP', sino / renorm_factor, angles, iterations=num_cp_iterations,
        data_term='kl', vol_att_in=vol_att_in, vol_att_out=vol_att_out,
        angles_detectors_rad=det_angles, psf=psf, lower_limit=0) * renorm_factor
    print('- Phantom power: %g, noise power: %g' % (
        np.sqrt(np.sum((ph) ** 2) / ph.size), np.sqrt(np.sum((rec_cpkl - ph) ** 2) / ph.size)))

    print('Reconstructing with CPTV - using KL w/ corrections')
    rec_cptvkl = corrct.reconstruct(
        'CPTV', sino / renorm_factor, angles, iterations=num_cptv_iterations,
        data_term='kl', vol_att_in=vol_att_in, vol_att_out=vol_att_out,
        angles_detectors_rad=det_angles, psf=psf, lower_limit=0, lambda_reg=2e-1) * renorm_factor
    print('- Phantom power: %g, noise power: %g' % (
        np.sqrt(np.sum((ph) ** 2) / ph.size), np.sqrt(np.sum((rec_cptvkl - ph) ** 2) / ph.size)))

else:
    print('Reconstructing with SART w/o corrections')
    rec_sart = corrct.reconstruct(
        'SART', sino, angles, lower_limit=0, data_term='l2', iterations=num_sart_iterations)
    print('- Phantom power: %g, noise power: %g' % (
        np.sqrt(np.sum((ph) ** 2) / ph.size), np.sqrt(np.sum((rec_sart - ph) ** 2) / ph.size)))

    print('Reconstructing with SIRT w/o corrections')
    rec_sirt = corrct.reconstruct(
        'SIRT', sino, angles, lower_limit=0, data_term='l2', iterations=num_sirt_iterations)
    print('- Phantom power: %g, noise power: %g' % (
        np.sqrt(np.sum((ph) ** 2) / ph.size), np.sqrt(np.sum((rec_sirt - ph) ** 2) / ph.size)))

    print('Reconstructing with CP - using KL w/o corrections')
    rec_cpkl = corrct.reconstruct(
        'CP', sino / renorm_factor, angles, iterations=num_cp_iterations,
        data_term='kl', lower_limit=0) * renorm_factor
    print('- Phantom power: %g, noise power: %g' % (
        np.sqrt(np.sum((ph) ** 2) / ph.size), np.sqrt(np.sum((rec_cpkl - ph) ** 2) / ph.size)))

    print('Reconstructing with CPTV - using KL w/o corrections')
    rec_cptvkl = corrct.reconstruct(
        'CPTV', sino / renorm_factor, angles, iterations=num_cptv_iterations,
        data_term='kl', lower_limit=0, lambda_reg=2e-1) * renorm_factor
    print('- Phantom power: %g, noise power: %g' % (
        np.sqrt(np.sum((ph) ** 2) / ph.size), np.sqrt(np.sum((rec_cptvkl - ph) ** 2) / ph.size)))

(f, axes) = plt.subplots(2, 3)
axes[0, 0].imshow(ph)
axes[0, 0].set_title('Phantom')
axes[1, 0].imshow(np.reshape(np.transpose(sino, [1, 0, 2]), [-1, ph.shape[0]]))
axes[1, 0].set_title('Sinogram')

axes[0, 1].imshow(rec_sart)
axes[0, 1].set_title('SART')
axes[0, 2].imshow(rec_sirt)
axes[0, 2].set_title('SIRT')
axes[1, 1].imshow(rec_cpkl)
axes[1, 1].set_title('CP-KL')
axes[1, 2].imshow(rec_cptvkl)
axes[1, 2].set_title('CP-KL-TV')

plt.show()
