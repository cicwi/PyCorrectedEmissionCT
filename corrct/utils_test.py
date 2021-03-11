# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:28:21 2020

This module provides useful functions for testing routines and functionalities.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from . import projectors


def roundup_to_pow2(x, p, data_type=np.int):
    """Rounds first argument to the power of 2 indicated by second argument.

    :param x: Number to round up
    :type x: int or float
    :param p: Power of 2
    :type p: int
    :param data_type: data type of the output
    :type data_type: `numpy.dtype`

    :return: Rounding up of input
    :rtype: dtype specified by data_type
    """
    return np.ceil(np.array(x) / (2 ** p)).astype(data_type) * (2 ** p)


def download_phantom():
    """Downloads the phantom generation module from Nicolas Barbey's repository
    on github.
    """
    phantom_url = "https://raw.githubusercontent.com/nbarbey/TomograPy/master/tomograpy/phantom.py"
    phantom_path = "./phantom.py"

    print(
        """This example uses the phantom definition from the package Tomograpy,
            developed by Nicolas Barbey. The needed module will be downloaded from: %s"""
        % phantom_url
    )

    import urllib

    urllib.request.urlretrieve(phantom_url, phantom_path)

    with open(phantom_path, "r") as f:
        file_content = f.read()
    with open(phantom_path, "w") as f:
        f.write(file_content.replace("xrange", "range"))


def phantom_assign_concentration(ph_or, data_type=np.float32):
    """Builds the phantom used in:
    - N. Viganò and V. A. Solé, “Physically corrected forward operators for
    induced emission tomography: a simulation study,” Meas. Sci. Technol., no.
    Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.

    :param ph_or: DESCRIPTION
    :type ph_or: TYPE
    :param data_type: DESCRIPTION, defaults to np.float32
    :type data_type: TYPE, optional

    :return: DESCRIPTION
    :rtype: TYPE
    """
    # ph_air = ph_or < 0.1
    ph_FeO = 0.5 < ph_or
    ph_CaO = np.logical_and(0.25 < ph_or, ph_or < 0.5)
    ph_CaC = np.logical_and(0.1 < ph_or, ph_or < 0.25)

    conv_mm_to_cm = 1e-1
    conv_um_to_mm = 1e-3
    voxel_size_um = 0.5
    voxel_size_cm = voxel_size_um * conv_um_to_mm * conv_mm_to_cm  # cm to micron
    print("Sample size: [%g %g] um" % (ph_or.shape[0] * voxel_size_um, ph_or.shape[1] * voxel_size_um))

    import xraylib

    xraylib.XRayInit()
    cp_fo = xraylib.GetCompoundDataNISTByName("Ferric Oxide")
    cp_co = xraylib.GetCompoundDataNISTByName("Calcium Oxide")
    cp_cc = xraylib.GetCompoundDataNISTByName("Calcium Carbonate")

    ca_an = xraylib.SymbolToAtomicNumber("Ca")
    ca_kal = xraylib.LineEnergy(ca_an, xraylib.KA_LINE)

    in_energy_keV = 20
    out_energy_keV = ca_kal

    ph_lin_att_in = (
        ph_FeO * xraylib.CS_Total_CP("Ferric Oxide", in_energy_keV) * cp_fo["density"]
        + ph_CaC * xraylib.CS_Total_CP("Calcium Carbonate", in_energy_keV) * cp_cc["density"]
        + ph_CaO * xraylib.CS_Total_CP("Calcium Oxide", in_energy_keV) * cp_co["density"]
    )

    ph_lin_att_out = (
        ph_FeO * xraylib.CS_Total_CP("Ferric Oxide", out_energy_keV) * cp_fo["density"]
        + ph_CaC * xraylib.CS_Total_CP("Calcium Carbonate", out_energy_keV) * cp_cc["density"]
        + ph_CaO * xraylib.CS_Total_CP("Calcium Oxide", out_energy_keV) * cp_co["density"]
    )

    vol_att_in = ph_lin_att_in * voxel_size_cm
    vol_att_out = ph_lin_att_out * voxel_size_cm

    ca_cs = xraylib.CS_FluorLine_Kissel(ca_an, xraylib.KA_LINE, in_energy_keV)  # fluo production for cm2/g
    ph_CaC_mass_fract = cp_cc["massFractions"][np.where(np.array(cp_cc["Elements"]) == ca_an)[0][0]]
    ph_CaO_mass_fract = cp_co["massFractions"][np.where(np.array(cp_co["Elements"]) == ca_an)[0][0]]

    ph = ph_CaC * ph_CaC_mass_fract * cp_cc["density"] + ph_CaO * ph_CaO_mass_fract * cp_co["density"]
    ph = ph * ca_cs * voxel_size_cm

    return (ph, vol_att_in, vol_att_out)


def create_sino(
    ph,
    num_angles,
    start_angle_deg=0,
    end_angle_deg=180,
    dwell_time_s=1,
    photon_flux=1e9,
    detectors_pos_rad=(np.pi / 2),
    vol_att_in=None,
    vol_att_out=None,
    psf=None,
    background_avg=None,
    background_std=None,
    add_poisson=False,
    readout_noise_std=None,
    data_type=np.float32,
):
    """Computes the sinogram from a given phantom

    :param ph: The phantom volume, with the expected average photon production per voxel per impinging photon
    :type ph: `numpy.array_like`
    :param num_angles: Number of angles
    :type num_angles: int
    :param start_angle_deg: Initial scan angle in degrees
    :type start_angle_deg: float
    :param end_angle_deg: Final scan angle in degrees
    :type end_angle_deg: float
    :param dwell_time_s: The acquisition time per sinogram point
    :type dwell_time_s: float
    :param photon_flux: The impinging photon flux per unit time (second)
    :type photon_flux: float
    :param detectors_pos_rad: Detector(s) positions in radians, with respect to incoming beam
    :type detectors_pos_rad: float or `numpy.array_like`
    :param vol_att_in: Attenuation volume for the incoming beam, defaults to None
    :type vol_att_in: `numpy.array_like`, optional
    :param vol_att_out: Attenuation volume for the outgoing beam, defaults to None
    :type vol_att_out: `numpy.array_like`, optional
    :param psf: Point spread function or probing beam profile, defaults to None
    :type psf: `numpy.array_like`, optional
    :param background_avg: Background average value, defaults to None
    :type background_avg: float, optional
    :param background_std: Background sigma, defaults to None
    :type background_std: float, optional
    :param add_poisson: Switch to turn on Poisson noise, defaults to False
    :type add_poisson: boolean, optional
    :param readout_noise_std: Read-out noise sigma, defaults to None
    :type readout_noise_std: float, optional
    :param data_type: Volumes' data type, defaults to np.float32
    :type data_type: `numpy.dtype`, optional

    :return: The sinogram (detecto readings), the angular positions, and the expected average phton production per voxel
    :rtype: tuple(`numpy.array_like`, `numpy.array_like`, `numpy.array_like`)
    """
    print("Creating Sino with %d angles" % num_angles)
    angles_deg = np.linspace(start_angle_deg, end_angle_deg, num_angles, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)
    print(angles_deg)

    num_photons = photon_flux * dwell_time_s
    detector_solidangle_sr = (2.4 / 2 / 16 / 2) ** 2

    if vol_att_in is None and vol_att_out is None:
        with projectors.ProjectorUncorrected(ph.shape, angles_rad) as p:
            sino = num_photons * detector_solidangle_sr * p.fp(ph)
    else:
        with projectors.ProjectorAttenuationXRF(
            ph.shape, angles_rad, att_in=vol_att_in, att_out=vol_att_out, angles_detectors_rad=detectors_pos_rad, psf=psf
        ) as p:
            sino = num_photons * detector_solidangle_sr * p.fp(ph)

    # Adding noise
    sino_noise = sino.copy()
    if background_avg is not None:
        if background_std is None:
            background_std = background_avg * 5e-2
        sino_noise += (
            num_photons * detector_solidangle_sr * np.abs(np.random.normal(background_avg, background_std, sino.shape))
        )

    background = np.mean((sino_noise - sino).flatten())

    if add_poisson:
        sino_noise = np.random.poisson(sino_noise).astype(np.float32)
    if readout_noise_std is not None:
        sino_noise += np.random.normal(0, readout_noise_std, sino.shape)

    return (sino_noise, angles_rad, ph * num_photons * detector_solidangle_sr, background)


def compute_error_power(expected_vol, computed_vol):
    """Computes the expected volume signal power, and computed volume error power.

    :param expected_vol: The expected volume
    :type expected_vol: `numpy.array_like`
    :param computed_vol: The computed volume
    :type computed_vol: `numpy.array_like`

    :return: The expected volume signal power, and the computed volume
    :rtype: tuple(float, float)
    """
    vol_power = np.sqrt(np.sum((expected_vol) ** 2) / expected_vol.size)
    error_power = np.sqrt(np.sum(np.abs(expected_vol - computed_vol) ** 2) / expected_vol.size)
    return (vol_power, error_power)
