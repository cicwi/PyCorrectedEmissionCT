#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provide utility functions for testing.

Created on Thu Jun  4 12:28:21 2020

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from . import projectors

try:
    from . import utils_phys

    __has_physics__ = True
except ImportError as exc:
    print("WARNING:", exc)
    print("Physics based phantom creation will not be available.")

    __has_physics__ = False


from typing import Union, Sequence, Optional, Tuple
from numpy.typing import ArrayLike, DTypeLike


def roundup_to_pow2(x: Union[int, float, ArrayLike], p: int, data_type: DTypeLike = int) -> Union[int, float, ArrayLike]:
    """Round first argument to the power of 2 indicated by second argument.

    Parameters
    ----------
    x : int | float | ArrayLike
        Number to round up.
    p : int
        Power of 2.
    data_type : DTypeLike, optional
        data type of the output. The default is int.

    Returns
    -------
    int | float | ArrayLike
        Rounding up of input.
    """
    return np.ceil(np.array(x) / (2 ** p)).astype(data_type) * (2 ** p)


def download_phantom():
    """Download the phantom generation module from Nicolas Barbey's repository on github."""
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


def phantom_assign_concentration(
    ph_or: ArrayLike, element: str = "Ca", em_line: str = "KA", in_energy_keV: float = 20.0
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Build an XRF phantom.

    The created phantom has been used in:
    - N. Viganò and V. A. Solé, “Physically corrected forward operators for
    induced emission tomography: a simulation study,” Meas. Sci. Technol., no.
    Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.

    Parameters
    ----------
    ph_or : ArrayLike
        The phases phantom map.
    element : str, optional
        Element symbol. The default is "Ca".
    em_line : str, optional
        Emission line. The default is "KA" (corresponding to K-alpha).
    in_energy_keV : float, optional
        Incoming beam energy in keV. The default is 20.0.

    Returns
    -------
    vol_fluo_yield : ArrayLike
        Voxel-wise fluorescence yields.
    vol_att_in : ArrayLike
        Voxel-wise attenuation at the incoming beam energy.
    vol_att_out : ArrayLike
        Voxel-wise attenuation at the emitted energy.
    """
    if not __has_physics__:
        raise RuntimeError("Physics module not available!")
    ph_air = ph_or < 0.1
    ph_FeO = 0.5 < ph_or
    ph_CaO = np.logical_and(0.25 < ph_or, ph_or < 0.5)
    ph_CaC = np.logical_and(0.1 < ph_or, ph_or < 0.25)

    conv_mm_to_cm = 1e-1
    conv_um_to_mm = 1e-3
    voxel_size_um = 0.5
    voxel_size_cm = voxel_size_um * conv_um_to_mm * conv_mm_to_cm  # cm to micron
    print("Sample size: [%g %g] um" % (ph_or.shape[0] * voxel_size_um, ph_or.shape[1] * voxel_size_um))

    phase_fractions = (ph_air, ph_FeO, ph_CaO, ph_CaC)
    phase_compound_names = ("Air, Dry (near sea level)", "Ferric Oxide", "Calcium Oxide", "Calcium Carbonate")

    phantom = utils_phys.VolumeMaterial(phase_fractions, phase_compound_names, voxel_size_cm)

    out_energy_keV, vol_fluo_yield = phantom.get_fluo_yield(element, in_energy_keV, em_line)

    vol_lin_att_in = phantom.get_attenuation(in_energy_keV)
    vol_lin_att_out = phantom.get_attenuation(out_energy_keV)

    return (vol_fluo_yield, vol_lin_att_in, vol_lin_att_out)


def create_sino(
    ph: ArrayLike,
    num_angles: int,
    start_angle_deg: float = 0,
    end_angle_deg: float = 180,
    dwell_time_s: float = 1,
    photon_flux: float = 1e9,
    detectors_pos_rad: Union[float, Sequence] = (np.pi / 2),
    vol_att_in: Optional[ArrayLike] = None,
    vol_att_out: Optional[ArrayLike] = None,
    psf: Optional[ArrayLike] = None,
    background_avg: Optional[float] = None,
    background_std: Optional[float] = None,
    add_poisson: bool = False,
    readout_noise_std: Optional[float] = None,
    data_type: DTypeLike = np.float32,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Compute the sinogram from a given phantom.

    Parameters
    ----------
    ph : ArrayLike
        The phantom volume, with the expected average photon production per voxel per impinging photon.
    num_angles : int
        The number of angles.
    start_angle_deg : float, optional
        Initial scan angle in degrees. The default is 0.
    end_angle_deg : float, optional
        Final scan angle in degrees. The default is 180.
    dwell_time_s : float, optional
        The acquisition time per sinogram point. The default is 1.
    photon_flux : float, optional
        The impinging photon flux per unit time (second). The default is 1e9.
    detectors_pos_rad : float | tuple | list | ArrayLike, optional
        Detector(s) positions in radians, with respect to incoming beam. The default is (np.pi / 2).
    vol_att_in : ArrayLike, optional
        Attenuation volume for the incoming beam. The default is None.
    vol_att_out : ArrayLike, optional
        Attenuation volume for the outgoing beam. The default is None.
    psf : ArrayLike, optional
        Point spread function or probing beam profile. The default is None.
    background_avg : float, optional
        Background average value. The default is None.
    background_std : float, optional
        Background standard deviation. The default is None.
    add_poisson : bool, optional
        Switch to turn on Poisson noise. The default is False.
    readout_noise_std : float, optional
        Read-out noise standard deviation. The default is None.
    data_type : numpy.dtype, optional
        Output datatype. The default is np.float32.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike, ArrayLike]
        The sinogram (detector readings), the angular positions, and the expected average phton production per voxel.
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


def compute_error_power(expected_vol: ArrayLike, computed_vol: ArrayLike) -> Tuple[float, float]:
    """Compute the expected volume signal power, and computed volume error power.

    Parameters
    ----------
    expected_vol : ArrayLike
        The expected volume.
    computed_vol : ArrayLike
        The computed volume.

    Returns
    -------
    Tuple[float, float]
        The expected volume signal power, and the computed volume.
    """
    vol_power = np.sqrt(np.sum((expected_vol) ** 2) / expected_vol.size)
    error_power = np.sqrt(np.sum(np.abs(expected_vol - computed_vol) ** 2) / expected_vol.size)
    return vol_power, error_power
