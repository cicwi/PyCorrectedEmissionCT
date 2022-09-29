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
    from . import physics

    __has_physics__ = True
except ImportError as exc:
    print("WARNING:", exc)
    print("Physics based phantom creation will not be available.")

    __has_physics__ = False


from typing import List, Union, Sequence, Optional, Tuple
from numpy.typing import DTypeLike, NDArray


NDArrayFloat = NDArray[np.floating]


def roundup_to_pow2(x: Union[int, float, NDArrayFloat], p: int, dtype: DTypeLike = int) -> Union[int, float, NDArrayFloat]:
    """Round first argument to the power of 2 indicated by second argument.

    Parameters
    ----------
    x : int | float | NDArrayFloat
        Number to round up.
    p : int
        Power of 2.
    dtype : DTypeLike, optional
        data type of the output. The default is int.

    Returns
    -------
    int | float | NDArrayFloat
        Rounding up of input.
    """
    return np.ceil(np.array(x) / (2**p)).astype(dtype) * (2**p)


def download_phantom():
    """Download the phantom generation module from Nicolas Barbey's repository on github."""
    phantom_url = "https://raw.githubusercontent.com/nbarbey/TomograPy/master/tomograpy/phantom.py"
    phantom_path = "./phantom.py"

    print(
        """This example uses the phantom definition from the package Tomograpy,
            developed by Nicolas Barbey. The needed module will be downloaded from: %s"""
        % phantom_url
    )

    import urllib.request as urlreq

    urlreq.urlretrieve(phantom_url, phantom_path)

    with open(phantom_path, "r") as f:
        file_content = f.read()
    with open(phantom_path, "w") as f:
        f.write(file_content.replace("xrange", "range"))


def phantom_assign_concentration(
    ph_or: NDArrayFloat, element: str = "Ca", em_line: str = "KA", in_energy_keV: float = 20.0, voxel_size_um: float = 0.5
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Build an XRF phantom.

    The created phantom has been used in:
    - N. Viganò and V. A. Solé, “Physically corrected forward operators for
    induced emission tomography: a simulation study,” Meas. Sci. Technol., no.
    Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.

    Parameters
    ----------
    ph_or : NDArrayFloat
        The phases phantom map.
    element : str, optional
        Element symbol. The default is "Ca".
    em_line : str, optional
        Emission line. The default is "KA" (corresponding to K-alpha).
    in_energy_keV : float, optional
        Incoming beam energy in keV. The default is 20.0.

    Returns
    -------
    vol_fluo_yield : NDArrayFloat
        Voxel-wise fluorescence yields.
    vol_att_in : NDArrayFloat
        Voxel-wise attenuation at the incoming beam energy.
    vol_att_out : NDArrayFloat
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
    voxel_size_cm = voxel_size_um * conv_um_to_mm * conv_mm_to_cm  # cm to micron
    print("Sample size: [%g %g] um" % (ph_or.shape[0] * voxel_size_um, ph_or.shape[1] * voxel_size_um))

    phase_fractions = (ph_air, ph_FeO, ph_CaO, ph_CaC)
    phase_compound_names = ("Air, Dry (near sea level)", "Ferric Oxide", "Calcium Oxide", "Calcium Carbonate")

    phantom = physics.VolumeMaterial(phase_fractions, phase_compound_names, voxel_size_cm)

    out_energy_keV, vol_fluo_yield = phantom.get_fluo_yield(element, in_energy_keV, em_line)

    vol_lin_att_in = phantom.get_attenuation(in_energy_keV)
    vol_lin_att_out = phantom.get_attenuation(out_energy_keV)

    return (vol_fluo_yield, vol_lin_att_in, vol_lin_att_out)


def phantom_assign_concentration_multi(
    ph_or: NDArrayFloat,
    elements: Sequence[str] = ["Ca", "Fe"],
    em_lines: Union[str, Sequence[str]] = "KA",
    in_energy_keV: float = 20.0,
    detectors_pos_rad: Optional[float] = None,
) -> Tuple[List[NDArrayFloat], NDArrayFloat, List[NDArrayFloat]]:
    """Build an XRF phantom.

    The created phantom has been used in:
    - N. Viganò and V. A. Solé, “Physically corrected forward operators for
    induced emission tomography: a simulation study,” Meas. Sci. Technol., no.
    Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.

    Parameters
    ----------
    ph_or : NDArrayFloat
        The phases phantom map.
    elements : Sequence[str], optional
        Element symbols. The default is ["Ca", "Fe"].
    em_lines : str | Sequence[str], optional
        Emission lines. The default is "KA" (corresponding to K-alpha).
    in_energy_keV : float, optional
        Incoming beam energy in keV. The default is 20.0.
    detectors_pos_rad : float | tuple | list | NDArrayFloat, optional
        Detector(s) positions in radians, with respect to incoming beam.
        If None, Compton is not produced. The default is None.

    Returns
    -------
    vol_yield : List[NDArrayFloat]
        Voxel-wise fluorescence and Compton yields.
    vol_att_in : NDArrayFloat
        Voxel-wise attenuation at the incoming beam energy.
    vol_att_out : List[NDArrayFloat]
        Voxel-wise attenuation at the emitted energy.
    """
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

    phantom = physics.VolumeMaterial(phase_fractions, phase_compound_names, voxel_size_cm)

    vol_lin_att_in = phantom.get_attenuation(in_energy_keV)

    num_vols_out = len(elements) + (detectors_pos_rad is not None)
    vol_yield = [np.array([])] * num_vols_out
    vol_lin_att_out = [np.array([])] * num_vols_out

    for ii, el in enumerate(elements):
        if isinstance(em_lines, str):
            line = em_lines
        else:
            line = em_lines[ii]
        out_energy_keV, vol_yield[ii] = phantom.get_fluo_yield(el, in_energy_keV, line)
        vol_lin_att_out[ii] = phantom.get_attenuation(out_energy_keV)

    if detectors_pos_rad:
        out_energy_keV, vol_yield[-1] = phantom.get_compton_scattering(in_energy_keV, angle_rad=detectors_pos_rad)
        vol_lin_att_out[-1] = phantom.get_attenuation(out_energy_keV)

    return (vol_yield, vol_lin_att_in, vol_lin_att_out)


def add_noise(
    img_clean: NDArray,
    num_photons: Union[int, float],
    add_poisson: bool = False,
    readout_noise_std: Optional[float] = None,
    background_avg: Optional[float] = None,
    background_std: Optional[float] = None,
    detection_efficiency: float = 1.0,
    dtype: DTypeLike = np.float32,
) -> Tuple[NDArray, NDArray, float]:
    """Add noise to an image (sinogram).

    Parameters
    ----------
    img_clean : NDArray
        The clean input image.
    num_photons : Union[int, float]
        Number of photons corresponding to the value 1.0 in the image.
    add_poisson : bool, optional
        Whether to add Poisson noise, by default False.
    readout_noise_std : Optional[float], optional
        Standard deviation of the readout noise, by default None.
    background_avg : Optional[float], optional
        Average value of the background, by default None.
    background_std : Optional[float], optional
        Standard deviation of the background, by default None.
    detection_efficiency : float, optional
        Efficiency of the detection (e.g. detector solid angle, inclination, etc), by default 1.0.
    dtype : DTypeLike, optional
        Data type of the volumes, by default np.float32.

    Returns
    -------
    Tuple[NDArray, NDArray, float]
        The noised and clean images (scaled by the photons and efficiency), and the background.
    """
    img_clean = num_photons * detection_efficiency * img_clean.copy().astype(dtype)

    img_noise = img_clean.copy()

    if background_avg is not None:
        if background_std is None:
            background_std = background_avg * 5e-2
        img_noise += (
            num_photons * detection_efficiency * np.abs(np.random.normal(background_avg, background_std, img_clean.shape))
        )

    background = float(np.mean((img_noise - img_clean).flatten()))

    if add_poisson:
        img_noise = np.random.poisson(img_noise).astype(dtype)
    if readout_noise_std is not None:
        img_noise += np.random.normal(0, readout_noise_std, img_clean.shape)

    return img_noise, img_clean, background


def create_sino(
    ph: NDArrayFloat,
    num_angles: int,
    start_angle_deg: float = 0.0,
    end_angle_deg: float = 180.0,
    dwell_time_s: float = 1.0,
    photon_flux: float = 1e9,
    detectors_pos_rad: Union[float, Sequence[float], NDArrayFloat] = (np.pi / 2),
    vol_att_in: Optional[NDArrayFloat] = None,
    vol_att_out: Optional[NDArrayFloat] = None,
    psf: Optional[NDArrayFloat] = None,
    background_avg: Optional[float] = None,
    background_std: Optional[float] = None,
    add_poisson: bool = False,
    readout_noise_std: Optional[float] = None,
    dtype: DTypeLike = np.float32,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, float]:
    """Compute the sinogram from a given phantom.

    Parameters
    ----------
    ph : NDArrayFloat
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
    detectors_pos_rad : float | Sequence[float] | NDArrayFloat, optional
        Detector(s) positions in radians, with respect to incoming beam. The default is (np.pi / 2).
    vol_att_in : NDArrayFloat, optional
        Attenuation volume for the incoming beam. The default is None.
    vol_att_out : NDArrayFloat, optional
        Attenuation volume for the outgoing beam. The default is None.
    psf : NDArrayFloat, optional
        Point spread function or probing beam profile. The default is None.
    background_avg : float, optional
        Background average value. The default is None.
    background_std : float, optional
        Background standard deviation. The default is None.
    add_poisson : bool, optional
        Switch to turn on Poisson noise. The default is False.
    readout_noise_std : float, optional
        Read-out noise standard deviation. The default is None.
    dtype : numpy.dtype, optional
        Output datatype. The default is np.float32.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, float]
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
            sino = p.fp(ph)
    else:
        with projectors.ProjectorAttenuationXRF(
            ph.shape, angles_rad, att_in=vol_att_in, att_out=vol_att_out, angles_detectors_rad=detectors_pos_rad, psf=psf
        ) as p:
            sino = p.fp(ph)

    # Adding noise
    sino_noise, sino, background = add_noise(
        sino,
        num_photons=num_photons,
        add_poisson=add_poisson,
        readout_noise_std=readout_noise_std,
        background_avg=background_avg,
        background_std=background_std,
        detection_efficiency=detector_solidangle_sr,
        dtype=dtype,
    )

    return (sino_noise, angles_rad, ph * num_photons * detector_solidangle_sr, background)


def create_sino_transmission(
    ph: NDArrayFloat,
    num_angles: int,
    start_angle_deg: float = 0,
    end_angle_deg: float = 180,
    dwell_time_s: float = 1,
    photon_flux: float = 1e9,
    psf: Optional[NDArrayFloat] = None,
    add_poisson: bool = False,
    readout_noise_std: Optional[float] = None,
    dtype: DTypeLike = np.float32,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Compute the sinogram from a given phantom.

    Parameters
    ----------
    ph : NDArrayFloat
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
    psf : NDArrayFloat, optional
        Point spread function or probing beam profile. The default is None.
    add_poisson : bool, optional
        Switch to turn on Poisson noise. The default is False.
    readout_noise_std : float, optional
        Read-out noise standard deviation. The default is None.
    dtype : numpy.dtype, optional
        Output datatype. The default is np.float32.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, float]
        The sinogram (detector readings), the flat-field, and the angular positions.
    """
    print("Creating attenuation Sino with %d angles" % num_angles)
    angles_deg = np.linspace(start_angle_deg, end_angle_deg, num_angles, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)
    print(angles_deg)

    num_photons = photon_flux * dwell_time_s

    with projectors.ProjectorUncorrected(ph.shape, angles_rad, psf=psf) as p:
        sino = np.exp(-p.fp(ph))

    # Adding noise
    sino_noise, _, _ = add_noise(
        sino,
        num_photons=num_photons,
        add_poisson=add_poisson,
        readout_noise_std=readout_noise_std,
        dtype=dtype,
    )
    flat_noise, _, _ = add_noise(
        np.ones_like(sino),
        num_photons=num_photons,
        add_poisson=add_poisson,
        readout_noise_std=readout_noise_std,
        dtype=dtype,
    )

    return (sino_noise, flat_noise, angles_rad, ph)


def compute_error_power(expected_vol: NDArrayFloat, computed_vol: NDArrayFloat) -> Tuple[float, float]:
    """Compute the expected volume signal power, and computed volume error power.

    Parameters
    ----------
    expected_vol : NDArrayFloat
        The expected volume.
    computed_vol : NDArrayFloat
        The computed volume.

    Returns
    -------
    Tuple[float, float]
        The expected volume signal power, and the computed volume.
    """
    vol_power = np.sqrt(np.sum((expected_vol) ** 2) / expected_vol.size)
    error_power = np.sqrt(np.sum(np.abs(expected_vol - computed_vol) ** 2) / expected_vol.size)
    return vol_power, error_power
