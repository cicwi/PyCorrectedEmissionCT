# -*- coding: utf-8 -*-
"""
Provides useful functions for testing routines and functionalities.

Created on Thu Jun  4 12:28:21 2020

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from . import projectors

from enum import Enum
from dataclasses import dataclass

from typing import Union, Sequence, Optional, Tuple
from numpy.typing import ArrayLike, DTypeLike

import copy as cp

try:
    import xraylib

    xraylib.XRayInit()

    __has_xraylib__ = True
except ImportError as exc:
    print("WARNING:", exc)

    __has_xraylib__ = False


def roundup_to_pow2(x: Union[int, float, ArrayLike], p: int, data_type: DTypeLike = int) -> Union[int, float, ArrayLike]:
    """Round first argument to the power of 2 indicated by second argument.

    Parameters
    ----------
    x : int | float | ArrayLike
        Number to round up.
    p : int
        Power of 2.
    data_type : DTypeLike, optional
        data type of the output. The default is np.intp.

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


class FluoLinesSiegbahn(Enum):
    KA1 = xraylib.KA1_LINE
    KA2 = xraylib.KA2_LINE
    KA3 = xraylib.KA3_LINE
    KB1 = xraylib.KB1_LINE
    KB2 = xraylib.KB2_LINE
    KB3 = xraylib.KB3_LINE
    KB4 = xraylib.KB4_LINE
    KB5 = xraylib.KB5_LINE
    LA1 = xraylib.LA1_LINE
    LA2 = xraylib.LA2_LINE
    LB1 = xraylib.LB1_LINE
    LB2 = xraylib.LB2_LINE
    LB3 = xraylib.LB3_LINE
    LB4 = xraylib.LB4_LINE
    LB5 = xraylib.LB5_LINE
    LB6 = xraylib.LB6_LINE
    LB7 = xraylib.LB7_LINE
    LB9 = xraylib.LB9_LINE
    LB10 = xraylib.LB10_LINE
    LB15 = xraylib.LB15_LINE
    LB17 = xraylib.LB17_LINE
    LG1 = xraylib.LG1_LINE
    LG2 = xraylib.LG2_LINE
    LG3 = xraylib.LG3_LINE
    LG4 = xraylib.LG4_LINE
    LG5 = xraylib.LG5_LINE
    LG6 = xraylib.LG6_LINE
    LG8 = xraylib.LG8_LINE
    LE = xraylib.LE_LINE
    LH = xraylib.LH_LINE
    LL = xraylib.LL_LINE
    LS = xraylib.LS_LINE
    LT = xraylib.LT_LINE
    LU = xraylib.LU_LINE
    LV = xraylib.LV_LINE
    MA1 = xraylib.MA1_LINE
    MA2 = xraylib.MA2_LINE
    MB = xraylib.MB_LINE
    MG = xraylib.MG_LINE

    @staticmethod
    def get_lines(line: str) -> Sequence:
        """
        Return the list of xraylib line macro definitions for the requested family.

        Parameters
        ----------
        line : str
            The requested line. It can be a whole shell (transition to that shell),
            or sub-shells.

        Returns
        -------
        Sequence
            List of corresponding lines.
        """
        return [f for f in FluoLinesSiegbahn if f.name[:len(line)] == line]

    @staticmethod
    def get_energy(element: Union[str, int], line: str, compute_average: bool = False) -> Union[float, Sequence[float]]:
        """
        Return the energy(ies) of the requested line for the given element.

        Parameters
        ----------
        element : Union[str, int]
            The requested element.
        line : str
            The requested line. It can be a whole shell (transition to that shell),
            or sub-shells.
        compute_average : bool, optional
            Weighted averaging the lines, using the radiation rate. The default is False.

        Returns
        -------
        energy_keV : Union[float, Sequence[float]]
            Either the average energy or the list of different energies.
        """
        if isinstance(element, str):
            el_sym = element
            el_num = xraylib.SymbolToAtomicNumber(element)
        else:
            el_sym = xraylib.AtomicNumberToSymbol(element)
            el_num = element

        lines = FluoLinesSiegbahn.get_lines(line)
        energy_keV = np.empty(len(lines), dtype=np.float32)
        for ii, line in enumerate(lines):
            try:
                energy_keV[ii] = xraylib.LineEnergy(el_num, line.value)
            except ValueError as exc:
                print(f"Energy {exc}: {el_num=} ({el_sym}) {line=}")
                energy_keV[ii] = 0

        if compute_average:
            rates = np.empty(energy_keV.shape)
            for ii, line in enumerate(lines):
                try:
                    rates[ii] = xraylib.RadRate(el_num, line.value)
                except ValueError as exc:
                    print(f"RadRate {exc}: {el_num=} ({el_sym}) {line=}")
                    rates[ii] = 0
            energy_keV = np.sum(energy_keV * rates / np.sum(rates))

        return energy_keV


@dataclass
class DetectorXRF(object):
    """Simple XRF detector model."""

    surface_mm2: float
    distance_mm: float
    angle_rad: float = (np.pi / 2)

    @property
    def solid_angle_sr(self) -> Union[float, ArrayLike]:
        """Compute the solid angle covered by the detector.

        Returns
        -------
        float | ArrayLike
            The computed solid angle of the detector geometry.
        """
        return self.surface_mm2 / (np.pi * self.distance_mm ** 2)


class Phantom(object):
    """Base phantom class."""

    def __init__(self, phase_fractions: Sequence, phase_compounds: Sequence, voxel_size_cm: float):
        if not __has_xraylib__:
            raise ValueError("Realistic phantoms not available without xraylib.")

        if len(phase_fractions) != len(phase_compounds):
            raise ValueError(
                "Phase fractions (# %d) and phase compounds (# %d) should have the same length"
                % (len(phase_fractions), len(phase_compounds))
            )
        if len(phase_fractions) == 0:
            raise ValueError("Phase list is empty")

        self.phase_fractions = list(phase_fractions)
        self.shape = np.array(self.phase_fractions[0].shape)
        for ii, ph in enumerate(self.phase_fractions):
            if len(self.shape) != len(ph.shape):
                raise ValueError("All phantom phase fraction volumes should have the same number of dimensions")
            if not np.all(self.shape == ph.shape):
                raise ValueError("Phantom phase fraction volumes should all have the same shape")
            if ph.dtype == bool:
                self.phase_fractions[ii] = ph.astype(np.float32)
        self.dtype = self.phase_fractions[0].dtype

        self.phase_compounds = [
            xraylib.GetCompoundDataNISTByName(cmp) if isinstance(cmp, str) else cmp for cmp in phase_compounds
        ]

        self.voxel_size_cm = voxel_size_cm

    @staticmethod
    def get_element_number(element: Union[str, int]) -> int:
        """Compute the element number from the symbol.

        Parameters
        ----------
        element : str | int
            The element symbol (or number, which won't be converted).

        Returns
        -------
        int
            The corresponding element number.
        """
        if isinstance(element, int):
            return element
        else:
            return xraylib.SymbolToAtomicNumber(element)

    def get_attenuation(self, energy_keV: float) -> ArrayLike:
        """Compute the local attenuation for each voxel.

        Parameters
        ----------
        energy_keV : float
            The X-ray energy.

        Returns
        -------
        ArrayLike
            The computed local attenuation volume.
        """
        ph_lin_att = np.zeros(self.shape, self.dtype)
        for ph, cmp in zip(self.phase_fractions, self.phase_compounds):
            try:
                cmp_cs = xraylib.CS_Total_CP(cmp["name"], energy_keV)
            except ValueError:
                cmp_cs = np.sum(
                    [xraylib.CS_Total(el, energy_keV) * cmp["massFractions"][ii] for ii, el in enumerate(cmp["Elements"])],
                    axis=0,
                )
            ph_lin_att += ph * cmp["density"] * cmp_cs
        return ph_lin_att * self.voxel_size_cm

    def get_element_mass_fraction(self, element: Union[str, int]) -> ArrayLike:
        """Compute the local element mass fraction through out all the phases.

        Parameters
        ----------
        element : str | int
            The element to look for.

        Returns
        -------
        mass_fraction : ArrayLike
            The local mass fraction in each voxel.
        """
        el_num = self.get_element_number(element)

        mass_fraction = np.zeros(self.shape, self.dtype)
        for ph, cmp in zip(self.phase_fractions, self.phase_compounds):
            if el_num in cmp["Elements"]:
                el_ind = np.where(np.array(cmp["Elements"]) == el_num)[0][0]
                mass_fraction += ph * cmp["density"] * cmp["massFractions"][el_ind]
        return mass_fraction

    def _check_parallax_detector(self, detector: DetectorXRF, tolerance: float = 1e-2) -> bool:
        half_sample_size = np.max(self.voxel_size_cm * self.shape) / 2

        if isinstance(detector.distance_mm, float) or (
            isinstance(detector.distance_mm, np.ndarray) and detector.distance_mm.size == 1
        ):
            dets = cp.deepcopy(detector)
            dets.distance_mm = dets.distance_mm + np.array([-half_sample_size, half_sample_size])
        else:
            dets = detector

        solid_angles = dets.solid_angle_sr
        max_parallax = np.max(solid_angles) - np.min(solid_angles)
        return max_parallax < tolerance

    def get_compton_scattering(
        self, energy_in_keV: float, angle_rad: Optional[float] = None, detector: Optional[DetectorXRF] = None
    ) -> Tuple[float, ArrayLike]:
        """Compute the local Compton scattering.

        Parameters
        ----------
        energy_in_keV : float
            Incoming beam energy.
        angle_rad : float, optional
            The detector angle, with respect to incoming beam direction. The default is None.
        detector : DetectorXRF, optional
            The detector object. The default is None.

        Raises
        ------
        ValueError
            In case neither of `angle_rad` nor `detector` have been passed.

        Returns
        -------
        energy_out_keV : float
            The energy of the Compton radiation received by the detector.
        cmptn_yield : ArrayLike
            Local yield of Compton radiation.

        Either `angle_rad` or `detector` need to be supplied.
        """
        if angle_rad is None and detector is None:
            raise ValueError("Either 'angle_rad' or 'detector' should be passed.")
        if detector:
            if not self._check_parallax_detector(detector):
                print("WARNING - detector parallax is above 1e-2")
            if angle_rad is None:
                angle_rad = detector.angle_rad

        cmptn_yield = np.zeros(self.shape, self.dtype)
        for ph, cmp in zip(self.phase_fractions, self.phase_compounds):
            try:
                cmptn_cmp_cs = xraylib.DCS_Compt_CP(cmp["name"], energy_in_keV, angle_rad)
            except ValueError:
                cmptn_cmp_cs = np.sum(
                    [
                        xraylib.DCS_Compt(el, energy_in_keV, angle_rad) * cmp["massFractions"][ii]
                        for ii, el in enumerate(cmp["Elements"])
                    ],
                    axis=0,
                )
            cmptn_yield += ph * cmp["density"] * cmptn_cmp_cs
        cmptn_yield *= self.voxel_size_cm

        if detector:
            cmptn_yield *= detector.solid_angle_sr

        energy_out_keV = xraylib.ComptonEnergy(energy_in_keV, angle_rad)
        return (energy_out_keV, cmptn_yield)

    def get_fluo_yield(
        self, element: Union[str, int], energy_in_keV: float, fluo_line: int, detector: Optional[DetectorXRF] = None
    ) -> Tuple[float, ArrayLike]:
        """Compute the local fluorescence yield, for the given line of the given element.

        Parameters
        ----------
        element : str | int
            The element to consider.
        energy_in_keV : float
            The incombing X-ray beam energy.
        fluo_line : int
            The fluorescence line to consider.
        detector : DetectorXRF, optional
            The detector geometry. The default is None.

        Returns
        -------
        energy_out_keV : float
            The emitted fluorescence energy.
        el_yield : ArrayLike
            The local fluorescence yield in each voxel.
        """
        if detector:
            if not self._check_parallax_detector(detector):
                print("WARNING - detector parallax is above 1e-2")

        el_num = self.get_element_number(element)

        el_cs = xraylib.CS_FluorLine_Kissel(el_num, fluo_line, energy_in_keV)  # fluo production for cm2/g
        el_yield = self.get_element_mass_fraction(el_num) * el_cs * self.voxel_size_cm

        if detector:
            el_yield *= detector.solid_angle_sr

        energy_out_keV = xraylib.LineEnergy(el_num, fluo_line)
        return (energy_out_keV, el_yield)


def phantom_assign_concentration(
    ph_or: ArrayLike, element: str = "Ca", in_energy_keV: float = 20.0
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

    phantom = Phantom(phase_fractions, phase_compound_names, voxel_size_cm)

    out_energy_keV, vol_fluo_yield = phantom.get_fluo_yield(element, in_energy_keV, xraylib.KA_LINE)

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
