#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provide X-ray physics related funtionality.

Created on Fri Jan  7 16:07:21 2022

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from dataclasses import dataclass

from typing import Union, Sequence, Optional, Tuple
from numpy.typing import ArrayLike, DTypeLike

import copy as cp

try:
    import xraylib

    xraylib.XRayInit()
except ImportError:
    print("WARNING: Physics support is only available when xraylib is installed!")
    raise


@dataclass
class FluoLine:
    """Fluorescence line description class."""

    name: str
    indx: int


class FluoLinesSiegbahn:
    """Siegbahn fluorescence lines collection class."""

    lines = [
        FluoLine(name="KA1", indx=xraylib.KA1_LINE),
        FluoLine(name="KA2", indx=xraylib.KA2_LINE),
        FluoLine(name="KA3", indx=xraylib.KA3_LINE),
        FluoLine(name="KB1", indx=xraylib.KB1_LINE),
        FluoLine(name="KB2", indx=xraylib.KB2_LINE),
        FluoLine(name="KB3", indx=xraylib.KB3_LINE),
        FluoLine(name="KB4", indx=xraylib.KB4_LINE),
        FluoLine(name="KB5", indx=xraylib.KB5_LINE),
        FluoLine(name="LA1", indx=xraylib.LA1_LINE),
        FluoLine(name="LA2", indx=xraylib.LA2_LINE),
        FluoLine(name="LB1", indx=xraylib.LB1_LINE),
        FluoLine(name="LB2", indx=xraylib.LB2_LINE),
        FluoLine(name="LB3", indx=xraylib.LB3_LINE),
        FluoLine(name="LB4", indx=xraylib.LB4_LINE),
        FluoLine(name="LB5", indx=xraylib.LB5_LINE),
        FluoLine(name="LB6", indx=xraylib.LB6_LINE),
        FluoLine(name="LB7", indx=xraylib.LB7_LINE),
        FluoLine(name="LB9", indx=xraylib.LB9_LINE),
        FluoLine(name="LB10", indx=xraylib.LB10_LINE),
        FluoLine(name="LB15", indx=xraylib.LB15_LINE),
        FluoLine(name="LB17", indx=xraylib.LB17_LINE),
        FluoLine(name="LG1", indx=xraylib.LG1_LINE),
        FluoLine(name="LG2", indx=xraylib.LG2_LINE),
        FluoLine(name="LG3", indx=xraylib.LG3_LINE),
        FluoLine(name="LG4", indx=xraylib.LG4_LINE),
        FluoLine(name="LG5", indx=xraylib.LG5_LINE),
        FluoLine(name="LG6", indx=xraylib.LG6_LINE),
        FluoLine(name="LG8", indx=xraylib.LG8_LINE),
        FluoLine(name="LE", indx=xraylib.LE_LINE),
        FluoLine(name="LH", indx=xraylib.LH_LINE),
        FluoLine(name="LL", indx=xraylib.LL_LINE),
        FluoLine(name="LS", indx=xraylib.LS_LINE),
        FluoLine(name="LT", indx=xraylib.LT_LINE),
        FluoLine(name="LU", indx=xraylib.LU_LINE),
        FluoLine(name="LV", indx=xraylib.LV_LINE),
        FluoLine(name="MA1", indx=xraylib.MA1_LINE),
        FluoLine(name="MA2", indx=xraylib.MA2_LINE),
        FluoLine(name="MB", indx=xraylib.MB_LINE),
        FluoLine(name="MG", indx=xraylib.MG_LINE),
    ]

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
        return [f for f in FluoLinesSiegbahn.lines if f.name[: len(line)] == line]

    @staticmethod
    def get_energy(
        element: Union[str, int], lines: Union[str, FluoLine, Sequence[FluoLine]], compute_average: bool = False
    ) -> Union[float, Sequence[float]]:
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

        if isinstance(lines, FluoLine):
            lines = [lines]
        elif isinstance(lines, str):
            lines = FluoLinesSiegbahn.get_lines(lines)
        elif len(lines) == 0:
            raise ValueError(f"No line was passed! lines={lines}")

        energy_keV = np.empty(len(lines), dtype=np.float32)
        for ii, line in enumerate(lines):
            try:
                energy_keV[ii] = xraylib.LineEnergy(el_num, line.indx)
            except ValueError as exc:
                print(f"Energy {exc}: el_num={el_num} ({el_sym}) line={line}")
                energy_keV[ii] = 0

        if compute_average:
            rates = np.empty(energy_keV.shape)
            for ii, line in enumerate(lines):
                try:
                    rates[ii] = xraylib.RadRate(el_num, line.indx)
                except ValueError as exc:
                    print(f"RadRate {exc}: el_num={el_num} ({el_sym}) line={line}")
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


class VolumeMaterial(object):
    """
    VolumeMaterial class, that can be used for predicting fluorescence and Compton yields, attenuation, etc.

    Parameters
    ----------
    phase_fractions : Sequence
        Concentration fractions of each phase for each voxel.
    phase_compounds : Sequence
        Compound description of each phase.
    voxel_size_cm : float
        Voxel size in cm.
    dtype : DTypeLike, optional
        Data type of the produced data. The default is None.

    Raises
    ------
    ValueError
        Raised in case of incorrect parameters.
    """

    def __init__(self, phase_fractions: Sequence, phase_compounds: Sequence, voxel_size_cm: float, dtype: DTypeLike = None):
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
                raise ValueError("All phase fraction volumes should have the same number of dimensions")
            if not np.all(self.shape == ph.shape):
                raise ValueError("Phase fraction volumes should all have the same shape")
            if ph.dtype == bool:
                self.phase_fractions[ii] = ph.astype(np.float32)

        if dtype is None:
            dtype = self.phase_fractions[0].dtype
        self.dtype = dtype
        for ii, ph in enumerate(self.phase_fractions):
            self.phase_fractions[ii] = ph.astype(self.dtype)

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
        self,
        element: Union[str, int],
        energy_in_keV: float,
        fluo_lines: Union[str, FluoLine, Sequence[FluoLine]],
        detector: Optional[DetectorXRF] = None,
    ) -> Tuple[float, ArrayLike]:
        """Compute the local fluorescence yield, for the given line of the given element.

        Parameters
        ----------
        element : str | int
            The element to consider.
        energy_in_keV : float
            The incombing X-ray beam energy.
        fluo_lines : str | FluoLine | Sequence[FluoLine]
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

        if isinstance(fluo_lines, FluoLine):
            fluo_lines = [fluo_lines]
        elif isinstance(fluo_lines, str):
            fluo_lines = FluoLinesSiegbahn.get_lines(fluo_lines)

        el_num = self.get_element_number(element)

        el_cs = np.empty((len(fluo_lines),), self.dtype)
        for ii, line in enumerate(fluo_lines):
            try:
                el_cs[ii] = xraylib.CS_FluorLine_Kissel(el_num, line.indx, energy_in_keV)  # fluo production for cm2/g
            except ValueError as exc:
                el_sym = xraylib.AtomicNumberToSymbol(el_num)
                print(f"Energy {exc}: el_num={el_num} ({el_sym}) line={line}")
                el_cs[ii] = 0
        el_yield = self.get_element_mass_fraction(el_num) * np.sum(el_cs) * self.voxel_size_cm

        if detector:
            el_yield *= detector.solid_angle_sr

        energy_out_keV = FluoLinesSiegbahn.get_energy(el_num, fluo_lines, compute_average=True)
        return (energy_out_keV, el_yield)
