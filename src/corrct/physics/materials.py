#!/usr/bin/env python3
"""
Materials support functions and classes.

@author: Nicola VIGANÒ, ESRF - The European Synchrotron, Grenoble, France,
and CEA-IRIG, Grenoble, France
"""

import copy as cp
from collections.abc import Sequence
from typing import Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from corrct.physics import xrf  # noqa: F401, F402
from corrct.physics.xraylib_helper import get_compound_cross_section
from corrct.physics.xraylib_helper import get_element_number
from corrct.physics.xraylib_helper import xraylib


class VolumeMaterial:
    """
    VolumeMaterial class, that can be used for predicting fluorescence and Compton productions, attenuation, etc.

    Parameters
    ----------
    material_fractions : Sequence
        Concentration fractions of each material for each voxel.
    material_compounds : Sequence
        Compound description of each material.
    voxel_size_cm : float
        Voxel size in cm.
    dtype : DTypeLike, optional
        Data type of the produced data. The default is None.

    Raises
    ------
    ValueError
        Raised in case of incorrect parameters.
    """

    def __init__(
        self,
        materials_fractions: Sequence[NDArray],
        materials_composition: Sequence,
        voxel_size_cm: float,
        dtype: DTypeLike = None,
        verbose: bool = False,
    ):
        if len(materials_fractions) != len(materials_composition):
            raise ValueError(
                f"Materials fractions (# {len(materials_fractions)}) and "
                f"materials composition (# {len(materials_composition)}) arrays should have the same length"
            )
        if len(materials_fractions) == 0:
            raise ValueError("Phase list is empty")

        self.materials_fractions = list(materials_fractions)
        self.shape = np.array(self.materials_fractions[0].shape)
        for ii, ph in enumerate(self.materials_fractions):
            if len(self.shape) != len(ph.shape):
                raise ValueError("All material fraction volumes should have the same number of dimensions")
            if not np.all(self.shape == ph.shape):
                raise ValueError("Materials fraction volumes should all have the same shape")
            if ph.dtype == bool:
                self.materials_fractions[ii] = ph.astype(np.float32)

        if dtype is None:
            dtype = self.materials_fractions[0].dtype
        self.dtype = dtype
        for ii, ph in enumerate(self.materials_fractions):
            self.materials_fractions[ii] = ph.astype(self.dtype)

        self.materials_compositions = [
            xraylib.GetCompoundDataNISTByName(cmp) if isinstance(cmp, str) else cmp for cmp in materials_composition
        ]

        self.voxel_size_cm = voxel_size_cm
        self.verbose = verbose

    def get_attenuation(self, energy_keV: float) -> NDArray:
        """
        Compute the local attenuation for each voxel.

        Parameters
        ----------
        energy_keV : float
            The X-ray energy.

        Returns
        -------
        NDArray
            The computed local attenuation volume.
        """
        ph_lin_att = np.zeros(self.shape, self.dtype)
        for ph, cmp in zip(self.materials_fractions, self.materials_compositions):
            cmp_cs = get_compound_cross_section(cmp, energy_keV)
            if self.verbose:
                print(f"Attenuation ({cmp['name']} at {energy_keV}):")
                print(
                    f" - cross-section * mass fraction = {cmp_cs}, density = {cmp['density']}, pixel-size {self.voxel_size_cm}"
                )
                print(f" - total {cmp['density'] * cmp_cs * self.voxel_size_cm} (assuming material mass fraction = 1)")
            ph_lin_att += ph * cmp["density"] * cmp_cs
        return ph_lin_att * self.voxel_size_cm

    def get_element_mass_fraction(self, element: Union[str, int]) -> NDArray:
        """Compute the local element mass fraction through out all the materials.

        Parameters
        ----------
        element : str | int
            The element to look for.

        Returns
        -------
        mass_fraction : NDArray
            The local mass fraction in each voxel.
        """
        el_num = get_element_number(element)

        mass_fraction = np.zeros(self.shape, self.dtype)
        for ph, cmp in zip(self.materials_fractions, self.materials_compositions):
            if el_num in cmp["Elements"]:
                el_ind = np.where(np.array(cmp["Elements"]) == el_num)[0][0]
                mass_fraction += ph * cmp["density"] * cmp["massFractions"][el_ind]
        return mass_fraction

    def _check_parallax_detector(self, detector: xrf.DetectorXRF, tolerance: float = 1e-2) -> bool:
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
        self, energy_in_keV: float, angle_rad: Union[float, None] = None, detector: Union[xrf.DetectorXRF, None] = None
    ) -> tuple[float, NDArray]:
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
        cmptn_prod : NDArray
            Local production of Compton radiation.

        Either `angle_rad` or `detector` need to be supplied.
        """
        if detector is None:
            if angle_rad is None:
                raise ValueError("Either 'angle_rad' or 'detector' should be passed.")
        else:
            if not self._check_parallax_detector(detector):
                print("WARNING - detector parallax is above 1e-2")
            if angle_rad is None:
                angle_rad = detector.angle_rad

        cmptn_prod = np.zeros(self.shape, self.dtype)
        for ph, cmp in zip(self.materials_fractions, self.materials_compositions):
            try:
                cmp_cs = xraylib.DCS_Compt_CP(cmp["name"], energy_in_keV, angle_rad)
            except ValueError:
                cmp_cs = np.sum(
                    [
                        xraylib.DCS_Compt(el, energy_in_keV, angle_rad) * cmp["massFractions"][ii]
                        for ii, el in enumerate(cmp["Elements"])
                    ],
                    axis=0,
                )
            if self.verbose:
                print(
                    f"Compton - {cmp['name']} at incoming energy {energy_in_keV} (keV),"
                    + f" outgoing angle {np.rad2deg(angle_rad)} (deg):\n"
                    + f" - cross-section * mass fraction = {cmp_cs}, density = {cmp['density']}"
                    + ", pixel-size {self.voxel_size_cm}"
                    + f" - total {cmp['density'] * cmp_cs * self.voxel_size_cm} (assuming material mass fraction = 1)"
                )
            cmptn_prod += ph * cmp["density"] * cmp_cs
        cmptn_prod *= self.voxel_size_cm

        if detector:
            cmptn_prod *= detector.solid_angle_sr

        energy_out_keV = xraylib.ComptonEnergy(energy_in_keV, angle_rad)
        return (energy_out_keV, cmptn_prod)

    def get_fluo_production(
        self,
        element: Union[str, int],
        energy_in_keV: float,
        fluo_lines: Union[str, xrf.FluoLine, Sequence[xrf.FluoLine]],
        detector: Union[xrf.DetectorXRF, None] = None,
    ) -> tuple[float, NDArray]:
        """Compute the local fluorescence production, for the given line of the given element.

        Using Eq. (1) from:
        - T. Schoonjans et al., "The xraylib library for X-ray-matter interactions. Recent developments," Spectrochim. Acta
        Part B At. Spectrosc., vol. 66, no. 11-12, pp. 776-784, Nov. 2011, doi: 10.1016/j.sab.2011.09.011.

        Parameters
        ----------
        element : str | int
            The element to consider.
        energy_in_keV : float
            The incoming X-ray beam energy.
        fluo_lines : str | FluoLine | Sequence[FluoLine]
            The fluorescence line to consider.
        detector : DetectorXRF, optional
            The detector geometry. The default is None.

        Returns
        -------
        energy_out_keV : float
            The emitted fluorescence energy.
        el_prod : NDArray
            The local fluorescence production in each voxel.
        """
        if detector:
            if not self._check_parallax_detector(detector):
                print("WARNING - detector parallax is above 1e-2")

        if isinstance(fluo_lines, xrf.FluoLine):
            fluo_lines = [fluo_lines]
        elif isinstance(fluo_lines, str):
            fluo_lines = xrf.LinesSiegbahn.get_lines(fluo_lines)

        el_num = get_element_number(element)

        el_yield = np.empty((len(fluo_lines),), self.dtype)
        for ii, line in enumerate(fluo_lines):
            try:
                el_yield[ii] = xraylib.CS_FluorLine_Kissel(el_num, line.indx, energy_in_keV)  # fluo production for cm2/g
            except ValueError as exc:
                el_sym = xraylib.AtomicNumberToSymbol(el_num)
                if self.verbose:
                    print(f"Energy {exc}: el_num={el_num} ({el_sym}) line={line}")
                el_yield[ii] = 0
        el_prod = self.get_element_mass_fraction(el_num) * np.sum(el_yield) * self.voxel_size_cm

        if detector:
            el_prod *= detector.solid_angle_sr

        energy_out_keV = xrf.get_energy(el_num, fluo_lines, compute_average=True)

        return float(energy_out_keV), el_prod
