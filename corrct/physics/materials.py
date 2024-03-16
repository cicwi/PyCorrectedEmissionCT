#!/usr/bin/env python3
"""
Materials support functions and classes.

@author: Nicola VIGANÒ, ESRF - The European Synchrotron, Grenoble, France,
and CEA-IRIG, Grenoble, France
"""

try:
    from . import xraylib_helper  # noqa: F401, F402
    from . import xrf  # noqa: F401, F402

    xraylib = xraylib_helper.xraylib

except ImportError:
    print("WARNING: Physics support is only available when xraylib is installed!")
    raise


import copy as cp
from collections.abc import Sequence
from typing import Union
import numpy as np
from numpy.typing import DTypeLike, NDArray
import matplotlib.pyplot as plt


def _get_compound_cross_section(compound: dict, mean_energy_keV: float) -> float:
    try:
        return xraylib.CS_Total_CP(compound["name"], mean_energy_keV)
    except ValueError:
        elemets_cs = [
            xraylib.CS_Total(el, mean_energy_keV) * compound["massFractions"][ii] for ii, el in enumerate(compound["Elements"])
        ]
        return np.sum(elemets_cs, axis=0)


def get_linear_attenuation_coefficient(
    compound: Union[str, dict], energy_keV: float, pixel_size_um: float, density: Union[float, None] = None
) -> float:
    """Compute the linear attenuation coefficient for given compound, energy, and pixel size.

    Parameters
    ----------
    compound : Union[str, dict]
        The compound for which we compute the linear attenuation coefficient
    energy_keV : float
        The energy of the photons
    pixel_size_um : float
        The pixel size in microns
    density : Union[float, None], optional
        The density of the compound (if different from the default value), by default None

    Returns
    -------
    float
        The linear attenuation coefficient
    """
    if isinstance(compound, str):
        compound = xraylib_helper.get_compound(compound)

    if density is not None:
        compound["density"] = density

    cmp_cs = _get_compound_cross_section(compound, energy_keV)
    return pixel_size_um * 1e-4 * compound["density"] * cmp_cs


def plot_effective_attenuation(
    compound: Union[str, dict],
    thickness_um: float,
    mean_energy_keV: float,
    fwhm_keV: float,
    line_shape: str = "lorentzian",
    num_points: int = 201,
) -> None:
    """Plot spectral attenuation of a given line.

    Parameters
    ----------
    compound : Union[str, dict]
        Compound to consider
    thickness_um : float
        Thickness of the compound (in microns)
    mean_energy_keV : float
        Average energy of the line
    fwhm_keV : float
        Full-width half-maximum of the line
    line_shape : str, optional
        Shape of the line, by default "lorentzian".
        Options are: "gaussian" | "lorentzian" | "sech**2".
    num_points : int, optional
        number of discretization points, by default 201

    Raises
    ------
    ValueError
        When an unsupported line is chosen.
    """
    xc = np.linspace(-0.5, 0.5, num_points)

    if line_shape.lower() == "gaussian":
        xc *= fwhm_keV * 3
        yg = np.exp(-4 * np.log(2) * (xc**2) / (fwhm_keV**2))
    elif line_shape.lower() == "lorentzian":
        xc *= fwhm_keV * 13
        hwhm_keV = fwhm_keV / 2
        yg = hwhm_keV / (xc**2 + hwhm_keV**2)
    elif line_shape.lower() == "sech**2":
        # doi: 10.1364/ol.20.001160
        xc *= fwhm_keV * 4
        tau = fwhm_keV / (2 * np.arccosh(np.sqrt(2)))
        yg = 1 / np.cosh(xc / tau) ** 2
    else:
        raise ValueError(f"Unknown beam shape: {line_shape.lower()}")

    nrgs_keV = xc + mean_energy_keV

    if isinstance(compound, str):
        compound = xraylib_helper.get_compound(compound)

    atts = np.empty_like(yg)

    for ii, nrg in enumerate(nrgs_keV):
        cmp_cs = _get_compound_cross_section(compound, nrg)
        atts[ii] = np.exp(-thickness_um * 1e-4 * compound["density"] * cmp_cs)

    yg = yg / np.max(yg)

    fig, axs_line = plt.subplots(1, 1)
    pl_line = axs_line.plot(nrgs_keV, yg, label="$I_0$", color="C0")
    axs_line.tick_params(axis="y", labelcolor="C0")
    axs_atts = axs_line.twinx()
    pl_atts = axs_atts.plot(nrgs_keV, atts, label="$\\mu (E)$", color="C1")
    pl_line_att = axs_atts.plot(nrgs_keV, yg * atts, label="$I_m$", color="C2")
    axs_atts.tick_params(axis="y", labelcolor="C1")
    all_pls = pl_line + pl_atts + pl_line_att
    axs_atts.legend(all_pls, [pl.get_label() for pl in all_pls])
    axs_line.grid()
    fig.tight_layout()

    I_lin = np.sum(yg * atts[num_points // 2])
    I_meas = yg.dot(atts)
    print(f"Expected intensity: {I_lin}, measured: {I_meas} ({I_meas / I_lin:%})")
    print(f"Mean energy {nrgs_keV.dot(yg / np.sum(yg) * (atts / atts[len(atts) // 2]))}, {nrgs_keV.dot(yg / np.sum(yg))}")


class VolumeMaterial:
    """
    VolumeMaterial class, that can be used for predicting fluorescence and Compton yields, attenuation, etc.

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
            cmp_cs = _get_compound_cross_section(cmp, energy_keV)
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
        el_num = xraylib_helper.get_element_number(element)

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
        cmptn_yield : NDArray
            Local yield of Compton radiation.

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

        cmptn_yield = np.zeros(self.shape, self.dtype)
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
            cmptn_yield += ph * cmp["density"] * cmp_cs
        cmptn_yield *= self.voxel_size_cm

        if detector:
            cmptn_yield *= detector.solid_angle_sr

        energy_out_keV = xraylib.ComptonEnergy(energy_in_keV, angle_rad)
        return (energy_out_keV, cmptn_yield)

    def get_fluo_yield(
        self,
        element: Union[str, int],
        energy_in_keV: float,
        fluo_lines: Union[str, xrf.FluoLine, Sequence[xrf.FluoLine]],
        detector: Union[xrf.DetectorXRF, None] = None,
    ) -> tuple[float, NDArray]:
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
        el_yield : NDArray
            The local fluorescence yield in each voxel.
        """
        if detector:
            if not self._check_parallax_detector(detector):
                print("WARNING - detector parallax is above 1e-2")

        if isinstance(fluo_lines, xrf.FluoLine):
            fluo_lines = [fluo_lines]
        elif isinstance(fluo_lines, str):
            fluo_lines = xrf.LinesSiegbahn.get_lines(fluo_lines)

        el_num = xraylib_helper.get_element_number(element)

        el_cs = np.empty((len(fluo_lines),), self.dtype)
        for ii, line in enumerate(fluo_lines):
            try:
                el_cs[ii] = xraylib.CS_FluorLine_Kissel(el_num, line.indx, energy_in_keV)  # fluo production for cm2/g
            except ValueError as exc:
                el_sym = xraylib.AtomicNumberToSymbol(el_num)
                if self.verbose:
                    print(f"Energy {exc}: el_num={el_num} ({el_sym}) line={line}")
                el_cs[ii] = 0
        el_yield = self.get_element_mass_fraction(el_num) * np.sum(el_cs) * self.voxel_size_cm

        if detector:
            el_yield *= detector.solid_angle_sr

        energy_out_keV = xrf.LinesSiegbahn.get_energy(el_num, fluo_lines, compute_average=True)

        return float(energy_out_keV), el_yield
