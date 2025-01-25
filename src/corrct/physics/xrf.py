#!/usr/bin/env python3
"""
XRF support functions.

@author: Nicola VIGANÒ, ESRF - The European Synchrotron, Grenoble, France,
and CEA-IRIG, Grenoble, France
"""

try:
    from . import xraylib_helper  # noqa: F401, F402

    xraylib = xraylib_helper.xraylib

except ImportError:
    print("WARNING: Physics support is only available when xraylib is installed!")
    raise


from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload, Literal, Union
import numpy as np
from numpy.typing import NDArray


@dataclass
class FluoLine:
    """Fluorescence line description class."""

    name: str
    indx: int


class LinesSiegbahn:
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
    def get_lines(line: str) -> Sequence[FluoLine]:
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
        return [f for f in LinesSiegbahn.lines if f.name[: len(line)] == line.upper()]


def _get_lines_list(lines) -> Sequence[FluoLine]:
    if isinstance(lines, FluoLine):
        return [lines]
    elif isinstance(lines, str):
        return LinesSiegbahn.get_lines(lines)
    elif len(lines) == 0:
        raise ValueError(f"No line was passed! lines={lines}")
    else:
        return lines


def get_radiation_rate(
    element: Union[str, int],
    lines: Union[str, FluoLine, Sequence[FluoLine]],
    verbose: bool = False,
) -> NDArray:
    """Return the radiation rates of the requested lines for the given element.

    Parameters
    ----------
    element : Union[str, int]
        The requested element
    lines : Union[str, FluoLine, Sequence[FluoLine]]
        The requested line. It can be a whole shell (transition to that shell),
        or sub-shells.
    verbose : bool, optional
        Whether to produce verbose output in case of errors, by default False

    Returns
    -------
    NDArray
        The list of radiation rates
    """
    el_sym, el_num = xraylib_helper.get_element_number_and_symbol(element)

    lines_list = _get_lines_list(lines)

    rates = np.empty(len(lines_list), dtype=np.float32)
    for ii, line in enumerate(lines_list):
        try:
            rates[ii] = xraylib.RadRate(el_num, line.indx)
        except ValueError as exc:
            if verbose:
                print(f"INFO - RadRate - {exc}: el_num={el_num} ({el_sym}) line={line}")
            rates[ii] = 0
    return rates


@overload
def get_energy(
    element: Union[str, int],
    lines: Union[str, FluoLine, Sequence[FluoLine]],
    *,
    compute_average: Literal[False] = False,
    verbose: bool = False,
) -> NDArray: ...


@overload
def get_energy(
    element: Union[str, int],
    lines: Union[str, FluoLine, Sequence[FluoLine]],
    *,
    compute_average: Literal[True] = True,
    verbose: bool = False,
) -> float: ...


def get_energy(
    element: Union[str, int],
    lines: Union[str, FluoLine, Sequence[FluoLine]],
    *,
    compute_average: bool = False,
    verbose: bool = False,
) -> Union[float, NDArray]:
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
    energy_keV : Union[float, NDArray]
        Either the average energy or the list of different energies.
    """
    el_sym, el_num = xraylib_helper.get_element_number_and_symbol(element)

    lines_list = _get_lines_list(lines)

    energy_keV = np.empty(len(lines_list), dtype=np.float32)
    for ii, line in enumerate(lines_list):
        try:
            energy_keV[ii] = xraylib.LineEnergy(el_num, line.indx)
        except ValueError as exc:
            if verbose:
                print(f"INFO - Energy - {exc}: el_num={el_num} ({el_sym}) line={line}")
            energy_keV[ii] = 0

    if compute_average:
        rates = get_radiation_rate(element, lines_list)
        energy_keV = float(np.sum(energy_keV * rates / np.sum(rates)))

    if verbose:
        print(f"{el_sym}-{lines} emission energy (keV):", energy_keV, "\n")

    return energy_keV


@dataclass
class DetectorXRF:
    """Simple XRF detector model."""

    surface_mm2: float
    distance_mm: Union[float, NDArray]
    angle_rad: float = np.pi / 2

    @property
    def solid_angle_sr(self) -> Union[float, NDArray]:
        """Compute the solid angle covered by the detector.

        Returns
        -------
        float | NDArray
            The computed solid angle of the detector geometry.
        """
        return self.surface_mm2 / (4 * np.pi * self.distance_mm**2)
