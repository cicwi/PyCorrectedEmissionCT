"""Units and conversion."""

from typing import overload, Union

import scipy.constants as spc
from numpy.typing import NDArray


class ConversionMetric:
    """Conversion factors between orders of magnitude of the metric units."""

    str_to_order = {
        "km": 1e-3,
        "m": 1e0,
        "cm": 1e2,
        "mm": 1e3,
        "um": 1e6,
        "nm": 1e9,
        "a": 1e10,
    }

    order_to_str = {
        1e-3: "km",
        1e0: "m",
        1e2: "cm",
        1e3: "mm",
        1e6: "um",
        1e9: "nm",
        1e10: "a",
    }

    @staticmethod
    def convert(from_unit: str, to_unit: str) -> float:
        """Convert numbers from the source unit to the destination unit.

        Parameters
        ----------
        from_unit : str
            The source unit
        to_unit : str
            The destination unit

        Returns
        -------
        float
            The conversion factor
        """
        return ConversionMetric.str_to_order[to_unit] / ConversionMetric.str_to_order[from_unit]


class ConversionEnergy:
    """Conversion factors between orders of magnitude of the energy units."""

    str_to_order = {
        "GeV": 1e-9,
        "MeV": 1e-6,
        "keV": 1e-3,
        "eV": 1e0,
        "meV": 1e3,
        "ueV": 1e6,
    }

    order_to_str = {
        1e-9: "GeV",
        1e-6: "MeV",
        1e-3: "keV",
        1e0: "eV",
        1e3: "meV",
        1e6: "ueV",
    }

    @staticmethod
    def convert(from_unit: str, to_unit: str) -> float:
        """Convert numbers from the source unit to the destination unit.

        Parameters
        ----------
        from_unit : str
            The source unit
        to_unit : str
            The destination unit

        Returns
        -------
        float
            The conversion factor
        """
        return ConversionEnergy.str_to_order[to_unit] / ConversionEnergy.str_to_order[from_unit]


@overload
def energy_to_wlength(energy: float, unit_wl: str = "m", unit_en: str = "keV") -> float: ...


@overload
def energy_to_wlength(energy: NDArray, unit_wl: str = "m", unit_en: str = "keV") -> NDArray: ...


def energy_to_wlength(energy: Union[float, NDArray], unit_wl: str = "m", unit_en: str = "keV") -> Union[float, NDArray]:
    """Convert from energy to wavelength.

    Parameters
    ----------
    energy : float | NDArray
        The energy
    unit_wl : str, optional
        The chosen unit for the output wavelength. The default is "m"
    unit_en : str, optional
        The chosen unit for the input energy. The default is "keV"

    Returns
    -------
    float | NDArray
        The wavelength in the chosen unit
    """
    factor_m = ConversionMetric.convert("m", unit_wl)
    factor_e = ConversionEnergy.convert("eV", unit_en)
    return factor_m * factor_e / (spc.physical_constants["electron volt-inverse meter relationship"][0] * energy)


@overload
def wlength_to_energy(w_length: float, unit_wl: str = "m", unit_en: str = "keV") -> float: ...


@overload
def wlength_to_energy(w_length: NDArray, unit_wl: str = "m", unit_en: str = "keV") -> NDArray: ...


def wlength_to_energy(w_length: Union[float, NDArray], unit_wl: str = "m", unit_en: str = "keV") -> Union[float, NDArray]:
    """Convert wavelength to energy.

    Parameters
    ----------
    w_length : float | NDArray
        The wavelength in the chosen unit
    unit : str, optional
        The chosen unit for the input wavelength. The default is "m"
    unit_en : str, optional
        The chosen unit for the output energy. The default is "keV"

    Returns
    -------
    float | NDArray
        The energy in the chosen unit
    """
    factor_m = ConversionMetric.convert("m", unit_wl)
    factor_e = ConversionEnergy.convert("eV", unit_en)
    return factor_m * factor_e / (spc.physical_constants["electron volt-inverse meter relationship"][0] * w_length)
