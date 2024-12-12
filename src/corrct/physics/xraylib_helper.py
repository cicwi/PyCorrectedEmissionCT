#!/usr/bin/env python3
"""
xraylib handling functions.

@author: Nicola VIGANÒ, CEA-IRIG, Grenoble, France
"""

from typing import Union

import numpy as np

import xraylib

xraylib.XRayInit()


def get_element_number(element: Union[str, int]) -> int:
    """Return the element number from the symbol.

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


def get_element_number_and_symbol(element: Union[str, int]) -> tuple[str, int]:
    """Return both the element symbol and number from either the symbol or the number.

    Parameters
    ----------
    element : str | int
        The element symbol (or number, which won't be converted).

    Returns
    -------
    tuple[str, int]
        The element symbol and number.
    """
    if isinstance(element, str):
        el_sym = element
        el_num = xraylib.SymbolToAtomicNumber(element)
    else:
        el_sym = xraylib.AtomicNumberToSymbol(element)
        el_num = element

    return el_sym, el_num


def get_compound(cmp_name: str, density: Union[float, None] = None) -> dict:
    """
    Build a compound from the compound composition string.

    Parameters
    ----------
    cmp_name : str
        Compund name / composition.
    density : float, optional
        The density of the compound. If not provided it will be approximated from the composing elements.
        The default is None.

    Returns
    -------
    cmp : dict
        The compound structure.
    """
    try:
        cmp = xraylib.GetCompoundDataNISTByName(cmp_name)
    except ValueError:
        cmp = xraylib.CompoundParser(cmp_name)

    cmp["name"] = cmp_name
    if density is None:
        density = 0
        for ii, el in enumerate(cmp["Elements"]):
            density += xraylib.ElementDensity(el) * cmp["massFractions"][ii]
    cmp["density"] = density
    return cmp


def get_compound_cross_section(compound: dict, mean_energy_keV: float) -> float:
    """Compute a compound cross section for the given incoming photon energy.

    Parameters
    ----------
    compound : dict
        The compound structure (as returned by `get_compound`)
    mean_energy_keV : float
        The average photon energy

    Returns
    -------
    float
        The computed cross-section
    """
    try:
        return xraylib.CS_Total_CP(compound["name"], mean_energy_keV)
    except ValueError:
        elemets_cs = [
            xraylib.CS_Total(el, mean_energy_keV) * compound["massFractions"][ii] for ii, el in enumerate(compound["Elements"])
        ]
        return np.sum(elemets_cs, axis=0)
