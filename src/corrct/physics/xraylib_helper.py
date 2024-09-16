#!/usr/bin/env python3
"""
xraylib handling functions.

@author: Nicola VIGANÒ, CEA-IRIG, Grenoble, France
"""

from typing import Union

try:
    import xraylib

    xraylib.XRayInit()
except ImportError:
    print("WARNING: Physics support is only available when _xraylib_ is installed!")
    raise


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
