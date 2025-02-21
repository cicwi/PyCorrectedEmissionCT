"""Physics module."""

__author__ = """Nicola VIGANÒ"""
__email__ = "N.R.Vigano@cwi.nl"


import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import attenuation  # noqa: F401, F402
from . import materials  # noqa: F401, F402
from . import phase  # noqa: F401, F402
from . import xraylib_helper  # noqa: F401, F402
from . import xrf  # noqa: F401, F402
from . import units

xraylib = xraylib_helper.xraylib
get_compound = xraylib_helper.get_compound
get_element_number = xraylib_helper.get_element_number

FluoLinesSiegbahn = xrf.LinesSiegbahn
VolumeMaterial = materials.VolumeMaterial

convert_energy_to_wlength = units.energy_to_wlength
convert_wlength_to_energy = units.wlength_to_energy


def pencil_beam_profile(
    voxel_size_um: float, beam_fwhm_um: float, profile_size: int = 1, beam_shape: str = "gaussian", verbose: bool = False
) -> NDArray:
    """
    Compute the pencil beam integration point spread function.

    Parameters
    ----------
    voxel_size_um : float
        The integration length.
    beam_fwhm_um : float
        The beam FWHM.
    profile_size : int, optional
        The number of pixels of the PSF. The default is 1.
    verbose : bool, optional
        Whether to print verbose information. The default is False.

    Returns
    -------
    NDArray
        The beam PSF.
    """
    y_points = profile_size * 2 + 1
    extent_um = y_points * voxel_size_um
    num_points = int(np.ceil(extent_um * 10) // 2) * 2 + 1
    half_voxel_size_um = voxel_size_um / 2

    eps = np.finfo(np.float32).eps

    xc = np.linspace(-extent_um / 2, extent_um / 2, num_points)

    # Box beam shape
    yb = np.abs(xc) < half_voxel_size_um
    yb[np.abs(np.abs(xc) - half_voxel_size_um) <= eps] = 0.5

    # Gaussian beam shape
    if beam_shape.lower() == "gaussian":
        yg = np.exp(-4 * np.log(2) * (xc**2) / (beam_fwhm_um**2))
    elif beam_shape.lower() == "lorentzian":
        beam_hwhm_um = beam_fwhm_um / 2
        yg = (beam_hwhm_um**2) / (xc**2 + beam_hwhm_um**2)
    elif beam_shape.lower() == "sech**2":
        # doi: 10.1364/ol.20.001160
        tau = beam_fwhm_um / (2 * np.arccosh(np.sqrt(2)))
        yg = 1 / np.cosh(xc / tau) ** 2
    else:
        raise ValueError(f"Unknown beam shape: {beam_shape.lower()}")

    yc = np.convolve(yb, yg, mode="same")
    yc = yc / np.max(yc)

    y = np.zeros((y_points,))
    for ii_p in range(y_points):
        # Finding the region that overlaps with the given adjacent voxel
        voxel_center_um = (ii_p - profile_size) * voxel_size_um
        yp = np.abs(xc - voxel_center_um) < half_voxel_size_um
        yp[(np.abs(xc - voxel_center_um) - half_voxel_size_um) < eps] = 0.5

        y[ii_p] = np.sum(yc * yp)

    # Renormalization
    y /= y.sum()

    if verbose:
        fig, ax = plt.subplots(1, 2, figsize=[10, 5])
        ax[0].plot(xc, yb, label="Integration length")  # type: ignore
        ax[0].plot(xc, yg, label=f"{beam_shape.capitalize()} beam shape")  # type: ignore
        ax[0].plot(xc, yc, label="Resulting beam shape")  # type: ignore
        ax[0].legend()  # type: ignore
        ax[0].grid()  # type: ignore

        pixels_pos = np.linspace(-(y_points - 1) / 2, (y_points - 1) / 2, y_points)
        ax[1].bar(pixels_pos, y, label="PSF values", color="C1", width=1, linewidth=1.5, edgecolor="k", alpha=0.75)  # type: ignore
        ax[1].plot(xc / extent_um * profile_size * 2, yc, label="Resulting beam shape")  # type: ignore
        ax[1].legend()  # type: ignore
        ax[1].grid()  # type: ignore
        fig.tight_layout()

    return y
