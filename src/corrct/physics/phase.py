#!/usr/bin/env python3
"""
Phase contrast support functions.

@author: Nicola VIGANÒ, CEA-IRIG, Grenoble, France
"""

try:
    from . import xraylib_helper  # noqa: F401, F402

    xraylib = xraylib_helper.xraylib

except ImportError:
    print("WARNING: Physics support is only available when xraylib is installed!")
    raise

from collections.abc import Sequence
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as spimg
from numpy.typing import DTypeLike, NDArray


def get_delta_beta(cmp_name: str, energy_keV: float, density: Union[float, None] = None) -> float:
    """Compute the delta-over-beta parameter for a specific compound.

    Parameters
    ----------
    cmp_name : str
        Molar composition of a compound
    energy_keV : float
        Energy at which the d/b value should be computed
    density : Optional[float], optional
        Density of the compound, by default None

    Returns
    -------
    float
        The computed delta-over-beta.
    """
    cmp = xraylib_helper.get_compound(cmp_name, density=density)
    ref_ind = xraylib.Refractive_Index(cmp_name, energy_keV, cmp["density"])
    return (1 - ref_ind.real) / ref_ind.imag


def get_delta_beta_curves(
    compounds: Sequence[str], energy_keV_range: tuple[float, float, int] = (1.0, 800.0, 500), plot: bool = True
) -> Sequence[NDArray]:
    """Compute the delta-over-beta curves for the listed compounds, in the requested energy range.

    Parameters
    ----------
    compounds : Sequence[str]
        Sequence of compounds.
    energy_keV_range : tuple[float, float, int], optional
        The energy range in keV, by default (1.0, 800.0, 500)
    plot : bool, optional
        Whether to plot the result, by default True

    Returns
    -------
    Sequence[NDArray]
        The computed delta-over-beta valueas.
    """
    energies = np.linspace(*energy_keV_range)
    delta_betas_cmps = []
    for cmp in compounds:
        delta_betas_cmp = np.array([get_delta_beta(cmp, e) for e in energies])
        delta_betas_cmps.append(delta_betas_cmp)

    if plot:
        fig, axs = plt.subplots(1, 1)
        for ii, cmp in enumerate(compounds):
            axs.plot(energies, delta_betas_cmps[ii], label=cmp)
        axs.grid()
        axs.legend(fontsize=13)
        axs.set_xlabel("Energy (keV)", fontsize=14)
        axs.set_ylabel("$\\delta/\\beta$", fontsize=14)
        axs.tick_params(labelsize=13)
        fig.tight_layout()
        plt.show(block=False)

    return delta_betas_cmps


def _tie_freq_response(k2: NDArray, dist_um: float, wlength_um: float, delta_beta: float) -> NDArray:
    return 1 + delta_beta * dist_um * wlength_um * np.pi * k2


def _ctf_freq_response(k2: NDArray, dist_um: float, wlength_um: float, delta_beta: float) -> NDArray:
    steps = dist_um * wlength_um * np.pi * k2
    return np.cos(steps) + delta_beta * np.sin(steps)


def plot_filter_responses(
    filter_length: int, pix_size_um: float, dist_um: float, wlength_um: float, delta_beta: float, domain: str = "fourier"
) -> tuple:
    """Plot frequency response of the wave propagation.

    Parameters
    ----------
    filter_length : int
        Length of the filter in pixels
    pix_size_um : float
        Pixel-wise of the detector in microns
    dist_um : float
        Distance of the detector from sample in microns
    wlength_um : float
        Wavelength of the wave in microns
    delta_beta : float
        Radio between the refraction index decrement and the absorption coefficient.
    domain : str
        Whether to plot Fourier or direct-space responses. By default, "Fourier".

    Returns
    -------
    tuple
        Figure and axes of the plot
    """
    k = np.fft.rfftfreq(filter_length, d=pix_size_um)
    k2 = k**2
    tie_resp = _tie_freq_response(k2, dist_um=dist_um, wlength_um=wlength_um, delta_beta=delta_beta)
    ctf_resp = _ctf_freq_response(k2, dist_um=dist_um, wlength_um=wlength_um, delta_beta=delta_beta)
    if domain.lower() == "fourier":
        step = k
    elif domain.lower() == "direct":
        tie_resp = np.fft.fftshift(np.fft.irfft(tie_resp))
        ctf_resp = np.fft.fftshift(np.fft.irfft(ctf_resp))
        p = np.fft.fftfreq(filter_length, d=1.0 / (pix_size_um * filter_length))
        p = np.fft.fftshift(p)
        step = p
    else:
        raise ValueError(f"Unknown domain {domain}, please choose one of 'Fourier' | 'direct'.")

    fig, axs = plt.subplots(1, 1, figsize=[8, 4])
    axs.set_title(f"Domain: {domain}")
    axs.plot(step, tie_resp, label="TIE")
    axs.plot(step, ctf_resp, label="CTF")
    axs.grid()
    axs.legend()
    fig.tight_layout()

    return fig, axs


def get_propagation_filter(
    img_shape: Union[Sequence[int], NDArray],
    pix_size_um: float,
    dist_um: float,
    wlength_um: float,
    delta_beta: float,
    filter_type: str = "ctf",
    use_rfft: bool = False,
    plot_result: bool = False,
    dtype: DTypeLike = np.float32,
) -> tuple[NDArray, NDArray]:
    """Compute the phase contrast propagation filter for the given parameters.

    Parameters
    ----------
    img_shape : Union[Sequence[int], NDArray]
        Shape of the target image
    pix_size_um : float
        Pixel size of the detector (in microns)
    dist_um : float
        Propagation distance (in microns)
    wlength_um : float
        Wavelength of the radiation (in microns)
    delta_beta : float
        Delta-over-beta value for the given material
    filter_type : str, optional
        Type of the filter, by default "ctf".
        Options: "ctf" (Contrast Transfer Function) | "tie" (Transport of Intensity Equation)
    use_rfft : bool, optional
        Whether to use the rfft, by default False
    plot_result : bool, optional
        Whether to plot the result, by default False
    dtype : DTypeLike, optional
        Data type to use, by default np.float32

    Returns
    -------
    tuple[NDArray, NDArray]
        The Fourier-space and real-space filters, respectively

    Raises
    ------
    ValueError
        When choosing an incorrect filter type
    """
    if use_rfft:
        coords = [np.fft.fftfreq(s, d=pix_size_um) for s in img_shape[:-1]] + [np.fft.rfftfreq(img_shape[-1], d=pix_size_um)]
    else:
        coords = [np.fft.fftfreq(s, d=pix_size_um) for s in img_shape]
    coords = np.meshgrid(*coords, indexing="ij")
    k2 = np.sum(np.stack(coords, axis=0) ** 2, axis=0)

    if filter_type.lower() == "ctf":
        filt_fourier = _ctf_freq_response(k2=k2, dist_um=dist_um, wlength_um=wlength_um, delta_beta=delta_beta)
    elif filter_type.lower() == "tie":
        filt_fourier = _tie_freq_response(k2=k2, dist_um=dist_um, wlength_um=wlength_um, delta_beta=delta_beta)
    else:
        raise ValueError(f"Invalid filter type: {filter_type}. Possible choices: 'ctf' | 'tie'")

    fft_axes = tuple(np.arange(-len(img_shape), 0))
    if use_rfft:
        filt_direct = np.fft.irfftn(filt_fourier, s=tuple(img_shape), axes=fft_axes)
    else:
        filt_direct = np.fft.ifftn(filt_fourier, s=tuple(img_shape), axes=fft_axes).real

    if plot_result:
        fig, axs = plt.subplots(
            2, 1, sharex=True, sharey=True, figsize=tuple(np.flip(img_shape) * 12 / np.max(img_shape) * np.array([1, 2]))
        )
        axs[0].imshow(filt_fourier)
        axs[1].imshow(np.fft.fftshift(filt_direct))
        fig.tight_layout()

    return filt_fourier.astype(dtype), np.fft.fftshift(filt_direct.astype(dtype))


def apply_propagation_filter(
    data_wvu: NDArray, pix_size_um: float, dist_um: float, wlength_um: float, delta_beta: float, filter_type: str = "tie"
) -> NDArray:
    """Apply a requested propagation filter to an image or a stack of images.

    Parameters
    ----------
    data_wvu : NDArray
        The Image or stack of images
    pix_size_um : float
        Pixel size of the detector (in microns)
    dist_um : float
        Propagation distance (in microns)
    wlength_um : float
        Wavelength of the radiation (in microns)
    delta_beta : float
        Delta-over-beta value for the given material
    filter_type : str, optional
        Type of the filter, by default "tie".
        Options: "ctf" (Contrast Transfer Function) | "tie" (Transport of Intensity Equation)

    Returns
    -------
    NDArray
        The filtered image
    """
    pad_width = [(0, 0) for _ in range(data_wvu.ndim)]
    pad_size = np.array(data_wvu.shape[-2:])
    pad_width[-2] = pad_size[-2] - pad_size[-2] // 2, pad_size[-2] // 2
    pad_width[-1] = pad_size[-1] - pad_size[-1] // 2, pad_size[-1] // 2
    data_p = np.pad(data_wvu, pad_width=pad_width, mode="edge")
    data_f = np.fft.rfftn(data_p, axes=(-2, -1))

    filt_f, _ = get_propagation_filter(
        data_p.shape[-2:],
        pix_size_um=pix_size_um,
        dist_um=dist_um,
        wlength_um=wlength_um,
        delta_beta=delta_beta,
        filter_type=filter_type,
        use_rfft=True,
    )

    data_f = spimg.fourier_shift(
        data_f,
        shift=(*([0] * (data_wvu.ndim - 2)), pad_size[-2] // 2 - pad_size[-2], pad_size[-1] // 2 - pad_size[-1]),
        n=data_p.shape[-1],
    )
    data_f /= filt_f

    return np.fft.irfftn(data_f, axes=(-2, -1))[..., : data_wvu.shape[-2], : data_wvu.shape[-1]]
