# -*- coding: utf-8 -*-
"""
Post-processing routines.

Created on Tue Mar 24 15:25:14 2020

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import scipy as sp

from typing import List, Sequence, Optional, Tuple
from numpy.typing import ArrayLike, NDArray

from .misc import azimuthal_integration, lines_intersection

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes


eps = np.finfo(np.float32).eps


def com(vol: NDArray, axes: Optional[ArrayLike] = None) -> NDArray:
    """
    Compute center-of-mass for given volume.

    Parameters
    ----------
    vol : NDArray
        The input volume.
    axes : ArrayLike, optional
        Axes on which to compute center-of-mass. The default is None.

    Returns
    -------
    NDArray
        The center-of-mass position.
    """
    if axes is None:
        axes = np.arange(len(vol.shape))
    else:
        axes = np.array(axes, ndmin=1)

    coords = [np.linspace(-(s - 1) / 2, (s - 1) / 2, s) for s in np.array(vol.shape)[list(axes)]]

    num_dims = len(vol.shape)
    com = np.empty((len(axes),))
    for ii, a in enumerate(axes):
        sum_axes = np.array(np.delete(np.arange(num_dims), a), ndmin=1, dtype=int)
        line = np.abs(vol).sum(axis=tuple(sum_axes))
        com[ii] = line.dot(coords[ii]) / line.sum()

    return com


def frc(
    img1: NDArray,
    img2: NDArray,
    snrt: float = 0.2071,
    axes: Optional[Sequence[int]] = None,
    smooth: Optional[int] = 5,
    supersampling: int = 1,
) -> Tuple[NDArray, NDArray]:
    """
    Compute the FRC/FSC (Fourier ring/shell correlation) between two images / volumes.

    Please refer to the following article for more information:
        M. van Heel and M. Schatz, “Fourier shell correlation threshold criteria,”
        J. Struct. Biol., vol. 151, no. 3, pp. 250–262, Sep. 2005.

    Parameters
    ----------
    img1 : NDArray
        First image / volume.
    img2 : NDArray
        Second image / volume.
    snrt : float, optional
        SNR to be used for generating the threshold curve for resolution definition.
        The SNR value of 0.4142 corresponds to the hald-bit curve for a full dataset.
        When splitting datasets in two sub-datasets, that value needs to be halved.
        The default is 0.2071, which corresponds to the half-bit threashold for half dataset.
    axes : Sequence[int], optional
        The axes over which we want to compute the FRC/FSC.
        If None, all axes will be used The default is None.
    smooth : Optional[int], optional
        Size of the Hann smoothing window. The default is 5.
    supersampling : int, optional
        Supersampling factor of the images.
        Larger values increase the high-frequency range of the FRC/FSC function.
        The default is 1, which corresponds to the Nyquist frequency.

    Raises
    ------
    ValueError
        Error returned when not passing images of the same shape.

    Returns
    -------
    NDArray
        The computed FRC/FSC.
    NDArray
        The threshold curve corresponding to the given threshod SNR.
    """
    img1_shape = np.array(img1.shape)

    if axes is None:
        axes = list(np.arange(-len(img1_shape), 0))

    if img2 is None:
        if np.any(img1_shape[axes] % 2 == 1):
            raise ValueError(f"Image shape {img1_shape} along the chosen axes {axes} needs to be even.")
        raise NotImplementedError("Self FRC not implemented, yet.")
    else:
        img2_shape = np.array(img2.shape)
        if len(img1_shape) != len(img2_shape) or np.any(img1_shape != img2_shape):
            raise ValueError(
                f"Image #1 size {img1_shape} and image #2 size {img2_shape} are different, while they should be equal."
            )

    if supersampling > 1:
        base_grid = [np.linspace(-(d - 1) / 2, (d - 1) / 2, d) for d in img1_shape]

        interp_grid = [np.linspace(-(d - 1) / 2, (d - 1) / 2, d) for d in img1_shape]
        for a in axes:
            d = img1_shape[a] * 2
            interp_grid[a] = np.linspace(-(d - 1) / 4, (d - 1) / 4, d)
        interp_grid = np.meshgrid(*interp_grid, indexing="ij")
        interp_grid = np.transpose(interp_grid, [*range(1, len(img1_shape) + 1), 0])

        img1 = sp.interpolate.interpn(base_grid, img1, interp_grid, bounds_error=False, fill_value=None)
        img2 = sp.interpolate.interpn(base_grid, img2, interp_grid, bounds_error=False, fill_value=None)

        img1_shape = np.array(img1.shape)

    axes_shape = img1_shape[list(axes)]
    cut_off = np.min(axes_shape) // 2

    img1_f = np.fft.fftn(img1, axes=axes)
    img2_f = np.fft.fftn(img2, axes=axes)

    fc = img1_f * np.conj(img2_f)
    f1 = np.abs(img1_f) ** 2
    f2 = np.abs(img2_f) ** 2

    fc_r_int = azimuthal_integration(fc.real, axes=axes, domain="fourier")
    fc_i_int = azimuthal_integration(fc.imag, axes=axes, domain="fourier")
    fc_int = np.sqrt((fc_r_int**2) + (fc_i_int**2))
    f1_int = azimuthal_integration(f1, axes=axes, domain="fourier")
    f2_int = azimuthal_integration(f2, axes=axes, domain="fourier")

    f1s_f2s = f1_int * f2_int
    f1s_f2s = f1s_f2s + (f1s_f2s == 0)
    frc = fc_int / np.sqrt(f1s_f2s)

    rings_size = azimuthal_integration(np.ones_like(img1), axes=axes, domain="fourier")
    # Alternatively:
    # # The number of pixels in a ring is given by the surface.
    # # We compute the n-dimensional hyper-sphere surface, where n is given by the number of axes.
    # n = len(axes)
    # num_surf = 2 * np.pi ** (n / 2)
    # den_surf = sp.special.gamma(n / 2)
    # rings_size = np.concatenate(((1.0, ), num_surf / den_surf * np.arange(1, len(frc)) ** (n - 1)))

    Tnum = snrt + (2 * np.sqrt(snrt) + 1) / np.sqrt(rings_size)
    Tden = snrt + 1 + 2 * np.sqrt(snrt) / np.sqrt(rings_size)
    Thb = Tnum / Tden

    if smooth is not None and smooth > 1:
        win = sp.signal.windows.hann(smooth)
        win /= np.sum(win)
        frc = sp.ndimage.convolve(frc, win, mode="nearest")

    return frc[..., :cut_off], Thb[..., :cut_off]


def plot_frcs(
    volume_pairs: Sequence[Tuple[NDArray, NDArray]],
    labels: Sequence[str],
    title: Optional[str] = None,
    smooth: Optional[int] = 5,
    snrt: float = 0.2071,
) -> Tuple[Figure, Axes]:
    """Compute and plot the FSCs / FRCs of some volumes.

    Parameters
    ----------
    volume_pairs : Sequence[Tuple[NDArray, NDArray]]
        A list of pairs of volumes to compute the FRCs on.
    labels : Sequence[str]
        The labels associated with each pair.
    title : Optional[str], optional
        The axes title, by default None.
    smooth : Optional[int], optional
        The size of the smoothing window for the computed curves, by default 5.
    snrt : float, optional
        The SNR of the T curve, by default 0.2071 - as per half-dataset SNR.
    """
    frcs = [np.array([])] * len(volume_pairs)
    xps: List[Optional[Tuple[float, float]]] = [(0.0, 0.0)] * len(volume_pairs)

    for ii, pair in enumerate(volume_pairs):
        frcs[ii], T = frc(pair[0], pair[1], snrt=snrt, smooth=smooth)
        xps[ii] = lines_intersection(frcs[ii], T, x_lims=(1, None))

    nyquist = len(frcs[0])
    xx = np.linspace(0, 1, nyquist)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    for f, l in zip(frcs, labels):
        ax.plot(xx, np.squeeze(f), label=l)
    ax.plot(xx, np.squeeze(T), label="T 1/2 bit", linestyle="dashed")
    for ii, p in enumerate(xps):
        if p is not None:
            res = p[0] / (nyquist - 1)
            ax.stem(res, p[1], label=f"Resolution ({labels[ii]}): {res:.3}", linefmt=f"C{ii}-.", markerfmt=f"C{ii}o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend()
    ax.grid()
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Spatial frequency / Nyquist")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    plt.show(block=False)

    return fig, ax
