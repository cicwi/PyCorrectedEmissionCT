# -*- coding: utf-8 -*-
"""
Pre-processing routines.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

import skimage.transform as skt

from typing import Sequence, Optional, Union, Tuple
from numpy.typing import DTypeLike, NDArray

import matplotlib.pyplot as plt


eps = np.finfo(np.float32).eps


def pad_sinogram(
    sinogram: NDArray, width: Union[int, Sequence[int], NDArray], pad_axis: int = -1, mode: str = "edge", **kwds
) -> NDArray:
    """
    Pad the sinogram.

    Parameters
    ----------
    sinogram : NDArray
        The sinogram to pad.
    width : Union[int, Sequence[int]]
        The width of the padding. Normally, it should either be an int or a tuple(int, int).
    pad_axis : int, optional
        The axis to pad. The default is -1.
    mode : str, optional
        The padding type (from numpy.pad). The default is "edge".
    **kwds :
        The numpy.pad arguments.

    Returns
    -------
    NDArray
        The padded sinogram.
    """
    width = np.array(width, ndmin=1)

    pad_size = np.zeros((len(sinogram.shape), len(width)), dtype=int)
    pad_size[pad_axis, :] = width

    return np.pad(sinogram, pad_size, mode=mode.lower(), **kwds)  # type: ignore


def apply_flat_field(
    projs: NDArray,
    flats: NDArray,
    darks: Optional[NDArray] = None,
    crop: Optional[Sequence[int]] = None,
    dtype: DTypeLike = np.float32,
) -> NDArray:
    """
    Apply flat field.

    Parameters
    ----------
    projs : NDArray
        Projections.
    flats : NDArray
        Flat fields.
    darks : Optional[NDArray], optional
        Dark noise. The default is None.
    crop : Optional[Sequence[int]], optional
        Crop region. The default is None.
    dtype : DTypeLike, optional
        Data type of the processed data. The default is np.float32.

    Returns
    -------
    NDArray
        Falt-field corrected and linearized projections.
    """
    projs = np.ascontiguousarray(projs, dtype=dtype)
    flats = np.ascontiguousarray(flats, dtype=dtype)

    if crop is not None:
        projs = projs[..., crop[0] : crop[2], crop[1] : crop[3]]
        flats = flats[..., crop[0] : crop[2], crop[1] : crop[3]]
        if darks is not None:
            darks = darks[..., crop[0] : crop[2], crop[1] : crop[3]]

    if darks is not None:
        darks = np.ascontiguousarray(darks, dtype=dtype)
        projs = projs - darks
        flats = flats - darks

    flats = np.mean(flats, axis=0)

    return projs / flats


def apply_minus_log(projs: NDArray) -> NDArray:
    """
    Apply -log.

    Parameters
    ----------
    projs : NDArray
        Projections.

    Returns
    -------
    NDArray
        Linearized projections.
    """
    return np.fmax(-np.log(projs), 0.0)


def rotate_proj_stack(data_vwu: NDArray, rot_angle_deg: float) -> NDArray:
    """
    Rotate the projection stack.

    Parameters
    ----------
    data_vwu : NDArray
        The projection stack, with dimensions [v, w, u] (vertical, omega / sample rotation, horizontal).
    rot_angle_deg : float
        The rotation angle in degrees.

    Returns
    -------
    NDArray
        The rotated projection stack.
    """
    data_vwu_r = np.empty_like(data_vwu)
    for ii in range(data_vwu.shape[-2]):
        data_vwu_r[:, ii, :] = skt.rotate(data_vwu[:, ii, :], -rot_angle_deg, clip=False)
    return data_vwu_r


def bin_imgs(imgs: NDArray, binning: Union[int, float], verbose: bool = True) -> NDArray:
    """Bin a stack of images.

    Parameters
    ----------
    imgs : NDArray
        The stack of images.
    binning : int | float
        The binning factor.
    verbose : bool, optional
        Whether to print the image shapes, by default True

    Returns
    -------
    NDArray
        The binned images
    """
    imgs_shape = imgs.shape

    if isinstance(binning, int):
        binned_shape = (*imgs_shape[:-2], imgs_shape[-2] // binning, imgs_shape[-1] // binning)
        imgs = imgs.reshape([*binned_shape[:-1], binning, binned_shape[-1], binning])
        imgs = imgs.mean(axis=(-3, -1))
    else:
        imgs = imgs.reshape([-1, *imgs_shape[-2:]])
        imgs = skt.rescale(imgs, 1 / binning, channel_axis=0)
        binned_shape = [*imgs_shape[:-2], *imgs.shape[-2:]]
        imgs = imgs.reshape(binned_shape)

    if verbose:
        print(f"Binning: {imgs_shape} => {binned_shape}")

    return imgs


def compute_eigen_flats(
    trans: NDArray, flats: NDArray, darks: NDArray, ndim: int = 2, plot: bool = False
) -> Tuple[NDArray, NDArray]:
    """Compute the eigen flats of a stack of transmission images.

    Parameters
    ----------
    trans : NDArray
        The stack of transmission images.
    flats : NDArray
        The flats without sample.
    darks : NDArray
        The darks.
    ndim : int, optional
        The number of dimensions of the images, by default 2
    plot : bool, optional
        Whether to plot the results, by default False

    Returns
    -------
    Tuple[NDArray, NDArray]
        The decomposition of the tranmissions of the sample and the flats.
    """
    trans_shape = trans.shape
    trans_num = np.prod(trans_shape[:-ndim])
    img_shape = trans_shape[-ndim:]

    darks_num = np.prod(darks.shape[:-ndim])

    imgs = np.concatenate((trans.reshape([-1, *img_shape]), flats.reshape([-1, *img_shape]), darks.reshape([-1, *img_shape])))
    imgs = imgs.reshape([-1, np.prod(img_shape)]).transpose()

    U, s, Vh = np.linalg.svd(imgs, full_matrices=False)

    last_t_ind = -darks_num + 1

    t: NDArray = np.matmul(U[..., 1:last_t_ind] * s[..., None, 1:last_t_ind], Vh[..., 1:last_t_ind, :])
    t = t.transpose().reshape([-1, *img_shape])[:trans_num]

    f: NDArray = np.matmul(U[..., 0:1:] * s[..., None, 0:1:], Vh[..., 0:1:, :])
    f = f.transpose().reshape([-1, *img_shape])[:trans_num]

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=[10, 3.75])
        ax[0].plot(s)
        ax[0].grid()
        ax[0].set_title("Singular values")
        ax[1].imshow(U[:, 0].reshape(img_shape))
        ax[1].set_title("Highest value component")
        ax[2].imshow(U[:, last_t_ind:].mean(axis=1).reshape(img_shape))
        ax[2].set_title("Noise average")
        fig.tight_layout()
        plt.show(block=False)

    return t.reshape(trans_shape), f.reshape(trans_shape)
