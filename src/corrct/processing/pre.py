"""
Pre-processing routines.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from collections.abc import Sequence
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.ndimage as spimg
import skimage.transform as skt
from numpy.polynomial import Polynomial
from numpy.typing import DTypeLike, NDArray
from skimage.measure import block_reduce


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
    projs_wvu: NDArray,
    flats_wvu: NDArray,
    darks_wvu: Optional[NDArray] = None,
    crop: Union[NDArray, Sequence[int], None] = None,
    cap_intensity: Optional[float] = None,
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
    cap_intensity: float | None, optional
        Cap the intensity to a given value. The default is None.
    dtype : DTypeLike, optional
        Data type of the processed data. The default is np.float32.

    Returns
    -------
    NDArray
        Flat-field corrected and linearized projections.
    """
    projs_wvu = np.ascontiguousarray(projs_wvu, dtype=dtype)
    flats_wvu = np.ascontiguousarray(flats_wvu, dtype=dtype)

    if crop is not None:
        projs_wvu = projs_wvu[..., crop[0] : crop[2], crop[1] : crop[3]]
        flats_wvu = flats_wvu[..., crop[0] : crop[2], crop[1] : crop[3]]
        if darks_wvu is not None:
            darks_wvu = darks_wvu[..., crop[0] : crop[2], crop[1] : crop[3]]

    if darks_wvu is not None:
        darks_wvu = np.ascontiguousarray(darks_wvu, dtype=dtype)
        projs_wvu = projs_wvu - darks_wvu
        flats_wvu = flats_wvu - darks_wvu

    if flats_wvu.ndim == 3:
        flats_wvu = np.mean(flats_wvu, axis=0)

    projs_wvu = projs_wvu / flats_wvu

    if cap_intensity is not None:
        projs_wvu = np.fmin(projs_wvu, cap_intensity)

    return projs_wvu


def apply_minus_log(projs: NDArray, lower_limit: float = -np.inf) -> NDArray:
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
    return np.fmax(-np.log(projs), lower_limit)


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


def shift_proj_stack(data_vwu: NDArray, shifts: NDArray, use_fft: bool = False) -> NDArray:
    """Shift each projection in a stack of projections, by projection dependent shifts.

    Parameters
    ----------
    data_vwu : NDArray
        The projection stack
    shifts : NDArray
        The shifts
    use_fft : bool, optional
        Whether to use fft shift or not, by default False

    Returns
    -------
    NDArray
        The shifted stack
    """
    new_data = np.empty_like(data_vwu)
    for ii in range(data_vwu.shape[-2]):
        if use_fft:
            img = data_vwu[..., ii, :]
            img_f = np.fft.rfftn(img)
            img_f = spimg.fourier_shift(img_f, shifts[..., ii], n=img.shape[-1])
            new_data[..., ii, :] = np.fft.irfftn(img_f)
        else:
            new_data[..., ii, :] = spimg.shift(data_vwu[..., ii, :], shifts[..., ii], order=1, mode="nearest")

    return new_data


def bin_imgs(
    imgs: NDArray, binning: Union[int, float], axes: Sequence[int] = (-2, -1), auto_crop: bool = False, verbose: bool = True
) -> NDArray:
    """Bin a stack of images.

    Parameters
    ----------
    imgs : NDArray
        The stack of images.
    binning : int | float
        The binning factor.
    auto_crop : bool, optional
        Whether to automatically crop the images to match, by default False
    verbose : bool, optional
        Whether to print the image shapes, by default True

    Returns
    -------
    NDArray
        The binned images
    """
    if auto_crop:
        imgs_shape = np.array(imgs.shape)
        excess_pixels = (imgs_shape[list(axes)] % binning).astype(int)
        crop_vu = (excess_pixels - excess_pixels // 2, imgs_shape[list(axes)] - excess_pixels // 2)
        slicing = [slice(None)] * len(imgs_shape)
        for ii, ax in enumerate(axes):
            if excess_pixels[ii] > 0:
                slicing[ax] = slice(crop_vu[0][ii], crop_vu[1][ii])
        imgs = imgs[tuple(slicing)]

        if verbose:
            print(f"Auto-cropping {crop_vu}: {imgs_shape} => {imgs.shape}")

    imgs_shape = imgs.shape

    if isinstance(binning, int):
        binning_shape = np.ones_like(imgs_shape)
        for ax in axes:
            binning_shape[ax] = binning
        imgs = block_reduce(imgs, tuple(binning_shape), np.mean)
        binned_shape = imgs.shape
    else:
        imgs = imgs.reshape([-1, *imgs_shape[-2:]])
        imgs = skt.rescale(imgs, 1 / binning, channel_axis=0 if len(imgs_shape) > 2 else None)
        binned_shape = [*imgs_shape[:-2], *imgs.shape[-2:]]
        imgs = imgs.reshape(binned_shape)

    if verbose:
        print(f"Binning {binning}: {imgs_shape} => {binned_shape}")

    return imgs


def background_from_margin(
    data_vwu: NDArray, margin: Union[int, Sequence[int], NDArray[np.integer]] = 4, poly_order: int = 0, plot: bool = False
) -> NDArray:
    """Compute background of the projection data, from the margins of the projections.

    Parameters
    ----------
    data_vwu : NDArray
        The projection data in the format [V]WU.
    margin : int | Sequence[int] | NDArray[np.integer], optional
        The size of the margin, by default 4
    poly_order : int, optional
        The order of the interpolation polynomial, by default 0

    Returns
    -------
    NDArray
        The computed background.

    Raises
    ------
    NotImplementedError
        Different margins per line are not supported, at the moment.
    ValueError
        In case the margins ar larger than the image size in U.
    """
    data_shape_u = data_vwu.shape[-1]
    data_shape_w = data_vwu.shape[-2]
    margin = np.array(margin, dtype=int, ndmin=1)
    if margin.size == 1:
        margin = np.tile(margin, [*np.ones(margin.ndim - 1), 2])
    if margin.ndim > 1:
        raise NotImplementedError("Complex masks support has not been implemented, yet.")
    if margin.sum() > data_shape_u:
        raise ValueError(f"Margin size ({margin}) should be smaller than the image size in U ({data_shape_u})")
    if poly_order > 0 and np.any(margin == 0):
        print("WARNING: parameter `poly_order` cannot be greater than 0 if one of the margins is 0")
        poly_order = 0

    if poly_order > 0:
        ydata = np.concatenate([data_vwu[..., : margin[0]], data_vwu[..., -margin[1] :]], axis=-1)
        xdata = np.concatenate([np.arange(0, margin[0]), np.arange(data_shape_u - margin[1], data_shape_u)])

        if data_vwu.ndim > 2:
            ydata = ydata.mean(axis=-3)

        background = np.empty([data_shape_w, data_shape_u], dtype=data_vwu.dtype)
        for ii_w in range(data_shape_w):
            poly = Polynomial.fit(xdata, ydata[ii_w], deg=poly_order)
            background[ii_w, :] = poly(np.arange(data_shape_u))

        if plot:
            fig, axs = plt.subplots(1, 1)
            axs.plot(background[0])
            axs.scatter(xdata, ydata[0])
            axs.grid()
            axs.set_ylim(0)
            fig.tight_layout()
            plt.show(block=False)

        if data_vwu.ndim > 2:
            background = np.tile(background[None, ...], [data_vwu.shape[-3], 1, 1])
        return background
    else:
        sum_vals = data_vwu[..., : margin[0]].sum(axis=-1) + data_vwu[..., -margin[1] :].sum(axis=-1)
        background: NDArray = sum_vals / margin.sum(axis=-1)

        if data_vwu.ndim > 2:
            background = background.mean(axis=-2, keepdims=True)

        return np.tile(background[..., None], [*np.ones(background.ndim, dtype=int), data_shape_u])


def destripe_wlf_vwu(
    data: NDArray,
    sigma: float = 0.005,
    level: int = 1,
    wavelet: str = "bior2.2",
    angle_axis: int = -2,
    other_axes: Union[Sequence[int], NDArray, None] = None,
) -> NDArray:
    """Remove stripes from sinogram, using the Wavelet-Fourier method.

    Parameters
    ----------
    data : NDArray
        The data to de-stripe
    sigma : float, optional
        Fourier space filter coefficient, by default 0.005
    level : int, optional
        The wavelet level to use, by default 1
    wavelet : str, optional
        The type of wavelet to use, by default "bior2.2"
    angle_axis : int, optional
        The axis of the Fourier transform, by default -2
    other_axes : Union[Sequence[int], NDArray, None], optional
        The axes of the wavelet decomposition, by default None

    Returns
    -------
    NDArray
        The de-striped data.
    """
    if other_axes is None:
        other_axes = np.arange(-data.ndim, 0)
    else:
        other_axes = np.array(other_axes)

    if angle_axis is other_axes:
        other_axes = np.delete(other_axes, angle_axis)

    level_power = 2**level

    data_shape = np.array(data.shape)
    target_shape = data_shape.copy()
    target_shape[list(other_axes)] = np.ceil(data_shape[list(other_axes)] / level_power) * level_power
    diff_size = target_shape - data_shape
    padding = np.stack((diff_size - diff_size // 2, diff_size // 2), axis=-1)
    data = np.pad(data, pad_width=padding, mode="edge")

    coeffs = pywt.swtn(data, wavelet=wavelet, axes=other_axes, level=level)
    for ii_l in range(level):
        for wl_label, coeffs_l_wl in coeffs[ii_l].items():
            if wl_label == "a" * len(other_axes):
                continue
            coeff_f = np.fft.rfft(coeffs_l_wl, axis=angle_axis)
            filt_f = 1 - np.exp(-(np.fft.rfftfreq(coeffs_l_wl.shape[angle_axis]) ** 2) / (2 * sigma**2))
            coeff_f *= filt_f[:, None]
            coeffs[ii_l][wl_label] = np.fft.irfft(coeff_f, axis=angle_axis, n=coeffs_l_wl.shape[angle_axis])

    data = pywt.iswtn(coeffs, wavelet=wavelet, axes=other_axes)
    slicing = [slice(padding[ii, 0], data.shape[ii] - padding[ii, 1]) for ii in range(data.ndim)]

    return data[tuple(slicing)]


def compute_eigen_flats(
    trans_wvu: NDArray,
    flats_wvu: Optional[NDArray] = None,
    darks_wvu: Optional[NDArray] = None,
    ndim: int = 2,
    plot: bool = False,
) -> tuple[NDArray, NDArray]:
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
        The decomposition of the transmissions of the sample and the flats.
    """
    trans_shape = trans_wvu.shape
    trans_num = np.prod(trans_shape[:-ndim])
    img_shape = trans_shape[-ndim:]

    stack_imgs = [trans_wvu.reshape([-1, *img_shape])]

    if flats_wvu is not None:
        stack_imgs.append(flats_wvu.reshape([-1, *img_shape]))

    stack_imgs = np.concatenate(stack_imgs)

    if darks_wvu is not None:
        if darks_wvu.ndim > 2:
            darks_wvu = darks_wvu.mean(axis=tuple(np.arange(darks_wvu.ndim - 2)))
        stack_imgs = np.fmax(stack_imgs - darks_wvu, np.finfo(np.float32).eps)

    stack_imgs = stack_imgs.reshape([-1, np.prod(img_shape)]).transpose()
    stack_imgs = np.log(stack_imgs)

    mat_u, sigma, mat_v_h = np.linalg.svd(stack_imgs, full_matrices=False)

    eigen_projs: NDArray = (mat_u[..., 1:] * sigma[..., None, 1:]) @ mat_v_h[..., 1:, :]
    eigen_projs = np.exp(eigen_projs)
    eigen_projs = eigen_projs.transpose().reshape([-1, *img_shape])[:trans_num]

    eigen_flats: NDArray = (mat_u[..., 0:1:] * sigma[..., None, 0:1:]) @ mat_v_h[..., 0:1:, :]
    eigen_flats = np.exp(eigen_flats)
    eigen_flats = eigen_flats.transpose().reshape([-1, *img_shape])[:trans_num]

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=[10, 3.75])
        axs[0].plot(sigma)
        axs[0].grid()
        axs[0].set_title("Singular values")
        axs[1].imshow(mat_u[:, 0].reshape(img_shape))
        axs[1].set_title("Highest value component")
        axs[2].plot(eigen_flats.mean(axis=(-2, -1)).flatten())
        axs[2].grid()
        axs[2].set_title("Eigen intensities")
        fig.tight_layout()
        plt.show(block=False)

    return eigen_projs.reshape(trans_shape), eigen_flats.reshape(trans_shape)
