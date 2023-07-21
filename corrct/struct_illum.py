#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provide structured illumination support.

Created on Sun Jan  9 17:39:02 2022

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
from scipy import linalg as spalg

from numpy.typing import NDArray, DTypeLike
from typing import Union, Sequence, Optional, Tuple

from abc import abstractmethod, ABC

import matplotlib.pyplot as plt

from . import operators

import copy as cp

NDArrayInt = NDArray[np.integer]


def reorder_masks(masks: NDArray, buckets: NDArray, shift: int) -> Tuple[NDArray, NDArray]:
    """Reorder masks, with a simple rot-n algorithm.

    Parameters
    ----------
    masks : NDArray
        The masks to re-order.
    buckets : NDArray
        The corresponding buckets.
    shift : int
        The length of the shift.

    Returns
    -------
    Tuple[NDArray, NDArray]
        The reordered masks and buckets.
    """
    masks_shape = masks.shape
    masks = masks.reshape([-1, *masks_shape[-2:]])

    masks = np.roll(masks, -shift, axis=0)
    buckets = np.roll(buckets, -shift)

    return masks.reshape(masks_shape), buckets


def decompose_qr_masks(masks: NDArray, verbose: bool = False) -> Tuple[NDArray, NDArray]:
    """Compute QR decomposition of the given masks.

    Parameters
    ----------
    masks : NDArray
        The masks to decompose
    verbose : bool, optional
        Whether to emite verbose output, by default False

    Returns
    -------
    Tuple[NDArray, NDArray]
        The Q and R components.
    """
    masks_shape = masks.shape
    masks = masks.reshape([-1, np.prod(masks_shape[-2:])])

    if verbose:
        print("Computing QR decomposition")
    Q, R = np.linalg.qr(masks.transpose())

    Rt = R.transpose()
    Qt = Q.transpose()
    if verbose:
        print("Computing inversion of R (fast, because triangular)")
    R1t = np.linalg.inv(Rt)

    return Qt.reshape(masks_shape), R1t


class MaskCollection:
    """Define mask collection class."""

    masks_enc: NDArray
    masks_dec: NDArray

    mask_dims: int
    mask_support: NDArrayInt

    mask_type: str

    def __init__(
        self,
        masks_enc: NDArray,
        masks_dec: Optional[NDArray] = None,
        mask_dims: int = 2,
        mask_type: str = "measured",
        mask_support: Union[None, Sequence[int], NDArrayInt] = None,
    ) -> None:
        """Initialize mask collection.

        Parameters
        ----------
        masks_enc : NDArray
            The encoding masks.
        masks_dec : NDArray | None, optional
            The decoding masks. The default (None) will assume them identical to the encoding masks.
        mask_dims : int, optional
            The dimensions of a single mask. The feault is 2.
        mask_type : str, optional
            The type of masks. The default is "measured".
        mask_support : Sequence[int] | NDArray[np.integer] | None, optional
            The extent of the mask support in pixels. The default (None) will consider it equal to the FoV.
        """
        self.mask_dims = mask_dims

        # TODO: check sizes of masks
        self.masks_enc = masks_enc
        self.masks_dec = masks_dec if masks_dec is not None else masks_enc
        # TODO: check the decoding masks are consistent with the encoding masks

        self.mask_type = mask_type

        # TODO: check support is honored
        if mask_support is not None:
            self.mask_support = np.array(mask_support, ndmin=1, dtype=int)
        else:
            self.mask_support = np.array(self.shape_FoV, ndmin=1, dtype=int)

    @property
    def shape_FoV(self) -> Sequence[int]:
        """Return the mask shape.

        Returns
        -------
        Sequence[int]
            The mask shape.
        """
        return self.masks_enc.shape[-self.mask_dims :]

    @property
    def shape_shifts(self) -> Sequence[int]:
        """Compute the shape of the available shifts.

        Returns
        -------
        Sequence[int]
            The shape of the available shifts.
        """
        return self.masks_enc.shape[: -self.mask_dims]

    @property
    def num_buckets(self) -> int:
        """Compute the total number of available buckets.

        Returns
        -------
        int
            The total number of buckets.
        """
        return int(np.prod(self.shape_shifts))

    def info(self) -> str:
        """
        Return the mask info.

        Returns
        -------
        str
            Mask info string.
        """
        return self.mask_type

    def upper(self) -> str:
        """
        Return the upper case name of the mask.

        Returns
        -------
        str
            Upper case string name of the mask.
        """
        return self.mask_type.upper()

    def lower(self) -> str:
        """
        Return the lower case name of the mask.

        Returns
        -------
        str
            Lower case string name of the mask.
        """
        return self.mask_type.lower()

    def get_mask(self, mask_inds_vu: Union[Sequence, NDArray], mask_encoding: bool = True) -> NDArray:
        """Return the requested mask.

        Parameters
        ----------
        mask_inds_vu : Union[Sequence, NDArray]
            The mask position.
        mask_encoding : bool, optional
            Whether it is an encoding or decoding mask, by default True

        Returns
        -------
        NDArray
            The requested mask.
        """
        if mask_encoding is True:
            mask = self.masks_enc
        else:
            mask = self.masks_dec

        return mask[tuple(mask_inds_vu)]

    def get_QR_decomposition(self, buckets: NDArray, shift: int = 0) -> Tuple["MaskCollection", NDArray]:
        """Compute and return the QR decomposition of the masks.

        Parameters
        ----------
        buckets : NDArray
            The buckets corresponding to the masks.
        shift : int, optional
            Index of the first mask, by default 0.

        Returns
        -------
        Tuple[MaskCollection, NDArray]
            The new mask collection and the matrix for modifying buckets accordingly.

        Raises
        ------
        ValueError
            In case the masks have encoding-decoding form.
        """
        if np.any(self.masks_dec != self.masks_enc):
            raise ValueError("Cannot compute QR decomposition of masks that have encoding and decoding forms.")

        new_masks = cp.deepcopy(self)
        new_buckets = cp.deepcopy(buckets)
        if shift != 0:
            new_masks.masks_enc, new_buckets = reorder_masks(new_masks.masks_enc, new_buckets, shift=shift)

        new_masks.masks_enc, R1t = decompose_qr_masks(new_masks.masks_enc)
        new_masks.masks_dec = new_masks.masks_enc
        new_buckets = R1t.dot(new_buckets)

        return new_masks, new_buckets

    def inspect_masks(self, mask_inds_vu: Union[None, Sequence[int], NDArrayInt] = None):
        """Inspect the encoding and decoding masks at the requested shifts.

        Parameters
        ----------
        mask_inds_vu : Sequence[int] | NDArray[np.integer] | None, optional
            The requested axes shifts. The default (None) is the first mask.
        """
        if mask_inds_vu is None:
            mask_inds_vu = [0] * len(self.shape_shifts)

        mask_enc = self.get_mask(mask_inds_vu, mask_encoding=True)
        mask_dec = self.get_mask(mask_inds_vu, mask_encoding=False)

        f, axs = plt.subplots(1, 2)
        f.suptitle("Base masks")

        axs[0].imshow(mask_enc)
        axs[0].set_title("Encoding")
        axs[1].imshow(mask_dec)
        axs[1].set_title("Decoding")

        plt.show(block=False)


class MaskGenerator(ABC):
    """Define mask generation interface."""

    shape_FoV: NDArrayInt
    shape_mask: NDArrayInt
    shape_shifts: NDArrayInt
    transmittance: float
    dtype: DTypeLike

    _enc_dec_mismatch: bool

    __mask_name__ = "generated"

    def __init__(
        self,
        shape_FoV: Union[Sequence[int], NDArrayInt],
        shape_mask: Union[Sequence[int], NDArrayInt],
        shape_shifts: Union[Sequence[int], NDArrayInt],
        transmittance: float = 1.0,
        dtype: DTypeLike = np.float32,
    ) -> None:
        """Initialize mask collection.

        Parameters
        ----------
        shape_FoV : Sequence[int] | NDArray[np.integer]
            The shape of the field-of-view.
        shape_mask : Sequence[int] | NDArray[np.integer]
            The shape of the masks.
        shape_shifts : Sequence[int] | NDArray[np.integer]
            The shape of the shifts to generate.
        transmittance : float
            The maximum transmittance of the structuring elements.
        dtype : DTypeLike
            The dtype of the created masks.
        """
        self.shape_FoV = np.array(shape_FoV, dtype=int)
        self.shape_mask = np.array(shape_mask, dtype=int)
        self.shape_shifts = np.array(shape_shifts, dtype=int)

        self.transmittance = transmittance

        self.dtype = dtype

        self._enc_dec_mismatch = False

    def info(self) -> str:
        """
        Return the mask info.

        Returns
        -------
        str
            Mask info string.
        """
        return self.__mask_name__

    def __repr__(self) -> str:
        """Produce the string representation of the object.

        Returns
        -------
        str
            The string representation.
        """
        return self.__class__.__name__ + " {\n" + ",\n".join([f"  {k} = {v}" for k, v in self.__dict__.items()]) + "\n}"

    @property
    def num_buckets(self) -> int:
        """Compute the number of buckets.

        Returns
        -------
        int
            The number of buckets.
        """
        return int(np.prod(self.shape_shifts))

    def _init_FoV_mm(self, FoV_size_mm: Union[float, Sequence[float], NDArray], req_res_mm: float) -> NDArrayInt:
        self.FoV_size_mm = np.array(FoV_size_mm, ndmin=1)
        num_points = np.ceil(self.FoV_size_mm / req_res_mm).astype(int)
        self.feature_size_mm = self.FoV_size_mm / num_points

        return num_points

    def generate_collection(self, buckets_fraction: float = 1, shift_type: str = "sequential") -> MaskCollection:
        """Generate the mask collection.

        Parameters
        ----------
        buckets_fraction : float, optional
            The fraction of buckets to generate, by default 1
        shift_type : str, optional
            The type of shift to implement, by default "sequential"
        abs_fraction : float, optional
            The attenuation fraction of the pixels

        Returns
        -------
        MaskCollection
            The generated mask collection.

        Raises
        ------
        ValueError
            In case of wrong shift type.
        """
        if shift_type.lower() == "random":
            num_chosen_buckets = np.ceil(self.num_buckets * buckets_fraction).astype(int)
            disp_v, disp_u = self.get_random_shifts(num_chosen_buckets)
        elif shift_type.lower() == "interval":
            interval = np.ceil(1 / buckets_fraction).astype(int)
            disp_v, disp_u = self.get_interval_shifts(interval)
            num_chosen_buckets = len(disp_v)
        elif shift_type.lower() == "sequential":
            num_chosen_buckets = np.ceil(self.num_buckets * buckets_fraction).astype(int)
            disp_v, disp_u = self.get_sequential_shifts(num_chosen_buckets)
        else:
            raise ValueError('Wrong shift_type: "%s". Available options: {random} | interval | sequential.' % shift_type)

        gen_masks_enc = self._generate_masks(disp_v, disp_u, mask_encoding=True)
        gen_masks_dec = self._generate_masks(disp_v, disp_u, mask_encoding=False) if self._enc_dec_mismatch else None

        print("Using %d masks over %d" % (num_chosen_buckets, self.num_buckets))

        return MaskCollection(
            gen_masks_enc,
            gen_masks_dec,
            mask_type=self.__mask_name__,
            mask_support=self.shape_mask,
            mask_dims=len(self.shape_FoV),
        )

    def _apply_transmission(self, masks: NDArray) -> NDArray:
        return 1 - (1 - masks) * self.transmittance

    @abstractmethod
    def generate_mask(self, mask_inds_vu: Union[Sequence, NDArray], mask_encoding: bool = True) -> NDArray:
        """Produce the shifted masks.

        Parameters
        ----------
        mask_inds_vu : tuple | list | NDArray
            The vertical and horizontal shifts.
        mask_encoding : bool, optional
            Is the mask encoding (False = decoding). The default is True.

        Returns
        -------
        NDArray
            The shifted mask.
        """

    def _generate_masks(
        self, shifts_v: Union[Sequence, NDArray], shifts_u: Union[Sequence, NDArray], mask_encoding: bool = True
    ) -> NDArray:
        """Produce all the masks.

        Parameters
        ----------
        shifts_v : tuple | list | NDArray
            List of all the vertical shifts.
        shifts_u : tuple | list | NDArray
            List of all the horizontal shifts.
        mask_encoding : bool, optional
            Are the masks encoding (False = decoding). The default is True.
        abs_fraction : float, optional
            Absorption fraction of the mask elements. The default is 1.

        Returns
        -------
        NDArray
            The collection of all the shifted masks.
        """
        masks = [self.generate_mask([v, u], mask_encoding) for v, u in zip(shifts_v, shifts_u)]
        return np.stack(masks, axis=0)  # .reshape([*self.shift_shape, *self.FoV_shape])

    def get_interval_shifts(
        self, interval: Union[int, Sequence[int], NDArray], axes_order: Sequence[int] = (-2, -1)
    ) -> Sequence[NDArray]:
        """Produce shifts for the "interval" shift type.

        Parameters
        ----------
        interval : int | tuple(int, int) | list(int, int)
            The shift interval.
        axes_order : int | tuple | list, optional
            Order of the axes to shift. The default is (-2, -1).

        Returns
        -------
        tuple
            The collection of shifts.
        """
        interval = np.array(interval, dtype=int)
        if interval.size == 1:
            interval = np.tile(interval, len(axes_order))

        disps = [np.arange(0, self.shape_shifts[ax], interval[ii]) for ii, ax in enumerate(axes_order)]
        disps = np.meshgrid(*disps, indexing="ij")
        return [disp.flatten() for disp in disps]

    def get_random_shifts(self, num_shifts: int, axes_order: Sequence[int] = (-2, -1)) -> Sequence[NDArray]:
        """Produce shifts for the "random" shift type.

        Parameters
        ----------
        num_shifts : int
            Number of shifts.

        Returns
        -------
        NDArray
            The collection of shifts.
        """
        max_disps = np.prod(self.shape_shifts)

        if num_shifts > max_disps:
            print("Warning, too many shifts. Truncating to: %d" % max_disps)
        num_shifts = np.fmin(num_shifts, max_disps)

        disps = self.get_interval_shifts(interval=1, axes_order=axes_order)
        perms = np.random.permutation(np.prod(self.shape_shifts))[:num_shifts]
        return [disp[perms] for disp in disps]

    def get_sequential_shifts(
        self, num_shifts: Optional[int] = None, axes_order: Sequence[int] = (-2, -1)
    ) -> Sequence[NDArray]:
        """Produce shifts for the "sequential" shift type.

        Parameters
        ----------
        num_shifts : int, optional
            Number of shifts. The default is None.
        axes_order : tuple | list | NDArray, optional
            Order of the axes to shift. The default is (-2, -1).

        Returns
        -------
        NDArray
            The collection of shifts.
        """
        disps = self.get_interval_shifts(interval=1, axes_order=axes_order)
        if num_shifts is not None:
            disps = [disp[:num_shifts] for disp in disps]
        return disps


class MaskGeneratorPoint(MaskGenerator):
    """Pencil beam masks generator class."""

    __mask_name__ = "pencil"

    def __init__(self, FoV_size_mm: Union[float, Sequence[float], NDArray], req_res_mm: float = 1.0):
        """Initialize the pencil beam mask collection.

        Parameters
        ----------
        FoV_size_mm : float
            Size of the Field-of-View in millimiters.
        req_res_mm : float
            Requested resolution in millimiters.
        """
        num_points = self._init_FoV_mm(FoV_size_mm, req_res_mm)
        super().__init__(num_points, [1, 1], num_points)

    def generate_mask(self, mask_inds_vu: Union[Sequence, NDArray], mask_encoding: bool = True) -> NDArray:
        """Produce the shifted masks.

        Parameters
        ----------
        mask_inds_vu : tuple | list | NDArray
            The vertical and horizontal shifts.
        mask_encoding : bool, optional
            Is the mask encoding (False = decoding). The default is True.

        Returns
        -------
        NDArray
            The shifted mask.
        """
        mask = np.zeros(self.shape_FoV, dtype=self.dtype)
        mask[mask_inds_vu[0], mask_inds_vu[1]] = 1.0
        return self._apply_transmission(mask)


class MaskGeneratorBernoulli(MaskGenerator):
    """Bernoulli mask generator class."""

    __mask_name__ = "bernoulli"

    def __init__(self, FoV_size_mm: Union[float, Sequence[float], NDArray], req_res_mm: float = 1.0):
        """
        Bernulli masks collection class.

        It computes and stores the original mask pattern for a given resolution and Field-ofView.

        Parameters
        ----------
        FoV_size_mm : float
            DESCRIPTION.
        req_res_mm : float
            DESCRIPTION.
        """
        num_points = self._init_FoV_mm(FoV_size_mm, req_res_mm)
        super().__init__(num_points, num_points, num_points)

    def generate_mask(self, mask_inds_vu: Union[Sequence, NDArray], mask_encoding: bool = True) -> NDArray:
        """Produce the shifted masks.

        Parameters
        ----------
        mask_inds_vu : tuple | list | NDArray
            The vertical and horizontal shifts.
        mask_encoding : bool, optional
            Is the mask encoding (False = decoding). The default is True.

        Returns
        -------
        NDArray
            The shifted mask.
        """
        mask = np.random.randint(0, 2, size=self.shape_FoV).astype(self.dtype)
        return self._apply_transmission(mask)


class MaskGeneratorHalfGaussian(MaskGenerator):
    """Half-Gaussian mask generator class."""

    __mask_name__ = "half-gaussian"

    def __init__(self, FoV_size_mm: Union[float, Sequence[float], NDArray], req_res_mm: float = 1.0):
        """
        Half Gaussian masks collection class.

        It computes and stores the original mask pattern for a given resolution and Field-ofView.

        Parameters
        ----------
        FoV_size_mm : float
            DESCRIPTION.
        req_res_mm : float
            DESCRIPTION.
        """
        num_points = self._init_FoV_mm(FoV_size_mm, req_res_mm)
        super().__init__(num_points, num_points, num_points)

    def generate_mask(self, mask_inds_vu: Union[Sequence, NDArray], mask_encoding: bool = True) -> NDArray:
        """Produce the shifted masks.

        Parameters
        ----------
        mask_inds_vu : tuple | list | NDArray
            The vertical and horizontal shifts.
        mask_encoding : bool, optional
            Is the mask encoding (False = decoding). The default is True.

        Returns
        -------
        NDArray
            The shifted mask.
        """
        mask = np.abs(np.random.randn(*self.shape_FoV)).astype(self.dtype)
        return self._apply_transmission(mask)


class MaskGeneratorMURA(MaskGenerator):
    """MURA mask generator class."""

    __mask_name__ = "mura"

    def __init__(self, FoV_size_mm: float, req_res_mm: float = 1.0):
        """
        MURA masks collection class.

        Parameters
        ----------
        FoV_size_mm : float
            DESCRIPTION.
        req_res_mm : float
            DESCRIPTION.
        """
        self.FoV_size_mm = np.array([FoV_size_mm, FoV_size_mm])
        base_points = int(np.ceil((FoV_size_mm / req_res_mm - 1) / 4))
        num_points = 4 * base_points + 1
        self.feature_size_mm = self.FoV_size_mm / num_points

        sq = np.mod(np.arange(num_points) ** 2, num_points)

        mask_1d = np.zeros(num_points)
        mask_1d[sq[1:]] = 1

        self.masks_enc = mask_1d[:, None] * mask_1d[None, :] + ((1 - mask_1d[:, None]) * (1 - mask_1d[None, :]))
        self.masks_enc[0, :] = 0
        self.masks_enc[1:, 0] = 1

        self.masks_dec = self.masks_enc.copy()
        self.masks_dec[0, 0] = 1

        super().__init__([num_points, num_points], [num_points, num_points], [num_points, num_points])
        self._enc_dec_mismatch = True

    def generate_mask(self, mask_inds_vu: Union[Sequence, NDArray], mask_encoding: bool = True) -> NDArray:
        """Produce the shifted masks.

        Parameters
        ----------
        mask_inds_vu : tuple | list | NDArray
            The vertical and horizontal shifts.
        mask_encoding : bool, optional
            Is the mask encoding (False = decoding). The default is True.

        Returns
        -------
        NDArray
            The shifted mask.
        """
        if mask_encoding:
            mask = self.masks_enc
        else:
            mask = self.masks_dec

        for ii, ind in enumerate(mask_inds_vu):
            mask = np.roll(mask, ind, axis=ii)

        return self._apply_transmission(mask)

    @staticmethod
    def compute_possible_mask_sizes(FoV_size: int) -> NDArray:
        """Compute MURA masks sizes.

        MURA masks require specific edge sizes: prime numbers _x_ that also
        satisfy the rule: _x_ = 4 * _l_ + 1, where _l_ is integer.

        Parameters
        ----------
        FoV_size : int
            Edge size of the FoV in pixels.

        Returns
        -------
        NDArray
            Array of all possible MURA mask sizes in that range.
        """

        def test_prime(x):
            div_val = x % np.arange(2, x // 2)
            return not np.any(div_val == 0)

        max_possible_val = (FoV_size - 1) // 4
        test_values = np.arange(1, max_possible_val) * 4 + 1
        primes = np.array([test_prime(x) for x in test_values])
        return test_values[primes]


class ProjectorGhostImaging(operators.ProjectorOperator):
    """Projector class for the ghost imaging reconstructions."""

    mc: MaskCollection

    def __init__(self, mask_collection: MaskCollection):
        """
        Initialize the Ghost Imaging projector class.

        Parameters
        ----------
        mask_collection : MaskCollection
            Container of the masks.
        """
        self.mc = mask_collection

        self._axes_shifts = np.arange(len(self.mc.shape_shifts))
        self.col_sum = np.abs(self.mc.masks_dec).sum(axis=tuple(self._axes_shifts))
        self._axes_fov = np.arange(-len(self.mc.shape_FoV), 0)
        self.row_sum = np.sqrt(np.abs(self.mc.masks_enc * self.mc.masks_dec)).sum(axis=tuple(self._axes_fov))

        self.vol_shape = self.mc.masks_enc.shape[-2:]
        self.prj_shape = np.array(self.mc.num_buckets, ndmin=1)
        super().__init__()

    def fp(self, image: NDArray) -> NDArray:
        """Compute forward-projection (prediction) of the bucket values.

        Parameters
        ----------
        image : NDArray
            The image for which we want to predict the bucket values.

        Returns
        -------
        NDArray
            The predicted bucket values.
        """
        masks_shape = [np.prod(self.mc.shape_shifts), np.prod(self.mc.shape_FoV)]
        image_shape = [*image.shape[: -self.mc.mask_dims], np.prod(image.shape[-self.mc.mask_dims :])]

        return np.squeeze(image.reshape(image_shape).dot(self.mc.masks_enc.reshape(masks_shape).T))
        # return np.sum(self.masks_enc * image, axis=(-2, -1))

    def bp(self, bucket_vals: NDArray) -> NDArray:
        """Compute back-projection of the bucket values.

        Parameters
        ----------
        bucket_vals : NDArray
            The list of bucket values.
        subtract_mean : bool, optional
            Whether to subtract the mean of the values. The default is False.

        Returns
        -------
        NDArray
            Back-projected image.
        """
        masks_shape = [np.prod(self.mc.shape_shifts), np.prod(self.mc.shape_FoV)]
        out_shape = [*bucket_vals.shape[:-1], *self.mc.shape_FoV]

        masks_in = self.mc.masks_dec.reshape(masks_shape)
        img_out: NDArray = bucket_vals.dot(masks_in)

        return img_out.reshape(out_shape)
        # return np.sum(bucket_vals[..., None, None] * self.masks_dec, axis=-3, keepdims=True)

    def adjust_sampling_scaling(self, image: NDArray) -> NDArray:
        """Adjust reconstruction scaling and bias, due to the undersampling.

        Parameters
        ----------
        image : NDArray
            Unscaled image.

        Returns
        -------
        NDArray
            Scaled image.
        """
        sampling_ratio = self.mc.num_buckets / np.prod(self.mc.shape_FoV)

        image_f: NDArray = np.fft.rfftn(image, axes=tuple(self._axes_fov), norm="ortho")
        rec_f_shape = np.array(image_f.shape)[list(self._axes_fov)]

        sampling_filter = np.ones(rec_f_shape, dtype=image.dtype) / sampling_ratio
        sampling_filter[0, 0] = 1
        image_f *= sampling_filter

        return np.fft.irfftn(image_f, s=self.mc.shape_FoV, axes=tuple(self._axes_fov), norm="ortho")

    def fbp(self, bucket_vals: NDArray, use_lstsq: bool = True, adjust_scaling: bool = True) -> NDArray:
        """Compute cross-correlation reconstruction of the bucket values.

        Parameters
        ----------
        bucket_vals : NDArray
            The bucket vales to reconstruct.

        Returns
        -------
        NDArray
            The reconstructed image.
        """
        # return self.bp(bucket_vals - np.mean(bucket_vals)) + np.mean(bucket_vals) * np.mean(self.col_sum)
        if np.any(self.mc.masks_enc != self.mc.masks_dec) and not use_lstsq:
            img = self.bp(bucket_vals)
        else:
            masks = self.mc.masks_enc.reshape([-1, np.prod(self.mc.shape_FoV)])
            result = spalg.lstsq(masks, bucket_vals)
            img: NDArray = result[0]
            img = img.reshape(self.mc.shape_FoV)

        if adjust_scaling:
            img = self.adjust_sampling_scaling(img)

        return img

    def absolute(self):
        """Compute the absolute value of the projection operator coefficients.

        Returns
        -------
        Op_a : ProjectorGhostImaging
            The absolute value of the projector.
        """
        Op_a = cp.deepcopy(self)
        Op_a.mc.masks_enc = np.abs(Op_a.mc.masks_enc)
        Op_a.mc.masks_dec = np.abs(Op_a.mc.masks_dec)
        return Op_a
