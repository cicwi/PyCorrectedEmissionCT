"""
Detector shifts finding classes.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from collections.abc import Mapping
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from . import fitting

eps = np.finfo(np.float32).eps


NDArrayFloat = NDArray[np.floating]


def _filter_shifts(shifts_vu: NDArrayFloat, max_shifts: NDArrayFloat) -> NDArrayFloat:
    invalid_shifts = np.abs(shifts_vu) > max_shifts

    shifts_vu_filt = shifts_vu.copy()
    if np.any(invalid_shifts):
        print(f"WARNING - Some shifts exceeded the maximum allowed magnitude (max: {max_shifts}). Setting them to 0.")
        shifts_vu_filt[invalid_shifts] = 0

    return shifts_vu_filt


class DetectorShiftsBase:
    """Compute the detector shifts for a given dataset."""

    data_vwu: NDArrayFloat
    angles_rad: NDArrayFloat

    def __init__(
        self,
        data_dvwu: NDArrayFloat,
        rot_angle_rad: Union[ArrayLike, NDArrayFloat],
        *,
        data_format: str = "dvwu",
        data_mask_dvwu: Optional[NDArray] = None,
        borders_dvwu: dict = {"d": None, "v": None, "w": None, "u": None},
        max_shifts: Union[float, NDArrayFloat, None] = None,
        precision_decimals: int = 2,
        verbose: bool = True,
    ):
        """Initialize the base class for detector shifts.

        Parameters
        ----------
        data_dvwu : NDArrayFloat
            The tomographic data.
        rot_angle_rad : ArrayLike | NDArrayFloat
            The rotation angles in radians.
        data_format : str, optional
            The data organization, by default "dvwu"
        data_mask_dvwu : NDArray | None, optional
            The mask of the tomographic data, by default None
        borders_dvwu : dict, optional
            The borders of the tomographic data, by default {"d": None, "v": None, "w": None, "u": None}
        max_shifts : float | NDArrayFloat | None, optional
            Maximum shifts allowed, by default None
        precision_decimals : int, optional
            The precision of the results, by default 2
        verbose : bool, optional
            Whether to be verbose, by default True

        Raises
        ------
        ValueError
            Raised when passing incoherent data and angles.
        """
        self.data_dims = len(data_dvwu.shape)

        self.data_shapes = dict(u=0, v=1, w=0, d=1)
        for ii in range(-self.data_dims, 0):
            self.data_shapes[data_format[ii]] = data_dvwu.shape[ii]

        self.num_dets = self.data_shapes["d"]

        self.angles_rad = np.array(np.squeeze(rot_angle_rad), ndmin=1, dtype=np.float32)
        if self.data_shapes["w"] != len(self.angles_rad):
            raise ValueError(
                f"Mismatch between rotation angles ({len(self.angles_rad)}),"
                f" and number of projections ({self.data_shapes['w']})."
            )

        slicing = [borders_dvwu[data_format[ii]] for ii in range(-self.data_dims, 0)]
        slicing = [slice(s, -s) if s is not None and s > 0 else slice(None) for s in slicing]
        self.slicing = tuple(slicing)

        if self.data_shapes["v"] > 1:
            self._align_coords = np.array([-3, -1])
        else:
            self._align_coords = np.array([-1])

        if max_shifts is None:
            max_shifts = np.array(data_dvwu.shape)[self._align_coords] / 2
        max_shifts = np.array(max_shifts, ndmin=1)
        self.max_shifts = max_shifts[:, None]

        self.decimals = precision_decimals
        self.verbose = verbose

        self.data_vwu = data_dvwu
        self.data_mask_vwu = data_mask_dvwu


class DetectorShiftsPRE(DetectorShiftsBase):
    """Compute the pre-alignment detector shifts for a given dataset."""

    def fit_v(
        self,
        use_derivative: bool = True,
        use_rfft: bool = True,
        normalize_fourier: bool = True,
    ) -> NDArrayFloat:
        """Compute the pre-alignment vertical shifts of a 3D dataset.

        The pre-alignment shifts are computed by cross-correlation of one
        projection against the others.
        The projections are integrated in the horizontal direction.

        In the vertical direction, it is suggested to use some high pass filter.
        The default option is to use the derivates of the intensity profiles.

        Parameters
        ----------
        use_derivative : bool, optional
            Whether to use the derivate of the vertical profile, by default True
        use_rfft : bool, optional
            Whether to use the `rfft` transform for the cross-correlation, by default True
        normalize_fourier : bool, optional
            Whether to normalize the cross-correlation in Fourier space, by default True

        Returns
        -------
        NDArrayFloat
            The vertical shifts.

        Raises
        ------
        ValueError
            If the dataset is 2D.
        """
        if self.data_shapes["v"] <= 1:
            raise ValueError("Vertical alignment not supported for 2D reconstructions.")

        if use_rfft:
            local_fft = np.fft.rfft
            local_ifft = np.fft.irfft
        else:
            local_fft = np.fft.fft
            local_ifft = np.fft.ifft

        data_vwu = self.data_vwu[self.slicing]
        if self.data_mask_vwu is None:
            data_vw = np.mean(data_vwu, axis=-1)
        else:
            data_mask_vwu = self.data_mask_vwu[self.slicing].astype(data_vwu.dtype)
            data_vwu = data_vwu * data_mask_vwu
            mask_sum_vw = data_mask_vwu.sum(axis=-1)
            data_vw = data_vwu.sum(axis=-1) / (mask_sum_vw + (mask_sum_vw == 0))
        if self.num_dets > 1:
            data_vw = np.mean(data_vw, axis=-3)

        mins = data_vw.min(axis=-2, keepdims=True)
        maxs = data_vw.max(axis=-2, keepdims=True)
        data_vw_d = (data_vw - mins) / (maxs - mins)

        pad_size = ((data_vw.shape[-2] // 2,), (0,))
        if use_derivative:
            data_vw_d = np.diff(data_vw_d, axis=-2)
            data_vw_p = np.pad(data_vw_d, pad_width=pad_size, mode="constant")
        else:
            data_vw_p = np.pad(data_vw_d, pad_width=pad_size, mode="linear_ramp")

        data_vw_f = local_fft(data_vw_p, axis=-2)

        ref_angle = len(self.angles_rad) // 2
        ccs_f = data_vw_f[:, [ref_angle]] * data_vw_f.conj()

        if normalize_fourier:
            ccs_f /= np.fmax(np.abs(ccs_f).max(axis=-2, keepdims=True), eps)

        cross_corr = local_ifft(ccs_f, axis=-2).real

        cc_coords = np.fft.fftfreq(cross_corr.shape[-2], 1 / cross_corr.shape[-2])
        f_vals, fc_ax = fitting.extract_peak_regions_1d(cross_corr, axis=-2, cc_coords=cc_coords)
        shifts_v = fitting.refine_max_position_1d(f_vals, decimals=self.decimals) + fc_ax[1, :]

        shifts_v = _filter_shifts(shifts_v, self.max_shifts[0, :])

        shifts_v -= np.mean(shifts_v)
        shifts_v = np.around(shifts_v, decimals=self.decimals)

        if self.verbose:
            fig, axs = plt.subplots(2, 2, figsize=[10, 5], sharex=True)
            axs[0, 0].imshow(data_vw)
            axs[0, 0].set_title("Data VW")
            axs[0, 0].set_xlabel("Coord. W (angular)")
            axs[0, 0].set_ylabel("Coord. V (vertical)")
            axs[0, 1].plot(shifts_v)
            axs[0, 1].plot(np.zeros_like(shifts_v))
            axs[0, 1].grid()
            axs[0, 1].set_title(f"Shifts V (wrt angle n.{ref_angle})")
            axs[0, 1].set_xlabel("Coord. W (angular)")
            axs[0, 1].set_ylabel("Coord. V (vertical)")
            axs[1, 0].imshow(data_vw_p)
            axs[1, 0].set_title("Data used for cross-correlation")
            axs[1, 1].imshow(np.fft.fftshift(cross_corr, axes=(-2,)))
            axs[1, 1].set_title("Cross-correlation")
            fig.tight_layout()
            plt.show(block=False)

        return shifts_v

    def fit_u(
        self,
        fit_l1: bool = False,
        background: Union[float, NDArray, None] = None,
        method: str = "com",
    ) -> tuple[NDArrayFloat, float]:
        """Compute the pre-alignment shifts for the horizontal dimension.

        The pre-alignment shifts, and center-of-rotation (CoR) are computed by
        fitting a sinusoid to the centers of mass of each angle in the sinogram.
        The bias of the sinusoid corresponds to the CoR, while the deviations
        from the fitted curve correspond to the shifts.

        Parameters
        ----------
        fit_l1 : bool, optional
            Computes the l1-min fit of the sinusoid, by default False.
        background : float | NDArray | None, optional
            Removes the given background, by default None.
        method : str, optional
            The method used for the identification of the fiducial marker position.
            Options are "com" (center-of-mass) | "max" (maximum value), by default "com".

        Returns
        -------
        Tuple[NDArrayFloat, float]
            The shifts and the CoR.
        """
        is_3d = self.data_shapes["v"] > 1

        data_vwu = self.data_vwu[self.slicing]

        if self.num_dets > 1:
            data_vwu = np.mean(data_vwu, axis=0)

        if is_3d:
            data_vwu = np.mean(data_vwu, axis=-3)

        if background is not None:
            data_vwu = data_vwu - background
        data_vwu = np.fmax(data_vwu, 0.0)

        fx_half_size = (data_vwu.shape[-1] - 1) / 2

        if method.lower() == "com":
            fx = np.linspace(-fx_half_size, fx_half_size, data_vwu.shape[-1])
            ref_points = -np.sum(data_vwu * fx, axis=-1) / np.sum(data_vwu, axis=-1)
        elif method.lower() == "max":
            ref_points = fx_half_size - np.argmax(data_vwu, axis=-1)
        else:
            raise ValueError(f"Unknown selected method {method}. Please choose one among: 'com' | 'max'")

        a, p, b = fitting.fit_sinusoid(self.angles_rad, ref_points, fit_l1=fit_l1)

        cor = np.around(b, decimals=self.decimals)

        if self.verbose:
            angles_deg = np.rad2deg(self.angles_rad)
            sort_angles_deg = np.sort(angles_deg)
            sort_angles_rad = np.sort(self.angles_rad)
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].scatter(angles_deg, ref_points, label="Centers of mass")
            axs[0].plot(sort_angles_deg, fitting.sinusoid(sort_angles_rad, a, p, b), c="C1", label="Fitted sinusoid")
            axs[0].plot(sort_angles_deg, np.ones_like(sort_angles_deg) * b, c="C2", label="Bias (CoR)")
            axs[0].legend()
            axs[0].grid()
            axs[0].set_xlabel("Coord. W (angular)")
            axs[0].set_ylabel("Coord. U (horizontal)")
            axs[0].xaxis.label.set_fontsize(16)
            axs[0].yaxis.label.set_fontsize(16)
            axs[1].imshow(data_vwu)
            axs[1].scatter(-ref_points + fx_half_size, np.arange(len(ref_points)), c="C1")
            axs[1].set_xlabel("Coord. U (horizontal)")
            axs[1].set_ylabel("Coord. W (angular)")
            axs[1].xaxis.label.set_fontsize(16)
            axs[1].yaxis.label.set_fontsize(16)
            fig.tight_layout()
            plt.show(block=False)
            print(f"amplitude = {a} (pix)")
            print(f"phase = {p} (rad)")
            print(f"bias = {b} (pix)")
            print(f" -> cor = {cor} (pix)")

        shifts_u: NDArrayFloat = ref_points - fitting.sinusoid(self.angles_rad, a, p, b)
        shifts_u = _filter_shifts(shifts_u, self.max_shifts[-1, :])
        shifts_u = np.around(shifts_u, decimals=self.decimals)

        return shifts_u, float(cor)


class DetectorShiftsXC(DetectorShiftsBase):
    """Compute the center-of-rotation for a given dataset, by cross correlation."""

    def fit_vu_accum_drifts(self, ref_data_dvwu: Optional[NDArrayFloat] = None) -> NDArray:
        """Fit static image drifts.

        Parameters
        ----------
        ref_data_dvwu : Optional[NDArrayFloat], optional
            Reference image, by default None. If None, the first image in the
            data stack will be used.

        Returns
        -------
        NDArray
            The shifts of the image stack.

        Raises
        ------
        ValueError
            When the number of reference images is either too many or not enough.
        """
        if ref_data_dvwu is None:
            ref_data_dvwu = self.data_vwu[..., [0], :]

        img_inds = np.arange(self.data_shapes["w"])
        if ref_data_dvwu.shape[-2] == 1:
            ref_inds = np.zeros_like(img_inds)
        else:
            ref_inds = np.arange(ref_data_dvwu.shape[-2])
        if img_inds.size != ref_inds.size:
            raise ValueError(
                f"Reference images should either be 1 or as many as the data images"
                f" ({img_inds.size}), but {ref_inds.size} were passed instead"
            )

        is_3d = self.data_shapes["v"] > 1
        num_dims = 1 + is_3d
        rel_shifts_vu = np.zeros((num_dims, img_inds.size))
        for ii, (ind_ref, ind_img) in enumerate(zip(tqdm(ref_inds, desc="Computing drifts"), img_inds)):
            rel_shifts_vu[..., [ii]] = self.find_shifts_vu(ref_data_dvwu[..., [ind_ref], :], self.data_vwu[..., [ind_img], :])

        return rel_shifts_vu

    def fit_vu(self, fit_l1: bool = False) -> NDArray:
        """Compute the pre-alignment vertical and horizontal shifts, using cross-correlation.

        Parameters
        ----------
        fit_l1 : bool, optional
            Computes the l1-min fit of the sinusoid, by default False.

        Returns
        -------
        NDArray
            Pre-alignment shifts in VU coordinates.
        """
        is_3d = self.data_shapes["v"] > 1
        num_dims = 1 + is_3d

        angles_order = np.argsort(self.angles_rad)
        sorted_angles_rad = self.angles_rad[angles_order]

        rel_shifts_vu = np.zeros((num_dims, len(self.angles_rad)))
        desc = "Computing adjacent images shifts"
        for ii, (a_prev, a_curr) in enumerate(zip(tqdm(angles_order[:-1], desc=desc), angles_order[1:])):
            rel_shifts_vu[..., [ii + 1]] = self.find_shifts_vu(
                self.data_vwu[..., [a_prev], :], self.data_vwu[..., [a_curr], :]
            )

        shifts_vu = np.cumsum(rel_shifts_vu, axis=-1)
        if is_3d:
            shifts_vu[0, :] -= np.mean(shifts_vu[0, :])
            shifts_vu[0, :] = np.around(shifts_vu[0, :], decimals=self.decimals)

        a, p, b = fitting.fit_sinusoid(sorted_angles_rad, shifts_vu[-1, ...], fit_l1=fit_l1)

        if self.verbose:
            coords = ["V", "U"] if is_3d else ["U"]
            fig, axs = plt.subplots(1, num_dims, figsize=[10, 5], sharex=True, squeeze=False)
            angles_deg = np.rad2deg(sorted_angles_rad)
            for ii, (s, coord) in enumerate(zip(shifts_vu, coords)):
                axs[0, ii].plot(angles_deg, s, label="Shifts")
                if coord == "V":
                    axs[0, ii].plot(angles_deg, np.zeros_like(s), label="Zero")
                else:
                    axs[0, ii].plot(angles_deg, np.ones_like(s) * b, label="Bias")
                    axs[0, ii].plot(angles_deg, fitting.sinusoid(sorted_angles_rad, a, p, b), label="Expected motion")
                axs[0, ii].legend()
                axs[0, ii].grid()
                axs[0, ii].set_title(f"Shifts {coord}")
            fig.tight_layout()
            plt.show(block=False)

        shifts_u: NDArrayFloat = shifts_vu[-1, ...] - fitting.sinusoid(sorted_angles_rad, a, p, b)
        shifts_u = _filter_shifts(shifts_u, self.max_shifts[-1, :])
        shifts_u = np.around(shifts_u, decimals=self.decimals)

        shifts_vu[-1] = shifts_u

        return shifts_vu[:, list(angles_order)]

    def fit_u_180(self) -> float:
        """Find the center-of-rotation, using the 0 and 180 degrees projections.

        Returns
        -------
        float
            The center-of-rotation.
        """
        angle_0 = self.angles_rad.min()
        angle_0_ind = np.argmin(np.abs(self.angles_rad - angle_0))
        angle_180_ind = np.argmin(np.abs(self.angles_rad - (angle_0 + np.pi)))

        angle_180 = self.angles_rad[angle_180_ind]
        if not np.isclose(angle_0 + np.pi, angle_180):
            print(
                f"WARNING - No opposite angles found ({np.rad2deg(angle_0)} and {np.rad2deg(angle_180)})."
                " Center-of-rotation will be inaccurate."
            )
        img_0 = self.data_vwu[..., [angle_0_ind], :]
        img_180 = np.flip(self.data_vwu[..., [angle_180_ind], :], axis=-1)
        # upsample_factor = 1 / (10 ** (-self.decimals))
        # return skr.phase_cross_correlation(img_0, img_180, upsample_factor=upsample_factor, return_error=False)
        shifts_vu = self.find_shifts_vu(img_0, img_180)
        return -shifts_vu[-1] / 2

    def fit_u_360(self) -> float:
        """Find the center of rotation over a 360 degrees scan, by taking the average of the 0-180 over all pairs of angles.

        Returns
        -------
        float
            The center-of-rotation.
        """
        # We should be checking whether the scan is really 360 or not.

        angles_boundary = self.angles_rad[0] + np.pi
        num_angles = np.sum(self.angles_rad < angles_boundary)

        a1s = self.angles_rad[:num_angles]
        a2s = a1s + np.pi

        iis_1 = np.arange(num_angles)
        iis_2 = np.argmin(np.abs(self.angles_rad[None, :] - a2s[:, None]), axis=-1)

        shifts_vu = np.empty([2, len(iis_1)])
        for ii, (ii1, ii2) in enumerate(tqdm(zip(iis_1, iis_2), total=num_angles)):
            img_1 = self.data_vwu[..., [ii1], :]
            img_2 = np.flip(self.data_vwu[..., [ii2], :], axis=-1)
            shifts_vu[..., [ii]] = self.find_shifts_vu(img_1, img_2)

        cors = -shifts_vu[-1, :] / 2

        # upsample_factor = int(1 / (10 ** (-self.decimals)))

        # cors = np.empty_like(a1s)
        # for ii_1, a1 in enumerate(tqdm(a1s)):
        #     ii_2 = np.argmin(np.abs(self.angles_rad - (a1 + np.pi)))
        #     # We should be handling non-redundant scans.

        #     img_0 = self.data_vwu[..., ii_1, :]
        #     img_180 = np.flip(self.data_vwu[..., ii_2, :], axis=-1)
        #     s = skr.phase_cross_correlation(img_0, img_180, upsample_factor=upsample_factor, return_error=False)

        #     cors[ii_1] = -s[-1] / 2

        cor = np.around(np.mean(cors), decimals=self.decimals)

        if self.verbose:
            fig, axs = plt.subplots(1, 1, figsize=[10, 5])
            axs.plot(cors)
            axs.plot(np.ones_like(cors) * cor)
            axs.grid()
            axs.set_title("Centers of rotation")
            fig.tight_layout()
            plt.show(block=False)

        return float(cor)

    def find_shifts_vu(
        self,
        data_dvwu: NDArrayFloat,
        proj_dvwu: NDArrayFloat,
        use_derivative: bool = False,
        xc_opts: Mapping = dict(normalize_fourier=False),
    ) -> NDArrayFloat:
        """Find shifts between two images or sets of lines.

        Parameters
        ----------
        data_dvwu : NDArrayFloat
            The reference data.
        proj_dvwu : NDArrayFloat
            The other data.
        use_derivative : bool, optional
            Whether to use derivatives over the horizontal (U) coordinate, by default False.

        Returns
        -------
        NDArrayFloat
            The shifts in vertical (optional) and horizontal coordinates ([V]U).
        """
        if self.num_dets == 1:
            data_dvwu = data_dvwu[None, ...]
            proj_dvwu = proj_dvwu[None, ...]

        shifts_vu_all = [np.array([])] * self.num_dets

        for ii_d in range(self.num_dets):
            data_vwu = data_dvwu[ii_d]
            proj_vwu = proj_dvwu[ii_d]

            if use_derivative:
                data_vwu = np.diff(data_vwu, axis=-1)
                proj_vwu = np.diff(proj_vwu, axis=-1)

            # Allow to choose different shift finding functions
            shifts_vu = fitting.fit_shifts_vu_xc(data_vwu, proj_vwu, decimals=self.decimals, **xc_opts)
            # shifts_vu = fitting.fit_shifts_u_sad(data_vwu, proj_vwu, decimals=self.decimals)

            shifts_vu_all[ii_d] = _filter_shifts(shifts_vu, self.max_shifts)

        return np.mean(shifts_vu_all, axis=0)
