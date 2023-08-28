#!/usr/bin/env python3
"""
Calibrate cone-beam reconstruction geometry.

@author: Nicola VIGANÒ, ESRF - The European Synchrotron, Grenoble, France,
and CEA-IRIG, Grenoble, France
"""

import json
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from . import fitting

from .. import models, projectors, solvers


def _class_to_json(obj: object) -> str:
    return json.dumps(obj, default=lambda o: {o.__class__.__name__: o.__dict__}, sort_keys=True, indent=4)


def _get_rot_axis_angle_rad(center_1_vu: Union[ArrayLike, NDArray], center_2_vu: Union[ArrayLike, NDArray]) -> float:
    diffs_vu = np.array(center_1_vu) - np.array(center_2_vu)
    angle_rad = np.arctan2(diffs_vu[-1], diffs_vu[-2])
    angle_rad = np.mod(angle_rad, 2 * np.pi)
    return angle_rad - np.pi


@dataclass
class ConeBeamGeometry:
    """Store the acquisition geometry parameters, used for creating reconstruction geometries.

    A description of the geometry / meaning of the fields can be found here:
    - Noo, F., Clackdoyle, R., Mennessier, C., White, T. A. & Roney, T. J. (2000). Phys. Med. Biol. 45, 3489–3508.
      doi: 10.1088/0031-9155/45/11/327
    """

    theta_deg: float = 0.0
    phi_deg: float = 0.0
    eta_deg: float = 0.0
    D: float = 0.0
    R: float = 0.0
    v0: float = 0.0
    u0: float = 0.0
    det_pix_v: int = 0
    det_pix_u: int = 0

    def __str__(self) -> str:
        """
        Return a human readable representation of the object.

        Returns
        -------
        str
            The human readable representation of the object.
        """
        descr = "AcquisitionGeometry(\n"
        for f, v in self.__dict__.items():
            descr += f"    {f} = {v}"
            if f.lower()[-3:] == "rad":
                descr += f" ({np.rad2deg(v)} deg)"
            descr += ",\n"
        return descr + ")"

    def get_prj_geom(self, translate_z_to_center: bool = True) -> models.ProjectionGeometry:
        """
        Create the geometry for reconstruction.

        Returns
        -------
        Dict
            The geometry to be used for reconstruction.
        """
        # Sample base vectors
        e_x_xyz = np.array([1, 0, 0])

        # NOTE: Y coordinate is flipped, with respect to the input!!!

        theta_rad = np.deg2rad(self.theta_deg)
        phi_rad = np.deg2rad(self.phi_deg)

        # Rotated detector base vectors
        alpha_xyz = np.array([-np.sin(phi_rad), -np.cos(phi_rad), 0])
        beta_xyz = np.array(
            [
                -np.sin(theta_rad) * np.cos(phi_rad),
                np.sin(theta_rad) * np.sin(phi_rad),
                np.cos(theta_rad),
            ]
        )

        # Detector normal
        e_n_xyz = np.array(
            [
                np.cos(theta_rad) * np.cos(phi_rad),
                -np.cos(theta_rad) * np.sin(phi_rad),
                np.sin(theta_rad),
            ]
        )

        det_pos_xyz = -e_n_xyz * self.D + e_x_xyz * self.R - alpha_xyz * self.u0 - beta_xyz * self.v0
        src_pos_xyz = e_x_xyz * self.R

        pix2vox_ratio = self.R / self.D * np.abs(np.dot(e_n_xyz, e_x_xyz))

        if translate_z_to_center:
            det_center_xyz = e_n_xyz * self.D + alpha_xyz * self.u0 + beta_xyz * self.v0

            translation_z = det_center_xyz[2] / np.abs(det_center_xyz[0]) * self.R
            src_pos_xyz[2] += translation_z
            det_pos_xyz[2] += translation_z

        rotation = Rotation.from_rotvec(-e_n_xyz * np.deg2rad(self.eta_deg))
        e_u_xyz = rotation.apply(alpha_xyz)
        e_v_xyz = rotation.apply(beta_xyz)

        return models.ProjectionGeometry(
            geom_type="cone",
            src_pos_xyz=src_pos_xyz,
            det_pos_xyz=det_pos_xyz,
            det_u_xyz=e_u_xyz,
            det_v_xyz=e_v_xyz,
            rot_dir_xyz=np.array([0, 0, 1]),
            pix2vox_ratio=pix2vox_ratio,
        )

    def get_vol_geom(self, up_sampling: int = 1) -> models.VolumeGeometry:
        """
        Generate volume geometry.

        Returns
        -------
        VolumeGeometry
            The volume geometry.
        """
        return models.VolumeGeometry(
            _vol_shape_xyz=np.array([self.det_pix_u, self.det_pix_u, self.det_pix_v], dtype=int) * up_sampling,
            vox_size=1 / up_sampling,
        )

    def update(self, field: str, val: float, is_relative: bool = True, decimals: Union[int, None] = 3) -> "ConeBeamGeometry":
        """
        Return a copy of the original data, with a replaced field.

        Parameters
        ----------
        field : str
            The field to replace.
        val : float
            The new value of the field.
        is_relative : bool, optional
            Whether the value is relative to the previous. The default is True.
        decimals : int | None, optional
            The number of decimals (precision) to use for the updated values, by default 3 decimals.

        Returns
        -------
        AcquisitionGeometry
            The updated geometry.
        """
        new_val = getattr(self, field) + val if is_relative else val
        if decimals is not None:
            new_val = np.around(new_val, decimals=decimals)
        return dc_replace(self, **{field: new_val})

    def get_tuning_params(
        self, field: str, val_range: Union[ArrayLike, NDArray], is_relative: bool = True
    ) -> Sequence["ConeBeamGeometry"]:
        """
        Generate sequences of acquisition geometries, with a slight variation over a field's value.

        Parameters
        ----------
        field : str
            The field to tune.
        val_range : ArrayLike
            The value range.
        is_relative : bool, optional
            Whether the values are relative. The default is True.

        Returns
        -------
        Sequence[AcquisitionGeometry]
            The list of new acquisition geometries.
        """
        return [self.update(field, val, is_relative) for val in np.array(val_range, ndmin=1)]

    def to_json(self) -> str:
        """
        Save instance to JSON.

        Returns
        -------
        str
            The JSON representation.
        """
        return _class_to_json(self)

    def from_json(self, data: str) -> None:
        """
        Load instance from JSON.

        Parameters
        ----------
        data : str
            The JSON data to load.

        Raises
        ------
        ValueError
            In case we were to load more than one instance, or different classes.
        """
        d = json.loads(data)
        if len(d.keys()) > 1:
            raise ValueError("Initialization from JSON: More than one class instance passed.")
        class_name = list(d.keys())[0]
        if list(d.keys())[0] != self.__class__.__name__:
            raise ValueError(
                f"Initialization from JSON: expecting {self.__class__.__name__} class instance, but {d.keys()[0]} passed."
            )
        d = d[class_name]
        for k in self.__dict__.keys():
            self.__dict__[k] = d[k]


class FitConeBeamGeometry:
    """Cone-beam geometry calibration object.

    This method is based on the following article:
    - Noo, F., Clackdoyle, R., Mennessier, C., White, T. A. & Roney, T. J. (2000). Phys. Med. Biol. 45, 3489–3508.
      doi: 10.1088/0031-9155/45/11/327
    """

    def __init__(
        self,
        prj_size_vu: Union[ArrayLike, NDArray],
        points_ell1: Union[ArrayLike, NDArray],
        points_ell2: Union[ArrayLike, NDArray],
        points_axis: Union[ArrayLike, NDArray, None] = None,
        verbose: bool = True,
        plot_result: bool = False,
    ):
        """Initialize a cone-beam geometry calibration object.

        Parameters
        ----------
        prj_size_vu : Union[ArrayLike, NDArray]
            Size of the projections.
        points_ell1 : Union[ArrayLike, NDArray]
            Points of first ellipse.
        points_ell2 : Union[ArrayLike, NDArray]
            Points of second ellipse.
        points_axis : Union[ArrayLike, NDArray, None], optional
            Points of the rotation axis, by default None
        verbose : bool, optional
            Whether to produce verbose output, by default True
        plot_result : bool, optional
            Whether to plot the results of the geometry, by default False
            It requires verbose to be True.
        """
        self.prj_size_vu = np.array(prj_size_vu)
        self.center_vu = self.prj_size_vu[:, None] / 2

        self.points_ell1 = np.array(points_ell1) - self.center_vu
        self.points_ell2 = np.array(points_ell2) - self.center_vu

        if points_axis is not None:
            points_axis = np.array(points_axis) - self.center_vu
        self.points_axis = points_axis

        self.acq_geom = ConeBeamGeometry(det_pix_v=int(self.prj_size_vu[0]), det_pix_u=int(self.prj_size_vu[1]))

        self.verbose = verbose
        self.plot_result = plot_result and verbose

        self._pre_fit()

    def _pre_fit(self, use_least_squares: bool = False) -> None:
        ell1_acq = fitting.Ellipse(self.points_ell1)
        ell2_acq = fitting.Ellipse(self.points_ell2)

        if self.points_axis is not None:
            # Using measured projected center, whenever available
            self.ell1_prj_center_vu = self.points_axis[:, 0]
            self.ell2_prj_center_vu = self.points_axis[:, 2]

            self.prj_origin_vu = self.points_axis[:, 1]
        else:
            self.ell1_prj_center_vu = ell1_acq.fit_prj_center(least_squares=use_least_squares)
            self.ell2_prj_center_vu = ell2_acq.fit_prj_center(least_squares=use_least_squares)

            self.prj_origin_vu = None

        self.acq_geom.eta_deg = np.rad2deg(_get_rot_axis_angle_rad(self.ell1_prj_center_vu, self.ell2_prj_center_vu))

        if self.verbose:
            print(f"Projected origin on the detector (pix): {self.prj_origin_vu}")
            print(f"Detector tilt around its normal (eta), fitted (deg): {self.acq_geom.eta_deg}")

        if np.abs(self.acq_geom.eta_deg) > 0.1:
            rot = Rotation.from_rotvec(-np.deg2rad(self.acq_geom.eta_deg) * np.array([0, 0, 1]))
            rot_mat = rot.as_matrix()[:2, :2]

            self.points_ell1_rot = rot_mat.dot(self.points_ell1)
            self.points_ell2_rot = rot_mat.dot(self.points_ell2)
        else:
            self.points_ell1_rot = self.points_ell1.copy()
            self.points_ell2_rot = self.points_ell2.copy()

        # Re-instatiate ellipse class, after rotation
        ell1_rot = fitting.Ellipse(self.points_ell1_rot)
        ell2_rot = fitting.Ellipse(self.points_ell2_rot)

        self.ell1_params = ell1_rot.fit_parameters(least_squares=use_least_squares)
        self.ell2_params = ell2_rot.fit_parameters(least_squares=use_least_squares)

        if self.plot_result:
            fig, axs = plt.subplots()
            axs.plot(self.points_ell1[1, :], self.points_ell1[0, :], "C0--", label="Ellipse 1 - Acquired")
            axs.plot(self.points_ell2[1, :], self.points_ell2[0, :], "C1--", label="Ellipse 2 - Acquired")
            axs.plot(self.points_ell1_rot[1, :], self.points_ell1_rot[0, :], "C0", label="Ellipse 1 - Rotated")
            axs.plot(self.points_ell2_rot[1, :], self.points_ell2_rot[0, :], "C1", label="Ellipse 2 - Rotated")
            ell1_acq_params = ell1_acq.fit_parameters(least_squares=use_least_squares)
            ell2_acq_params = ell2_acq.fit_parameters(least_squares=use_least_squares)
            axs.plot([ell1_acq_params[-1], ell2_acq_params[-1]], [ell1_acq_params[-2], ell2_acq_params[-2]], "C2--")
            axs.plot([self.ell1_params[-1], self.ell2_params[-1]], [self.ell1_params[-2], self.ell2_params[-2]], "C2")
            if self.points_axis is not None:
                axs.scatter(self.points_axis[1], self.points_axis[0], c="C2", marker="*", label="Centers - Acquired")
            axs.legend()
            axs.grid()
            fig.tight_layout()
            plt.show(block=False)

        self.acq_geom.D = self._fit_distance_det2src(self.ell1_params, self.ell2_params)

        if self.verbose:
            print(f"Fitted detector distance from source (pix): {self.acq_geom.D}")

    def fit(self, r: float, e: float = 1) -> ConeBeamGeometry:
        """
        Fit the cone-beam geometry parameters, that will be used for producing the projection geometry.

        Parameters
        ----------
        r : float
            The radius of the circle performed by the spheres.
        e : float, optional
            Either 1 or -1, indicating whether the source is between the circles or not. The default is 1.

        Raises
        ------
        ValueError
            In case of flipped ellipses.
        """
        b1, a1, c1, v1, u1 = self.ell1_params
        b2, a2, c2, v2, u2 = self.ell2_params

        sign_z1 = -1
        sign_z2 = sign_z1 * -e

        def get_v0(vk, bk, ak, ck, D, sign_zk) -> float:
            return vk - sign_zk * np.sqrt(ak + (ak**2) * (D**2)) / np.sqrt(ak * bk - (ck**2))

        def get_denom(bk, ak, ck, D) -> float:
            return np.sqrt(ak * bk + (ak**2) * bk * (D**2) - (ck**2))

        def get_rho(bk, ak, ck, D) -> float:
            return np.sqrt(ak * bk - (ck**2)) / get_denom(bk, ak, ck, D)

        def get_zeta(bk, ak, ck, D, sign_zk) -> float:
            return D * sign_zk * ak * np.sqrt(ak) / get_denom(bk, ak, ck, D)

        v01 = get_v0(v1, b1, a1, c1, self.acq_geom.D, sign_z1)
        v02 = get_v0(v2, b2, a2, c2, self.acq_geom.D, sign_z2)

        self.acq_geom.v0 = np.array([v01, v02]).mean()
        self.acq_geom.u0 = (
            np.mean([u1, u2]) + c1 / (2 * a1) * (v1 - self.acq_geom.v0) + c2 / (2 * a2) * (v2 - self.acq_geom.v0)
        )

        if self.verbose:
            print(f"Ellipses' positions:\n- upper (v1={v1}, u1={u1})\n- lower (v2={v2}, u2={u2})")
            print(f"Fitted source position over detector: v0={self.acq_geom.v0}, u0={self.acq_geom.u0}")
            print(f"- Separately fitted v: v01={v01}, v02={v02}")

        if np.linalg.norm(v01 - v02) > np.linalg.norm(v1 - v2):
            raise ValueError(
                f"Obtained: v01={v01}, v02={v02}, while v1={v1}, v2={v2}. Probably wrong order of ellipses (please flip them!)"
            )

        rho1 = get_rho(b1, a1, c1, self.acq_geom.D)
        rho2 = get_rho(b2, a2, c2, self.acq_geom.D)

        zeta1 = get_zeta(b1, a1, c1, self.acq_geom.D, sign_z1)
        zeta2 = get_zeta(b2, a2, c2, self.acq_geom.D, sign_z2)

        self.acq_geom.phi_deg = np.rad2deg(np.arcsin(-c1 / (2 * a1) * zeta1 - c2 / (2 * a2) * zeta2))

        R1 = r / rho1
        R2 = r / rho2

        self.z1 = R1 * zeta1
        self.z2 = R2 * zeta2

        z_full = self.z1 - self.z2

        self.acq_geom.theta_deg = 0.0
        # self.acq_geom.theta_rad = np.arcsin((R1 - R2) / z_full) / np.mean(self.prj_size_vu)

        self.acq_geom.R = (-self.z2 * R1 + self.z1 * R2) / z_full

        if self.prj_origin_vu is None:
            self.prj_origin_vu = (-self.z2 * self.ell1_prj_center_vu + self.z1 * self.ell2_prj_center_vu) / z_full
            if self.verbose:
                print(f"Projected origin on the detector (pix): {self.prj_origin_vu}")

        if self.verbose:
            print(f"Fitted distances between source and rotation axis (pix):\n- R1={R1}, R={self.acq_geom.R}, R2={R2}")
            print(f"Fitted heights of the two ellipses, with respect to the source (pix): z1={self.z1}, z2={self.z2}")
            print(f"Fitted polar angle of the detector (phi deg): {self.acq_geom.phi_deg}")
            print(f"Fitted azimuthal angle of the detector (theta deg): {self.acq_geom.theta_deg}")

        return self.acq_geom

    @staticmethod
    def _fit_distance_det2src(
        ellipse_1: Union[ArrayLike, NDArray], ellipse_2: Union[ArrayLike, NDArray], e: float = 1
    ) -> float:
        b1, a1, c1, v1, _ = np.array(ellipse_1)
        b2, a2, c2, v2, _ = np.array(ellipse_2)

        ecc2 = np.sqrt(b2 - c2**2 / a2)
        ecc1 = np.sqrt(b1 - c1**2 / a1)

        m0 = (v2 - v1) * ecc2
        m1 = ecc2 / ecc1

        m0m1 = 2 * m0 * m1
        m1_2 = m1**2

        n0 = (1 - m0**2 - m1_2) / m0m1
        n1 = (a2 - a1 * m1_2) / m0m1

        n0n1 = n0 * n1
        n1_2 = n1**2

        s2d_2 = ((a1 - 2 * n0n1) - e * np.sqrt(a1**2 + 4 * n1_2 - 4 * n0n1 * a1)) / (2 * n1_2)
        return np.sqrt(s2d_2)


def tune_acquisition_geometry(
    acq_geom_init: ConeBeamGeometry,
    data: Union[ArrayLike, NDArray],
    angles_rot_rad: Union[ArrayLike, NDArray],
    params: dict[str, Union[ArrayLike, NDArray]],
    data_mask: Union[ArrayLike, NDArray, None] = None,
    verbose: bool = True,
) -> ConeBeamGeometry:
    """
    Tune the acquisition geometry, based on calibration data self-consistency.

    Parameters
    ----------
    acq_geom : ConeBeamGeometry
        The cone-beam geometry to refine.
    data : ArrayLike | NDArray
        The calibration projection data.
    angles : ArrayLike | NDArray
        Angles of the projections.
    params : dict[str, ArrayLike | NDArray]
        Parameters to tune as a dictionary.
        The acquisition parameters to tune are the keys, and their test values are the dictionary values.
    data_mask : ArrayLike | NDArray | None, optional
        Pixel mask of the data, to mask out dead or hot pixels. The default is None.
    verbose : bool, optional
        Whether to output verbose information or not, by default False.
    """
    data = np.array(data)
    angles_rot_rad = np.array(angles_rot_rad)
    if data_mask is not None:
        data_mask = np.array(data_mask)

    solver = solvers.SIRT(tolerance=0)

    acq_geom_tuned = acq_geom_init

    for par_name, par_vals in params.items():
        par_vals = np.array(par_vals)
        desc = f"Tuning '{par_name}': "
        residuals = np.atleast_1d(np.empty(len(par_vals)))
        for par_ind, acq_geom in enumerate(tqdm(acq_geom_tuned.get_tuning_params(par_name, par_vals), desc=desc)):
            vol_geom = acq_geom.get_vol_geom()
            prj_geom = acq_geom.get_prj_geom()
            with projectors.ProjectorUncorrected(vol_geom, angles_rot_rad, prj_geom=prj_geom) as prj:
                _, info = solver(prj, data, iterations=100, b_mask=data_mask)
                residuals[par_ind] = info.residuals[-1]

        min_par, min_res, fit_info = fitting.fit_parabola_min(par_vals, residuals, decimals=6)

        if verbose:
            old_par_val = getattr(acq_geom_tuned, par_name)
            print(f"Min of {par_name}: {old_par_val}" + f" -> {old_par_val + min_par} (diff: {min_par})\n")
            fig, axs = plt.subplots()
            axs.plot(par_vals, residuals)
            if fit_info is not None:
                x = np.linspace(fit_info[1][0], fit_info[1][2])
                y = fit_info[0][0] + x * (fit_info[0][1] + x * fit_info[0][2])
                axs.plot(x, y)
            axs.scatter(min_par, min_res, 10, "r")
            axs.set_title(par_name)
            axs.grid()
            fig.tight_layout()
            plt.show(block=False)

        acq_geom_tuned = acq_geom_tuned.update(par_name, min_par)

    return acq_geom_tuned


def cm2inch(dims: Union[ArrayLike, NDArray]) -> tuple[float]:
    """Convert cm into inch.

    Parameters
    ----------
    dims : Union[ArrayLike, NDArray]
        The dimentions of the object in cm

    Returns
    -------
    tuple[float]
        The output dimensions in inch
    """
    return tuple(np.array(dims) / 2.54)


class MarkerVisualizer:
    """Plotting class to assess the calibration quality."""

    def __init__(
        self,
        fitted_positions_vu: Union[ArrayLike, NDArray],
        imgs: NDArray,
        disk: NDArray,
        ell_params: Union[ArrayLike, NDArray, None] = None,
    ) -> None:
        self.positions_vu = np.array(fitted_positions_vu)
        self.imgs = imgs
        self.disk = disk
        self.global_lims = False

        if ell_params is not None:
            ell_params = np.array(ell_params)
        self.ell_params = ell_params

        self.curr_pos = 0

        if self.ell_params is not None:
            us = np.sort(self.positions_vu[1, :])
            self.v_1, self.v_2 = fitting.Ellipse.predict_v(self.ell_params, us)

        self.fig, self.axs = plt.subplots(1, 3, figsize=cm2inch([36, 12]))  # , sharex=True, sharey=True
        self.axs[2].imshow(self.disk)
        self.axs[0].set_xlim(0, self.imgs.shape[-1])
        self.axs[0].set_ylim(self.imgs.shape[-3], 0)
        self.fig.tight_layout()
        self.update()

        self.fig.canvas.mpl_connect("key_press_event", self._key_event)
        self.fig.canvas.mpl_connect("scroll_event", self._scroll_event)

    def update(self) -> None:
        self.curr_pos = self.curr_pos % self.imgs.shape[-2]

        for img in self.axs[0].get_images():
            img.remove()
        x_lims = self.axs[0].get_xlim()
        y_lims = self.axs[0].get_ylim()
        self.axs[0].cla()
        self.axs[0].set_xlim(x_lims[0], x_lims[1])
        self.axs[0].set_ylim(y_lims[0], y_lims[1])

        for img in self.axs[1].get_images():
            img.remove()
        self.axs[1].cla()

        self.axs[0].plot(self.positions_vu[1, :], self.positions_vu[0, :], "bo-", markersize=4)
        self.axs[0].scatter(self.positions_vu[1, self.curr_pos], self.positions_vu[0, self.curr_pos], c="r")

        if self.ell_params is not None:
            us = np.sort(self.positions_vu[1, :])
            self.axs[0].plot(us, self.v_1, "g")
            self.axs[0].plot(us, self.v_2, "g")
        self.axs[0].grid()

        if self.global_lims:
            vmin = self.imgs.min()
            vmax = self.imgs.max()
        else:
            vmin = self.imgs[:, self.curr_pos, :].min()
            vmax = self.imgs[:, self.curr_pos, :].max()

        img = self.axs[1].imshow(self.imgs[:, self.curr_pos, :], vmin=vmin, vmax=vmax)
        self.axs[1].scatter(self.positions_vu[1, self.curr_pos], self.positions_vu[0, self.curr_pos], c="r")
        self.axs[1].set_title(f"Range: [{vmin}, {vmax}]")
        # plt.colorbar(im, ax=self.axs[1])
        self.fig.canvas.draw()

    def _key_event(self, evnt) -> None:
        if evnt.key == "right":
            self.curr_pos += 1
        elif evnt.key == "left":
            self.curr_pos -= 1
        elif evnt.key == "up":
            self.curr_pos += 1
        elif evnt.key == "down":
            self.curr_pos -= 1
        elif evnt.key == "pageup":
            self.curr_pos += 10
        elif evnt.key == "pagedown":
            self.curr_pos -= 10
        elif evnt.key == "escape":
            plt.close(self.fig)
        elif evnt.key == "ctrl+l":
            self.global_lims = not self.global_lims
        else:
            print(evnt.key)
            return

        self.update()

    def _scroll_event(self, evnt) -> None:
        if evnt.button == "up":
            self.curr_pos += 1
        elif evnt.button == "down":
            self.curr_pos -= 1
        else:
            print(evnt.key)
            return

        self.update()
