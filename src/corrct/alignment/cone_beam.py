#!/usr/bin/env python3
"""
Calibrate cone-beam reconstruction geometry.

@author: Nicola VIGANÒ, ESRF - The European Synchrotron, Grenoble, France,
and CEA-IRIG, Grenoble, France
"""

import json
from dataclasses import dataclass, replace as dc_replace
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from corrct.models import ProjectionGeometry, VolumeGeometry
from corrct.projectors import ProjectorUncorrected
from corrct.solvers import SIRT
from corrct.alignment.fitting import Ellipse, fit_parabola_min, fit_ellipse, fit_ellipse_center


def _class_to_json(obj: object) -> str:
    return json.dumps(obj, default=lambda o: {o.__class__.__name__: o.__dict__}, sort_keys=True, indent=4)


def _get_rot_axis_angle_deg(
    center_1_vu: Sequence[float] | NDArray,
    center_2_vu: Sequence[float] | NDArray,
    decimals: int | None = 4,
    dtype: DTypeLike = np.float32,
) -> float:
    center_1_vu = np.squeeze(np.array(center_1_vu, dtype=dtype))
    center_2_vu = np.squeeze(np.array(center_2_vu, dtype=dtype))

    # Check if the arrays are 2D
    if center_1_vu.ndim != 1 or center_2_vu.ndim != 1:
        raise ValueError("Input arrays must be 1D")

    # Check if the first dimension has length 2
    if center_1_vu.shape[0] != 2 or center_2_vu.shape[0] != 2:
        raise ValueError("Input arrays must have length 2")

    diffs_vu = center_1_vu - center_2_vu
    angle_rad = np.arctan2(diffs_vu[-1], diffs_vu[-2])
    angle_rad = np.mod(angle_rad, 2 * np.pi)

    angle_deg = np.rad2deg(angle_rad - np.pi)

    if decimals is not None:
        angle_deg = np.around(angle_deg, decimals=decimals)

    return float(angle_deg)


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
    D_pix: float = 0.0
    R_pix: float = 0.0
    v0_pix: float = 0.0
    u0_pix: float = 0.0
    det_size_v_pix: int = 0
    det_size_u_pix: int = 0
    pix_size_um: float = 0.0

    def __str__(self) -> str:
        """
        Return a human readable representation of the object.

        Returns
        -------
        str
            The human readable representation of the object.
        """
        descr = "AcquisitionGeometry(\n"
        for field, value in self.__dict__.items():
            descr += f"    {field} = {value}"
            if field.lower()[-3:] == "deg":
                descr += " [deg]"
            elif field.lower()[-3:] == "pix":
                descr += " [pix]"
                if self.pix_size_um > 0.0:
                    descr += f" ({value * self.pix_size_um} [um])"
            elif field.lower()[-2:] == "um":
                descr += " [um]"
            descr += ",\n"
        return descr + ")"

    def get_prj_geom(self, translate_z_to_center: bool = True) -> ProjectionGeometry:
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
        beta_xyz = np.array([-np.sin(theta_rad) * np.cos(phi_rad), np.sin(theta_rad) * np.sin(phi_rad), np.cos(theta_rad)])

        # Detector normal
        e_n_xyz = np.array([np.cos(theta_rad) * np.cos(phi_rad), -np.cos(theta_rad) * np.sin(phi_rad), np.sin(theta_rad)])

        det_pos_xyz = -e_n_xyz * self.D_pix + e_x_xyz * self.R_pix - alpha_xyz * self.u0_pix - beta_xyz * self.v0_pix
        src_pos_xyz = e_x_xyz * self.R_pix

        pix2vox_ratio = self.R_pix / self.D_pix * np.abs(np.dot(e_n_xyz, e_x_xyz))

        if translate_z_to_center:
            det_center_xyz = e_n_xyz * self.D_pix + alpha_xyz * self.u0_pix + beta_xyz * self.v0_pix

            translation_z = det_center_xyz[2] / np.abs(det_center_xyz[0]) * self.R_pix
            src_pos_xyz[2] += translation_z
            det_pos_xyz[2] += translation_z

        rotation = Rotation.from_rotvec(-e_n_xyz * np.deg2rad(self.eta_deg))
        e_u_xyz = rotation.apply(alpha_xyz)
        e_v_xyz = rotation.apply(beta_xyz)

        return ProjectionGeometry(
            geom_type="cone",
            src_pos_xyz=src_pos_xyz,
            det_pos_xyz=det_pos_xyz,
            det_u_xyz=e_u_xyz,
            det_v_xyz=e_v_xyz,
            rot_dir_xyz=np.array([0, 0, 1]),
            pix2vox_ratio=pix2vox_ratio,
            det_shape_vu=np.array((self.det_size_v_pix, self.det_size_u_pix)),
        )

    def get_vol_geom(self, up_sampling: int = 1) -> VolumeGeometry:
        """
        Generate volume geometry.

        Returns
        -------
        VolumeGeometry
            The volume geometry.
        """
        return VolumeGeometry(
            _vol_shape_xyz=np.array([self.det_size_u_pix, self.det_size_u_pix, self.det_size_v_pix], dtype=int) * up_sampling,
            vox_size=1 / up_sampling,
        )

    def update(self, field: str, val: float, is_relative: bool = True, decimals: int | None = 3) -> "ConeBeamGeometry":
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
        self, field: str, val_range: Sequence[float] | NDArray, is_relative: bool = True
    ) -> Sequence["ConeBeamGeometry"]:
        """
        Generate sequences of acquisition geometries, with a slight variation over a field's value.

        Parameters
        ----------
        field : str
            The field to tune.
        val_range : Sequence[float] | NDArray
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

    def from_json(self, data_json: str) -> None:
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
        data_tree = json.loads(data_json)
        if len(data_tree.keys()) > 1:
            raise ValueError("Initialization from JSON: More than one class instance passed.")

        class_name = list(data_tree.keys())[0]
        if list(data_tree.keys())[0] != self.__class__.__name__:
            raise ValueError(
                f"Initialization from JSON: expecting {self.__class__.__name__} class instance,"
                f" but {data_tree.keys()[0]} passed."
            )

        data_dict = data_tree[class_name]
        for key in self.__dict__.keys():
            self.__dict__[key] = data_dict[key]


class FitConeBeamGeometry:
    """Cone-beam geometry calibration object.

    This method is based on the following article:
    - Noo, F., Clackdoyle, R., Mennessier, C., White, T. A. & Roney, T. J. (2000). Phys. Med. Biol. 45, 3489–3508.
      doi: 10.1088/0031-9155/45/11/327
    """

    acq_geom: ConeBeamGeometry

    def __init__(
        self,
        prj_size_vu: Sequence[int] | NDArray,
        points_ell1: Sequence[Sequence[float]] | NDArray,
        points_ell2: Sequence[Sequence[float]] | NDArray,
        points_axis: Sequence[Sequence[float]] | NDArray | None = None,
        pix_size_um: float | None = None,
        use_l1_norm: bool = False,
        verbose: bool = True,
        plot_result: bool = False,
    ):
        """Initialize a cone-beam geometry calibration object.

        Parameters
        ----------
        prj_size_vu : Sequence[int] | NDArray
            Size of the projections.
        points_ell1 : Sequence[Sequence[float]] | NDArray
            Points of first ellipse.
        points_ell2 : Sequence[Sequence[float]] | NDArray
            Points of second ellipse.
        points_axis : Sequence[Sequence[float]] | NDArray | None, optional
            Points of the rotation axis, by default None
        pix_size_um : float | None, optional
            The size of the pixel edge in micrometers. Default is None.
        use_l1_norm : bool, optional
            Whether to use the l1-norm or the least-squares (l2-norm) fit for optimization.
            Default is False.
        verbose : bool, optional
            Whether to produce verbose output, by default True
        plot_result : bool, optional
            Whether to plot the results of the geometry, by default False
            It requires verbose to be True.
        """
        prj_size_vu = np.array(prj_size_vu)
        if prj_size_vu.ndim != 1 or len(prj_size_vu) != 2:
            if len(prj_size_vu) == 1:
                prj_size_vu = np.tile(prj_size_vu, 2)
            else:
                raise ValueError("prj_size_vu must be a 1D array with 2 elements")
        self.prj_size_vu = prj_size_vu

        self.center_vu = np.squeeze(self.prj_size_vu)[:, None] / 2
        self.prj_origin_vu = None

        points_ell1 = np.array(points_ell1)
        if points_ell1.ndim != 2:
            raise ValueError("points_ell1 must be a 2D array")
        if points_ell1.shape[0] != 2:
            raise ValueError("points_ell1 must have a first dimension equal to 2")
        self.points_ell1 = points_ell1 - self.center_vu

        points_ell2 = np.array(points_ell2)
        if points_ell2.ndim != 2:
            raise ValueError("points_ell2 must be a 2D array")
        if points_ell2.shape[0] != 2:
            raise ValueError("points_ell2 must have a first dimension equal to 2")
        self.points_ell2 = points_ell2 - self.center_vu

        if points_axis is not None:
            points_axis = np.array(points_axis) - self.center_vu
        self.points_axis = points_axis

        self.acq_geom = ConeBeamGeometry(det_size_v_pix=int(self.prj_size_vu[0]), det_size_u_pix=int(self.prj_size_vu[1]))
        if pix_size_um is not None:
            self.acq_geom.pix_size_um = pix_size_um

        self.verbose = verbose
        self.plot_result = plot_result and verbose

        self._initialize(use_l1_norm=use_l1_norm)

    def _initialize(self, use_l1_norm: bool) -> None:
        ell1_fit_prj_c_vu = fit_ellipse_center(self.points_ell1, use_l1_norm=use_l1_norm)
        ell2_fit_prj_c_vu = fit_ellipse_center(self.points_ell2, use_l1_norm=use_l1_norm)
        fit_eta_deg = _get_rot_axis_angle_deg(ell1_fit_prj_c_vu, ell2_fit_prj_c_vu)

        if self.verbose:
            print("Fitted / measured values:")

        if self.points_axis is not None:
            # Using measured projected center, whenever available
            ell1_acq_prj_c_vu = self.points_axis[:, 0]
            ell2_acq_prj_c_vu = self.points_axis[:, 2]
            acq_eta_deg = _get_rot_axis_angle_deg(ell1_acq_prj_c_vu, ell2_acq_prj_c_vu)

            if self.verbose:
                print(f"- Ellipse 1 center: fitted = {ell1_fit_prj_c_vu} vs acquired = {ell1_acq_prj_c_vu} [pix]")
                print(f"- Ellipse 2 center: fitted = {ell2_fit_prj_c_vu} vs acquired = {ell2_acq_prj_c_vu} [pix]")
                print("- Detector tilt around its normal (eta):")
                print(f"  * fitted = {fit_eta_deg:.4} vs acq = {acq_eta_deg:.4} [deg] <= Using acquired!")

            self.ell1_prj_c_vu = ell1_acq_prj_c_vu
            self.ell2_prj_c_vu = ell2_acq_prj_c_vu

            self.prj_origin_vu = self.points_axis[:, 1]
            self.acq_geom.eta_deg = acq_eta_deg
        else:
            if self.verbose:
                print(f"- Ellipse 1 center: fitted = {ell1_fit_prj_c_vu} [pix]")
                print(f"- Ellipse 2 center: fitted = {ell2_fit_prj_c_vu} [pix]")

            self.ell1_prj_c_vu = ell1_fit_prj_c_vu
            self.ell2_prj_c_vu = ell2_fit_prj_c_vu

            self.prj_origin_vu = None
            self.acq_geom.eta_deg = fit_eta_deg
            if self.verbose:
                print(f"- Detector tilt around its normal (eta), fitted: {self.acq_geom.eta_deg:.4} [deg]")

        if np.abs(self.acq_geom.eta_deg) > 120:
            raise ValueError(
                "The order of the ellipses seems to have been inverted."
                f" (it suggests an eta of {self.acq_geom.eta_deg}). Please swap them."
            )

        pix_size_um = self.acq_geom.pix_size_um
        if self.verbose and self.prj_origin_vu is not None:
            print(
                f"- Projected origin on the detector: {self.prj_origin_vu} [pix]",
                f"({self.prj_origin_vu * pix_size_um} [um])" if pix_size_um > 0.0 else "",
            )

        if np.abs(self.acq_geom.eta_deg) > 0.1:
            rot = Rotation.from_rotvec(-np.deg2rad(self.acq_geom.eta_deg) * np.array([0, 0, 1]))
            rot_mat = rot.as_matrix()[:2, :2]

            points_ell1_rot = rot_mat.dot(self.points_ell1)
            points_ell2_rot = rot_mat.dot(self.points_ell2)
        else:
            points_ell1_rot = self.points_ell1.copy()
            points_ell2_rot = self.points_ell2.copy()

        # Re-instantiate ellipse class, after rotation
        self.ell1_rot: Ellipse = fit_ellipse(points_ell1_rot, use_l1_norm=use_l1_norm)
        self.ell2_rot: Ellipse = fit_ellipse(points_ell2_rot, use_l1_norm=use_l1_norm)

        if self.plot_result:
            fig, axs = plt.subplots()
            axs.plot(self.points_ell1[1, :], self.points_ell1[0, :], "C0--", label="Ellipse 1 - Acquired")
            axs.plot(self.points_ell2[1, :], self.points_ell2[0, :], "C1--", label="Ellipse 2 - Acquired")
            axs.plot(points_ell1_rot[1, :], points_ell1_rot[0, :], "C0", label="Ellipse 1 - Rotated")
            axs.plot(points_ell2_rot[1, :], points_ell2_rot[0, :], "C1", label="Ellipse 2 - Rotated")
            axs.plot([ell1_fit_prj_c_vu[1], ell2_fit_prj_c_vu[1]], [ell1_fit_prj_c_vu[0], ell2_fit_prj_c_vu[0]], "C2--")
            axs.plot([self.ell1_rot.u, self.ell2_rot.u], [self.ell1_rot.v, self.ell2_rot.v], "C2")
            if self.points_axis is not None:
                axs.scatter(self.points_axis[1], self.points_axis[0], c="C2", marker="*", label="Centers - Acquired")
            axs.legend()
            axs.grid()
            fig.tight_layout()
            plt.show(block=False)

    def fit(self, r: float, e: float = 1, meas_D_pix: float | None = None) -> ConeBeamGeometry:
        """
        Fit the cone-beam geometry parameters, that will be used for producing the projection geometry.

        Parameters
        ----------
        r : float
            The radius of the circle performed by the spheres in pixels.
        e : float, optional
            Either 1 or -1, indicating whether the source is between the circles or not. The default is 1.
        meas_D_pix : float, optional
            The measured source-detector distance in pixels. This parameter is only necessary when the
            computed source-detector distance is invalid or zero.

        Raises
        ------
        ValueError
            In case of flipped ellipses or invalid computed source-detector distance.
        """

        def get_v0(v: float, b: float, a: float, c: float, D: float, sign_z: float) -> float:
            return v - sign_z * np.sqrt(a + (a**2) * (D**2)) / np.sqrt(a * b - (c**2))

        def get_denom(b: float, a: float, c: float, D: float) -> float:
            return np.sqrt(a * b + (a**2) * b * (D**2) - (c**2))

        def get_rho(b: float, a: float, c: float, D: float) -> float:
            return np.sqrt(a * b - (c**2)) / get_denom(b, a, c, D)

        def get_zeta(b: float, a: float, c: float, D: float, sign_zk: float) -> float:
            return D * sign_zk * a * np.sqrt(a) / get_denom(b, a, c, D)

        b1, a1, c1, v1, u1 = self.ell1_rot.parameters
        b2, a2, c2, v2, u2 = self.ell2_rot.parameters

        if self.verbose:
            print("Fitted values from the calibration scan parameters:")
            print("- Ellipses' parameters:")
            print(f"  * upper ({a1 = :.6}, {b1 = :.6}, {c1 = :.6}, {v1 = :.6}, {u1 = :.6})")
            print(f"  * lower ({a2 = :.6}, {b2 = :.6}, {c2 = :.6}, {v2 = :.6}, {u2 = :.6})")

        pix_size_um = self.acq_geom.pix_size_um

        comp_D_pix = self._fit_distance_det2src(self.ell1_rot, self.ell2_rot, e=e)
        if meas_D_pix is None:
            if np.isnan(comp_D_pix) or np.isclose(comp_D_pix, 0.0):
                raise ValueError(
                    f"The computed source-detector distance is invalid ({comp_D_pix}), please enter a measured value"
                )

            self.acq_geom.D_pix = comp_D_pix

            if self.verbose:
                print(
                    f"- Computed source-detector distance: {self.acq_geom.D_pix:.6} [pix]",
                    f"({self.acq_geom.D_pix * pix_size_um:.4e} [um])" if pix_size_um > 0.0 else "",
                )
        else:
            self.acq_geom.D_pix = meas_D_pix

            if self.verbose:
                print("- Source-detector distance (using measured):")
                print(f"  * Measured = {meas_D_pix:.6} vs computed = {comp_D_pix:.6} [pix]")
                print(f"  * Measured = {meas_D_pix*pix_size_um:.6} vs computed = {comp_D_pix*pix_size_um:.6} [um]")

        sign_z1 = -1
        sign_z2 = sign_z1 * -e

        v01 = get_v0(v1, b1, a1, c1, self.acq_geom.D_pix, sign_z1)
        v02 = get_v0(v2, b2, a2, c2, self.acq_geom.D_pix, sign_z2)

        tilt_ratio1 = c1 / (2 * a1)
        tilt_ratio2 = c2 / (2 * a2)

        self.acq_geom.v0_pix = np.array([v01, v02]).mean()
        self.acq_geom.u0_pix = (
            (u1 + u2) / 2 + tilt_ratio1 * (v1 - self.acq_geom.v0_pix) + tilt_ratio2 * (v2 - self.acq_geom.v0_pix)
        )

        if self.verbose:
            print(f"- Source position over detector: v0 = {self.acq_geom.v0_pix:.6}, u0 = {self.acq_geom.u0_pix:.6}")
            print(f"  * Separately fitted v (from the two ellipses): {v01 = :.6}, {v02 = :.6}")

        if np.linalg.norm(v01 - v02) > np.linalg.norm(v1 - v2):
            raise ValueError(
                f"Obtained: {v01 = }, {v02 = }, while {v1 = }, {v2 = }. Probably wrong order of ellipses (please flip them!)"
            )

        rho1 = get_rho(b1, a1, c1, self.acq_geom.D_pix)
        rho2 = get_rho(b2, a2, c2, self.acq_geom.D_pix)

        zeta1 = get_zeta(b1, a1, c1, self.acq_geom.D_pix, sign_z1)
        zeta2 = get_zeta(b2, a2, c2, self.acq_geom.D_pix, sign_z2)

        sin_phi1 = -tilt_ratio1 * zeta1
        sin_phi2 = -tilt_ratio2 * zeta2
        self.acq_geom.phi_deg = np.rad2deg(np.arcsin(sin_phi1 + sin_phi2))

        R_e1 = r / rho1
        R_e2 = r / rho2

        z1 = R_e1 * zeta1
        z2 = R_e2 * zeta2

        z_full = z1 - z2

        self.acq_geom.R_pix = (-z2 * R_e1 + z1 * R_e2) / z_full

        if np.isnan(self.acq_geom.R_pix) or np.isclose(self.acq_geom.R_pix, 0.0):
            raise ValueError(f"The computed source-origin distance is invalid ({self.acq_geom.R_pix})")

        if self.prj_origin_vu is None:
            self.prj_origin_vu = (-z2 * self.ell1_prj_c_vu + z1 * self.ell2_prj_c_vu) / z_full
            if self.verbose:
                print(f"- Projected origin on the detector: {self.prj_origin_vu} [pix]")

        self.acq_geom.theta_deg = 0.0
        # self.acq_geom.theta_rad = np.arcsin((R1 - R2) / z_full) / np.mean(self.prj_size_vu)

        if self.verbose:
            print("- Distances between source and rotation axis:")
            print(f"  * R1 = {R_e1:.6}, R = {self.acq_geom.R_pix:.6}, R2 = {R_e2:.6} [pix]")
            if pix_size_um > 0.0:
                print(
                    f"  * R1 = {R_e1 * pix_size_um:.4e},",
                    f"R = {self.acq_geom.R_pix * pix_size_um:.4e},",
                    f"R2 = {R_e2 * pix_size_um:.4e} [um]",
                )
            print("- Heights of the two ellipses, with respect to the source:")
            print(f"  * {z1 = :.6}, {z2 = :.6} [pix]")
            if pix_size_um > 0.0:
                print(f"  * z1 = {z1 * pix_size_um:.4e}, z2 = {z2 * pix_size_um:.4e} [um]")
            print(f"- Polar angle of the detector (phi deg): {self.acq_geom.phi_deg}")
            print(f"- Azimuthal angle of the detector (theta deg): {self.acq_geom.theta_deg}")

        return self.acq_geom

    @staticmethod
    def _fit_distance_det2src(ellipse_1: Ellipse, ellipse_2: Ellipse, e: float = 1) -> float:
        b1, a1, c1, v1, _ = ellipse_1.parameters
        b2, a2, c2, v2, _ = ellipse_2.parameters

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
    data: NDArray,
    angles_rot_rad: Sequence[float] | NDArray,
    params: dict[str, Sequence[float] | NDArray],
    data_mask: NDArray | None = None,
    verbose: bool = True,
) -> ConeBeamGeometry:
    """
    Tune the acquisition geometry, based on calibration data self-consistency.

    Parameters
    ----------
    acq_geom : ConeBeamGeometry
        The cone-beam geometry to refine.
    data : NDArray
        The calibration projection data.
    angles : Sequence[float] | NDArray
        Angles of the projections.
    params : dict[str, Sequence[float] | NDArray]
        Parameters to tune as a dictionary.
        The acquisition parameters to tune are the keys, and their test values are the dictionary values.
    data_mask : NDArray | None, optional
        Pixel mask of the data, to mask out dead or hot pixels. The default is None.
    verbose : bool, optional
        Whether to output verbose information or not, by default False.
    """
    data = np.array(data)
    angles_rot_rad = np.array(angles_rot_rad)
    if data_mask is not None:
        data_mask = np.array(data_mask)

    solver = SIRT(tolerance=0.0)

    acq_geom_tuned = acq_geom_init

    for par_name, par_vals in params.items():
        par_vals = np.array(par_vals)
        desc = f"Tuning '{par_name}': "
        residuals = np.atleast_1d(np.empty(len(par_vals)))
        for par_ind, acq_geom in enumerate(tqdm(acq_geom_tuned.get_tuning_params(par_name, par_vals), desc=desc)):
            vol_geom = acq_geom.get_vol_geom()
            prj_geom = acq_geom.get_prj_geom()
            with ProjectorUncorrected(vol_geom, angles_rot_rad, prj_geom=prj_geom) as prj:
                _, info = solver(prj, data, iterations=100, b_mask=data_mask)
                residuals[par_ind] = info.residuals[-1]

        min_par, min_res, fit_info = fit_parabola_min(par_vals, residuals, decimals=6)

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
