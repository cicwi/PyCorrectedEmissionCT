#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define all the models used through-out the code.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Sequence

import scipy.spatial.transform as spt

from dataclasses import dataclass, replace as dc_replace

from abc import ABC


class Geometry(ABC):
    """Base geometry class."""

    def __str__(self) -> str:
        """
        Return a human readable representation of the object.

        Returns
        -------
        str
            The human readable representation of the object.
        """
        descr = f"{self.__class__.__name__}(\n"
        for f, v in self.__dict__.items():
            descr += f"    {f} = {v},\n"
        return descr + ")"


@dataclass
class ProjectionGeometry(Geometry):
    """Store the projection geometry."""

    geom_type: str
    src_pos_xyz: ArrayLike
    det_pos_xyz: ArrayLike
    det_u_xyz: ArrayLike
    det_v_xyz: ArrayLike
    rot_dir_xyz: ArrayLike
    pix2vox_ratio: float = 1
    det_shape_vu: Optional[ArrayLike] = None

    @staticmethod
    def get_default_parallel(*, geom_type: str = "3d", rot_axis_shift_pix: Optional[ArrayLike] = None) -> "ProjectionGeometry":
        """
        Generate the default geometry for parallel beam.

        Parameters
        ----------
        geom_type : str, optional
            The geometry type. The default is "parallel3d".
        rot_axis_shift_pix : Optional[ArrayLike], optional
            Rotation axis shift in pixels. The default is None.

        Returns
        -------
        ProjectionGeometry
            The default paralle-beam geometry.
        """
        if rot_axis_shift_pix is None:
            det_pos_xyz = np.array([0, 0, 0])
        else:
            rot_axis_shift_pix = np.array(rot_axis_shift_pix, ndmin=1)
            det_pos_xyz = np.concatenate([rot_axis_shift_pix[:, None], np.zeros((len(rot_axis_shift_pix), 2))], axis=-1)

        return ProjectionGeometry(
            geom_type="parallel" + geom_type,
            src_pos_xyz=np.array([0, -1, 0]),
            det_pos_xyz=det_pos_xyz,
            det_u_xyz=np.array([1, 0, 0]),
            det_v_xyz=np.array([0, 0, 1]),
            rot_dir_xyz=np.array([0, 0, -1]),
        )

    def set_detector_shifts_vu(self, det_pos_vu: ArrayLike) -> None:
        """
        Set the detector position in XYZ, from VU (vertical, horizontal) coordinates.

        Parameters
        ----------
        det_pos_vu : ArrayLike
            Detector vertical and horizontal positions. Vertical is optional.
        """
        det_pos_vu = np.array(det_pos_vu, ndmin=2)
        self.det_pos_xyz = np.zeros((det_pos_vu.shape[-1], 3))
        self.det_pos_xyz[:, 0] = det_pos_vu[-1, :]
        if int(self.geom_type[-2]) == 3 and det_pos_vu.shape[0] == 2:
            self.det_pos_xyz[:, 2] = det_pos_vu[-2, :]

    def rotate(self, angles_w_rad: ArrayLike) -> "ProjectionGeometry":
        """
        Rotate the geometry by the given angle(s).

        Parameters
        ----------
        angles_w_rad : ArrayLike
            Rotation angle(s) in radians.

        Returns
        -------
        ProjectionGeometry
            The rotated geometry.
        """
        angles = np.array(angles_w_rad, ndmin=1)[:, None]

        # Deadling with ASTRA's incoherent 2D and 3D coordinate systems.
        if int(self.geom_type[-2]) == 2:
            angles = -angles

        rotations = spt.Rotation.from_rotvec(angles * self.rot_dir_xyz)

        return dc_replace(
            self,
            src_pos_xyz=rotations.apply(self.src_pos_xyz),
            det_pos_xyz=rotations.apply(self.det_pos_xyz),
            det_u_xyz=rotations.apply(self.det_u_xyz),
            det_v_xyz=rotations.apply(self.det_v_xyz),
        )

    def get_field_scaled(self, field_name: str) -> ArrayLike:
        """
        Return the a field content, scaled by the pix2vox ratio.

        Parameters
        ----------
        field_name : str
            Name of the field to access.

        Returns
        -------
        ArrayLike
            The scaled field.
        """
        field_value = getattr(self, field_name) / self.pix2vox_ratio
        if int(self.geom_type[-2]) == 2:
            return field_value[:, :-1]
        else:
            return field_value


@dataclass
class VolumeGeometry(Geometry):
    """Store the volume geometry."""

    vol_shape_xyz: ArrayLike
    vox_size: float = 1

    @property
    def shape(self) -> ArrayLike:
        """
        Return the volume shape.

        Returns
        -------
        ArrayLike
            Shape of the volume.
        """
        return self.vol_shape_xyz

    @property
    def extent(self) -> Sequence[float]:
        """
        Return extent of the volume.

        Returns
        -------
        Sequence[float]
            The extent of the volume [-x, +x, -y, +y, [-z, +z]].
        """
        half_size_xyz = self.vol_shape_xyz * self.vox_size / 2
        return [hs * sign for hs in half_size_xyz for sign in [-1, +1]]

    def is_3D(self) -> bool:
        """
        Tell whether this is a 3D geometry.

        Returns
        -------
        bool
            Whether this is a 3D geometry or not.
        """
        return len(self.vol_shape_xyz) == 3 and self.vol_shape_xyz[-1] > 1

    @staticmethod
    def get_default_from_data(data_vwu: ArrayLike) -> "VolumeGeometry":
        """
        Generate a default volume geometry from the data shape.

        Parameters
        ----------
        data_vwu : ArrayLike
            The data.

        Returns
        -------
        VolumeGeometry
            The default volume geometry.
        """
        return VolumeGeometry([data_vwu.shape[-1], data_vwu.shape[-1], *np.flip(data_vwu.shape[:-2])])
