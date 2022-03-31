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

from dataclasses import dataclass

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
    def get_default_parallel3d(rot_axis_shift_pix: Optional[ArrayLike] = None) -> "ProjectionGeometry":
        """
        Generate the default geometry for 3D parallel beam.

        Parameters
        ----------
        rot_axis_shift_pix : Optional[ArrayLike], optional
            Rotation axis shift in pixels. The default is None.

        Returns
        -------
        ProjectionGeometry
            The default paralle-beam geometry.
        """
        rot_axis_shift_pix = np.array(rot_axis_shift_pix, ndmin=1)
        det_pos_xyz = np.concatenate([rot_axis_shift_pix[:, None], np.zeros((len(rot_axis_shift_pix), 2))], axis=-1)
        return ProjectionGeometry(
            geom_type="parallel3d",
            src_pos_xyz=np.array([0, -1, 0]),
            det_pos_xyz=det_pos_xyz,
            det_u_xyz=np.array([1, 0, 0]),
            det_v_xyz=np.array([0, 0, 1]),
            rot_dir_xyz=np.array([0, 0, -1])
        )


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
