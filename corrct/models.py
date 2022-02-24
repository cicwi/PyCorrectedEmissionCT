#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define all the models used through-out the code.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

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
