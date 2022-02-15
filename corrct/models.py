#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define all the models used through-out the code.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from numpy.typing import ArrayLike

from dataclasses import dataclass


@dataclass
class ProjectionGeometry:
    """Store the projection geometry."""

    geom_type: str
    src_pos_xyz: ArrayLike
    det_pos_xyz: ArrayLike
    det_u_xyz: ArrayLike
    det_v_xyz: ArrayLike
    rot_dir_xyz: ArrayLike
    pix2vox_ratio: float = 1

    def __str__(self) -> str:
        """
        Return a human readable representation of the object.

        Returns
        -------
        str
            The human readable representation of the object.
        """
        descr = "ProjectionGeometry(\n"
        for f, v in self.__dict__.items():
            descr += f"    {f} = {v},\n"
        return descr + ")"
