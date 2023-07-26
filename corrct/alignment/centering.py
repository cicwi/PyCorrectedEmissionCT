# -*- coding: utf-8 -*-
"""
Volume centering classes.

@author: Nicola VIGANÒ, CEA-IRIG and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from typing import Union
from numpy.typing import ArrayLike, NDArray

from . import fitting

from .. import models
from ..processing import post as post_proc


class RecenterVolume:
    """Volume re-centering class."""

    def __init__(
        self, proj_geom: models.ProjectionGeometry, angles_rad: Union[NDArray, ArrayLike], precision: int = 2
    ) -> None:
        """Volume recentering class, that shifts the detector position, in order to meet a certain volume position.

        Parameters
        ----------
        proj_geom : models.ProjectionGeometry
            The projection geometry.
        angles_rad : NDArray | ArrayLike
            The rotation angles to consider.
        precision : int, optional
            Number of decimals to use, by default 2.
        """
        self.prj_geom = proj_geom.rotate(angles_rad)
        self.precision = precision

    def _apply_displacement_vu(self, shifts_vu: NDArray, displacemenet_zyx: NDArray) -> NDArray:
        shifts_vu_corrs = self.prj_geom.project_displacement_to_detector(displacemenet_zyx)
        return np.around(shifts_vu + shifts_vu_corrs, decimals=self.precision)

    def to_com(self, shifts_vu: Union[ArrayLike, NDArray], volume: NDArray, com_ref_zyx: Union[ArrayLike, NDArray]) -> NDArray:
        """Recenter to a given center-of-mass (CoM).

        Parameters
        ----------
        shifts_vu : ArrayLike | NDArray
            The current VU shifts.
        volume : NDArray
            The volume to shift.
        com_ref_zyx : ArrayLike | NDArray
            The destination CoM.

        Returns
        -------
        NDArray
            The corrected VU shifts.
        """
        com_rec_zyx = post_proc.com(volume)
        displacemenet_zyx = np.array(com_ref_zyx) - com_rec_zyx
        return self._apply_displacement_vu(np.array(shifts_vu), displacemenet_zyx)

    def as_reference(self, shifts_vu: NDArray, volume: NDArray, reference: NDArray, method: str = "com") -> NDArray:
        """Recenter with respect to a given volume.

        Parameters
        ----------
        shifts_vu : NDArray
            Current VU shifts.
        volume : NDArray
            The volume to shift.
        reference : NDArray
            The reference volume.
        method : str, optional
            The method to use out of "com" | "xc" (cross-correlation), by default "com"

        Returns
        -------
        NDArray
            The corrected VU shifts.

        Raises
        ------
        ValueError
            In case of wrong method.
        """
        if method.lower() == "com":
            com_ref_zyx = post_proc.com(reference)
            return self.to_com(shifts_vu, volume, com_ref_zyx)
        elif method.lower() == "xc":
            displacemenet_zyx = fitting.fit_shifts_zyx_xc(reference, volume, decimals=self.precision)
            return self._apply_displacement_vu(shifts_vu, displacemenet_zyx)
        else:
            raise ValueError(f"Method (passed: {method}) should be one of: 'com' | 'xc'.")
