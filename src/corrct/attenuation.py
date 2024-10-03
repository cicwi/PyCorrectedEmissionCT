# -*- coding: utf-8 -*-
"""
Incident beam and emidded radiation attenuation support.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from matplotlib.axes._axes import Axes
import numpy as np

from . import _projector_backends as prj_backends
from . import models

import concurrent.futures as cf
import multiprocessing as mp

from tqdm.auto import tqdm

from typing import Dict, Optional, Sequence, Union, List
from numpy.typing import ArrayLike, DTypeLike, NDArray


num_threads = round(np.log2(mp.cpu_count() + 1))

NDArrayFloat = NDArray[np.floating]
NDArrayInt = NDArray[np.integer]


class AttenuationVolume:
    """Attenuation volume computation class."""

    incident_local: Union[NDArrayFloat, None]
    emitted_local: Union[NDArrayFloat, None]
    angles_rot_rad: NDArrayFloat
    angles_det_rad: NDArrayFloat

    dtype: DTypeLike

    vol_shape_zyx: NDArray
    maps: NDArray

    def __init__(
        self,
        incident_local: Union[NDArrayFloat, None],
        emitted_local: Union[NDArrayFloat, None],
        angles_rot_rad: ArrayLike,
        angles_det_rad: Union[NDArrayFloat, ArrayLike, float] = np.pi / 2,
        dtype: DTypeLike = np.float32,
    ):
        """
        Initialize the AttenuationVolume class.

        Raises
        ------
        ValueError
            In case no volumes were passed, or if they differed in shape.
        """
        self.incident_local = incident_local
        self.emitted_local = emitted_local
        self.angles_rot_rad = np.array(angles_rot_rad, ndmin=1)
        self.angles_det_rad = np.array(angles_det_rad, ndmin=1)

        self.dtype = dtype

        if self.incident_local is not None:
            self.vol_shape_zyx = np.array(self.incident_local.shape)

            if self.emitted_local is not None and np.any(self.vol_shape_zyx != self.emitted_local.shape):
                raise ValueError(
                    f"Incident volume shape ({self.incident_local.shape}) does not"
                    + f" match the emitted volume shape ({self.emitted_local.shape})"
                )
        elif self.emitted_local is not None:
            self.vol_shape_zyx = np.array(self.emitted_local.shape)
        else:
            raise ValueError("No attenuation volumes were given.")

        self.vol_shape_zyx = np.array(self.vol_shape_zyx, ndmin=1)

        num_dims = len(self.vol_shape_zyx)
        if num_dims not in [2, 3]:
            raise ValueError(f"Maps can only be 2D or 3D Arrays. A {num_dims}-dimensional was passed ({self.vol_shape_zyx}).")

    def _compute_attenuation_angle_in(self, local_att: NDArrayFloat, angle_rad: float) -> NDArray:
        return prj_backends.compute_attenuation(local_att, angle_rad, invert=False)[None, ...]

    def _compute_attenuation_angle_out(self, local_att: NDArrayFloat, angle_rad: float) -> NDArray:
        angle_det = angle_rad + self.angles_det_rad
        atts = np.empty(self.maps.shape[1:], dtype=self.dtype)
        for ii, a in enumerate(angle_det):
            atts[ii, ...] = prj_backends.compute_attenuation(local_att, a, invert=True)
        return atts

    def compute_maps(self, use_multithreading: bool = True, verbose: bool = True) -> None:
        """
        Compute the correction maps for each angle.

        Parameters
        ----------
        use_multithreading : bool, optional
            Use multi-threading for computing the attenuation maps. The default is True.
        verbose : bool, optional
            Show verbose output. The default is True.
        """
        num_rot_angles = len(self.angles_rot_rad)
        self.maps = np.ones([num_rot_angles, len(self.angles_det_rad), *self.vol_shape_zyx], dtype=self.dtype)

        if self.incident_local is not None:
            description = "Computing attenuation maps for incident beam: "
            if use_multithreading:
                r = []
                with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # angle_atts = executor.map(self._compute_attenuation_angle_in, self.angles_rot_rad)
                    for a in self.angles_rot_rad:
                        r.append(executor.submit(self._compute_attenuation_angle_in, self.incident_local, a))
                    for ii in tqdm(range(num_rot_angles), desc=description, disable=(not verbose)):
                        self.maps[ii, ...] *= r[ii].result()
            else:
                for ii, a in enumerate(tqdm(self.angles_rot_rad, desc=description, disable=(not verbose))):
                    self.maps[ii, ...] *= self._compute_attenuation_angle_in(self.incident_local, a)

        if self.emitted_local is not None:
            description = "Computing attenuation maps for emitted photons: "
            if use_multithreading:
                r = []
                with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    for a in self.angles_rot_rad:
                        r.append(executor.submit(self._compute_attenuation_angle_out, self.emitted_local, a))
                    for ii in tqdm(range(num_rot_angles), desc=description, disable=(not verbose)):
                        self.maps[ii, ...] *= r[ii].result()
            else:
                for ii, a in enumerate(tqdm(self.angles_rot_rad, desc=description, disable=(not verbose))):
                    self.maps[ii, ...] *= self._compute_attenuation_angle_out(self.emitted_local, a)

    def plot_map(
        self,
        ax: Axes,
        rot_ind: int,
        det_ind: int = 0,
        slice_ind: Optional[int] = None,
        axes: Union[Sequence[int], NDArrayInt] = (-2, -1),
    ) -> List[float]:
        """
        Plot the requested attenuation map.

        Parameters
        ----------
        ax : matplotlib axes
            The axes where to plot.
        rot_ind : int
            Rotation angle index.
        det_ind : int, optional
            Detector angle index. The default is 0.
        slice_ind : Optional[int], optional
            Volume slice index (for 3D volumes). The default is None.
        axes : Sequence[int] | NDArray, optional
            Axes of the slice. The default is (-2, -1).

        Returns
        -------
        List[float]
            The extent of the axes plot (min-max coords).

        Raises
        ------
        ValueError
            In case a slice index is not passed for a 3D volume.
        """
        att_map = np.squeeze(self.get_maps(rot_ind=rot_ind, det_ind=det_ind))
        other_dim = np.squeeze(np.delete(np.arange(-3, 0), axes))
        if len(att_map.shape) == 3:
            if slice_ind is None:
                raise ValueError("Slice index is needed for 3D volumes. None was passed.")

            att_map = np.take(att_map, slice_ind, axis=int(other_dim))

        slice_shape = self.vol_shape_zyx[list(axes)]
        coords = [(-(s - 1) / 2, (s - 1) / 2) for s in slice_shape]

        extent = list(np.concatenate(coords))
        ax.imshow(att_map, extent=extent)

        if other_dim == -3:
            arrow_length = np.linalg.norm(slice_shape) / np.pi
            arrow_args = dict(
                width=arrow_length / 25,
                head_width=arrow_length / 8,
                head_length=arrow_length / 6,
                length_includes_head=True,
            )

            prj_geom = models.ProjectionGeometry.get_default_parallel()
            beam_i_geom = prj_geom.rotate(-self.angles_rot_rad[rot_ind])
            beam_e_geom = prj_geom.rotate(-(self.angles_rot_rad[rot_ind] + self.angles_det_rad[det_ind]))

            beam_i_dir = beam_i_geom.src_pos_xyz[0, :2] * arrow_length
            beam_i_orig = -beam_i_dir
            beam_e_dir = beam_e_geom.src_pos_xyz[0, :2] * arrow_length
            beam_e_orig = np.array([0, 0])

            ax.arrow(*beam_i_orig, *beam_i_dir, **arrow_args, fc="r", ec="r")
            ax.arrow(*beam_e_orig, *beam_e_dir, **arrow_args, fc="k", ec="k")

        return extent

    def get_maps(
        self,
        roi: Optional[ArrayLike] = None,
        rot_ind: Union[int, slice, Sequence[int], NDArrayInt, None] = None,
        det_ind: Union[int, slice, Sequence[int], NDArrayInt, None] = None,
    ) -> NDArray:
        """
        Return the attenuation maps.

        Parameters
        ----------
        roi : ArrayLike, optional
            The region-of-interest to select. The default is None.
        rot_ind : int, optional
            A specific rotation index, if only one is to be desired. The default is None.
        det_ind : int, optional
            A specific detector index, if only one is to be desired. The default is None.

        Returns
        -------
        NDArray
            The attenuation maps.
        """
        maps = self.maps

        if rot_ind is not None:
            if isinstance(rot_ind, int):
                rot_ind = slice(rot_ind, rot_ind + 1, 1)
            maps = maps[rot_ind, ...]

        if det_ind is not None:
            if isinstance(det_ind, int):
                det_ind = slice(det_ind, det_ind + 1, 1)
            maps = maps[:, det_ind, ...]

        if roi is not None:
            raise NotImplementedError("Extracting a region of interest is not supported, yet.")

        return maps

    def get_projector_args(
        self,
        roi: Optional[ArrayLike] = None,
        rot_ind: Union[int, slice, Sequence[int], NDArrayInt, None] = None,
        det_ind: Union[int, slice, Sequence[int], NDArrayInt, None] = None,
    ) -> Dict[str, NDArray]:
        """
        Return the projector arguments.

        Parameters
        ----------
        roi : ArrayLike, optional
            The region-of-interest to select. The default is None.
        rot_ind : int, optional
            A specific rotation index, if only one is to be desired. The default is None.
        det_ind : int, optional
            A specific detector index, if only one is to be desired. The default is None.

        Returns
        -------
        Dict[str, NDArray]
            A dictionary containing the attenuation maps and the detector angle.
        """
        if det_ind is None:
            det_angles = self.angles_det_rad
        else:
            det_angles = self.angles_det_rad[det_ind]
        return dict(att_maps=self.get_maps(roi=roi, rot_ind=rot_ind, det_ind=det_ind), angles_detectors_rad=det_angles)
