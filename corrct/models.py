#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define all the models used through-out the code.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Sequence, Union, Any

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
    src_pos_xyz: NDArray
    det_pos_xyz: NDArray
    det_u_xyz: NDArray
    det_v_xyz: NDArray
    rot_dir_xyz: NDArray
    pix2vox_ratio: float = 1
    det_shape_vu: Optional[NDArray] = None

    def __post_init__(self) -> None:
        self.src_pos_xyz = np.array(self.src_pos_xyz)
        self.det_pos_xyz = np.array(self.det_pos_xyz)
        self.det_u_xyz = np.array(self.det_u_xyz)
        self.det_v_xyz = np.array(self.det_v_xyz)
        self.rot_dir_xyz = np.array(self.rot_dir_xyz)

    def __getitem__(self, indx: Any):
        """
        Slice projection geometry along the angular direction.

        Parameters
        ----------
        indx : Any
            Indices of the slicing.
        """

        def slice_array(vecs_arr: NDArray, indx: Any):
            if len(vecs_arr.shape) > 1:
                return vecs_arr[indx, :]
            else:
                return vecs_arr

        return dc_replace(
            self,
            src_pos_xyz=slice_array(self.src_pos_xyz, indx),
            det_pos_xyz=slice_array(self.det_pos_xyz, indx),
            det_u_xyz=slice_array(self.det_u_xyz, indx),
            det_v_xyz=slice_array(self.det_v_xyz, indx),
        )

    @staticmethod
    def get_default_parallel(
        *,
        geom_type: str = "3d",
        rot_axis_shift_pix: Optional[ArrayLike] = None,
        rot_axis_dir: Union[str, ArrayLike] = "clockwise",
    ) -> "ProjectionGeometry":
        """
        Generate the default geometry for parallel beam.

        Parameters
        ----------
        geom_type : str, optional
            The geometry type. The default is "parallel3d".
        rot_axis_shift_pix : Optional[ArrayLike], optional
            Rotation axis shift in pixels. The default is None.
        rot_axis_dir : Union[str, ArrayLike], optional
            Rotation axis direction. It can be either a string or a direction. The default is "clockwise".

        Returns
        -------
        ProjectionGeometry
            The default paralle-beam geometry.
        """
        return get_prj_geom_parallel(geom_type=geom_type, rot_axis_shift_pix=rot_axis_shift_pix, rot_axis_dir=rot_axis_dir)

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

    def rotate(self, angles_w_rad: ArrayLike, patch_astra_2d: bool = False) -> "ProjectionGeometry":
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
        if patch_astra_2d and int(self.geom_type[-2]) == 2:
            angles = -angles

        rotations = spt.Rotation.from_rotvec(angles * self.rot_dir_xyz)  # type: ignore

        return dc_replace(
            self,
            src_pos_xyz=rotations.apply(self.src_pos_xyz),
            det_pos_xyz=rotations.apply(self.det_pos_xyz),
            det_u_xyz=rotations.apply(self.det_u_xyz),
            det_v_xyz=rotations.apply(self.det_v_xyz),
        )

    def get_field_scaled(self, field_name: str) -> NDArray:
        """
        Return the a field content, scaled by the pix2vox ratio.

        Parameters
        ----------
        field_name : str
            Name of the field to access.

        Returns
        -------
        NDArray
            The scaled field.
        """
        field_value = getattr(self, field_name) / self.pix2vox_ratio
        if self.geom_type.lower() != "cone" and int(self.geom_type[-2]) == 2:
            return field_value[:, :-1]
        else:
            return field_value

    def project_displacement_to_detector(self, disp_zyx: ArrayLike) -> NDArray:
        """Project a given displacement vector in the volume coordinates, over the detector.

        Parameters
        ----------
        disp_zyx : ArrayLike
            The displacement vector in volume coordinates.

        Returns
        -------
        NDArray
            The projection on u (and if applicable v) coordinates.

        Raises
        ------
        ValueError
            When projection geometry and vector dimensions don match.
        """
        geom_dims = int(self.geom_type[-2])
        disp_zyx = np.array(disp_zyx, ndmin=1)

        disp_dims = len(disp_zyx)
        if geom_dims != disp_dims:
            raise ValueError(f"Geometry is {geom_dims}d, while passed displacement is {disp_dims}d.")

        disp_xyz = np.flip(disp_zyx)

        if geom_dims == 2:
            return self.det_u_xyz[..., :geom_dims].dot(disp_xyz)
        else:
            return np.stack(
                [self.det_v_xyz[..., :geom_dims].dot(disp_xyz), self.det_u_xyz[..., :geom_dims].dot(disp_xyz)], axis=0
            )


class VolumeGeometry(Geometry):
    """Store the volume geometry."""

    _vol_shape_xyz: NDArray
    vox_size: float

    def __init__(self, vol_shape_xyz: ArrayLike, vox_size: float = 1.0):
        """Initialize the input parameters."""
        self._vol_shape_xyz = np.array(vol_shape_xyz, ndmin=1)
        self.vox_size = vox_size

    def is_square(self) -> bool:
        """Compute whether the volume is square in XY.

        Returns
        -------
        bool
            True is the volume is square in XY.
        """
        return self._vol_shape_xyz[0] == self._vol_shape_xyz[1]

    @property
    def shape_xyz(self) -> NDArray:
        """
        Return the volume shape (XYZ).

        Returns
        -------
        NDArray
            Shape of the volume (XYZ).
        """
        return self._vol_shape_xyz

    @property
    def shape_zxy(self) -> NDArray:
        """
        Return the volume shape (ZXY).

        The swap between X and Y is imposed by the astra-toolbox.

        Returns
        -------
        NDArray
            Shape of the volume (ZXY).
        """
        vol_shape_zyx = np.flip(self._vol_shape_xyz)
        return np.array([*vol_shape_zyx[:-2], vol_shape_zyx[-1], vol_shape_zyx[-2]], dtype=int)

    @property
    def mask_shape(self) -> NDArray:
        """Return the XY volume shape for circular masks.

        Returns
        -------
        NDArray
            Shape of the XY volume.
        """
        return self.shape_xyz[:2]

    @property
    def extent(self) -> Sequence[float]:
        """
        Return extent of the volume.

        Returns
        -------
        Sequence[float]
            The extent of the volume [-x, +x, -y, +y, [-z, +z]].
        """
        half_size_xyz = self._vol_shape_xyz * self.vox_size / 2
        return [hs * sign for hs in half_size_xyz for sign in [-1, +1]]

    def is_3D(self) -> bool:
        """
        Tell whether this is a 3D geometry.

        Returns
        -------
        bool
            Whether this is a 3D geometry or not.
        """
        return len(self._vol_shape_xyz) == 3 and self._vol_shape_xyz[-1] > 1

    @staticmethod
    def get_default_from_data(data: NDArray, data_format: str = "dvwu") -> "VolumeGeometry":
        """
        Generate a default volume geometry from the data shape.

        Parameters
        ----------
        data : NDArray
            The data.
        data_format : str, optional
            The ordering and meaning of the dimensions in the data. The deault is "dvwu".

        Returns
        -------
        VolumeGeometry
            The default volume geometry.
        """
        return get_vol_geom_from_data(data=data, data_format=data_format)

    @staticmethod
    def get_default_from_volume(volume: NDArray) -> "VolumeGeometry":
        """
        Generate a default volume geometry from the given volume.

        Parameters
        ----------
        volume : NDArray
            The volume.

        Returns
        -------
        VolumeGeometry
            The default volume geometry.
        """
        return get_vol_geom_from_volume(volume=volume)


def get_prj_geom_parallel(
    *,
    geom_type: str = "3d",
    rot_axis_shift_pix: Optional[ArrayLike] = None,
    rot_axis_dir: Union[str, ArrayLike] = "clockwise",
) -> ProjectionGeometry:
    """
    Generate the default geometry for parallel beam.

    Parameters
    ----------
    geom_type : str, optional
        The geometry type. The default is "parallel3d".
    rot_axis_shift_pix : Optional[ArrayLike], optional
        Rotation axis shift in pixels. The default is None.
    rot_axis_dir : Union[str, ArrayLike], optional
        Rotation axis direction. It can be either a string or a direction. The default is "clockwise".

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

    if isinstance(rot_axis_dir, str):
        if rot_axis_dir.lower() == "clockwise":
            rot_axis_dir = np.array([0, 0, -1])
        else:
            rot_axis_dir = np.array([0, 0, 1])
    else:
        rot_axis_dir = np.array(rot_axis_dir, ndmin=1)

    return ProjectionGeometry(
        geom_type="parallel" + geom_type,
        src_pos_xyz=np.array([0, -1, 0]),
        det_pos_xyz=det_pos_xyz,
        det_u_xyz=np.array([1, 0, 0]),
        det_v_xyz=np.array([0, 0, 1]),
        rot_dir_xyz=rot_axis_dir,
    )


def get_vol_geom_from_data(data: NDArray, data_format: str = "dvwu") -> VolumeGeometry:
    """
    Generate a default volume geometry from the data shape.

    Parameters
    ----------
    data : NDArray
        The data.
    data_format : str, optional
        The ordering and meaning of the dimensions in the data. The deault is "dvwu".

    Returns
    -------
    VolumeGeometry
        The default volume geometry.
    """
    dims = dict(u=[], v=[], w=[], d=[])
    for ii in range(-len(data.shape), 0):
        dims[data_format[ii]] = [data.shape[ii]]
    return VolumeGeometry([*(dims["u"] * 2), *dims["v"]])


def get_vol_geom_from_volume(volume: NDArray) -> VolumeGeometry:
    """
    Generate a default volume geometry from the given volume.

    Parameters
    ----------
    volume : NDArray
        The volume.

    Returns
    -------
    VolumeGeometry
        The default volume geometry.
    """
    vol_shape_zxy = volume.shape
    if len(vol_shape_zxy) < 2:
        raise ValueError(f"The volume should be at least 2-dimensional, but the following shape was passed: {vol_shape_zxy}")
    return VolumeGeometry([vol_shape_zxy[-2], vol_shape_zxy[-1], *np.flip(vol_shape_zxy[:-2])])
