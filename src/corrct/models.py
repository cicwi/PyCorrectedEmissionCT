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

from copy import deepcopy


ROT_DIRS_VALID = ("clockwise", "counter-clockwise")


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
        self.geom_type = self.geom_type.lower()
        self.src_pos_xyz = np.array(self.src_pos_xyz, ndmin=2)
        self.det_pos_xyz = np.array(self.det_pos_xyz, ndmin=2)
        self.det_u_xyz = np.array(self.det_u_xyz, ndmin=2)
        self.det_v_xyz = np.array(self.det_v_xyz, ndmin=2)
        self.rot_dir_xyz = np.array(self.rot_dir_xyz, ndmin=2)

    def __getitem__(self, indx: Any):
        """
        Slice projection geometry along the angular direction.

        Parameters
        ----------
        indx : Any
            Indices of the slicing.
        """

        def slice_array(vecs_arr: NDArray, indx: Any):
            if len(vecs_arr.shape) > 1 and vecs_arr.shape[0] > 1:
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

    def copy(self) -> "ProjectionGeometry":
        """Deepcopy an existing geometry.

        Returns
        -------
        ProjectionGeometry
            The new instance of ProjectionGeometry
        """
        return deepcopy(self)

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
            The default parallel-beam geometry.
        """
        return get_prj_geom_parallel(geom_type=geom_type, rot_axis_shift_pix=rot_axis_shift_pix, rot_axis_dir=rot_axis_dir)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the geometry.

        Returns
        -------
        int
            The number of dimensions.
        """
        if "parallel" in self.geom_type:
            return int(self.geom_type[-2])
        elif self.geom_type.lower() == "cone":
            return 3
        elif self.geom_type.lower() == "fanflat":
            return 2
        else:
            raise ValueError(
                f"Geometry ({self.geom_type}) needs to be one of: 'parallel2d' | 'parallel3d' | 'cone' | 'fanflat'."
            )

    def get_3d(self) -> "ProjectionGeometry":
        """Return the 3D version of the geometry.

        Returns
        -------
        ProjectionGeometry
            The new geometry.
        """
        if self.ndim == 2:
            if self.det_shape_vu is not None:
                new_det_shape_vu = np.ones(2, dtype=int)
                new_det_shape_vu[-len(self.det_shape_vu) :] = self.det_shape_vu
            else:
                new_det_shape_vu = None
            return dc_replace(self, geom_type=self.geom_type.replace("2d", "3d"), det_shape_vu=new_det_shape_vu)
        else:
            return dc_replace(self)

    def set_detector_shape_vu(self, vu: Union[int, Sequence[int], NDArray]) -> None:
        """Set the detector VU shape.

        Parameters
        ----------
        vu : int | Sequence[int] | NDArray
            The VU shape of the projection data.
        """
        self.det_shape_vu = np.array(vu, ndmin=1)

    def set_detector_shifts_vu(
        self,
        det_pos_vu: Union[ArrayLike, NDArray, None] = None,
        cor_pos_u: Union[float, None] = None,
        det_dist_y: ArrayLike = 0.0,
    ) -> None:
        """
        Set the detector position in XZ, from VU (vertical, horizontal) coordinates.

        Parameters
        ----------
        det_pos_vu : ArrayLike | NDArray | None
            Detector vertical and horizontal positions. Vertical is optional.
        cor_pos_u : float | None
            Center of rotation position along U.
        det_dist_y : ArrayLike, optional
            Detector distance from origin along Y. The default is 0.0.
        """
        det_pos_vu = np.array(det_pos_vu if det_pos_vu is not None else 0.0, ndmin=2, dtype=np.float64)
        if cor_pos_u is not None:
            det_pos_vu[-1, ...] = det_pos_vu[-1, ...] + cor_pos_u

        det_dist_y = np.array(det_dist_y, ndmin=1, dtype=np.float64)
        if det_dist_y.size > 1 and (
            det_dist_y.ndim > 1 or (det_pos_vu.shape[1] > 1 and det_dist_y.size != det_pos_vu.shape[1])
        ):
            raise ValueError(
                f"Detector distance along Y (shape: {det_dist_y.shape}) should either be a scalar or a 1D array of the "
                f"same length as the detector positions (shape: {det_pos_vu.shape}), if detector positions are more than 1."
            )

        if self.det_pos_xyz.shape[0] > 1 and det_pos_vu.shape[-1] > 1 and self.det_pos_xyz.shape[0] != det_pos_vu.shape[-1]:
            raise ValueError(
                f"Current number of angles ({self.det_pos_xyz.shape[-2]}) and new number of "
                f"angles ({det_pos_vu.shape[-1]}) differ!"
            )

        self.det_pos_xyz = np.zeros((det_pos_vu.shape[-1], 3), dtype=np.float64)
        self.det_pos_xyz += self.det_u_xyz * det_pos_vu[-1, :].reshape([-1, 1])
        if self.ndim == 3 and det_pos_vu.shape[0] == 2:
            self.det_pos_xyz += self.det_v_xyz * det_pos_vu[-2, :].reshape([-1, 1])

        self.det_pos_xyz[:, 1] += det_dist_y

    def set_source_shifts_vu(self, src_pos_vu: Union[ArrayLike, NDArray, None] = None) -> None:
        """
        Set the source position in XZ, from VU (vertical, horizontal) coordinates.

        Parameters
        ----------
        src_pos_vu : ArrayLike | NDArray | None
            Source vertical and horizontal positions. Vertical is optional.
        """
        if src_pos_vu is None:
            return

        src_pos_vu = np.array(src_pos_vu, ndmin=2)

        if self.src_pos_xyz.shape[0] > 1 and src_pos_vu.shape[-1] > 1 and self.src_pos_xyz.shape[0] != src_pos_vu.shape[-1]:
            raise ValueError(
                f"Current number of angles ({self.src_pos_xyz.shape[-2]}) and new number of angles ({src_pos_vu.shape[-1]}) differ!"
            )
        src_pos_y = self.src_pos_xyz[:, 1].copy()
        self.src_pos_xyz = np.zeros((src_pos_vu.shape[-1], 3))
        self.src_pos_xyz[:, 0] = src_pos_vu[-1, :]
        self.src_pos_xyz[:, 1] = src_pos_y
        if self.ndim == 3 and src_pos_vu.shape[0] == 2:
            self.src_pos_xyz[:, 2] = src_pos_vu[-2, :]

    def set_detector_tilt(
        self,
        angles_t_rad: Union[ArrayLike, NDArray],
        tilt_axis: Union[Sequence[float], NDArray] = (0, 1, 0),
        tilt_source: bool = False,
    ) -> None:
        """
        Rotate the detector by the given angle(s) and axis(axes).

        Parameters
        ----------
        angles_t_rad : ArrayLike | NDArray
            Rotation angle(s) in radians.
        tilt_axis : Sequence[float] | NDArray, optional
            The tilt axis or axes. The default is (0, 1, 0)
        tilt_source : bool, optional
            Whether to also tilt the source. The default is False.

        Notes
        -----
        When applying multiple axes, they will be applied in order. This means
        that the application is not going to be independent.
        """
        angles = np.array(angles_t_rad, ndmin=1)[:, None]
        tilt_axis = np.array(tilt_axis, ndmin=1)
        if tilt_axis.shape[-1] != 3:
            raise ValueError(
                f"Tilt axis/axes should be three-dimensional, along the last dimension. Current shape: {tilt_axis.shape}"
            )
        if tilt_axis.ndim == 1:
            tilt_axis = tilt_axis[None, :]
        elif tilt_axis.ndim > 2:
            raise ValueError(
                f"Tilt axis/axes should be three-dimensional, along the last dimension. Current shape: {tilt_axis.shape}"
            )
        elif angles.size > 1 and tilt_axis.shape[0] != angles.shape[0]:
            raise ValueError(
                "Tilt axes and tilt angles multiplicity should match. "
                f"Current shapes: {tilt_axis.shape = }, {angles.shape = }"
            )

        for angle, axis in zip(angles, tilt_axis):
            rotations = spt.Rotation.from_rotvec(angle * axis)  # type: ignore

            if tilt_source:
                self.src_pos_xyz = rotations.apply(self.src_pos_xyz)

            self.det_u_xyz = rotations.apply(self.det_u_xyz)
            self.det_v_xyz = rotations.apply(self.det_v_xyz)

            self.det_pos_xyz = rotations.apply(self.det_pos_xyz)

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

        # Dealing with ASTRA's incoherent 2D and 3D coordinate systems.
        if patch_astra_2d and self.ndim == 2:
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
        disp_zyx = np.array(disp_zyx, ndmin=1)

        disp_dims = len(disp_zyx)
        if self.ndim != disp_dims:
            raise ValueError(f"Geometry is {self.ndim}d, while passed displacement is {disp_dims}d.")

        disp_xyz = np.flip(disp_zyx)

        if self.ndim == 2:
            return self.det_u_xyz[..., : self.ndim].dot(disp_xyz)
        else:
            return np.stack(
                [self.det_v_xyz[..., : self.ndim].dot(disp_xyz), self.det_u_xyz[..., : self.ndim].dot(disp_xyz)], axis=0
            )

    def get_pre_weights(self, det_shape_vu: Union[Sequence[int], NDArray, None] = None) -> Union[NDArray, None]:
        """Compute the pre-weights of the projector geometry (notably for cone-beam geometries).

        Parameters
        ----------
        det_shape_vu : Sequence[int] | NDArray | None, optional
            Shape of the detector in [V]U coordinates, by default None

        Returns
        -------
        NDArray | None
            The computed detector weights
        """
        if self.geom_type != "cone":
            return None
        else:
            if det_shape_vu is None:
                if self.det_shape_vu is None:
                    print("WARNING: pre-weights cannot be computed because detector shape is None.")
                    return None
                else:
                    det_shape_vu = self.det_shape_vu
            det_shape_vu = np.array(det_shape_vu, dtype=int)
            if self.det_shape_vu is not None and np.any(det_shape_vu != self.det_shape_vu):
                print("WARNING: overriding the detector shape in the computation of the pre-weights.")

            src2det_xyz = self.det_pos_xyz + self.src_pos_xyz

            pixel_coords_vu = [np.linspace(-s / 2, s / 2, int(s)) for s in det_shape_vu]
            pixel_coords_vu = np.meshgrid(*pixel_coords_vu, indexing="ij")
            pixel_coords_vu = [coords[..., None, None] for coords in pixel_coords_vu]

            pixel_coords_xyz = pixel_coords_vu[-1] * self.det_u_xyz
            if len(pixel_coords_vu) > 1:
                pixel_coords_xyz += pixel_coords_vu[-2] * self.det_v_xyz

            src2pixel_dict = np.linalg.norm(pixel_coords_xyz + src2det_xyz, axis=-1)
            src2det_dist = np.linalg.norm(src2det_xyz, axis=-1)

            pre_weights = src2det_dist / (src2pixel_dict + (src2pixel_dict == 0))
            return pre_weights.swapaxes(-2, -1)


@dataclass
class VolumeGeometry(Geometry):
    """Store the volume geometry."""

    _vol_shape_xyz: NDArray
    vox_size: float = 1.0

    def __post_init__(self):
        """Initialize the input parameters."""
        self._vol_shape_xyz = np.array(self._vol_shape_xyz, ndmin=1)

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

    def get_3d(self) -> "VolumeGeometry":
        """Return the 3D version of the geometry.

        Returns
        -------
        VolumeGeometry
            The new geometry.
        """
        if len(self._vol_shape_xyz) == 2:
            return dc_replace(self, _vol_shape_xyz=np.concatenate((self._vol_shape_xyz, [1])))
        else:
            return dc_replace(self)

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


def combine_shifts_vu(shifts_v: NDArray, shifts_u: NDArray) -> NDArray:
    """Combine vertical and horizontal shifts.

    Parameters
    ----------
    shifts_v : NDArray
        The vertical shifts
    shifts_u : NDArray
        The horizontal shifts

    Returns
    -------
    NDArray
        The combined shifts
    """
    if np.sum(np.array(shifts_v.shape) > 1) > 1:
        raise ValueError(f"Expected 1-dimensional array for vertical shifts, but an array {shifts_v.shape = } was passed")
    if np.sum(np.array(shifts_u.shape) > 1) > 1:
        raise ValueError(f"Expected 1-dimensional array for horizontal shifts, but an array {shifts_u.shape = } was passed")
    if shifts_v.size != shifts_u.size:
        raise ValueError(f"Number of vertical shifts ({shifts_v.size}) and horizontal shifts ({shifts_u.size}) should match")

    return np.stack([np.squeeze(shifts_v), np.squeeze(shifts_u)], axis=-2)


def get_rot_axis_dir(rot_axis_dir: Union[str, ArrayLike, NDArray] = "clockwise") -> NDArray:
    """Process the requested rotation axis direction and return a meaningful value.

    Parameters
    ----------
    rot_axis_dir : Union[str, ArrayLike, NDArray], optional
        The requested direction, by default "clockwise"

    Returns
    -------
    NDArray
        The vector corresponding to the rotation direction.

    Raises
    ------
    ValueError
        In case of malformed direction.
    """
    if isinstance(rot_axis_dir, str):
        if rot_axis_dir.lower() not in ROT_DIRS_VALID:
            raise ValueError(f"Rotation axis direction {rot_axis_dir} not allowed. It should be one of: {ROT_DIRS_VALID}")

        if rot_axis_dir.lower() == "clockwise":
            return np.array([0.0, 0.0, -1.0])
        else:
            return np.array([0.0, 0.0, 1.0])
    else:
        return np.array(rot_axis_dir, ndmin=1)


def _get_data_dims(data_shape: Union[Sequence[int], NDArray], data_format: str = "dvwu") -> dict[str, Union[int, None]]:
    dims: dict[str, Union[int, None]] = dict(u=None, v=None, w=None, d=None)
    for ii in range(-len(data_shape), 0):
        dims[data_format[ii]] = data_shape[ii]
    return dims


def get_prj_geom_parallel(
    *,
    geom_type: str = "3d",
    rot_axis_shift_pix: Union[ArrayLike, NDArray, None] = None,
    rot_axis_dir: Union[str, ArrayLike, NDArray] = "clockwise",
    data_shape: Union[Sequence[int], NDArray, None] = None,
    data_format: str = "dvwu",
) -> ProjectionGeometry:
    """
    Generate the default geometry for parallel beam.

    Parameters
    ----------
    geom_type : str, optional
        The geometry type. The default is "parallel3d".
    rot_axis_shift_pix : ArrayLike | NDArray | None, optional
        Rotation axis shift in pixels. The default is None.
    rot_axis_dir : str | ArrayLike | NDArray, optional
        Rotation axis direction. It can be either a string or a direction. The default is "clockwise".

    Returns
    -------
    ProjectionGeometry
        The default parallel-beam geometry.
    """
    geom_type = geom_type.lower()

    prj_geom = ProjectionGeometry(
        geom_type="parallel" + geom_type,
        src_pos_xyz=np.array([0.0, -1.0, 0.0]),
        det_pos_xyz=np.zeros(3),
        det_u_xyz=np.array([1.0, 0.0, 0.0]),
        det_v_xyz=np.array([0.0, 0.0, 1.0]),
        rot_dir_xyz=get_rot_axis_dir(rot_axis_dir),
    )

    if rot_axis_shift_pix is not None:
        rot_axis_shift_pix = np.array(rot_axis_shift_pix)
        if rot_axis_shift_pix.size == 1:
            prj_geom.set_detector_shifts_vu(cor_pos_u=float(rot_axis_shift_pix))
        else:
            prj_geom.set_detector_shifts_vu(det_pos_vu=rot_axis_shift_pix)

    if data_shape is not None:
        data_dims = _get_data_dims(data_shape, data_format)
        if data_dims["u"] is None:
            raise ValueError(
                "Could not determine data dimensions. Coordinate U cannot be undetermined."
                f" Data shape: {data_shape}, data format: {data_format}"
            )
        if geom_type == "3d":
            if data_dims["v"] is None:
                raise ValueError(
                    "Could not determine data dimensions. Coordinate V cannot be undetermined in a 3D geometry."
                    f" Data shape: {data_shape}, data format: {data_format}"
                )
            prj_geom.set_detector_shape_vu([data_dims["v"], data_dims["u"]])
        else:
            prj_geom.set_detector_shape_vu([data_dims["u"]])

    return prj_geom


def get_prj_geom_cone(
    *,
    src_to_sam_dist: float,
    rot_axis_shift_pix: Union[ArrayLike, NDArray, None] = None,
    rot_axis_dir: Union[str, ArrayLike, NDArray] = "clockwise",
    data_shape: Union[Sequence[int], NDArray, None] = None,
    data_format: str = "dvwu",
) -> ProjectionGeometry:
    """
    Generate the default geometry for parallel beam.

    Parameters
    ----------
    geom_type : str, optional
        The geometry type. The default is "parallel3d".
    rot_axis_shift_pix : ArrayLike | NDArray | None, optional
        Rotation axis shift in pixels. The default is None.
    rot_axis_dir : str | ArrayLike | NDArray, optional
        Rotation axis direction. It can be either a string or a direction. The default is "clockwise".

    Returns
    -------
    ProjectionGeometry
        The default cone-beam geometry.
    """
    prj_geom = ProjectionGeometry(
        geom_type="cone",
        src_pos_xyz=np.array([0.0, -src_to_sam_dist, 0.0]),
        det_pos_xyz=np.zeros(3),
        det_u_xyz=np.array([1.0, 0.0, 0.0]),
        det_v_xyz=np.array([0.0, 0.0, 1.0]),
        rot_dir_xyz=get_rot_axis_dir(rot_axis_dir),
    )

    if rot_axis_shift_pix is not None:
        rot_axis_shift_pix = np.array(rot_axis_shift_pix)
        if rot_axis_shift_pix.size == 1:
            prj_geom.set_detector_shifts_vu(cor_pos_u=float(rot_axis_shift_pix))
        else:
            prj_geom.set_detector_shifts_vu(det_pos_vu=rot_axis_shift_pix)

    if data_shape is not None:
        data_dims = _get_data_dims(data_shape, data_format)
        if data_dims["v"] is None or data_dims["u"] is None:
            raise ValueError(
                "Could not determine data dimensions. Coordinates UV cannot be undetermined in a cone-beam geometry."
                f" Data shape: {data_shape}, data format: {data_format}"
            )
        prj_geom.set_detector_shape_vu([data_dims["v"], data_dims["u"]])

    return prj_geom


def get_vol_geom_from_data(
    data: NDArray, padding_u: Union[int, Sequence[int], NDArray] = 0, data_format: str = "dvwu", super_sampling: int = 1
) -> VolumeGeometry:
    """
    Generate a default volume geometry from the data shape.

    Parameters
    ----------
    data : NDArray
        The data.
    padding_u : int | Sequence[int]
    data_format : str, optional
        The ordering and meaning of the dimensions in the data. The default is "dvwu".
    super_sampling: int, optional
        The super-sampling size of the voxels. The default is 1.

    Returns
    -------
    VolumeGeometry
        The default volume geometry.
    """
    data_dims = _get_data_dims(data.shape, data_format)
    if data_dims["u"] is None:
        raise ValueError(
            "Could not determine data dimensions. Coordinate U cannot be undetermined."
            f" Data shape: {data.shape}, data format: {data_format}"
        )

    if isinstance(padding_u, (Sequence, np.ndarray)):
        if len(padding_u) != 2:
            raise ValueError(
                f"Padding along U can only either be an integer or a Sequence/NDArray of 2 values. {padding_u} passed instead."
            )
        data_dims["u"] -= padding_u[0] + padding_u[1]
    else:
        data_dims["u"] -= padding_u * 2

    dims_xyz = [data_dims["u"]] * 2
    if data_dims["v"] is not None:
        dims_xyz.append(data_dims["v"])

    return VolumeGeometry(np.array(dims_xyz) * super_sampling, vox_size=1 / super_sampling)


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
    return VolumeGeometry(np.array([vol_shape_zxy[-2], vol_shape_zxy[-1], *np.flip(vol_shape_zxy[:-2])]))
