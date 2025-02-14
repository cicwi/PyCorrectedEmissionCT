# -*- coding: utf-8 -*-
"""
Tomographic projector backends.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import skimage
import skimage.transform as skt

from . import filters
from .models import ProjectionGeometry, VolumeGeometry

from typing import Optional, Sequence, Union, Mapping, List
from numpy.typing import ArrayLike, NDArray

from abc import ABC, abstractmethod

try:
    import astra

    has_astra = True
    has_cuda = astra.astra.use_cuda()
    if not has_cuda:
        print(
            "WARNING: CUDA is not available. Only 2D operations on CPU are available (scikit-image will be used as default)."
        )
except ImportError:
    has_astra = False
    has_cuda = False
    print("WARNING: ASTRA is not available. Only 2D operations on CPU are available (scikit-image will be used).")

if has_cuda:
    try:
        import astra.experimental

        has_astra_direct = True
    except ImportError:
        has_astra_direct = False
        print("WARNING: the experimental ASTRA direct interface is not available. The traditional interface will be used.")


def compute_attenuation(vol: NDArray, angle_rad: float, invert: bool = False) -> NDArray:
    """
    Compute the attenuation volume for the given local attenuation, and angle.

    Parameters
    ----------
    vol : NDArray
        The local attenuation volume.
    angle_rad : float
        The angle along which to compute the attenuation.
    invert : bool, optional
        Whether to invert propagation direction. The default is False.

    Returns
    -------
    NDArray
        The attenuation volume.
    """

    def pad_vol(vol: NDArray, edges: Sequence[int]):
        paddings = np.zeros((len(vol.shape), 1), dtype=int)
        paddings[-2], paddings[-1] = edges[0], edges[1]
        return np.pad(vol, paddings, mode="constant")

    def compute_cumsum(vol: NDArray, rot_angle_deg: float):
        vol = skt.rotate(vol, -rot_angle_deg, order=1, clip=False)

        vol += np.roll(vol, 1, axis=-2)
        vol = np.cumsum(vol / 2, axis=-2)

        return skt.rotate(vol, rot_angle_deg, order=1, clip=False)

    size_lims = np.array(vol.shape[-2:])
    min_size = np.ceil(np.sqrt(np.sum(size_lims**2)))
    edges = np.ceil((min_size - size_lims) / 2).astype(int)

    if invert:
        angle_rad += np.pi

    rot_angle_deg = np.rad2deg(angle_rad)

    cum_arr = pad_vol(vol, edges)

    if cum_arr.ndim > 2:
        prev_shape = np.array(cum_arr.shape, ndmin=1)
        num_slices = np.prod(prev_shape[:-2])
        cum_arr = cum_arr.reshape([num_slices, *prev_shape[-2:]])
        for ii in range(num_slices):
            cum_arr[ii] = compute_cumsum(cum_arr[ii], rot_angle_deg)
        cum_arr = cum_arr.reshape(prev_shape)
    else:
        cum_arr = compute_cumsum(cum_arr, rot_angle_deg)
    cum_arr = cum_arr[..., edges[0] : -edges[0], edges[1] : -edges[1]]

    cum_arr = np.exp(-cum_arr)

    return cum_arr


class ProjectorBackend(ABC):
    """Base abstract projector backend class. All backends should inherit from this class."""

    vol_geom: VolumeGeometry
    vol_shape_zxy: NDArray[np.integer]

    angles_w_rad: NDArray[np.floating]

    prj_shape_vwu: NDArray[np.integer]
    prj_shape_vu: NDArray[np.integer]

    is_initialized: bool
    is_ready: bool

    def __init__(self):
        """Initialize base abstract projector backend class."""
        self.is_initialized = False
        self.is_ready = False

    def initialize_geometry(
        self,
        vol_geom: VolumeGeometry,
        angles_rot_rad: Union[ArrayLike, NDArray],
        rot_axis_shift_pix: Union[ArrayLike, NDArray, None] = None,
        prj_geom: Optional[ProjectionGeometry] = None,
        create_single_projs: bool = False,
    ):
        """Initialize the projector geometry.

        Parameters
        ----------
        vol_geom : VolumeGeometry
            The volume geometry.
        angles_rot_rad : ArrayLike | NDArray
            The projection angles.
        rot_axis_shift_pix : float | NDArray | None, optional
            Relative position of the rotation center with respect to the volume center. The default is None.
        prj_geom : ProjectionGeometry, optional
            The fully specified projection geometry.
            When active, the rotation axis shift is ignored. The default is None.
        create_single_projs : bool, optional
            Whether to create projectors for single projections. Used for corrections and SART. The default is False.
        """
        self.vol_geom = vol_geom

        self.vol_shape_zxy = self.vol_geom.shape_zxy
        self.angles_w_rad = np.array(angles_rot_rad, ndmin=1, dtype=np.float32)

        self.vol_eff_shape_zxy = np.floor(self.vol_shape_zxy * self.vol_geom.vox_size).astype(int)

        # Basic sizes, unless overridden
        self.prj_shape_vwu = np.array(
            [*self.vol_eff_shape_zxy[:-2], len(self.angles_w_rad), self.vol_eff_shape_zxy[-1]], dtype=int
        )
        self.prj_shape_vu = np.array([*self.vol_eff_shape_zxy[:-2], 1, self.vol_eff_shape_zxy[-1]], dtype=int)

        self.has_individual_projs = create_single_projs

        self.is_initialized = True

    def get_vol_shape(self) -> NDArray:
        """Return the expected and produced volume shape (in ZXY coordinates).

        Returns
        -------
        NDArray
            The volume shape.
        """
        return self.vol_shape_zxy

    def get_prj_shape(self) -> NDArray:
        """Return the expected and produced projection shape (in VWU coordinates).

        Returns
        -------
        NDArray
            The projection shape.
        """
        return self.prj_shape_vwu

    def make_ready(self) -> None:
        """Initialize the projector.

        It should make sure that all the resources have been allocated.
        """
        self.is_ready = True

    def dispose(self) -> None:
        """De-initialize the projector.

        It should make sure that all the resources have been de-allocated.
        """
        self.is_ready = False

    def __del__(self):
        """De-initialize projector on deletion."""
        if self.is_ready:
            self.dispose()

    def __repr__(self) -> str:
        """Build a string representation of the projector backend.

        Returns
        -------
        str
            The representation of the projector.
        """
        class_name = f"{self.__class__.__name__}: "
        if self.is_initialized:
            return (
                class_name
                + "{\n"
                + f"  Shape vol ZXY: {self.get_vol_shape()}\n"
                + f"  Shape prj VWU: {self.get_prj_shape()}\n"
                + f"  Angles (deg): {np.rad2deg(self.angles_w_rad)}\n"
                + "}"
            )
        else:
            return class_name + "{ Not initialized! }"

    @abstractmethod
    def fp(self, vol: NDArray, angle_ind: Optional[int] = None) -> NDArray:
        """Forward-project volume.

        Forward-projection interface. Derived backends need to implement this method.

        Parameters
        ----------
        vol : NDArray
            The volume to forward-project.
        angle_ind : int, optional
            The angle index to foward project. The default is None.
        """

    @abstractmethod
    def bp(self, prj: NDArray, angle_ind: Optional[int] = None) -> NDArray:
        """Back-project data.

        Back-projection interface. Derived backends need to implement this method.

        Parameters
        ----------
        prj : NDArray
            The sinogram to back-project or a single line.
        angle_ind : int, optional
            The angle index to foward project. The default is None.
        """


class ProjectorBackendSKimage(ProjectorBackend):
    """Projector backend based on scikit-image."""

    def __init__(self) -> None:
        """Initialize scikit-image projector backend class."""
        super().__init__()
        self.is_ready = True

    def initialize_geometry(
        self,
        vol_geom: VolumeGeometry,
        angles_rot_rad: Union[ArrayLike, NDArray],
        rot_axis_shift_pix: Union[ArrayLike, NDArray, None] = None,
        prj_geom: Optional[ProjectionGeometry] = None,
        create_single_projs: bool = False,
    ):
        """Initialize projector backend based on scikit-image.

        Parameters
        ----------
        vol_geom : VolumeGeometry
            The volume shape.
        angles_rot_rad : ArrayLike
            The projection angles.
        rot_axis_shift_pix : float | NDArray | None, optional
            Relative position of the rotation center with respect to the volume center. The default is None.
            NOT SUPPORTED: if anything else than None is passed, it will throw an error!
        prj_geom : ProjectionGeometry, optional
            The fully specified projection geometry.
            When active, the rotation axis shift is ignored. The default is None.
            NOT SUPPORTED: if anything else than None is passed, it will throw an error!
        create_single_projs : bool, optional
            Whether to create projectors for single projections. Used for corrections and SART. The default is False.

        Raises
        ------
        ValueError
            In case the volume dimensionality is larger than 2D, and if a rotation axis shift is passed.
        """
        if vol_geom.is_3D():
            raise ValueError("With the scikit-image backend, only 2D volumes are allowed!")
        if rot_axis_shift_pix is not None:
            raise ValueError("With the scikit-image backend, rotation axis shifts are not supported!")
        if prj_geom is not None:
            raise ValueError("With the scikit-image backend, `ProjectionGeometry` is not supported!")

        super().initialize_geometry(vol_geom, angles_rot_rad, create_single_projs=create_single_projs)

        self.angles_w_deg = np.rad2deg(self.angles_w_rad)

    @staticmethod
    def _set_filter_name(filt):
        if skimage.__version__ >= "0.18":
            return dict(filter_name=filt)
        else:
            return dict(filter=filt)

    @staticmethod
    def _set_bpj_size(output_size):
        return dict(circle=False, output_size=output_size)

    def fp(self, vol: NDArray, angle_ind: Optional[int] = None) -> NDArray:
        """Forward-projection of the volume to the sinogram or a single sinogram line.

        Parameters
        ----------
        vol : NDArray
            The volume to forward-project.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        NDArray
            The forward-projected sinogram or sinogram line.
        """
        if angle_ind is None:
            prj = np.empty(self.prj_shape_vwu, dtype=vol.dtype)
            for ii_a, a in enumerate(self.angles_w_deg):
                prj[ii_a, :] = np.squeeze(skt.radon(vol, [a]))
            return prj
        else:
            return np.squeeze(skt.radon(vol, self.angles_w_deg[angle_ind : angle_ind + 1 :]))

    def bp(self, prj: NDArray, angle_ind: Optional[int] = None) -> NDArray:
        """Back-projection of a single sinogram line to the volume.

        Parameters
        ----------
        prj : NDArray
            The sinogram to back-project or a single line.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        NDArray
            The back-projected volume.
        """
        filter_name = self._set_filter_name(None)
        bpj_size = self._set_bpj_size(self.vol_shape_zxy[-1])
        if angle_ind is None:
            vol = np.empty([self.prj_shape_vwu[-2], *self.vol_shape_zxy], dtype=prj.dtype)
            for ii_a, a in enumerate(self.angles_w_deg):
                vol[ii_a, ...] = skt.iradon(prj[ii_a, :, np.newaxis], [a], **bpj_size, **filter_name)
            vol = vol.sum(axis=0)
        else:
            vol = skt.iradon(prj[:, np.newaxis], self.angles_w_deg[angle_ind : angle_ind + 1 :], **bpj_size, **filter_name)
        return vol * 2 / np.pi


class ProjectorBackendASTRA(ProjectorBackend):
    """Projector backend based on astra-toolbox."""

    proj_id: List

    astra_vol_geom: Mapping
    proj_geom_ind: Sequence[Mapping]
    proj_geom_all: Mapping

    def __init__(self, super_sampling: int = 1):
        """Initialize the ASTRA projector backend.

        Parameters
        ----------
        super_sampling : int, optional
            Super sampling factor for the pixels and voxels, by default 1.
        """
        super().__init__()
        self.super_sampling = super_sampling

    def initialize_geometry(
        self,
        vol_geom: VolumeGeometry,
        angles_rot_rad: Union[ArrayLike, NDArray],
        rot_axis_shift_pix: Union[ArrayLike, NDArray, None] = None,
        prj_geom: Optional[ProjectionGeometry] = None,
        create_single_projs: bool = False,
    ):
        """Initialize geometry of projector backend based on astra-toolbox.

        Parameters
        ----------
        vol_geom : VolumeGeometry
            The volume shape.
        angles_rot_rad : ArrayLike
            The projection angles.
        rot_axis_shift_pix : float | NDArray | None, optional
            Relative position of the rotation center with respect to the volume center. The default is None.
        prj_geom : ProjectionGeometry, optional
            The fully specified projection geometry.
            When active, the rotation axis shift is ignored. The default is None.
        create_single_projs : bool, optional
            Whether to create projectors for single projections. Used for corrections and SART. The default is False.

        Raises
        ------
        ValueError
            In case the volume dimensionality is larger than 2D and CUDA is not available.
        """
        if vol_geom.is_3D() and not has_cuda:
            raise ValueError("CUDA is not available: only 2D volumes are allowed!")
        if not (rot_axis_shift_pix is None or isinstance(rot_axis_shift_pix, (int, float, Sequence, np.ndarray))):
            raise ValueError(
                "Rotation axis shift should either be None or one of the following: int, a float or a sequence of floats"
                + f" ({type(rot_axis_shift_pix)} given instead)."
            )

        super().initialize_geometry(vol_geom, angles_rot_rad, create_single_projs=create_single_projs)

        self.proj_id = []
        self.dispose()

        num_angles = self.angles_w_rad.size

        if self.vol_geom.is_3D():
            self.astra_vol_geom = astra.create_vol_geom(*vol_geom.shape_xyz[list([1, 0, 2])], *self.vol_geom.extent)
            if prj_geom is None:
                prj_geom = ProjectionGeometry.get_default_parallel(geom_type="3d", rot_axis_shift_pix=rot_axis_shift_pix)

            if prj_geom.det_shape_vu is None:
                prj_geom.det_shape_vu = np.array(self.prj_shape_vwu[list([-3, -1])], dtype=int)
            else:
                # Here the projections are supposed to be larger or smaller than the sample size
                self.prj_shape_vwu = np.array([*prj_geom.det_shape_vu[:-1], num_angles, prj_geom.det_shape_vu[-1]])
                self.prj_shape_vu = np.array([*prj_geom.det_shape_vu[:-1], 1, prj_geom.det_shape_vu[-1]])

            rot_geom = prj_geom.rotate(self.angles_w_rad)

            vectors = np.empty([num_angles, 12])
            # source / beam direction
            vectors[:, 0:3] = rot_geom.get_field_scaled("src_pos_xyz")
            # center of detector
            vectors[:, 3:6] = rot_geom.get_field_scaled("det_pos_xyz")
            # vector from detector pixel (0, 0) to (0, 1)
            vectors[:, 6:9] = rot_geom.get_field_scaled("det_u_xyz")
            # vector from detector pixel (0, 0) to (1, 0)
            vectors[:, 9:12] = rot_geom.get_field_scaled("det_v_xyz")

            geom_type_str = prj_geom.geom_type
        else:
            self.astra_vol_geom = astra.create_vol_geom(*vol_geom.shape_xyz[list([1, 0])], *self.vol_geom.extent)
            if prj_geom is None:
                prj_geom = ProjectionGeometry.get_default_parallel(geom_type="2d", rot_axis_shift_pix=rot_axis_shift_pix)

            if prj_geom.det_shape_vu is None:
                prj_geom.det_shape_vu = np.array(self.prj_shape_vwu[list([-1])], dtype=int)
            else:
                # Here the projections are supposed to be larger or smaller than the sample size
                self.prj_shape_vwu = np.array([num_angles, prj_geom.det_shape_vu[-1]])
                self.prj_shape_vu = np.array([1, prj_geom.det_shape_vu[-1]])

            rot_geom = prj_geom.rotate(self.angles_w_rad, patch_astra_2d=True)

            vectors = np.empty([num_angles, 6])
            # source / beam direction
            vectors[:, 0:2] = rot_geom.get_field_scaled("src_pos_xyz")
            # center of detector
            vectors[:, 2:4] = rot_geom.get_field_scaled("det_pos_xyz")
            # vector from detector pixel 0 to 1
            vectors[:, 4:6] = rot_geom.get_field_scaled("det_u_xyz")

            geom_type_str = prj_geom.geom_type[:-2]

        vectors /= vol_geom.vox_size

        if self.has_individual_projs:
            self.proj_geom_ind = [
                astra.create_proj_geom(geom_type_str + "_vec", *prj_geom.det_shape_vu, vectors[ii : ii + 1 :, :])
                for ii in range(num_angles)
            ]

        self.proj_geom_all = astra.create_proj_geom(geom_type_str + "_vec", *prj_geom.det_shape_vu, vectors)

    # def get_vol_shape(self) -> NDArray:
    #     """Return the expected and produced volume shape (in ZYX coordinates).

    #     Returns
    #     -------
    #     NDArray
    #         The volume shape.
    #     """
    #     return astra.functions.geom_size(self.astra_vol_geom)

    # def get_prj_shape(self) -> NDArray:
    #     """Return the expected and produced projection shape (in VWU coordinates).

    #     Returns
    #     -------
    #     NDArray
    #         The projection shape.
    #     """
    #     return astra.functions.geom_size(self.proj_geom_all)

    def make_ready(self) -> None:
        """Initialize the ASTRA projectors."""
        if not self.is_ready:
            if self.vol_geom.is_3D():
                projector_type = "cuda3d"
                self.algo_type = "3D_CUDA"
                self.data_mod = astra.data3d
            else:
                if has_cuda:
                    projector_type = "cuda"
                    self.algo_type = "_CUDA"
                else:
                    projector_type = "linear"
                    self.algo_type = ""
                self.data_mod = astra.data2d

            voxel_sampling = int(self.super_sampling * np.fmax(1, self.vol_geom.vox_size))
            pixel_sampling = int(self.super_sampling / np.fmin(1, self.vol_geom.vox_size))
            opts = {"VoxelSuperSampling": voxel_sampling, "DetectorSuperSampling": pixel_sampling}

            if self.has_individual_projs:
                self.proj_id = [
                    astra.create_projector(projector_type, pg, self.astra_vol_geom, opts) for pg in self.proj_geom_ind
                ]

            self.proj_id.append(astra.create_projector(projector_type, self.proj_geom_all, self.astra_vol_geom, opts))

        super().make_ready()

    def _check_data(self, x: NDArray, expected_shape: Union[Sequence[int], NDArray[np.integer]]) -> NDArray:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        try:
            return x.reshape(np.array(expected_shape, ndmin=1))
        except ValueError:
            print(f"Could not reshape input data of shape={x.shape} into expected shape={expected_shape}")
            raise

    def _check_prj_shape(self, prj: NDArray) -> None:
        inds_vu = [ind for ind in range(-len(self.prj_shape_vu), 0) if self.prj_shape_vu[ind] > 1]
        expected_prj_shape_vu = self.prj_shape_vu[inds_vu]
        given_prj_shape_vu = np.array(prj.shape)[inds_vu]
        if np.any(given_prj_shape_vu != expected_prj_shape_vu):
            raise ValueError(
                f"Expected prj shape (VU: {expected_prj_shape_vu} - likely inferred from `vol_geom`),"
                f" and passed prj shape (VU: {given_prj_shape_vu}) from the projection data, do not match.\n"
                "If you intend to use projection shapes different from the volume shape, please adjust"
                " the `prj_geom` object's `det_shape_vu` accordingly."
            )

    def dispose(self) -> None:
        """De-initialize the ASTRA projectors."""
        for p in self.proj_id:
            astra.projector.delete(p)
        self.proj_id = []

        super().dispose()

    def fp(self, vol: NDArray, angle_ind: Optional[int] = None) -> NDArray:
        """Apply forward-projection of the volume to the sinogram or a single sinogram line.

        Parameters
        ----------
        vol : NDArray
            The volume to forward-project.
        angle_ind : int | None, optional
            The angle index to forward project. The default is None.

        Returns
        -------
        NDArray
            The forward-projected sinogram or sinogram line.
        """
        self.make_ready()

        vol = self._check_data(vol, self.vol_shape_zxy)

        if angle_ind is None:
            prj = np.empty(self.prj_shape_vwu, dtype=np.float32)
            prj_geom = self.proj_geom_all
            proj_id = self.proj_id[-1]
        else:
            if not self.has_individual_projs:
                raise ValueError("Individual projectors not available!")

            prj = np.empty(self.prj_shape_vu, dtype=np.float32)
            prj_geom = self.proj_geom_ind[angle_ind]
            proj_id = self.proj_id[angle_ind]

        vid = self.data_mod.link("-vol", self.astra_vol_geom, vol)
        try:
            sid = self.data_mod.link("-sino", prj_geom, prj)
            try:
                cfg = astra.creators.astra_dict("FP" + self.algo_type)
                cfg["ProjectionDataId"] = sid
                cfg["VolumeDataId"] = vid
                cfg["ProjectorId"] = proj_id

                fp_id = astra.algorithm.create(cfg)
                try:
                    astra.algorithm.run(fp_id)
                finally:
                    astra.algorithm.delete(fp_id)
            finally:
                self.data_mod.delete([sid])
        finally:
            self.data_mod.delete([vid])

        return np.squeeze(prj)

    def bp(self, prj: NDArray, angle_ind: Optional[int] = None) -> NDArray:
        """Apply back-projection of a single sinogram line to the volume.

        Parameters
        ----------
        prj : NDArray
            The sinogram to back-project or a single line.
        angle_ind : int | None, optional
            The angle index to forward project. The default is None.

        Returns
        -------
        NDArray
            The back-projected volume.
        """
        self.make_ready()

        self._check_prj_shape(prj)

        vol = np.empty(self.vol_shape_zxy, dtype=np.float32)
        if angle_ind is None:
            prj = self._check_data(prj, self.prj_shape_vwu)
            prj_geom = self.proj_geom_all
            proj_id = self.proj_id[-1]
        else:
            if not self.has_individual_projs:
                raise ValueError("Individual projectors not available!")

            prj = self._check_data(prj, self.prj_shape_vu)
            prj_geom = self.proj_geom_ind[angle_ind]
            proj_id = self.proj_id[angle_ind]

        vid = self.data_mod.link("-vol", self.astra_vol_geom, vol)
        try:
            sid = self.data_mod.link("-sino", prj_geom, prj)

            try:
                cfg = astra.creators.astra_dict("BP" + self.algo_type)
                cfg["ProjectionDataId"] = sid
                cfg["ReconstructionDataId"] = vid
                cfg["ProjectorId"] = proj_id
                bp_id = astra.algorithm.create(cfg)
                try:
                    astra.algorithm.run(bp_id)
                finally:
                    astra.algorithm.delete(bp_id)
            finally:
                self.data_mod.delete([sid])
        finally:
            self.data_mod.delete([vid])

        return vol


class ProjectorBackendDirectASTRA(ProjectorBackendASTRA):
    """Experimental astra-toolbox functions projector."""

    astra_vol_shape: Sequence
    astra_prj_shape: Sequence
    astra_angle_prj_shape: Sequence
    angle_prj_shape: Sequence

    def initialize_geometry(
        self,
        vol_geom: VolumeGeometry,
        angles_rot_rad: Union[ArrayLike, NDArray],
        rot_axis_shift_pix: Union[ArrayLike, NDArray, None] = None,
        prj_geom: Optional[ProjectionGeometry] = None,
        create_single_projs: bool = False,
    ):
        """Initialize projector backend based on experimental astra-toolbox functions.

        Parameters
        ----------
        vol_geom : VolumeGeometry
            The volume shape.
        angles_rot_rad : ArrayLike
            The projection angles.
        rot_axis_shift_pix : float, optional
            Relative position of the rotation center with respect to the volume center. The default is 0.0.
        geom : ProjectionGeometry, optional
            The fully specified projection geometry.
            When active, the rotation axis shift is ignored. The default is None.
        create_single_projs : bool, optional
            Whether to create projectors for single projections. Used for corrections and SART. The default is False.

        Raises
        ------
        ValueError
            In case CUDA is not available.
        """
        if not has_cuda:
            raise ValueError("CUDA is not available, but it is required for the direct functions!")
        if not has_astra_direct:
            raise ValueError("ASTRA direct is not available, but it is required!")
        if not (rot_axis_shift_pix is None or isinstance(rot_axis_shift_pix, (int, float, Sequence, np.ndarray))):
            raise ValueError(
                "Rotation axis shift should either be None or one of the following: int, a float or a sequence of floats"
                + f" ({type(rot_axis_shift_pix)} given instead)."
            )

        ProjectorBackend.initialize_geometry(self, vol_geom, angles_rot_rad, create_single_projs=create_single_projs)

        self.proj_id = []
        self.dispose()

        num_angles = self.angles_w_rad.size

        vol_geom_tmp3d = self.vol_geom.get_3d()

        self.astra_vol_geom = astra.create_vol_geom(*vol_geom_tmp3d.shape_xyz[list([1, 0, 2])], *vol_geom_tmp3d.extent)
        if prj_geom is None:
            prj_geom = ProjectionGeometry.get_default_parallel(geom_type="3d", rot_axis_shift_pix=rot_axis_shift_pix)
        else:
            if prj_geom.det_shape_vu is not None:
                # Here the projections are supposed to be larger or smaller than the sample size
                # We use the original version because it makes sure that 2D is respected.
                self.prj_shape_vwu = np.array([*prj_geom.det_shape_vu[:-1], num_angles, prj_geom.det_shape_vu[-1]])
                self.prj_shape_vu = np.array([*prj_geom.det_shape_vu[:-1], 1, prj_geom.det_shape_vu[-1]])

            prj_geom = prj_geom.get_3d()

        if prj_geom.det_shape_vu is None:
            prj_shape_vu = np.delete(self.prj_shape_vwu, obj=-2)
            prj_geom.det_shape_vu = np.ones(2, dtype=int)
            prj_geom.det_shape_vu[-len(prj_shape_vu) :] = prj_shape_vu

        rot_geom = prj_geom.rotate(self.angles_w_rad)

        vectors = np.empty([num_angles, 12])
        # source / beam direction
        vectors[:, 0:3] = rot_geom.get_field_scaled("src_pos_xyz")
        # center of detector
        vectors[:, 3:6] = rot_geom.get_field_scaled("det_pos_xyz")
        # vector from detector pixel (0, 0) to (0, 1)
        vectors[:, 6:9] = rot_geom.get_field_scaled("det_u_xyz")
        # vector from detector pixel (0, 0) to (1, 0)
        vectors[:, 9:12] = rot_geom.get_field_scaled("det_v_xyz")

        geom_type_str = prj_geom.geom_type

        if self.has_individual_projs:
            self.proj_geom_ind = [
                astra.create_proj_geom(geom_type_str + "_vec", *prj_geom.det_shape_vu, vectors[ii : ii + 1 :, :])
                for ii in range(num_angles)
            ]

        self.proj_geom_all = astra.create_proj_geom(geom_type_str + "_vec", *prj_geom.det_shape_vu, vectors)

        self.astra_vol_shape = tuple(vol_geom_tmp3d.shape_zxy)
        self.astra_prj_shape = (prj_geom.det_shape_vu[-2], num_angles, prj_geom.det_shape_vu[-1])
        self.astra_angle_prj_shape = (prj_geom.det_shape_vu[-2], 1, prj_geom.det_shape_vu[-1])
        self.angle_prj_shape = (prj_geom.det_shape_vu[-2], prj_geom.det_shape_vu[-1])

    def make_ready(self):
        """Initialize the ASTRA projectors."""
        if not self.is_ready:
            voxel_sampling = int(self.super_sampling * np.fmax(1, self.vol_geom.vox_size))
            pixel_sampling = int(self.super_sampling / np.fmin(1, self.vol_geom.vox_size))
            opts = {"VoxelSuperSampling": voxel_sampling, "DetectorSuperSampling": pixel_sampling}

            if self.has_individual_projs:
                self.proj_id = [astra.create_projector("cuda3d", pg, self.astra_vol_geom, opts) for pg in self.proj_geom_ind]

            self.proj_id.append(astra.create_projector("cuda3d", self.proj_geom_all, self.astra_vol_geom, opts))

        ProjectorBackend.make_ready(self)

    def fp(self, vol: NDArray, angle_ind: Optional[int] = None):
        """Apply forward-projection of the volume to the sinogram or a single sinogram line.

        Parameters
        ----------
        vol : NDArray
            The volume to forward-project.
        angle_ind : int | None, optional
            The angle index to forward project. The default is None.

        Returns
        -------
        NDArray
            The forward-projected sinogram or sinogram line.
        """
        self.make_ready()

        if angle_ind is None:
            prj = np.zeros(self.astra_prj_shape, dtype=np.float32)
            proj_id = self.proj_id[-1]
            out_shape = self.prj_shape_vwu
        else:
            if not self.has_individual_projs:
                raise ValueError("Individual projectors not available!")
            prj = np.zeros(self.astra_angle_prj_shape, dtype=np.float32)
            proj_id = self.proj_id[angle_ind]
            out_shape = self.angle_prj_shape

        vol = self._check_data(vol, self.astra_vol_shape)
        astra.experimental.direct_FP3D(proj_id, vol, prj)
        return prj.reshape(out_shape)

    def bp(self, prj: NDArray, angle_ind: Optional[int] = None):
        """Apply back-projection of a single sinogram line to the volume.

        Parameters
        ----------
        prj : NDArray
            The sinogram to back-project or a single line.
        angle_ind : int | None, optional
            The angle index to forward project. The default is None.

        Returns
        -------
        NDArray
            The back-projected volume.
        """
        self.make_ready()

        self._check_prj_shape(prj)

        if angle_ind is None:
            prj = self._check_data(prj, self.astra_prj_shape)
            proj_id = self.proj_id[-1]
        else:
            if not self.has_individual_projs:
                raise ValueError("Individual projectors not available!")
            prj = self._check_data(prj, self.astra_angle_prj_shape)
            proj_id = self.proj_id[angle_ind]

        vol = np.zeros(self.astra_vol_shape, dtype=np.float32)
        astra.experimental.direct_BP3D(proj_id, vol, prj)
        return vol.reshape(self.vol_shape_zxy)
