# -*- coding: utf-8 -*-
"""
Tomographic projector backends.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import skimage
import skimage.transform as skt

from .models import ProjectionGeometry, VolumeGeometry

from typing import Optional
from numpy.typing import ArrayLike

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


class ProjectorBackend(ABC):
    """Initialize base abstract projector backend class. All backends should inherit from this class.

    Parameters
    ----------
    vol_geom : VolumeGeometry
        The volume geometry.
    angles_rot_rad : tuple, list or ArrayLike
        The projection angles.
    """

    def __init__(self, vol_geom: VolumeGeometry, angles_rot_rad: ArrayLike):
        self.vol_geom = vol_geom

        self.vol_shape_zxy = [*self.vol_geom.shape[2:], self.vol_geom.shape[0], self.vol_geom.shape[1]]
        self.angles_w_rad = np.array(angles_rot_rad, ndmin=1)

        # Basic sizes, unless overridden
        self.prj_shape_vwu = [*self.vol_geom.shape[2:], len(self.angles_w_rad), self.vol_geom.shape[1]]
        self.prj_shape_vu = [*self.vol_geom.shape[2:], 1, self.vol_geom.shape[1]]

        self.is_initialized = False

    def get_vol_shape(self) -> ArrayLike:
        """Return the expected and produced volume shape (in ZXY coordinates).

        Returns
        -------
        tuple
            The volume shape.
        """
        return self.vol_shape_zxy

    def get_prj_shape(self) -> ArrayLike:
        """Return the expected and produced projection shape (in VWU coordinates).

        Returns
        -------
        tuple
            The projection shape.
        """
        return self.prj_shape_vwu

    def initialize(self) -> None:
        """Initialize the projector.

        It should make sure that all the resources have been allocated.
        """
        self.is_initialized = True

    def dispose(self) -> None:
        """De-initialize the projector.

        It should make sure that all the resources have been de-allocated.
        """
        self.is_initialized = False

    def __del__(self):
        """De-initialize projector on deletion."""
        if self.is_initialized:
            self.dispose()

    @abstractmethod
    def fp(self, vol: ArrayLike, angle_ind: int = None) -> ArrayLike:
        """Forward-project volume.

        Forward-projection interface. Derived backends need to implement this method.

        Parameters
        ----------
        vol : ArrayLike
            The volume to forward-project.
        angle_ind : int, optional
            The angle index to foward project. The default is None.
        """
        raise NotImplementedError("Method FP not implemented.")

    @abstractmethod
    def bp(self, prj: ArrayLike, angle_ind: int = None) -> ArrayLike:
        """Back-project data.

        Back-projection interface. Derived backends need to implement this method.

        Parameters
        ----------
        prj : ArrayLike
            The sinogram to back-project or a single line.
        angle_ind : int, optional
            The angle index to foward project. The default is None.
        """
        raise NotImplementedError("Method BP not implemented.")

    @abstractmethod
    def fbp(self, prj: ArrayLike, fbp_filter) -> ArrayLike:
        """Apply FBP.

        Filtered back-projection interface. Derived backends need to implement this method.

        Parameters
        ----------
        prj : ArrayLike
            The sinogram or stack of sinograms.
        fbp_filter : str
            The filter to use in the filtered back-projection.
        """
        raise NotImplementedError("Method FBP not implemented.")

    @staticmethod
    def compute_attenuation(vol: ArrayLike, angle_rad: float, invert: bool = False) -> ArrayLike:
        """
        Compute the attenuation volume for the given local attenuation, and angle.

        Parameters
        ----------
        vol : ArrayLike
            The local attenuation volume.
        angle_rad : float
            The angle along which to compute the attenuation.
        invert : bool, optional
            Whether to invert propagation direction. The default is False.

        Returns
        -------
        ArrayLike
            The attenuation volume.
        """
        def pad_vol(vol, edges):
            paddings = [(0,)] * len(vol.shape)
            paddings[-2], paddings[-1] = (edges[0],), (edges[1],)
            return np.pad(vol, paddings, mode="constant")

        def compute_cumsum(vol, angle_deg):
            vol = skt.rotate(vol, -rot_angle_deg, order=1, clip=False)

            vol += np.roll(vol, 1, axis=-2)
            vol = np.cumsum(vol / 2, axis=-2)

            return skt.rotate(vol, rot_angle_deg, order=1, clip=False)

        size_lims = np.array(vol.shape[-2:])
        min_size = np.ceil(np.sqrt(np.sum(size_lims ** 2)))
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


class ProjectorBackendSKimage(ProjectorBackend):
    """Initialize projector backend based on scikit-image.

    Parameters
    ----------
    vol_geom : VolumeGeometry
        The volume shape.
    angles_rot_rad : tuple, list or ArrayLike
        The projection angles.
    rot_axis_shift_pix : float, optional
        Relative position of the rotation center with respect to the volume center. The default is 0.0.

    Raises
    ------
    ValueError
        In case the volume dimensionality is larger than 2D, and if a rotation axis shift is passed.
    """

    def __init__(self, vol_geom: VolumeGeometry, angles_rot_rad: ArrayLike, rot_axis_shift_pix: float = 0.0):
        if vol_geom.is_3D():
            raise ValueError("With the scikit-image backend only 2D volumes are allowed!")
        if not float(rot_axis_shift_pix) == 0.0:
            raise ValueError("With the scikit-image rotation axis shift is not supported!")

        super().__init__(vol_geom, angles_rot_rad)

        self.angles_w_deg = np.rad2deg(self.angles_w_rad)
        self.is_initialized = True

    @staticmethod
    def _set_filter_name(filt):
        if skimage.__version__ >= "0.18":
            return dict(filter_name=filt)
        else:
            return dict(filter=filt)

    @staticmethod
    def _set_bpj_size(output_size):
        return dict(circle=False, output_size=output_size)

    def fp(self, vol: ArrayLike, angle_ind: int = None) -> ArrayLike:
        """Forward-projection of the volume to the sinogram or a single sinogram line.

        Parameters
        ----------
        vol : ArrayLike
            The volume to forward-project.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        ArrayLike
            The forward-projected sinogram or sinogram line.
        """
        if angle_ind is None:
            prj = np.empty(self.prj_shape_vwu, dtype=vol.dtype)
            for ii_a, a in enumerate(self.angles_w_deg):
                prj[ii_a, :] = np.squeeze(skt.radon(vol, [a]))
            return prj
        else:
            return np.squeeze(skt.radon(vol, self.angles_w_deg[angle_ind : angle_ind + 1 :]))

    def bp(self, prj: ArrayLike, angle_ind: int = None) -> ArrayLike:
        """Back-projection of a single sinogram line to the volume.

        Parameters
        ----------
        prj : ArrayLike
            The sinogram to back-project or a single line.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        ArrayLike
            The back-projected volume.
        """
        filter_name = self._set_filter_name(None)
        bpj_size = self._set_bpj_size(self.vol_shape_zxy[-1])
        if angle_ind is None:
            vol = np.empty([self.prj_shape_vwu[-2], *self.vol_shape_zxy], dtype=prj.dtype)
            for ii_a, a in enumerate(self.angles_w_deg):
                vol[ii_a, ...] = skt.iradon(prj[ii_a, :, np.newaxis], [a], **bpj_size, **filter_name)
            return vol.sum(axis=0)
        else:
            return skt.iradon(prj[:, np.newaxis], self.angles_w_deg[angle_ind : angle_ind + 1 :], **bpj_size, **filter_name)

    def fbp(self, prj: ArrayLike, fbp_filter: str) -> ArrayLike:
        """Apply filtered back-projection of a sinogram or stack of sinograms.

        Parameters
        ----------
        prj : ArrayLike
            The sinogram or stack of sinograms.
        fbp_filter : str
            The filter to use in the filtered back-projection.

        Returns
        -------
        vol : ArrayLike
            The reconstructed volume.
        """
        filter_name = self._set_filter_name(fbp_filter.lower())
        bpj_size = self._set_bpj_size(self.vol_shape_zxy[-1])
        if len(prj.shape) > 2:
            num_lines = prj.shape[1]
            vol = np.empty([num_lines, *self.vol_shape_zxy], dtype=prj.dtype)

            for ii_v in range(num_lines):
                vol[ii_v, ...] = skt.iradon(prj[ii_v, ...].transpose(), self.angles_w_deg, **bpj_size, **filter_name)
            return vol
        else:
            return skt.iradon(prj.transpose(), self.angles_w_deg, **bpj_size, **filter_name)


class ProjectorBackendASTRA(ProjectorBackend):
    """Initialize projector backend based on astra-toolbox.

    Parameters
    ----------
    vol_geom : VolumeGeometry
        The volume shape.
    angles_rot_rad : tuple, list or ArrayLike
        The projection angles.
    rot_axis_shift_pix : float, optional
        Relative position of the rotation center with respect to the volume center. The default is 0.0.
    prj_geom : ProjectionGeometry, optional
        The fully specified projection geometry.
        When active, the rotation axis shift is ignored. The default is None.
    create_single_projs : bool, optional
        Whether to create projectors for single projections. Used for corrections and SART. The default is True.
    super_sampling : int, optional
        pixel and voxel super-sampling. The default is 1.

    Raises
    ------
    ValueError
        In case the volume dimensionality is larger than 2D and CUDA is not available.
    """

    def __init__(
        self,
        vol_geom: VolumeGeometry,
        angles_rot_rad: ArrayLike,
        rot_axis_shift_pix: float = 0.0,
        prj_geom: Optional[ProjectionGeometry] = None,
        create_single_projs: bool = True,
        super_sampling: int = 1,
    ):
        if vol_geom.is_3D() and not has_cuda:
            raise ValueError("CUDA is not available: only 2D volumes are allowed!")
        if not isinstance(rot_axis_shift_pix, (int, float, np.ndarray)):
            raise ValueError(
                "Rotation axis shift should either be an int, a float or a sequence of floats"
                + f" ({type(rot_axis_shift_pix)} given instead)."
            )

        super().__init__(vol_geom, angles_rot_rad)

        self.proj_id = []
        self.has_individual_projs = create_single_projs
        self.super_sampling = super_sampling
        self.dispose()

        num_angles = self.angles_w_rad.size

        if self.vol_geom.is_3D():
            self.astra_vol_geom = astra.create_vol_geom(*vol_geom.shape[list([1, 0, 2])], *self.vol_geom.extent)
            if prj_geom is None:
                prj_geom = ProjectionGeometry.get_default_parallel(geom_type="3d", rot_axis_shift_pix=rot_axis_shift_pix)

            if prj_geom.det_shape_vu is None:
                prj_geom.det_shape_vu = np.array(self.vol_geom.shape[list([2, 1])], dtype=int)
            else:
                self.prj_shape_vwu = [prj_geom.det_shape_vu[0], num_angles, prj_geom.det_shape_vu[1]]
                self.prj_shape_vu = [prj_geom.det_shape_vu[0], 1, prj_geom.det_shape_vu[1]]

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
            self.astra_vol_geom = astra.create_vol_geom(*vol_geom.shape[list([1, 0])], *self.vol_geom.extent)
            if prj_geom is None:
                prj_geom = ProjectionGeometry.get_default_parallel(geom_type="2d", rot_axis_shift_pix=rot_axis_shift_pix)

            if prj_geom.det_shape_vu is None:
                prj_geom.det_shape_vu = np.array(self.vol_geom.shape[list([1])], dtype=int)

            rot_geom = prj_geom.rotate(self.angles_w_rad)

            vectors = np.empty([num_angles, 6])
            # source / beam direction
            vectors[:, 0:2] = rot_geom.get_field_scaled("src_pos_xyz")
            # center of detector
            vectors[:, 2:4] = rot_geom.get_field_scaled("det_pos_xyz")
            # vector from detector pixel 0 to 1
            vectors[:, 4:6] = rot_geom.get_field_scaled("det_u_xyz")

            geom_type_str = prj_geom.geom_type[:-2]

        if self.has_individual_projs:
            self.proj_geom_ind = [
                astra.create_proj_geom(geom_type_str + "_vec", *prj_geom.det_shape_vu, vectors[ii : ii + 1 :, :])
                for ii in range(num_angles)
            ]

        self.proj_geom_all = astra.create_proj_geom(geom_type_str + "_vec", *prj_geom.det_shape_vu, vectors)

    def get_vol_shape(self) -> ArrayLike:
        """Return the expected and produced volume shape (in ZYX coordinates).

        Returns
        -------
        tuple
            The volume shape.
        """
        return astra.functions.geom_size(self.astra_vol_geom)

    def get_prj_shape(self) -> ArrayLike:
        """Return the expected and produced projection shape (in VWU coordinates).

        Returns
        -------
        tuple
            The projection shape.
        """
        return astra.functions.geom_size(self.proj_geom_all)

    def initialize(self) -> None:
        """Initialize the ASTRA projectors."""
        if not self.is_initialized:
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

        super().initialize()

    def _check_data(self, x: ArrayLike, expected_shape: ArrayLike) -> ArrayLike:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        try:
            return x.reshape(expected_shape)
        except ValueError:
            print(f"Could not reshape input data of shape={x.shape} into expected shape={expected_shape}")
            raise

    def dispose(self) -> None:
        """De-initialize the ASTRA projectors."""
        for p in self.proj_id:
            astra.projector.delete(p)
        self.proj_id = []

        super().dispose()

    def fp(self, vol: ArrayLike, angle_ind: int = None) -> ArrayLike:
        """Apply forward-projection of the volume to the sinogram or a single sinogram line.

        Parameters
        ----------
        vol : ArrayLike
            The volume to forward-project.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        ArrayLike
            The forward-projected sinogram or sinogram line.
        """
        self.initialize()

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

        try:
            vid = self.data_mod.link("-vol", self.astra_vol_geom, vol)
            sid = self.data_mod.link("-sino", prj_geom, prj)

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
            self.data_mod.delete([vid, sid])

        return np.squeeze(prj)

    def bp(self, prj: ArrayLike, angle_ind: int = None) -> ArrayLike:
        """Apply back-projection of a single sinogram line to the volume.

        Parameters
        ----------
        prj : ArrayLike
            The sinogram to back-project or a single line.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        ArrayLike
            The back-projected volume.
        """
        self.initialize()

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

        try:
            vid = self.data_mod.link("-vol", self.astra_vol_geom, vol)
            sid = self.data_mod.link("-sino", prj_geom, prj)

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
            self.data_mod.delete([vid, sid])

        return vol

    def fbp(self, prj: ArrayLike, fbp_filter: str) -> ArrayLike:
        """Apply filtered back-projection of a sinogram or stack of sinograms.

        Parameters
        ----------
        prj : ArrayLike
            The sinogram or stack of sinograms.
        fbp_filter : str
            The filter to use in the filtered back-projection.

        Returns
        -------
        vol : ArrayLike
            The reconstructed volume.
        """
        self.initialize()

        if has_cuda:
            fbp_type = "FBP_CUDA"
        else:
            fbp_type = "FBP"
        cfg = astra.astra_dict(fbp_type)
        cfg["ProjectorId"] = self.proj_id[-1]
        cfg["FilterType"] = fbp_filter

        proj_geom = astra.projector.projection_geometry(self.proj_id[-1])
        vol_geom = astra.projector.volume_geometry(self.proj_id[-1])

        if len(prj.shape) > 2:
            num_lines = prj.shape[1]
            vols = [None] * num_lines

            for ii_v in range(num_lines):
                sino_id = astra.data2d.link("-sino", proj_geom, prj[:, ii_v, :])
                vol_id = astra.data2d.create("-vol", vol_geom, 0)

                cfg["ProjectionDataId"] = sino_id
                cfg["ReconstructionDataId"] = vol_id
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
                astra.algorithm.delete(alg_id)

                vols[ii_v] = astra.data2d.get(vol_id)
                astra.data2d.delete(vol_id)
                astra.data2d.delete(sino_id)

            return np.stack(vols, axis=0)
        else:
            sino_id = astra.data2d.link("-sino", proj_geom, prj)
            vol_id = astra.data2d.create("-vol", vol_geom, 0)

            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = vol_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            astra.algorithm.delete(alg_id)

            vol = astra.data2d.get(vol_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)

            return vol
