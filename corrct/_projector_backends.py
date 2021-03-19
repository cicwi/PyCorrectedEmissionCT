# -*- coding: utf-8 -*-
"""
Tomographic projector backends.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import skimage.transform as skt

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


class ProjectorBackend(object):

    def __init__(self, vol_shape_yxz, angles_rot_rad):
        """
        Base projector backend class. All backends should inherit from this class.

        Parameters
        ----------
        vol_shape_yxz : tuple, list or np.ndarray
            The volume shape.
        angles_rot_rad : tuple, list or np.ndarray
            The projection angles.
        """
        self.vol_shape = [*vol_shape_yxz[2:], vol_shape_yxz[0], vol_shape_yxz[1]]
        self.angles_rot_rad = angles_rot_rad
        self.prj_shape = [*vol_shape_yxz[2:], len(self.angles_rot_rad), vol_shape_yxz[1]]

        self.is_initialized = False

    def get_vol_shape(self):
        """
        Returns the expected and produced volume shape (in ZYX coordinates).

        Returns
        -------
        tuple
            The volume shape.
        """
        return self.vol_shape

    def get_prj_shape(self):
        """
        Returns the expected and produced projection shape (in VWU coordinates).

        Returns
        -------
        tuple
            The projection shape.
        """
        return self.prj_shape

    def initialize(self):
        """
        Initialization of the projector.

        It should make sure that all the resources have been allocated.
        """
        self.is_initialized = True

    def dispose(self):
        """
        De-initialization of the projector.

        It should make sure that all the resources have been de-allocated.
        """
        self.is_initialized = False

    def __del__(self):
        if self.is_initialized:
            self.dispose()

    def fp(self, vol, angle_ind: int = None):
        """
        Forward-projection interface. Derived backends need to implement this method.

        Parameters
        ----------
        vol : numpy.array_like
            The volume to forward-project.
        angle_ind : int, optional
            The angle index to foward project. The default is None.
        """
        raise NotImplementedError("Method FP not implemented.")

    def bp(self, prj, angle_ind: int = None):
        """
        Back-projection interface. Derived backends need to implement this method.

        Parameters
        ----------
        prj : numpy.array_like
            The sinogram to back-project or a single line.
        angle_ind : int, optional
            The angle index to foward project. The default is None.
        """
        raise NotImplementedError("Method BP not implemented.")

    def fbp(self, prj, fbp_filter):
        """
        Filtered back-projection interface. Derived backends need to implement this method.

        Parameters
        ----------
        prj : numpy.array_like
            The sinogram or stack of sinograms.
        fbp_filter : str
            The filter to use in the filtered back-projection.
        """
        raise NotImplementedError("Method FBP not implemented.")


class ProjectorBackendSKimage(ProjectorBackend):

    def __init__(self, vol_shape, angles_rot_rad, rot_axis_shift_pix: float = 0.0):
        """
        Projector backend based on scikit-image.

        Parameters
        ----------
        vol_shape : tuple, list or np.ndarray
            The volume shape.
        angles_rot_rad : tuple, list or np.ndarray
            The projection angles.
        rot_axis_shift_pix : float, optional
            Relative position of the rotation center with respect to the volume center. The default is 0.0.

        Raises
        ------
        ValueError
            In case the volume dimensionality is larger than 2D, and if a rotation axis shift is passed.
        """
        if len(vol_shape) == 3:
            raise ValueError("With the scikit-image backend only 2D volumes are allowed!")
        if not float(rot_axis_shift_pix) == 0.0:
            raise ValueError("With the scikit-image rotation axis shift is not supported!")

        super().__init__(vol_shape, np.array(angles_rot_rad))

        self.angles_rot_deg = np.rad2deg(angles_rot_rad)
        self.is_initialized = True

    def fp(self, vol: np.ndarray, angle_ind: int = None):
        """
        Forward-projection of the volume to the sinogram or a single sinogram line.

        Parameters
        ----------
        vol : numpy.array_like
            The volume to forward-project.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        numpy.array_like
            The forward-projected sinogram or sinogram line.
        """
        if angle_ind is None:
            prj = np.empty(self.prj_shape, dtype=vol.dtype)
            for ii_a, a in enumerate(self.angles_rot_deg):
                prj[ii_a, :] = np.squeeze(skt.radon(vol, [a]))
            return prj
        else:
            return np.squeeze(skt.radon(vol, self.angles_rot_deg[angle_ind:angle_ind+1:]))

    def bp(self, prj: np.ndarray, angle_ind: int = None):
        """
        Back-projection of a single sinogram line to the volume.

        Parameters
        ----------
        prj : numpy.array_like
            The sinogram to back-project or a single line.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        numpy.array_like
            The back-projected volume.
        """
        if angle_ind is None:
            vol = np.empty([self.prj_shape[-1], *self.vol_shape], dtype=prj.dtype)
            for ii_a, a in enumerate(self.angles_rot_deg):
                vol[ii_a, ...] = skt.iradon(prj[ii_a, :, np.newaxis], [a], filter=None)
            return vol.sum(axis=0)
        else:
            return skt.iradon(prj[:, np.newaxis], self.angles_rot_deg[angle_ind:angle_ind+1:], filter=None)

    def fbp(self, prj: np.ndarray, fbp_filter: str):
        """
        Filtered back-projection of a sinogram or stack of sinograms

        Parameters
        ----------
        prj : numpy.array_like
            The sinogram or stack of sinograms.
        fbp_filter : str
            The filter to use in the filtered back-projection.

        Returns
        -------
        vol : numpy.array_like
            The reconstructed volume.
        """
        fbp_filter = fbp_filter.lower()
        if len(prj.shape) > 2:
            num_lines = prj.shape[1]
            vol = np.empty([num_lines, *self.vol_shape], dtype=prj.dtype)

            for ii_v in range(num_lines):
                vol[ii_v, ...] = skt.iradon(prj[ii_v, ...].transpose(), self.angles_rot_deg, filter=fbp_filter)
            return vol
        else:
            return skt.iradon(prj.transpose(), self.angles_rot_deg, filter=fbp_filter)


class ProjectorBackendASTRA(ProjectorBackend):

    def __init__(
        self,
        vol_shape,
        angles_rot_rad,
        rot_axis_shift_pix: float = 0.0,
        create_single_projs: bool = True,
        super_sampling: int = 1
    ):
        """
        Projector backend based on astra-toolbox.

        Parameters
        ----------
        vol_shape : tuple, list or np.ndarray
            The volume shape.
        angles_rot_rad : tuple, list or np.ndarray
            The projection angles.
        rot_axis_shift_pix : float, optional
            Relative position of the rotation center with respect to the volume center. The default is 0.0.
        create_single_projs : bool, optional
            Whether to create projectors for single projections. Used for corrections and SART. The default is True.
        super_sampling : int, optional
            pixel and voxel super-sampling. The default is 1.

        Raises
        ------
        ValueError
            In case the volume dimensionality is larger than 2D and CUDA is not available.
        """
        if len(vol_shape) == 3 and not has_cuda:
            raise ValueError("CUDA is not available: only 2D volumes are allowed!")

        super().__init__(vol_shape, np.array(angles_rot_rad))

        self.proj_id = []
        self.has_individual_projs = create_single_projs
        self.super_sampling = super_sampling
        self.dispose()

        num_angles = self.angles_rot_rad.size

        self.is_3d = len(vol_shape) == 3
        if self.is_3d:
            self.vol_geom = astra.create_vol_geom((vol_shape[1], vol_shape[0], vol_shape[2]))

            vectors = np.empty([num_angles, 12])
            # source
            vectors[:, 0] = -np.sin(self.angles_rot_rad)
            vectors[:, 1] = -np.cos(self.angles_rot_rad)
            vectors[:, 2] = 0
            # vector from detector pixel (0,0) to (0,1)
            vectors[:, 6] = np.cos(self.angles_rot_rad)
            vectors[:, 7] = -np.sin(self.angles_rot_rad)
            vectors[:, 8] = 0
            # center of detector
            vectors[:, 3:6] = rot_axis_shift_pix * vectors[:, 6:9]
            # vector from detector pixel (0,0) to (1,0)
            vectors[:, 9] = 0
            vectors[:, 10] = 0
            vectors[:, 11] = 1

            if self.has_individual_projs:
                self.proj_geom_ind = [
                    astra.create_proj_geom(
                        "parallel3d_vec", vol_shape[2], vol_shape[0], np.tile(np.reshape(vectors[ii, :], [1, -1]), [2, 1])
                    )
                    for ii in range(num_angles)
                ]

            self.proj_geom_all = astra.create_proj_geom("parallel3d_vec", vol_shape[2], vol_shape[0], vectors)
        else:
            self.vol_geom = astra.create_vol_geom((vol_shape[1], vol_shape[0]))

            vectors = np.empty([num_angles, 6])
            # source
            vectors[:, 0] = np.sin(self.angles_rot_rad)
            vectors[:, 1] = -np.cos(self.angles_rot_rad)
            # vector from detector pixel 0 to 1
            vectors[:, 4] = np.cos(self.angles_rot_rad)
            vectors[:, 5] = np.sin(self.angles_rot_rad)
            # center of detector
            vectors[:, 2:4] = rot_axis_shift_pix * vectors[:, 4:6]

            if self.has_individual_projs:
                self.proj_geom_ind = [
                    astra.create_proj_geom(
                        "parallel_vec", vol_shape[0], np.tile(np.reshape(vectors[ii, :], [1, -1]), [2, 1])
                    )
                    for ii in range(num_angles)
                ]

            self.proj_geom_all = astra.create_proj_geom("parallel_vec", vol_shape[0], vectors)

    def get_vol_shape(self):
        """
        Returns the expected and produced volume shape (in ZYX coordinates).

        Returns
        -------
        tuple
            The volume shape.
        """
        return astra.functions.geom_size(self.vol_geom)

    def get_prj_shape(self):
        """
        Returns the expected and produced projection shape (in VWU coordinates).

        Returns
        -------
        tuple
            The projection shape.
        """
        return astra.functions.geom_size(self.proj_geom_all)

    def initialize(self):
        """Initialization of the ASTRA projectors."""
        if not self.is_initialized:
            if self.is_3d:
                projector_type = "cuda3d"
            else:
                if has_cuda:
                    projector_type = "cuda"
                else:
                    projector_type = "linear"

            opts = {"VoxelSuperSampling": self.super_sampling, "DetectorSuperSampling": self.super_sampling}

            if self.has_individual_projs:
                self.proj_id = [astra.create_projector(projector_type, pg, self.vol_geom, opts) for pg in self.proj_geom_ind]
                self.W_ind = [astra.OpTomo(p_id) for p_id in self.proj_id]

            self.proj_id.append(astra.create_projector(projector_type, self.proj_geom_all, self.vol_geom, opts))
            self.W_all = astra.OpTomo(self.proj_id[-1])

        super().initialize()

    def dispose(self):
        """De-initialization of the ASTRA projectors."""
        for p in self.proj_id:
            astra.projector.delete(p)
        self.proj_id = []
        self.W_ind = []
        self.W_all = []

        super().dispose()

    def fp(self, vol, angle_ind: int = None):
        """
        Forward-projection of the volume to the sinogram or a single sinogram line.

        Parameters
        ----------
        vol : numpy.array_like
            The volume to forward-project.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        numpy.array_like
            The forward-projected sinogram or sinogram line.
        """
        self.initialize()

        if angle_ind is None:
            return self.W_all.FP(vol)
        else:
            if not self.has_individual_projs:
                raise ValueError("Individual projectors not available!")
            return self.W_ind[angle_ind].FP(vol)[0, ...]

    def bp(self, prj, angle_ind: int = None):
        """
        Back-projection of a single sinogram line to the volume.

        Parameters
        ----------
        prj : numpy.array_like
            The sinogram to back-project or a single line.
        angle_ind : int, optional
            The angle index to foward project. The default is None.

        Returns
        -------
        numpy.array_like
            The back-projected volume.
        """
        self.initialize()

        if angle_ind is None:
            return self.W_all.BP(prj)
        else:
            if not self.has_individual_projs:
                raise ValueError("Individual projectors not available!")
            sino = np.empty([2, *np.squeeze(prj).shape], dtype=prj.dtype)
            sino[0, ...] = prj
            sino[1, ...] = 0
            return self.W_ind[angle_ind].BP(sino)

    def fbp(self, prj, fbp_filter):
        """
        Filtered back-projection of a sinogram or stack of sinograms

        Parameters
        ----------
        prj : numpy.array_like
            The sinogram or stack of sinograms.
        fbp_filter : str
            The filter to use in the filtered back-projection.

        Returns
        -------
        vol : numpy.array_like
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
                sino_id = astra.data2d.create("-sino", proj_geom, prj[:, ii_v, :])
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
            sino_id = astra.data2d.create("-sino", proj_geom, prj)
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
