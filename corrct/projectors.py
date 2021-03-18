# -*- coding: utf-8 -*-
"""
Tomographic projectors.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

import scipy.ndimage as spimg
import scipy.signal as spsig

import copy as cp

from . import operators

try:
    import astra

    has_astra = True
    has_cuda = astra.astra.use_cuda()
    if not has_cuda:
        print("WARNING: CUDA is not available. Only 2D operations on CPU are available.")
except ImportError:
    has_astra = False
    print("WARNING: ASTRA is not available. Only 2D operations on CPU are available (scikit-image will be used).")

import skimage.transform as skt


class ProjectorBase(operators.ProjectorOperator):
    """Basic projection class.

    It implements the forward and back projection of the single lines of a sinogram.
    It takes care of initializing and disposing the ASTRA projectors when used in a *with* statement.
    It supports both 2D and 3D geometries.
    """

    class ProjectorBackend(object):

        def __init__(self, vol_shape_yxz, angles_rot_rad):
            self.vol_shape = [*vol_shape_yxz[2:], vol_shape_yxz[0], vol_shape_yxz[1]]
            self.angles_rot_rad = angles_rot_rad
            self.prj_shape = [*vol_shape_yxz[2:], len(self.angles_rot_rad), vol_shape_yxz[1]]

        def get_vol_shape(self):
            return self.vol_shape

        def get_prj_shape(self):
            return self.prj_shape

        def initialize(self):
            pass

        def dispose(self):
            pass

        def fp(self, vol, angle_ind: int = None):
            raise NotImplementedError("Method FP not implemented.")

        def bp(self, prj, angle_ind: int = None):
            raise NotImplementedError("Method BP not implemented.")

    class ProjectorBackendSKimage(ProjectorBackend):

        def __init__(self, vol_shape, angles_rot_rad, rot_axis_shift_pix: float = 0.0):
            if len(vol_shape) == 3:
                raise ValueError("With the scikit-image backend only 2D volumes are allowed!")
            if not float(rot_axis_shift_pix) == 0.0:
                raise ValueError("With the scikit-image rotation axis shift is not supported!")

            self.angles_rot_deg = np.rad2deg(angles_rot_rad)

            super().__init__(vol_shape, np.array(angles_rot_rad))

        def fp(self, vol, angle_ind: int = None):
            if angle_ind is None:
                prj = np.empty(self.prj_shape, dtype=vol.dtype)
                for ii_a, a in enumerate(self.angles_rot_deg):
                    prj[ii_a, :] = np.squeeze(skt.radon(vol, [a]))
                return prj
            else:
                return np.squeeze(skt.radon(vol, self.angles_rot_deg[angle_ind:angle_ind+1:]))

        def bp(self, prj, angle_ind: int = None):
            if angle_ind is None:
                vol = np.empty([self.prj_shape[-1], *self.vol_shape], dtype=prj.dtype)
                for ii_a, a in enumerate(self.angles_rot_deg):
                    vol[ii_a, ...] = skt.iradon(prj[ii_a, :, np.newaxis], [a], filter=None)
                return prj.sum(axis=0)
            else:
                return skt.iradon(prj[:, np.newaxis], self.angles_rot_deg[angle_ind:angle_ind+1:], filter=None)

    class ProjectorBackendASTRA(ProjectorBackend):

        def __init__(
            self,
            vol_shape,
            angles_rot_rad,
            rot_axis_shift_pix: float = 0.0,
            create_single_projs: bool = True,
            super_sampling: int = 1
        ):
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
            return astra.functions.geom_size(self.vol_geom)

        def get_prj_shape(self):
            return astra.functions.geom_size(self.proj_geom_all)

        def initialize(self):
            """Initialization of the ASTRA projectors."""
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

        def dispose(self):
            """Disposal of the ASTRA projectors."""
            for p in self.proj_id:
                astra.projector.delete(p)
            self.proj_id = []
            self.W_ind = []
            self.W_all = []

        def fp(self, vol, angle_ind: int = None):
            if angle_ind is None:
                return self.W_all.FP(vol)
            else:
                if not self.has_individual_projs:
                    raise ValueError("Individual projectors not available!")
                return self.W_ind[angle_ind].FP(vol)[0, ...]

        def bp(self, prj, angle_ind: int = None):
            if angle_ind is None:
                return self.W_all.BP(prj)
            else:
                if not self.has_individual_projs:
                    raise ValueError("Individual projectors not available!")
                sino = np.empty([2, *np.squeeze(prj).shape], dtype=prj.dtype)
                sino[0, ...] = prj
                sino[1, ...] = 0
                return self.W_ind[angle_ind].BP(sino)

    def __init__(
        self,
        vol_shape,
        angles_rot_rad,
        rot_axis_shift_pix: float = 0.0,
        proj_intensities: np.ndarray = None,
        use_astra: bool = True,
        create_single_projs: bool = True,
        super_sampling: int = 1
    ):
        """
        ProjectorBase class initialization.

        Parameters
        ----------
        vol_shape : numpy.array_like
            The volume shape in Y X and optionally Z.
        angles_rot_rad : numpy.array_like
            The rotation angles.
        rot_axis_shift_pix : float or numpy.array_like, optional
            The rotation axis shift(s) in pixels. The default is 0.
        proj_intensities : float or numpy.array_like, optional
            Projection scaling factor. The default is None.
        use_astra : bool, optional
            Whether to use ASTRA or fall back to scikit-image. The default is True.
        create_single_projs : bool, optional
            Whether to create projectors for single projections. Used for corrections and SART. The default is True.
        super_sampling : int, optional
            pixel and voxel super-sampling. The default is 1.

        Raises
        ------
        ValueError
            When the geometry is not correct.

        Returns
        -------
        None.
        """
        if not has_astra and use_astra:
            use_astra = False
            print("WARNING: ASTRA requested but not available. Falling back to scikit-image.")

        if len(vol_shape) < 2 or len(vol_shape) > 3:
            raise ValueError("Only 2D or 3D volumes")
        if not vol_shape[0] == vol_shape[1]:
            raise ValueError("Only square volumes")

        if use_astra:
            self.projector_backend = ProjectorBase.ProjectorBackendASTRA(
                vol_shape,
                angles_rot_rad,
                rot_axis_shift_pix=rot_axis_shift_pix,
                create_single_projs=create_single_projs,
                super_sampling=super_sampling
            )
        else:
            self.projector_backend = ProjectorBase.ProjectorBackendSKimage(
                vol_shape, angles_rot_rad, rot_axis_shift_pix=rot_axis_shift_pix
            )

        self.angles_rot_rad = angles_rot_rad
        self.is_3d = len(vol_shape) == 3
        self.proj_intensities = proj_intensities

        self.vol_shape = self.projector_backend.get_vol_shape()
        self.proj_shape = self.projector_backend.get_prj_shape()
        super().__init__()

    def __enter__(self):
        self.projector_backend.initialize()
        return self

    def __exit__(self, *args):
        self.projector_backend.dispose()

    def fp_angle(self, vol, angle_ind):
        """Forward-projection of the volume to a single sinogram line.

        :param vol: The volume to forward-project (numpy.array_like)
        :param angle_ind: The angle index to foward project (int)

        :returns: The forward-projected sinogram line
        :rtype: numpy.array_like
        """
        x = self.projector_backend.fp(vol, angle_ind)
        if self.proj_intensities is not None:
            x *= self.proj_intensities[angle_ind]
        return x

    def bp_angle(self, sino_line, angle_ind):
        """Back-projection of a single sinogram line to the volume. It only
        applies the attenuation corrections if the projector is symmetric.

        :param sino_line: The sinogram to back-project or a single line (numpy.array_like)
        :param angle_ind: The angle index to foward project (int)

        :returns: The back-projected volume
        :rtype: numpy.array_like
        """
        if self.proj_intensities is not None:
            sino_line = sino_line * self.proj_intensities[angle_ind]  # It will make a copy
        return self.projector_backend.bp(sino_line, angle_ind)

    def fp(self, vol):
        """Forward-projection of the volume to the projection data.

        :param vol: The volume to forward-project.
        :type vol: numpy.array_like

        :returns: The forward-projected projection data
        :rtype: numpy.array_like
        """
        x = self.projector_backend.fp(vol)
        if self.proj_intensities is not None:
            x *= self.proj_intensities[:, np.newaxis]
        return x

    def bp(self, data):
        """Back-projection of the projection data to the volume.

        :param data: The projection data to back-project (numpy.array_like)

        :returns: The back-projected volume
        :rtype: numpy.array_like
        """
        if self.proj_intensities is not None:
            data = data * self.proj_intensities[:, np.newaxis]  # Needs copy
        return self.projector_backend.bp(data)


class ProjectorAttenuationXRF(ProjectorBase):
    """Attenuation corrected projection class for XRF, with multi-detector support.
    It includes the computation of the attenuation volumes.
    """

    def __init__(
        self,
        vol_shape,
        angles_rot_rad,
        rot_axis_shift_pix=0,
        proj_intensities=None,
        super_sampling=1,
        att_in=None,
        att_out=None,
        angles_detectors_rad=(np.pi / 2),
        weights_detectors=None,
        psf=None,
        precompute_attenuation=True,
        is_symmetric=False,
        weights_angles=None,
        data_type=np.float32,
    ):
        """
        :param vol_shape: The volume shape in X Y and optionally Z
        :type vol_shape: numpy.array_like
        :param angles_rot_rad: The rotation angles
        :type angles_rot_rad: numpy.array_like
        :param rot_axis_shift_pix: The rotation axis shift(s) in pixels, defaults to 0
        :type rot_axis_shift_pix: float or numpy.array_like, optional
        :param proj_intensities: Projection scaling factor, defaults to None
        :type proj_intensities: float or numpy.array_like, optional
        :param super_sampling: pixel and voxel super-sampling, defaults to 1
        :type super_sampling: int, optional
        :param att_in: Attenuation volume of the incoming beam, defaults to None
        :type att_in: numpy.array_like, optional
        :param att_out: Attenuation volume of the outgoing beam, defaults to None
        :type att_out: TYPE, optional
        :param angles_detectors_rad: Angles of the detector elements with respect to incident beam, defaults to (np.pi/2)
        :type angles_detectors_rad: numpy.array_like, optional
        :param weights_detectors: Weights (e.g. solid angle, efficiency, etc) of the detector elements, defaults to None
        :type weights_detectors: numpy.array_like, optional
        :param psf: Optical system's point spread function (PSF), defaults to None
        :type psf: numpy.array_like, optional
        :param precompute_attenuation: Whether to precompute attenuation or not, defaults to True
        :type precompute_attenuation: boolean, optional
        :param is_symmetric: Whether the projector is symmetric or not, defaults to False
        :type is_symmetric: boolean, optional
        :param weights_angles: Projection weight for a given element at a given angle, defaults to None
        :type weights_angles: numpy.array_like, optional
        :param data_type: Data type, defaults to np.float32
        :type data_type: numpy.dtype, optional
        """
        ProjectorBase.__init__(
            self,
            vol_shape,
            angles_rot_rad,
            rot_axis_shift_pix,
            proj_intensities=proj_intensities,
            super_sampling=super_sampling,
        )

        self.data_type = data_type

        if precompute_attenuation:
            if att_in is None and att_out is None:
                print("Turning off precomputation of attenuation.")
                precompute_attenuation = False

        self.att_in = att_in
        self.att_out = att_out
        self.angles_det_rad = np.array(angles_detectors_rad, ndmin=1)
        num_det_angles = len(self.angles_det_rad)
        if weights_detectors is None:
            weights_detectors = np.ones_like(self.angles_det_rad)
        self.weights_det = np.array(weights_detectors, ndmin=1)

        num_det_weights = len(self.weights_det)
        if num_det_angles > 1:
            if num_det_weights == 1:
                self.weights_det = np.tile(self.weights_det, [num_det_angles])
            elif num_det_weights > 1 and not num_det_weights == num_det_angles:
                raise ValueError(
                    "Number of detector weights differs from number of"
                    + " detector angles: %d vs %d" % (num_det_weights, num_det_angles)
                )

        if weights_angles is None:
            weights_angles = np.ones((len(angles_rot_rad), num_det_angles))
        self.weights_angles = weights_angles

        self.precompute_attenuation = precompute_attenuation
        self.is_symmetric = is_symmetric

        self.psf = psf

        if self.precompute_attenuation:
            self.compute_attenuation_volumes()
        else:
            self.att_vol_angles = None

    @staticmethod
    def _compute_attenuation(vol, direction, sampling=1, invert=False):
        def pad_vol(vol, edges):
            paddings = [(0,)] * len(vol.shape)
            paddings[-2], paddings[-1] = (edges[0],), (edges[1],)
            return np.pad(vol, paddings, mode="constant")

        size_lims = np.array(vol.shape[-2:])
        min_size = np.ceil(np.sqrt(np.sum(size_lims ** 2)))
        edges = np.ceil((min_size - size_lims) / 2).astype(np.intp)

        direction = np.array(direction)
        if invert:
            direction = -direction
        direction = direction / np.sqrt(np.sum(direction ** 2))

        rot_angle = np.rad2deg(np.arctan2(direction[1], direction[0]))

        cum_arr = pad_vol(vol, edges)

        cum_arr = skt.rotate(cum_arr, rot_angle, order=1)
        cum_arr += np.roll(cum_arr, 1, axis=-1)
        cum_arr = np.cumsum(cum_arr / 2, axis=-1)

        cum_arr = skt.rotate(cum_arr, -rot_angle, order=1)
        cum_arr = cum_arr[..., edges[0] : -edges[0], edges[1] : -edges[1]]

        cum_arr = np.exp(-cum_arr)

        return cum_arr

    def compute_attenuation(self, vol, direction, sampling=1, invert=False):
        """Computes the attenuation experienced by the photons emitted in every
        point of the volume, along a certain direction.
        """

        vol = np.array(vol)
        if not len(vol.shape) in [2, 3]:
            raise ValueError("Maps can only be 2D or 3D Arrays. A %d-dimensional was passed" % (len(vol.shape)))
        if not np.all(np.equal(vol.shape[-2:], self.vol_shape[:2])):
            raise ValueError(
                "Mismatching volume shape of input volume (%s) with vol_shape (%s in 2D -> %s)"
                % (
                    " ".join(("%d" % x for x in vol.shape)),
                    " ".join(("%d" % x for x in self.vol_shape)),
                    " ".join(("%d" % x for x in self.vol_shape[:2])),
                )
            )

        return self._compute_attenuation(vol, direction, sampling=1, invert=False)

    def _compute_attenuation_angle_in(self, angle):
        direction_in = [np.sin(angle), np.cos(angle)]
        return self.compute_attenuation(self.att_in, direction_in)[np.newaxis, ...]

    def _compute_attenuation_angle_out(self, angle):
        angle_det = angle - self.angles_det_rad
        atts = np.zeros(self.att_vol_angles.shape[1:], dtype=self.data_type)
        for ii, a in enumerate(angle_det):
            direction_out = [np.sin(a), np.cos(a)]
            atts[ii, ...] = self.compute_attenuation(self.att_out, direction_out, invert=True)
        return atts

    def compute_attenuation_volumes(self):
        """Computes the corrections for each angle.
        """
        if self.att_in is None and self.att_out is None:
            raise ValueError("No attenuation volumes were given")

        self.att_vol_angles = np.ones(
            [len(self.angles_rot_rad), len(self.angles_det_rad), *self.vol_shape[:3]], dtype=self.data_type
        )

        if self.att_in is not None:
            for ii, a in enumerate(self.angles_rot_rad):
                self.att_vol_angles[ii, ...] *= self._compute_attenuation_angle_in(a)

        if self.att_out is not None:
            for ii, a in enumerate(self.angles_rot_rad):
                self.att_vol_angles[ii, ...] *= self._compute_attenuation_angle_out(a)

        if self.is_3d:
            self.att_vol_angles = self.att_vol_angles[:, :, np.newaxis, ...]

    def collapse_detectors(self):
        """Converts multi-detector configurations into single-detector."""
        weights = np.reshape(self.weights_det, [1, -1, 1, 1]) / np.sum(self.weights_det)
        self.att_vol_angles = np.sum(self.att_vol_angles * weights, axis=1)

        weights = np.squeeze(weights)
        self.angles_det_rad = np.sum(self.angles_det_rad * weights, keepdims=True)
        self.weights_angles = np.sum(self.weights_angles * weights, axis=1, keepdims=True)
        self.weights_det = np.sum(self.weights_det, keepdims=True)

    def fp_angle(self, vol, angle_ind):
        """Forward-projection of the volume to a single sinogram line. It
        applies the attenuation corrections.

        :param vol: The volume to forward-project (numpy.array_like)
        :param angle_ind: The angle index to foward project (int)

        :returns: The forward-projected sinogram line
        :rtype: numpy.array_like
        """
        temp_vol = cp.deepcopy(vol)[np.newaxis, ...]
        temp_vol = np.tile(temp_vol, (len(self.angles_det_rad), *((1,) * len(self.vol_shape))))

        if self.precompute_attenuation:
            temp_vol *= self.att_vol_angles[angle_ind, ...]
        else:
            a = self.angles_rot_rad[angle_ind]
            if self.att_in is not None:
                temp_vol *= self._compute_attenuation_angle_in(a)
            if self.att_out is not None:
                temp_vol *= self._compute_attenuation_angle_out(a)

        weights = self.weights_det * self.weights_angles[angle_ind, :]
        sino_line = [
            weights[ii] * ProjectorBase.fp_angle(self, temp_vol[ii], angle_ind) for ii in range(len(self.angles_det_rad))
        ]
        sino_line = np.stack(sino_line, axis=0)

        if self.psf is not None:
            sino_line = spsig.convolve(sino_line, self.psf, mode="same")

        if sino_line.shape[0] == 1:
            sino_line = np.squeeze(sino_line, axis=0)

        return sino_line

    def bp_angle(self, sino, angle_ind, single_line=False):
        """Back-projection of a single sinogram line to the volume. It only
        applies the attenuation corrections if the projector is symmetric.

        :param sino: The sinogram to back-project or a single line (numpy.array_like)
        :param angle_ind: The angle index to foward project (int)
        :param single_line: Whether the input is a single sinogram line (boolean, default: False)

        :returns: The back-projected volume
        :rtype: numpy.array_like
        """
        if single_line:
            sino_line = sino
        else:
            sino_line = sino[angle_ind, ...]

        if self.psf is not None:
            sino_line = spsig.convolve(sino_line, self.psf, mode="same")

        sino_line = np.reshape(sino_line, [len(self.weights_det), *sino_line.shape[-(len(self.vol_shape) - 1) :]])
        weights = self.weights_det * self.weights_angles[angle_ind, :]
        vol = [
            weights[ii] * ProjectorBase.bp_angle(self, sino_line[ii, ...], angle_ind) for ii in range(len(self.angles_det_rad))
        ]
        vol = np.stack(vol, axis=0)

        if self.is_symmetric:
            if self.precompute_attenuation:
                vol *= self.att_vol_angles[angle_ind, ...]
            else:
                a = self.angles_rot_rad[angle_ind]
                if self.att_in is not None:
                    vol *= self._compute_attenuation_angle_in(a)
                if self.att_out is not None:
                    vol *= self._compute_attenuation_angle_out(a)

        return np.sum(vol, axis=0)

    def fp(self, vol):
        """Forward-projection of the volume to the sinogram. It applies the
        attenuation corrections.

        :param vol: The volume to forward-project (numpy.array_like)

        :returns: The forward-projected sinogram
        :rtype: numpy.array_like
        """
        return np.stack([self.fp_angle(vol, ii) for ii in range(len(self.angles_rot_rad))], axis=0)

    def bp(self, sino):
        """Back-projection of the sinogram to the volume.

        :param sino: The sinogram to back-project (numpy.array_like)

        :returns: The back-projected volume
        :rtype: numpy.array_like
        """
        if self.is_symmetric:
            return np.sum([self.bp_angle(sino, ii) for ii in range(len(self.angles_rot_rad))], axis=0)
        else:
            return ProjectorBase.bp(self, sino)


class ProjectorUncorrected(ProjectorBase):
    """Uncorrected projection class, which implements the forward and back
    projection of the whole projection data.
    It takes care of initializing and disposing the ASTRA projectors when used
    in a *with* statement.
    It supports both 2D and 3D geometries.
    """

    def __init__(
        self,
        vol_shape,
        angles_rot_rad,
        rot_axis_shift_pix=0,
        proj_intensities=None,
        create_single_projs=False,
        super_sampling=1,
    ):
        """
        :param vol_shape: The volume shape in X Y and optionally Z
        :type vol_shape: numpy.array_like
        :param angles_rot_rad: The rotation angles
        :type angles_rot_rad: numpy.array_like
        :param rot_axis_shift_pix: The rotation axis shift(s) in pixels, defaults to 0
        :type rot_axis_shift_pix: float or numpy.array_like, optional
        :param proj_intensities: Projection scaling factor, defaults to None
        :type proj_intensities: float or numpy.array_like, optional
        :param create_single_projs: Specifies whether to create projectors for single projections.
        Useful for corrections and SART, defaults to False
        :type create_single_projs: boolean, optional
        :param super_sampling: pixel and voxel super-sampling, defaults to 1
        :type super_sampling: int, optional
        """
        ProjectorBase.__init__(
            self,
            vol_shape,
            angles_rot_rad,
            rot_axis_shift_pix,
            proj_intensities=proj_intensities,
            create_single_projs=create_single_projs,
            super_sampling=super_sampling,
        )

    def _astra_fbp(self, projs, fbp_filter):
        if has_cuda:
            fbp_type = "FBP_CUDA"
        else:
            fbp_type = "FBP"
        cfg = astra.astra_dict(fbp_type)
        cfg["ProjectorId"] = self.proj_id[-1]
        cfg["FilterType"] = fbp_filter

        proj_geom = astra.projector.projection_geometry(self.proj_id[-1])
        vol_geom = astra.projector.volume_geometry(self.proj_id[-1])

        if len(projs.shape) > 2:
            num_lines = projs.shape[1]
            vols = [None] * num_lines

            for ii_v in range(num_lines):
                sino_id = astra.data2d.create("-sino", proj_geom, projs[:, ii_v, :])
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
            sino_id = astra.data2d.create("-sino", proj_geom, projs)
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

    def fbp(self, projs, fbp_filter="shepp-logan"):
        """Computes the filtered back-projection of the projection data to the
        volume.
        The data could either be a sinogram, or a stack of sinograms.

        :param projs: The projection data to back-project
        :type projs: numpy.array_like
        :param fbp_filter: The FBP filter to use
        :type fbp_filter: str or function

        :returns: The FBP reconstructed volume
        :rtype: numpy.array_like
        """
        if self.is_3d:
            raise ValueError("FBP not supported with 3D projector")

        if isinstance(fbp_filter, str):
            return self._astra_fbp(projs, fbp_filter)
        else:
            projs = fbp_filter(projs, self)
            return self.bp(projs) / self.angles_rot_rad.size


class FilterMR(object):
    """Data dependent FBP filter. This is a simplified implementation from:

    [1] Pelt, D. M., & Batenburg, K. J. (2014). Improving filtered backprojection
    reconstruction by data-dependent filtering. Image Processing, IEEE
    Transactions on, 23(11), 4750-4762.

    Code inspired by: https://github.com/dmpelt/pymrfbp
    """

    def __init__(
        self, sinogram_pixels_num=None, sinogram_angles_num=None, start_exp_binning=2, lambda_smooth=None, data_type=np.float32
    ):
        """
        :param sinogram_pixels_num: Number of sinogram pixels (int)
        :param sinogram_angles_num: Number of sinogram angles (int)
        :param start_exp_binning: From which distance to start exponentional binning (int)
        :param lambda_smooth: Smoothing parameter (float)
        :param data_type: Filter data type (numpy data type)
        """
        self.data_type = data_type
        self.start_exp_binning = start_exp_binning
        self.lambda_smooth = lambda_smooth
        self.is_initialized = False
        self.sinogram_pixels_num = sinogram_pixels_num
        self.sinogram_angles_num = sinogram_angles_num

        if sinogram_pixels_num is not None and sinogram_angles_num is not None:
            self.initialize()

    def initialize(self):
        """ Filter initialization function.
        """
        if self.sinogram_pixels_num is None:
            raise ValueError("No sinogram pixels number was given!")
        if self.sinogram_angles_num is None:
            raise ValueError("No sinogram angles number was given!")

        filter_center = np.floor(self.sinogram_pixels_num / 2).astype(np.int)
        self.filter_size = filter_center * 2 + 1

        window_size = 1
        window_position = filter_center

        self.basis = []
        count = 0
        while window_position < self.filter_size:
            basis_tmp = np.zeros(self.filter_size, dtype=self.data_type)

            # simmetric exponential binning
            l_bound = window_position
            r_bound = np.fmin(l_bound + window_size, self.filter_size)
            basis_tmp[l_bound:r_bound] = 1

            r_bound = self.filter_size - window_position
            l_bound = np.fmax(r_bound - window_size, 0)
            basis_tmp[l_bound:r_bound] = 1

            self.basis.append(basis_tmp)
            window_position += window_size

            count += 1
            if self.start_exp_binning is not None and count > self.start_exp_binning:
                window_size = 2 * window_size

        self.is_initialized = True

    def compute_filter(self, sinogram, projector):
        """ Computes the filter.

        :param sinogram: The sinogram (np.array_like)
        :param projector: The projector used in the FBP (object)

        :returns: The computed filter
        :rtype: numpy.array_like
        """
        sino_size = self.sinogram_angles_num * self.sinogram_pixels_num
        nrows = sino_size
        ncols = len(self.basis)

        if self.lambda_smooth:
            grad_vol_size = self.sinogram_pixels_num * (self.sinogram_pixels_num - 1)
            nrows += 2 * grad_vol_size

        A = np.zeros((nrows, ncols), dtype=self.data_type)

        for ii, bas in enumerate(self.basis):
            img = self.apply_filter(sinogram, bas)
            img = projector.bp(img) / self.sinogram_angles_num

            astra.extrautils.clipCircle(img)
            A[:sino_size, ii] = projector.fp(img).flatten()
            if self.lambda_smooth:
                dx = np.diff(img, axis=0)
                dy = np.diff(img, axis=1)
                d = np.concatenate((dx.flatten(), dy.flatten()))
                A[sino_size:, ii] = self.lambda_smooth * d

        b = np.zeros((nrows,), dtype=self.data_type)
        b[:sino_size] = sinogram.flatten()
        fitted_components = np.linalg.lstsq(A, b, rcond=None)

        computed_filter = np.zeros((self.filter_size,), dtype=self.data_type)
        for ii, bas in enumerate(self.basis):
            computed_filter += fitted_components[0][ii] * bas
        return computed_filter

    def apply_filter(self, sinogram, computed_filter):
        """ Applies the filter to the sinogram.

        :param sinogram: The sinogram (np.array_like)
        :param computed_filter: The computed filter (np.array_like)

        :returns: The filtered sinogram
        :rtype: numpy.array_like
        """
        return spsig.fftconvolve(sinogram, computed_filter[np.newaxis, ...], "same")

    def __call__(self, sinogram, projector):
        """ Filters the sinogram, by first computing the filter, and then
        applying it.

        :param sinogram: The sinogram (np.array_like)
        :param projector: The projector used in the FBP (object)

        :returns: The filtered sinogram
        :rtype: numpy.array_like
        """
        if not self.is_initialized:
            self.sinogram_angles_num = sinogram.shape[0]
            self.sinogram_pixels_num = sinogram.shape[1]
            self.initialize()

        computed_filter = self.compute_filter(sinogram, projector)
        return self.apply_filter(sinogram, computed_filter)
