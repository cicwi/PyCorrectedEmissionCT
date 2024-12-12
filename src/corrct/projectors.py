"""
Tomographic projectors.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import concurrent.futures as cf
import multiprocessing as mp
from collections.abc import Sequence
from typing import Optional
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from corrct import _projector_backends as prj_backends
from corrct import models
from corrct import operators
from corrct.physics.attenuation import AttenuationVolume


num_threads = round(np.log2(mp.cpu_count() + 1))
astra_available = prj_backends.has_astra and prj_backends.has_cuda


class ProjectorMatrix(operators.ProjectorOperator):
    """Projector that uses an explicit projection matrix."""

    A: Union[NDArray, spmatrix]

    def __init__(self, A: Union[NDArray, spmatrix], vol_shape: ArrayLike, prj_shape: ArrayLike) -> None:
        """
        Initialize the matrix projector.

        Parameters
        ----------
        A : NDArray | spmatrix
            The projection matrix.
        vol_shape : ArrayLike
            Volume shape.
        prj_shape : ArrayLike
            Projection shape.
        """
        self.vol_shape = np.array(vol_shape, ndmin=1, dtype=int)
        self.prj_shape = np.array(prj_shape, ndmin=1, dtype=int)

        self.A = A
        super().__init__()

    def _transpose(self) -> operators.ProjectorOperator:
        """
        Create the transpose operator.

        Returns
        -------
        operators.ProjectorOperator
            The transpose operator.
        """
        return ProjectorMatrix(self.A.transpose(), self.prj_shape, self.vol_shape)

    def absolute(self) -> operators.ProjectorOperator:
        """
        Return the projection operator using the absolute value of the projection coefficients.

        Returns
        -------
        operators.ProjectorOperator
            The absolute value operator.
        """
        return ProjectorMatrix(np.abs(self.A), self.vol_shape, self.prj_shape)  # type: ignore

    def fp(self, x: NDArray) -> NDArray:
        """
        Define the interface for the forward-projection.

        Parameters
        ----------
        x : NDArray
            Input volume.

        Returns
        -------
        NDArray
            The projection data.
        """
        return self.A.dot(x.flatten()).reshape(self.prj_shape)

    def bp(self, x: NDArray) -> NDArray:
        """
        Define the interface for the back-projection.

        Parameters
        ----------
        x : NDArray
            Input projection data.

        Returns
        -------
        NDArray
            The back-projected volume.
        """
        return self.A.transpose().dot(x.flatten()).reshape(self.vol_shape)


class ProjectorUncorrected(operators.ProjectorOperator):
    """Base projection class."""

    vol_geom: models.VolumeGeometry
    projector_backend: prj_backends.ProjectorBackend

    prj_intensities: Union[NDArray[np.floating], None]
    psf: Union[NDArray[np.floating], float, None]

    def __init__(
        self,
        vol_geom: Union[Sequence[int], NDArray[np.integer], models.VolumeGeometry],
        angles_rot_rad: Union[Sequence[float], NDArray],
        rot_axis_shift_pix: Union[float, ArrayLike, NDArray, None] = None,
        *,
        prj_geom: Optional[models.ProjectionGeometry] = None,
        prj_intensities: Optional[ArrayLike] = None,
        psf: Optional[ArrayLike] = None,
        backend: Union[str, prj_backends.ProjectorBackend] = "astra" if astra_available else "skimage",
        create_single_projs: bool = False,
    ):
        """Initialize the base projection class.

        It implements the forward and back projection of the single lines of a sinogram.
        It takes care of initializing and disposing the ASTRA projectors when used in a *with* statement.
        It supports both 2D and 3D geometries.

        Parameters
        ----------
        vol_geom : Sequence[int] | NDArray[np.integer] | models.VolumeGeometry
            The volume shape in Y X and optionally Z.
        angles_rot_rad : Sequence[float] | NDArray
            The rotation angles.
        rot_axis_shift_pix : float | ArrayLike | NDArray, optional
            The rotation axis shift(s) in pixels, by default None.
        prj_geom : models.ProjectionGeometry | None, optional
            The fully specified projection geometry.
            When active, the rotation axis shift is ignored, by default None.
        prj_intensities : ArrayLike | NDArray | None, optional
            Projection scaling factor, by default None.
        psf : ArrayLike | NDArray | None, optional
            The "point spread function" of the detector, by default None.
        backend : bool, optional
            Whether to use ASTRA or fall back to scikit-image.
            The default is True if CUDA and ASTRA are available, otherwise False.
        create_single_projs : bool, optional
            Whether to create projectors for single projections.
            Used for corrections and SART, by default False.

        Raises
        ------
        ValueError
            When the geometry is not correct.
        """
        if not astra_available:
            astra_status = f"astra: {prj_backends.has_astra}, cuda: {prj_backends.has_cuda}"
            if isinstance(backend, str) and backend == "astra":
                backend = "skimage"
                print(f"WARNING: ASTRA backend requested but not available ({astra_status}). Falling back to scikit-image.")
            elif isinstance(backend, prj_backends.ProjectorBackendASTRA):
                raise ValueError(f"Passed ASTRA projector, but astra not available ({astra_status}).")

        if not isinstance(vol_geom, models.VolumeGeometry):
            vol_geom = models.VolumeGeometry(_vol_shape_xyz=np.array(vol_geom))
        self.vol_geom = vol_geom
        self.prj_geom = prj_geom

        if not len(self.vol_geom.shape_xyz) in (2, 3):
            raise ValueError("Only 2D or 3D volumes are valid")
        if not self.vol_geom.is_square():
            raise ValueError("Only square volumes are valid")

        if isinstance(backend, str):
            if backend == "astra":
                if prj_backends.has_astra_direct:
                    self.projector_backend = prj_backends.ProjectorBackendDirectASTRA()
                else:
                    self.projector_backend = prj_backends.ProjectorBackendASTRA()
            elif backend == "skimage":
                self.projector_backend = prj_backends.ProjectorBackendSKimage()
            else:
                raise ValueError(f"Unknown backend: {backend}. Available options are: 'astra', 'skimage'.")
        else:
            self.projector_backend = backend

        if not (prj_geom is None or isinstance(self.projector_backend, prj_backends.ProjectorBackendASTRA)):
            raise ValueError("Using class `ProjectionGeometry` requires using astra-toolbox.")

        self.projector_backend.initialize_geometry(
            vol_geom=vol_geom,
            angles_rot_rad=angles_rot_rad,
            rot_axis_shift_pix=rot_axis_shift_pix,
            prj_geom=prj_geom,
            create_single_projs=create_single_projs,
        )

        if prj_intensities is not None:
            prj_intensities = np.array(prj_intensities, dtype=np.floating)
        self.prj_intensities = prj_intensities

        self._set_psf(psf)

        self.vol_shape = np.array(self.projector_backend.get_vol_shape(), ndmin=1)
        self.prj_shape = np.array(self.projector_backend.get_prj_shape(), ndmin=1)
        super().__init__()

    @property
    def angles_rot_rad(self) -> NDArray:
        """Simplify access to the rotation angles (in radians).

        Returns
        -------
        NDArray
            The rotation angles (in radians).
        """
        return self.projector_backend.angles_w_rad

    def __enter__(self):
        """Initialize the with statement block."""
        self.projector_backend.make_ready()
        return self

    def __exit__(self, *args):
        """De-initialize the with statement block."""
        self.projector_backend.dispose()

    def _set_psf(self, psf: Optional[ArrayLike], is_conv_symm: bool = False) -> None:
        if psf is not None:
            psf = np.squeeze(np.array(psf))
            if len(psf.shape) >= len(self.vol_geom.shape_xyz):
                raise ValueError(
                    "PSF should either be 1D (for 2D and 3D reconstructions) or 2D (for 3D reconstructions)."
                    + f" Passed PSF has shape: {psf.shape}, and the reconstruction is {len(self.vol_geom.shape_xyz)}D."
                )
            # This redundancy is required, due to the different results between the single-angle and multi-angle projections
            prj_shape_vu = [*self.projector_backend.prj_shape_vu[:-2], self.projector_backend.prj_shape_vu[-1]]
            prj_shape_vu = np.array(prj_shape_vu, ndmin=1)
            self.psf_vu = operators.TransformConvolution(prj_shape_vu, kernel=psf, is_symm=is_conv_symm)
            prj_shape_vwu = np.array(self.projector_backend.prj_shape_vwu, ndmin=1)
            self.psf_vwu = operators.TransformConvolution(prj_shape_vwu, kernel=psf[..., None, :], is_symm=is_conv_symm)
        else:
            self.psf_vu = self.psf_vwu = None

    def get_pre_weights(self) -> Union[NDArray, None]:
        """Compute the pre-weights of the projector geometry (notably for cone-beam geometries).

        Returns
        -------
        Union[NDArray, None]
            The computed detector weights
        """
        if self.prj_geom is None:
            return None
        else:
            return self.prj_geom.get_pre_weights([*self.prj_shape[-3:-2], self.prj_shape[-1]])

    def fp_angle(self, vol: NDArray, angle_ind: int) -> NDArray:
        """Forward-project a volume to a single sinogram line.

        Parameters
        ----------
        vol : NDArray
            The volume to forward-project.
        angle_ind : int
            The angle index to forward project.

        Returns
        -------
        x : NDArray
            The forward-projected sinogram line.
        """
        prj_vu = self.projector_backend.fp(vol, angle_ind)
        if self.prj_intensities is not None:
            prj_vu *= self.prj_intensities[angle_ind]
        if self.psf_vu is not None:
            prj_vu = self.psf_vu(prj_vu)
        return prj_vu

    def bp_angle(self, prj_vu: NDArray, angle_ind: int) -> NDArray:
        """Back-project a single sinogram line to the volume.

        Parameters
        ----------
        prj_vu : NDArray
            The sinogram to back-project or a single line.
        angle_ind : int
            The angle index to forward project.

        Returns
        -------
        NDArray
            The back-projected volume.
        """
        if self.prj_intensities is not None:
            prj_vu = prj_vu * self.prj_intensities[angle_ind]  # It will make a copy
        if self.psf_vu is not None:
            prj_vu = self.psf_vu.T(prj_vu)
        return self.projector_backend.bp(prj_vu, angle_ind)

    def fp(self, vol: NDArray) -> NDArray:
        """
        Forward-projection of the volume to the projection data.

        Parameters
        ----------
        vol : NDArray
            The volume to forward-project.

        Returns
        -------
        NDArray
            The forward-projected projection data.
        """
        prj_vwu = self.projector_backend.fp(vol)
        if self.prj_intensities is not None:
            prj_vwu *= self.prj_intensities[:, np.newaxis]
        if self.psf_vwu is not None:
            prj_vwu = self.psf_vwu(prj_vwu)
        return prj_vwu

    def bp(self, prj_vwu: NDArray) -> NDArray:
        """
        Back-projection of the projection data to the volume.

        Parameters
        ----------
        prj_vwu : NDArray
            The projection data to back-project.

        Returns
        -------
        NDArray
            The back-projected volume.
        """
        if self.prj_intensities is not None:
            prj_vwu = prj_vwu * self.prj_intensities[:, np.newaxis]  # Needs copy
        if self.psf_vwu is not None:
            prj_vwu = self.psf_vwu.T(prj_vwu)
        return self.projector_backend.bp(prj_vwu)


class ProjectorAttenuationXRF(ProjectorUncorrected):
    """
    Attenuation corrected projection class for XRF, with multi-detector support.

    It includes the computation of the attenuation volumes.
    """

    att_vol_angles: NDArray[np.floating]

    executor: Union[cf.ThreadPoolExecutor, None]

    def __init__(
        self,
        vol_geom: Union[Sequence[int], NDArray[np.integer], models.VolumeGeometry],
        angles_rot_rad: Union[Sequence[float], NDArray],
        rot_axis_shift_pix: Union[float, ArrayLike, NDArray, None] = None,
        *,
        prj_geom: Optional[models.ProjectionGeometry] = None,
        prj_intensities: Optional[ArrayLike] = None,
        backend: Union[str, prj_backends.ProjectorBackend] = "astra" if astra_available else "skimage",
        att_maps: Optional[NDArray[np.floating]] = None,
        att_in: Optional[NDArray[np.floating]] = None,
        att_out: Optional[NDArray[np.floating]] = None,
        angles_detectors_rad: Union[float, ArrayLike] = (np.pi / 2),
        weights_detectors: Optional[ArrayLike] = None,
        psf: Optional[ArrayLike] = None,
        is_symmetric: bool = False,
        weights_angles: Optional[ArrayLike] = None,
        use_multithreading: bool = True,
        data_type: DTypeLike = np.float32,
        verbose: bool = True,
    ):
        """
        Initialize the (attenuation corrected) XRF dedicated projector.

        Parameters
        ----------
        vol_geom : Sequence[int] | NDArray[np.integer] | models.VolumeGeometry
            The volume shape in X Y and optionally Z.
        angles_rot_rad : Sequence[float] | NDArray
            The rotation angles.
        rot_axis_shift_pix : float | ArrayLike | NDArray | None, optional
            The rotation axis shift(s) in pixels. The default is None.
        prj_geom : Optional[models.ProjectionGeometry], optional
            The fully specified projection geometry.
            When active, the rotation axis shift is ignored. The default is None.
        prj_intensities : Optional[ArrayLike], optional
            Projection scaling factor. The default is None.
        backend : str | prj_backends.ProjectorBackend, optional
            Projector backend to use, by default "astra" if astra is available, otherwise "skimage".
        att_maps : Optional[NDArray[np.floating]], optional
            Precomputed attenuation maps for each angle, by default None
        att_in : Optional[ArrayLike], optional
            Attenuation volume of the incoming beam. The default is None.
        att_out : Optional[ArrayLike], optional
            Attenuation volume of the outgoing beam. The default is None.
        angles_detectors_rad : float | ArrayLike, optional
            Angles of the detector elements with respect to incident beam. The default is (np.pi / 2).
        weights_detectors : Optional[ArrayLike], optional
            Weights (e.g. solid angle, efficiency, etc) of the detector elements. The default is None.
        psf : Optional[ArrayLike], optional
            Optical system's point spread function (PSF). The default is None.
        is_symmetric : bool, optional
            Whether the projector is symmetric or not. The default is False.
        weights_angles : Optional[ArrayLike], optional
            Projection weight for a given element at a given angle. The default is None.
        use_multithreading : bool, optional
            Whether to use multiple threads or not. The default is True.
        data_type : DTypeLike, optional
            Output data type. The default is np.float32.
        verbose : bool, optional
            Whether to produce verbose output. The default is True.

        Raises
        ------
        ValueError
            When given inconsistent numbers of detector weights and detector angles.
        """
        ProjectorUncorrected.__init__(
            self,
            vol_geom=vol_geom,
            angles_rot_rad=angles_rot_rad,
            rot_axis_shift_pix=rot_axis_shift_pix,
            prj_geom=prj_geom,
            psf=psf,
            prj_intensities=prj_intensities,
            backend=backend,
            create_single_projs=True,
        )

        self.data_type = data_type
        self.use_multithreading = use_multithreading
        self.verbose = verbose
        self.is_symmetric = is_symmetric

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
                    f" detector angles: {num_det_weights} vs {num_det_angles}"
                )

        if weights_angles is None:
            weights_angles = np.ones((len(self.angles_rot_rad), num_det_angles))
        else:
            weights_angles = np.array(weights_angles)
        self.weights_angles = weights_angles

        if att_maps is None:
            m = AttenuationVolume(att_in, att_out, self.angles_rot_rad, self.angles_det_rad)
            m.compute_maps(use_multithreading=self.use_multithreading, verbose=self.verbose)
            self.att_vol_angles = m.get_maps()
        else:
            if att_maps.shape[1] != num_det_angles:
                raise ValueError(f"Number of maps ({att_maps.shape[1]}) differs from number of detectors ({num_det_angles})")
            if not np.all(np.equal(att_maps.shape[-2:], self.vol_shape[-2:])):
                raise ValueError(
                    f"Mismatching volume shape of input volume ({att_maps.shape[-2:]})"
                    + f" with vol_shape ({self.vol_shape} in 2D -> {self.vol_shape[-2:]})"
                )
            self.att_vol_angles = att_maps

        self.executor = None

    def __enter__(self):
        """Initialize the with statement block."""
        if self.use_multithreading and isinstance(self.projector_backend, prj_backends.ProjectorBackendSKimage):
            if self.executor is None:
                self.executor = cf.ThreadPoolExecutor(max_workers=num_threads)
        return super().__enter__()

    def __exit__(self, *args):
        """De-initialize the with statement block."""
        super().__exit__()
        if self.use_multithreading and isinstance(self.projector_backend, prj_backends.ProjectorBackendSKimage):
            if self.executor is not None:
                self.executor.shutdown()
                self.executor = None

    def collapse_detectors(self) -> None:
        """Convert multi-detector configurations into single-detector."""
        weights = np.reshape(self.weights_det, [1, -1, 1, 1]) / np.sum(self.weights_det)
        self.att_vol_angles = np.sum(self.att_vol_angles * weights, axis=1)

        weights = np.squeeze(weights)
        self.angles_det_rad = np.sum(self.angles_det_rad * weights, keepdims=True)
        self.weights_angles = np.sum(self.weights_angles * weights, axis=1, keepdims=True)
        self.weights_det = np.sum(self.weights_det, keepdims=True)

    def fp_angle(self, vol: NDArray, angle_ind: int) -> NDArray:
        """Forward-project the volume to a single sinogram line.

        It applies the attenuation corrections.

        Parameters
        ----------
        vol : NDArray
            The volume to forward-project.
        angle_ind : int
            The angle index to forward project.

        Returns
        -------
        sino_line : NDArray
            The forward-projected sinogram line.
        """
        temp_vol = vol * self.att_vol_angles[angle_ind, ...]

        weights = self.weights_det * self.weights_angles[angle_ind, :]
        sino_line = [
            weights[ii] * ProjectorUncorrected.fp_angle(self, temp_vol[ii], angle_ind)
            for ii in range(len(self.angles_det_rad))
        ]
        sino_line = np.ascontiguousarray(np.stack(sino_line, axis=0))

        if sino_line.shape[0] == 1:
            sino_line = np.squeeze(sino_line, axis=0)

        return sino_line

    def bp_angle(self, sino: NDArray, angle_ind: int, single_line: bool = False) -> NDArray:
        """Back-project a single sinogram line to the volume.

        It only applies the attenuation corrections if the projector is symmetric.

        Parameters
        ----------
        sino : NDArray
            The sinogram to back-project or a single line.
        angle_ind : int
            The angle index to forward project.
        single_line : bool, optional
            Whether the input is a single sinogram line. The default is False.

        Returns
        -------
        NDArray
            The back-projected volume.
        """
        if single_line:
            sino_line = sino
        else:
            sino_line = sino[..., angle_ind, :]

        sino_line = np.reshape(sino_line, [len(self.weights_det), *sino_line.shape[-(len(self.vol_shape) - 1) :]])
        weights = self.weights_det * self.weights_angles[angle_ind, :]
        vol = [
            weights[ii] * ProjectorUncorrected.bp_angle(self, sino_line[ii, ...], angle_ind)
            for ii in range(len(self.angles_det_rad))
        ]
        vol = np.stack(vol, axis=0)

        if self.is_symmetric:
            vol *= self.att_vol_angles[angle_ind, ...]

        return np.sum(vol, axis=0)

    def fp(self, vol: NDArray) -> NDArray:
        """Forward-project the volume to the sinogram.

        It applies the attenuation corrections.

        Parameters
        ----------
        vol : NDArray
            The volume to forward-project.

        Returns
        -------
        NDArray
            The forward-projected sinogram.
        """
        if self.use_multithreading and isinstance(self.projector_backend, prj_backends.ProjectorBackendSKimage):
            sino_lines = self.executor.map(lambda x: self.fp_angle(vol, x), range(len(self.angles_rot_rad)))
            sino_lines = [*sino_lines]
        else:
            sino_lines = [self.fp_angle(vol, ii) for ii in range(len(self.angles_rot_rad))]

        sino_lines = np.ascontiguousarray(np.stack(sino_lines, axis=-2))

        if sino_lines.shape[0] == 1:
            sino_lines = np.squeeze(sino_lines, axis=0)

        return sino_lines

    def bp(self, sino: NDArray) -> NDArray:
        """Back-projection of the sinogram to the volume.

        Parameters
        ----------
        sino : NDArray
            The sinogram to back-project.

        Returns
        -------
        NDArray
            The back-projected volume.
        """
        if self.is_symmetric:
            if self.use_multithreading and isinstance(self.projector_backend, prj_backends.ProjectorBackendSKimage):
                vols = self.executor.map(lambda x: self.bp_angle(sino, x), range(len(self.angles_rot_rad)))
                vols = [*vols]
            else:
                vols = [self.bp_angle(sino, ii) for ii in range(len(self.angles_rot_rad))]

            return np.sum(vols, axis=0)
        else:
            sino = np.reshape(sino, [len(self.weights_det), *self.prj_shape])
            vol = [ProjectorUncorrected.bp(self, sino[ii, ...]) for ii in range(len(self.angles_det_rad))]
            return np.sum(np.stack(vol, axis=0), axis=0)
