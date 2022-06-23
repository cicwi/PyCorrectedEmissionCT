# -*- coding: utf-8 -*-
"""
Tomographic projectors.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

import scipy.signal as spsig

from . import operators
from . import _projector_backends as prj_backends
from . import utils_proc
from . import models
from .attenuation import AttenuationVolume

import concurrent.futures as cf
import multiprocessing as mp

from typing import Union, Sequence, Optional, Callable
from numpy.typing import ArrayLike, DTypeLike


num_threads = round(np.log2(mp.cpu_count() + 1))


class ProjectorMatrix(operators.ProjectorOperator):
    """
    Projector that uses an explicit projection matrix.

    Parameters
    ----------
    A : ArrayLike
        The projection matrix.
    vol_shape : Sequence[int]
        Volume shape.
    prj_shape : Sequence[int]
        Projection shape.
    """

    def __init__(self, A: ArrayLike, vol_shape: Union[Sequence[int], ArrayLike], prj_shape: Union[Sequence[int], ArrayLike]):
        self.vol_shape = vol_shape
        self.prj_shape = prj_shape

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
        return ProjectorMatrix(np.abs(self.A), self.vol_shape, self.prj_shape)

    def fp(self, x: ArrayLike) -> ArrayLike:
        """
        Define the interface for the forward-projection.

        Parameters
        ----------
        x : ArrayLike
            Input volume.

        Returns
        -------
        ArrayLike
            The projection data.
        """
        return self.A.dot(x.flatten()).reshape(self.prj_shape)

    def bp(self, x: ArrayLike) -> ArrayLike:
        """
        Define the interface for the back-projection.

        Parameters
        ----------
        x : ArrayLike
            Input projection data.

        Returns
        -------
        ArrayLike
            The back-projected volume.
        """
        return self.A.transpose().dot(x.flatten()).reshape(self.vol_shape)


class ProjectorUncorrected(operators.ProjectorOperator):
    """Base projection class.

    It implements the forward and back projection of the single lines of a sinogram.
    It takes care of initializing and disposing the ASTRA projectors when used in a *with* statement.
    It supports both 2D and 3D geometries.

    Parameters
    ----------
    vol_shape : ArrayLike
        The volume shape in Y X and optionally Z.
    angles_rot_rad : ArrayLike
        The rotation angles.
    rot_axis_shift_pix : float | ArrayLike, optional
        The rotation axis shift(s) in pixels. The default is 0.
    prj_geom : ProjectionGeometry, optional
        The fully specified projection geometry.
        When active, the rotation axis shift is ignored. The default is None.
    prj_intensities : float | ArrayLike, optional
        Projection scaling factor. The default is None.
    psf: ArrayLike | None, optional
        The "point spread function" of the detector. The default is None.
    use_astra : bool, optional
        Whether to use ASTRA or fall back to scikit-image.
        The default is True if CUDA is available, otherwise False.
    create_single_projs : bool, optional
        Whether to create projectors for single projections.
        Used for corrections and SART. The default is True.
    super_sampling : int, optional
        pixel and voxel super-sampling. The default is 1.

    Raises
    ------
    ValueError
        When the geometry is not correct.
    """

    def __init__(
        self,
        vol_geom: Union[Sequence[int], ArrayLike, models.VolumeGeometry],
        angles_rot_rad: Union[Sequence[float], ArrayLike],
        rot_axis_shift_pix: float = 0.0,
        *,
        prj_geom: Optional[models.ProjectionGeometry] = None,
        prj_intensities: Optional[ArrayLike] = None,
        psf: Optional[ArrayLike] = None,
        use_astra: bool = prj_backends.has_cuda,
        create_single_projs: bool = True,
        super_sampling: int = 1,
    ):
        if not prj_backends.has_astra and use_astra:
            use_astra = False
            print("WARNING: ASTRA requested but not available. Falling back to scikit-image.")

        if not isinstance(vol_geom, models.VolumeGeometry):
            vol_geom = models.VolumeGeometry(vol_shape_xyz=np.array(vol_geom))
        self.vol_geom = vol_geom

        vol_shape = self.vol_geom.shape

        if len(vol_shape) < 2 or len(vol_shape) > 3:
            raise ValueError("Only 2D or 3D volumes")
        if not vol_shape[0] == vol_shape[1]:
            raise ValueError("Only square volumes")

        if prj_geom is not None and not use_astra:
            raise ValueError("Using class `ProjectionGeometry` requires astra-toolbox.")

        angles_rot_rad = np.array(np.squeeze(angles_rot_rad), ndmin=1)

        if use_astra:
            self.projector_backend = prj_backends.ProjectorBackendASTRA(
                vol_geom,
                angles_rot_rad,
                rot_axis_shift_pix=rot_axis_shift_pix,
                prj_geom=prj_geom,
                create_single_projs=create_single_projs,
                super_sampling=super_sampling,
            )
        else:
            self.projector_backend = prj_backends.ProjectorBackendSKimage(
                vol_geom, angles_rot_rad, rot_axis_shift_pix=rot_axis_shift_pix
            )

        self.angles_rot_rad = angles_rot_rad
        self.prj_intensities = prj_intensities

        self._set_psf(psf)

        self.vol_shape = self.projector_backend.get_vol_shape()
        self.prj_shape = self.projector_backend.get_prj_shape()
        super().__init__()

    def __enter__(self):
        """Initialize the with statement block."""
        self.projector_backend.initialize()
        return self

    def __exit__(self, *args):
        """De-initialize the with statement block."""
        self.projector_backend.dispose()

    def _set_psf(self, psf: ArrayLike, is_conv_symm: bool = False) -> None:
        if psf is not None:
            psf = np.squeeze(psf)
            if len(psf.shape) >= len(self.vol_geom.shape):
                raise ValueError(
                    "PSF should either be 1D (for 2D and 3D reconstructions) or 2D (for 3D reconstructions)."
                    + f" Passed PSF has shape: {psf.shape}, and the reconstruction is {len(self.vol_geom.shape)}D."
                )
            # This redundancy is required, due to the different results between the single-angle and multi-angle projections
            prj_shape_vu = [*self.projector_backend.prj_shape_vu[:-2], self.projector_backend.prj_shape_vu[-1]]
            prj_shape_vu = np.array(prj_shape_vu, ndmin=1)
            self.psf_vu = operators.TransformConvolution(prj_shape_vu, kernel=psf, is_symm=is_conv_symm)
            prj_shape_vwu = np.array(self.projector_backend.prj_shape_vwu, ndmin=1)
            self.psf_vwu = operators.TransformConvolution(prj_shape_vwu, kernel=psf[..., None, :], is_symm=is_conv_symm)
        else:
            self.psf_vu = self.psf_vwu = None

    def fp_angle(self, vol: ArrayLike, angle_ind: int) -> ArrayLike:
        """Forward-project a volume to a single sinogram line.

        Parameters
        ----------
        vol : ArrayLike
            The volume to forward-project.
        angle_ind : int
            The angle index to foward project.

        Returns
        -------
        x : ArrayLike
            The forward-projected sinogram line.
        """
        prj_vu = self.projector_backend.fp(vol, angle_ind)
        if self.prj_intensities is not None:
            prj_vu *= self.prj_intensities[angle_ind]
        if self.psf_vu is not None:
            prj_vu = self.psf_vu(prj_vu)
        return prj_vu

    def bp_angle(self, prj_vu: ArrayLike, angle_ind: int) -> ArrayLike:
        """Back-project a single sinogram line to the volume.

        Parameters
        ----------
        prj_vu : ArrayLike
            The sinogram to back-project or a single line.
        angle_ind : int
            The angle index to foward project.

        Returns
        -------
        ArrayLike
            The back-projected volume.
        """
        if self.prj_intensities is not None:
            prj_vu = prj_vu * self.prj_intensities[angle_ind]  # It will make a copy
        if self.psf_vu is not None:
            prj_vu = self.psf_vu.T(prj_vu)
        return self.projector_backend.bp(prj_vu, angle_ind)

    def fp(self, vol: ArrayLike) -> ArrayLike:
        """
        Forward-projection of the volume to the projection data.

        Parameters
        ----------
        vol : ArrayLike
            The volume to forward-project.

        Returns
        -------
        ArrayLike
            The forward-projected projection data.
        """
        prj_vwu = self.projector_backend.fp(vol)
        if self.prj_intensities is not None:
            prj_vwu *= self.prj_intensities[:, np.newaxis]
        if self.psf_vwu is not None:
            prj_vwu = self.psf_vwu(prj_vwu)
        return prj_vwu

    def bp(self, prj_vwu: ArrayLike) -> ArrayLike:
        """
        Back-projection of the projection data to the volume.

        Parameters
        ----------
        prj_vwu : ArrayLike
            The projection data to back-project.

        Returns
        -------
        ArrayLike
            The back-projected volume.
        """
        if self.prj_intensities is not None:
            prj_vwu = prj_vwu * self.prj_intensities[:, np.newaxis]  # Needs copy
        if self.psf_vwu is not None:
            prj_vwu = self.psf_vwu.T(prj_vwu)
        return self.projector_backend.bp(prj_vwu)

    def fbp(self, projs: ArrayLike, fbp_filter: Union[str, Callable] = "ramp", pad_mode: str = "constant") -> ArrayLike:
        """
        Compute the filtered back-projection of the projection data to the volume.

        The data could either be a sinogram, or a stack of sinograms.

        Parameters
        ----------
        projs : ArrayLike
            The projection data to back-project.
        fbp_filter : str | Callable, optional
            The FBP filter to use. The default is "ramp".
        pad_mode: str, optional
            The padding mode to use for the linear convolution. The default is "constant".

        Raises
        ------
        ValueError
            When trying to use fbp with a 3D projection geometry.

        Returns
        -------
        ArrayLike
            The FBP reconstructed volume.
        """
        if isinstance(fbp_filter, str):
            return self.projector_backend.fbp(projs, fbp_filter=fbp_filter, pad_mode=pad_mode)
        else:
            projs = fbp_filter(projs, self)
            return self.bp(projs) / self.angles_rot_rad.size


class ProjectorAttenuationXRF(ProjectorUncorrected):
    """
    Attenuation corrected projection class for XRF, with multi-detector support.

    It includes the computation of the attenuation volumes.

    Parameters
    ----------
    vol_shape : Union[Sequence[int], ArrayLike]
        The volume shape in X Y and optionally Z.
    angles_rot_rad : ArrayLike
        The rotation angles.
    rot_axis_shift_pix : float, optional
        The rotation axis shift(s) in pixels. The default is 0.
    prj_geom : ProjectionGeometry, optional
        The fully specified projection geometry.
        When active, the rotation axis shift is ignored. The default is None.
    prj_intensities : Optional[ArrayLike], optional
        Projection scaling factor. The default is None.
    super_sampling : int, optional
        Pixel and voxel super-sampling. The default is 1.
    att_in : Optional[ArrayLike], optional
        Attenuation volume of the incoming beam. The default is None.
    att_out : Optional[ArrayLike], optional
        Attenuation volume of the outgoing beam. The default is None.
    angles_detectors_rad : Union[float, ArrayLike], optional
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

    def __init__(
        self,
        vol_shape: Union[Sequence[int], ArrayLike],
        angles_rot_rad: ArrayLike,
        rot_axis_shift_pix: float = 0,
        *,
        prj_geom: Optional[models.ProjectionGeometry] = None,
        prj_intensities: Optional[ArrayLike] = None,
        super_sampling: int = 1,
        att_maps: Optional[ArrayLike] = None,
        att_in: Optional[ArrayLike] = None,
        att_out: Optional[ArrayLike] = None,
        angles_detectors_rad: Union[float, ArrayLike] = (np.pi / 2),
        weights_detectors: Optional[ArrayLike] = None,
        psf: Optional[ArrayLike] = None,
        is_symmetric: bool = False,
        weights_angles: Optional[ArrayLike] = None,
        use_multithreading: bool = True,
        data_type: DTypeLike = np.float32,
        verbose: bool = True,
    ):
        ProjectorUncorrected.__init__(
            self,
            vol_shape,
            angles_rot_rad,
            rot_axis_shift_pix,
            prj_geom=prj_geom,
            psf=psf,
            prj_intensities=prj_intensities,
            super_sampling=super_sampling,
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
                    + " detector angles: %d vs %d" % (num_det_weights, num_det_angles)
                )

        if weights_angles is None:
            weights_angles = np.ones((len(angles_rot_rad), num_det_angles))
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

    def __enter__(self):
        """Initialize the with statement block."""
        if self.use_multithreading and isinstance(self.projector_backend, prj_backends.ProjectorBackendSKimage):
            self.executor = cf.ThreadPoolExecutor(max_workers=num_threads)
        return super().__enter__()

    def __exit__(self, *args):
        """De-initialize the with statement block."""
        super().__exit__()
        if self.use_multithreading and isinstance(self.projector_backend, prj_backends.ProjectorBackendSKimage):
            self.executor.shutdown()

    def collapse_detectors(self) -> None:
        """Convert multi-detector configurations into single-detector."""
        weights = np.reshape(self.weights_det, [1, -1, 1, 1]) / np.sum(self.weights_det)
        self.att_vol_angles = np.sum(self.att_vol_angles * weights, axis=1)

        weights = np.squeeze(weights)
        self.angles_det_rad = np.sum(self.angles_det_rad * weights, keepdims=True)
        self.weights_angles = np.sum(self.weights_angles * weights, axis=1, keepdims=True)
        self.weights_det = np.sum(self.weights_det, keepdims=True)

    def fp_angle(self, vol: ArrayLike, angle_ind: int) -> ArrayLike:
        """Forward-project the volume to a single sinogram line.

        It applies the attenuation corrections.

        Parameters
        ----------
        vol : ArrayLike
            The volume to forward-project.
        angle_ind : int
            The angle index to foward project.

        Returns
        -------
        sino_line : ArrayLike
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

    def bp_angle(self, sino: ArrayLike, angle_ind: int, single_line: bool = False) -> ArrayLike:
        """Back-project a single sinogram line to the volume.

        It only applies the attenuation corrections if the projector is symmetric.

        Parameters
        ----------
        sino : ArrayLike
            The sinogram to back-project or a single line.
        angle_ind : int
            The angle index to foward project.
        single_line : bool, optional
            Whether the input is a single sinogram line. The default is False.

        Returns
        -------
        ArrayLike
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

    def fp(self, vol: ArrayLike) -> ArrayLike:
        """Forward-project the volume to the sinogram.

        It applies the attenuation corrections.

        Parameters
        ----------
        vol : ArrayLike
            The volume to forward-project.

        Returns
        -------
        ArrayLike
            The forward-projected sinogram.
        """
        if self.use_multithreading and isinstance(self.projector_backend, prj_backends.ProjectorBackendSKimage):
            sino_lines = self.executor.map(lambda x: self.fp_angle(vol, x), range(len(self.angles_rot_rad)))
            sino_lines = [*sino_lines]
        else:
            sino_lines = [self.fp_angle(vol, ii) for ii in range(len(self.angles_rot_rad))]

        return np.ascontiguousarray(np.stack(sino_lines, axis=-2))

    def bp(self, sino: ArrayLike) -> ArrayLike:
        """Back-projection of the sinogram to the volume.

        Parameters
        ----------
        sino : ArrayLike
            The sinogram to back-project.

        Returns
        -------
        ArrayLike
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


class FilterMR(object):
    """Data dependent FBP filter.

    This is a simplified implementation from:

    [1] Pelt, D. M., & Batenburg, K. J. (2014). Improving filtered backprojection
    reconstruction by data-dependent filtering. Image Processing, IEEE
    Transactions on, 23(11), 4750-4762.

    Code inspired by: https://github.com/dmpelt/pymrfbp

    Parameters
    ----------
    sinogram_pixels_num : int, optional
        Number of sinogram pixels. The default is None.
    sinogram_angles_num : int, optional
        Number of sinogram angles. The default is None.
    start_exp_binning : int, optional
        From which distance to start exponentional binning. The default is 2.
    lambda_smooth : float, optional
        Smoothing parameter. The default is None.
    data_type : DTypeLike, optional
        Filter data type. The default is np.float32.
    """

    def __init__(
        self,
        sinogram_pixels_num: int = None,
        sinogram_angles_num: int = None,
        start_exp_binning: int = 2,
        lambda_smooth: float = None,
        data_type=np.float32,
    ):
        self.data_type = data_type
        self.start_exp_binning = start_exp_binning
        self.lambda_smooth = lambda_smooth
        self.is_initialized = False
        self.sinogram_pixels_num = sinogram_pixels_num
        self.sinogram_angles_num = sinogram_angles_num

        if sinogram_pixels_num is not None and sinogram_angles_num is not None:
            self.initialize()

    def initialize(self) -> None:
        """Filter initialization function."""
        if self.sinogram_pixels_num is None:
            raise ValueError("No sinogram pixels number was given!")
        if self.sinogram_angles_num is None:
            raise ValueError("No sinogram angles number was given!")

        filter_center = np.floor(self.sinogram_pixels_num / 2).astype(int)
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

    def compute_filter(self, sinogram: ArrayLike, projector: operators.ProjectorOperator) -> ArrayLike:
        """Compute the filter.

        Parameters
        ----------
        sinogram : ArrayLike
            The sinogram.
        projector : operators.ProjectorOperator
            The projector used in the FBP.

        Returns
        -------
        computed_filter : ArrayLike
            The computed filter.
        """
        sino_size = self.sinogram_angles_num * self.sinogram_pixels_num
        nrows = sino_size
        ncols = len(self.basis)

        if self.lambda_smooth:
            grad_vol_size = self.sinogram_pixels_num * (self.sinogram_pixels_num - 1)
            nrows += 2 * grad_vol_size

        A = np.zeros((nrows, ncols), dtype=self.data_type)
        vol_mask = utils_proc.get_circular_mask([self.sinogram_pixels_num] * 2)

        for ii, bas in enumerate(self.basis):
            img = self.apply_filter(sinogram, bas)
            img = projector.bp(img) / self.sinogram_angles_num
            img *= vol_mask

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

    def apply_filter(self, sinogram: ArrayLike, computed_filter: ArrayLike) -> ArrayLike:
        """Apply the filter to the sinogram.

        Parameters
        ----------
        sinogram : ArrayLike
            The sinogram.
        computed_filter : ArrayLike
            The computed filter.

        Returns
        -------
        ArrayLike
            The filtered sinogram.
        """
        return spsig.fftconvolve(sinogram, computed_filter[np.newaxis, ...], "same")

    def __call__(self, sinogram: ArrayLike, projector: operators.ProjectorOperator) -> ArrayLike:
        """Filter the sinogram, by first computing the filter, and then applying it.

        Parameters
        ----------
        sinogram : ArrayLike
            The unfiltered sinogram.
        projector : operators.ProjectorOperator
            The projector used in the FBP.

        Returns
        -------
        ArrayLike
            The filtered sinogram.
        """
        if not self.is_initialized:
            self.sinogram_angles_num = sinogram.shape[0]
            self.sinogram_pixels_num = sinogram.shape[1]
            self.initialize()

        computed_filter = self.compute_filter(sinogram, projector)
        return self.apply_filter(sinogram, computed_filter)
