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

import concurrent.futures as cf
import multiprocessing as mp

from tqdm import tqdm

from typing import Union, Sequence, Optional, Callable
from numpy.typing import ArrayLike, DTypeLike

from dataclasses import dataclass, field


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
    rot_axis_shift_pix : float or ArrayLike, optional
        The rotation axis shift(s) in pixels. The default is 0.
    prj_geom : ProjectionGeometry, optional
        The fully specified projection geometry.
        When active, the rotation axis shift is ignored. The default is None.
    prj_intensities : float or ArrayLike, optional
        Projection scaling factor. The default is None.
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

        self.psf = psf

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
        x = self.projector_backend.fp(vol, angle_ind)
        if self.prj_intensities is not None:
            x *= self.prj_intensities[angle_ind]
        if self.psf is not None:
            x = spsig.convolve(x, self.psf, mode="same")
        return x

    def bp_angle(self, sino_line: ArrayLike, angle_ind: int) -> ArrayLike:
        """Back-project a single sinogram line to the volume.

        Parameters
        ----------
        sino_line : ArrayLike
            The sinogram to back-project or a single line.
        angle_ind : int
            The angle index to foward project.

        Returns
        -------
        ArrayLike
            The back-projected volume.
        """
        if self.prj_intensities is not None:
            sino_line = sino_line * self.prj_intensities[angle_ind]  # It will make a copy
        return self.projector_backend.bp(sino_line, angle_ind)

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
        x = self.projector_backend.fp(vol)
        if self.prj_intensities is not None:
            x *= self.prj_intensities[:, np.newaxis]
        if self.psf is not None:
            x = spsig.convolve(x, self.psf[..., None, :], mode="same")
        return x

    def bp(self, data: ArrayLike) -> ArrayLike:
        """
        Back-projection of the projection data to the volume.

        Parameters
        ----------
        data : ArrayLike
            The projection data to back-project.

        Returns
        -------
        ArrayLike
            The back-projected volume.
        """
        if self.prj_intensities is not None:
            data = data * self.prj_intensities[:, np.newaxis]  # Needs copy
        return self.projector_backend.bp(data)

    def fbp(self, projs: ArrayLike, fbp_filter: Union[str, Callable] = "shepp-logan") -> ArrayLike:
        """
        Compute the filtered back-projection of the projection data to the volume.

        The data could either be a sinogram, or a stack of sinograms.

        Parameters
        ----------
        projs : ArrayLike
            The projection data to back-project.
        fbp_filter : str | Callable, optional
            The FBP filter to use. The default is "shepp-logan".

        Raises
        ------
        ValueError
            When trying to use fbp with a 3D projection geometry.

        Returns
        -------
        ArrayLike
            The FBP reconstructed volume.
        """
        if self.vol_geom.is_3D():
            raise ValueError("FBP not supported with 3D projector")

        if isinstance(fbp_filter, str):
            return self.projector_backend.fbp(projs, fbp_filter)
        else:
            projs = fbp_filter(projs, self)
            return self.bp(projs) / self.angles_rot_rad.size


@dataclass
class AttenuationVolume:
    """Attenuation volume computation class."""

    incident_local: ArrayLike
    emitted_local: ArrayLike
    angles_rot_rad: ArrayLike
    angles_det_rad: ArrayLike = np.pi / 2

    dtype: DTypeLike = np.float32

    vol_shape_zyx: ArrayLike = field(init=False)
    maps: ArrayLike = field(init=False, default=None)

    def __post_init__(self):
        """
        Initialize the AttenuationVolume class.

        Raises
        ------
        ValueError
            In case no volumes were passed, or if they differed in shape.
        """
        self.angles_det_rad = np.array(self.angles_det_rad, ndmin=1)

        if self.incident_local is not None:
            self.vol_shape_zyx = self.incident_local.shape

            if self.emitted_local is not None and np.any(self.vol_shape_zyx != self.emitted_local.shape):
                raise ValueError(
                    f"Incident volume shape ({self.incident_local.shape}) does not"
                    + f" match the emitted volume shape ({self.emitted_local.shape})"
                )
        elif self.emitted_local is not None:
            self.vol_shape_zyx = self.emitted_local.shape
        else:
            raise ValueError("No attenuation volumes were given.")

        self.vol_shape_zyx = np.array(self.vol_shape_zyx, ndmin=1)

        num_dims = len(self.vol_shape_zyx)
        if num_dims not in [2, 3]:
            raise ValueError(f"Maps can only be 2D or 3D Arrays. A {num_dims}-dimensional was passed ({self.vol_shape_zyx}).")

    @staticmethod
    def _compute_attenuation(vol: ArrayLike, angle_rad: float, invert: bool = False) -> ArrayLike:
        return prj_backends.ProjectorBackend.compute_attenuation(vol, angle_rad, invert)

    def _compute_attenuation_angle_in(self, angle_rad: float) -> ArrayLike:
        return self._compute_attenuation(self.incident_local, angle_rad)[None, ...]

    def _compute_attenuation_angle_out(self, angle_rad: float) -> ArrayLike:
        angle_det = angle_rad + self.angles_det_rad
        atts = np.empty(self.maps.shape[1:], dtype=self.dtype)
        for ii, a in enumerate(angle_det):
            atts[ii, ...] = self._compute_attenuation(self.emitted_local, a, invert=True)
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
                r = [None] * num_rot_angles
                with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # angle_atts = executor.map(self._compute_attenuation_angle_in, self.angles_rot_rad)
                    for ii, a in enumerate(self.angles_rot_rad):
                        r[ii] = executor.submit(self._compute_attenuation_angle_in, a)
                    for ii in tqdm(range(num_rot_angles), desc=description, disable=(not verbose)):
                        self.maps[ii, ...] *= r[ii].result()
            else:
                for ii, a in enumerate(tqdm(self.angles_rot_rad, desc=description, disable=(not verbose))):
                    self.maps[ii, ...] *= self._compute_attenuation_angle_in(a)

        if self.emitted_local is not None:
            description = "Computing attenuation maps for emitted photons: "
            if use_multithreading:
                r = [None] * num_rot_angles
                with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    for ii, a in enumerate(self.angles_rot_rad):
                        r[ii] = executor.submit(self._compute_attenuation_angle_out, a)
                    for ii in tqdm(range(num_rot_angles), desc=description, disable=(not verbose)):
                        self.maps[ii, ...] *= r[ii].result()
            else:
                for ii, a in enumerate(tqdm(self.angles_rot_rad, desc=description, disable=(not verbose))):
                    self.maps[ii, ...] *= self._compute_attenuation_angle_out(a)

    def plot_map(
        self, ax, rot_ind: int, det_ind: int = 0, slice_ind: Optional[int] = None, axes: ArrayLike = (-2, -1)
    ) -> None:
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
        axes : ArrayLike, optional
            Axes of the slice. The default is (-2, -1).

        Raises
        ------
        ValueError
            In case a slice index is not passed for a 3D volume.
        """
        slice_shape = self.vol_shape_zyx[list(axes)]
        coords = [(-(s - 1) / 2, (s - 1) / 2) for s in slice_shape]

        att_map = np.squeeze(self.get_maps(rot_ind=rot_ind, det_ind=det_ind))
        other_dim = np.squeeze(np.delete(np.arange(-3, 0), axes))
        if len(att_map.shape) == 3:
            if slice_ind is None:
                raise ValueError("Slice index is needed for 3D volumes. None was passed.")

            att_map = np.take(att_map, slice_ind, axis=other_dim)

        ax.imshow(att_map, extent=np.concatenate(coords))

        if other_dim == -3:
            arrow_length = np.linalg.norm(slice_shape) / np.pi
            arrow_args = dict(
                width=arrow_length / 25, head_width=arrow_length / 8, head_length=arrow_length / 6, length_includes_head=True,
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

    def get_maps(
        self, roi: Optional[ArrayLike] = None, rot_ind: Optional[int] = None, det_ind: Optional[int] = None
    ) -> ArrayLike:
        """
        Return the attenuation maps.

        Parameters
        ----------
        roi : Optional[ArrayLike], optional
            The region-of-interest to select. The default is None.

        Returns
        -------
        ArrayLike
            The attenuation maps.
        """
        maps = self.maps

        if rot_ind is not None:
            maps = maps[rot_ind : rot_ind + 1 :, ...]
        if det_ind is not None:
            maps = maps[:, det_ind : det_ind + 1 :, ...]
        if roi is not None:
            raise NotImplementedError("Extracting a region of interest is not supported, yet.")

        return maps


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

        if self.psf is not None and self.is_symmetric:
            sino_line = spsig.convolve(sino_line, self.psf, mode="same")

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
