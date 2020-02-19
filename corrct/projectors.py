# -*- coding: utf-8 -*-
"""
Tomographic projectors.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands
"""

import numpy as np

import scipy.ndimage as spimg
import scipy.signal as spsig

import copy

import astra

class ProjectorBase(object):
    """Basic projection class, which implements the forward and back projection
    of the single lines of a sinogram.
    It takes care of initializing and disposing the ASTRA projectors when used
    in a *with* statement.
    It includes the computation of the attenuation volumes.
    """

    def __init__(self, vol_shape, angles_rot_rad, rot_axis_shift_pix=0):
        self.proj_id = []
        self.dispose_projectors()

        self.vol_shape = vol_shape
        self.angles_rot_rad = angles_rot_rad

        self.vol_geom = astra.create_vol_geom(vol_shape)
        self.initialize_geometry(vol_shape, angles_rot_rad, rot_axis_shift_pix)

    def initialize_geometry(self, vol_shape, angles_rot_rad, rot_axis_shift_pix):

        if len(vol_shape) < 2 or len(vol_shape) > 3:
            raise ValueError("Only 2D or 3D volumes")
        if not vol_shape[0] == vol_shape[1]:
            raise ValueError("Only square volumes")

        num_angles = angles_rot_rad.size

        self.is_3d = len(vol_shape) == 3
        if self.is_3d:
            vectors = np.empty([num_angles, 12])
            # source
            vectors[:, 0] = -np.sin(angles_rot_rad)
            vectors[:, 1] = -np.cos(angles_rot_rad)
            vectors[:, 2] = 0
            # vector from detector pixel (0,0) to (0,1)
            vectors[:, 6] = np.cos(angles_rot_rad)
            vectors[:, 7] = -np.sin(angles_rot_rad)
            vectors[:, 8] = 0
            # center of detector
            vectors[:, 3:6] = rot_axis_shift_pix * vectors[:, 6:9]
            # vector from detector pixel (0,0) to (1,0)
            vectors[:, 9] = 0
            vectors[:, 10] = 0
            vectors[:, 11] = 1

            self.proj_geom = [
                    astra.create_proj_geom('parallel3d_vec', vol_shape[2], vol_shape[0],
                                           np.tile(np.reshape(vectors[ii, :], [1, -1]), [2, 1]))
                    for ii in range(num_angles)]
        else:
            vectors = np.empty([num_angles, 6])
            # source
            vectors[:, 0] = np.sin(angles_rot_rad)
            vectors[:, 1] = -np.cos(angles_rot_rad)
            # vector from detector pixel 0 to 1
            vectors[:, 4] = np.cos(angles_rot_rad)
            vectors[:, 5] = np.sin(angles_rot_rad)
            # center of detector
            vectors[:, 2:4] = rot_axis_shift_pix * vectors[:, 4:6]

            self.proj_geom = [
                    astra.create_proj_geom('parallel_vec', vol_shape[0],
                                           np.tile(np.reshape(vectors[ii, :], [1, -1]), [2, 1]))
                    for ii in range(num_angles)]

    def initialize_projectors(self):
        """Initialization of the ASTRA projectors.
        """
        if self.is_3d:
            projector_type = 'linear3d'
        else:
            projector_type = 'linear'
        self.proj_id = [astra.create_projector(projector_type, pg, self.vol_geom) for pg in self.proj_geom]
        self.W = [astra.OpTomo(p_id) for p_id in self.proj_id]

    def dispose_projectors(self):
        """Disposal of the ASTRA projectors.
        """
        for p in self.proj_id:
            astra.projector.delete(p)
        self.proj_id = []
        self.W = []

    def __enter__(self):
        self.initialize_projectors()
        return self

    def __exit__(self, *args):
        self.dispose_projectors()

    def fp_angle(self, vol, ii):
        """Forward-projection of the volume to a single sinogram line.
        """
        return self.W[ii].FP(vol)[0, ...]

    def bp_angle(self, sino_line, ii):
        """Back-projection of a single sinogram line to the volume.
        """
        sino = np.empty([2, *np.squeeze(sino_line).shape], dtype=sino_line.dtype)
        sino[0, ...] = sino_line
        sino[1, ...] = 0
        return self.W[ii].BP(sino)


class AttenuationProjector(ProjectorBase):
    """Attenuation corrected projection class, with multi-detector support.
    """

    def __init__(
            self, vol_shape, angles_rot_rad, rot_axis_shift_pix=0,
            att_in=None, att_out=None,
            angles_detectors_rad=(np.pi/2), weights_detectors=None, psf=None,
            precompute_attenuation=True, is_symmetric=False, weights_angles=None,
            data_type=np.float32 ):
        ProjectorBase.__init__(self, vol_shape, angles_rot_rad, rot_axis_shift_pix)

        self.data_type = data_type

        if precompute_attenuation:
            if att_in is None and att_out is None:
                print('Turning off precomputation of attenuation.')
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
                        'Number of detector weights differs from number of' +
                        ' detector angles: %d vs %d' % (num_det_weights, num_det_angles))

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

    def compute_attenuation(self, vol, direction, sampling=1, invert=False):
        """Computes the attenuation experienced by the photons emitted in every
        point of the volume, along a certain direction.
        """
        vol = np.array(vol)
        if len(vol.shape) > 2:
            raise ValueError("Maps can only be 2D Arrays")
        if not np.all(vol.shape == self.vol_shape[1:3]):
            raise ValueError("Mismatching volume shape of input volume with vol_shape")

        size_lims = np.array(vol.shape)
        min_size = np.ceil(np.sqrt(np.sum(size_lims ** 2)))
        edges = np.ceil((min_size - size_lims) / 2).astype(np.intp)

        direction = np.array(direction)
        if invert:
            direction = -direction
        direction = direction / np.sqrt(np.sum(direction ** 2))

        rot_angle = np.rad2deg(np.arctan2(direction[1], direction[0]))

        cum_arr = np.pad(vol, ((edges[0], ), (edges[1], )), mode='constant')

        cum_arr = spimg.interpolation.rotate(cum_arr, rot_angle, reshape=False, order=1)
        cum_arr += np.roll(cum_arr, 1, axis=-1)
        cum_arr = np.cumsum(cum_arr / 2, axis=-1)

        cum_arr = spimg.interpolation.rotate(cum_arr, -rot_angle, reshape=False, order=1)
        cum_arr = cum_arr[edges[0]:-edges[0], edges[1]:-edges[1]]

        cum_arr = np.exp(- cum_arr)

        return cum_arr

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
            raise ValueError('No attenuation volumes were given')

        self.att_vol_angles = np.ones(
                [len(self.angles_rot_rad), len(self.angles_det_rad), *self.vol_shape[:3]],
                dtype=self.data_type)

        if self.att_in is not None:
            for ii, a in enumerate(self.angles_rot_rad):
                self.att_vol_angles[ii, ...] *= self._compute_attenuation_angle_in(a)

        if self.att_out is not None:
            for ii, a in enumerate(self.angles_rot_rad):
                self.att_vol_angles[ii, ...] *= self._compute_attenuation_angle_out(a)

        if self.is_3d:
            self.att_vol_angles = self.att_vol_angles[:, :, np.newaxis, ...]

    def collapse_detectors(self):
        weights = np.reshape(self.weights_det, [1, -1, 1, 1]) / np.sum(self.weights_det)

        self.att_vol_angles = np.sum(self.att_vol_angles * weights, axis=1)
        self.angles_det_rad = np.sum(self.angles_det_rad * np.squeeze(weights), keepdims=True)
        self.weights_angles = np.sum(self.weights_angles * np.squeeze(weights), axis=1, keepdims=True)
        self.weights_det = np.sum(self.weights_det, keepdims=True)

    def fp_angle(self, vol, angle_ind):
        """Forward-projection of the volume to a single sinogram line. It
        applies the attenuation corrections.

        :param vol: The volume to forward-project (numpy.array_like)
        :param angle_ind: The angle index to foward project (int)

        :returns: The forward-projected sinogram line
        :rtype: (numpy.array_like)
        """
        temp_vol = copy.deepcopy(vol)[np.newaxis, ...]
        temp_vol = np.tile(temp_vol, (len(self.angles_det_rad), *((1, ) * len(self.vol_shape))))

        if self.precompute_attenuation:
            temp_vol *= self.att_vol_angles[angle_ind, ...]
        else:
            a = self.angles_rot_rad[angle_ind]
            if self.att_in is not None:
                temp_vol *= self._compute_attenuation_angle_in(a)
            if self.vol_att_out is not None:
                temp_vol *= self._compute_attenuation_angle_out(a)

        weights = self.weights_det * self.weights_angles[angle_ind, :]
        sino_line = [
                weights[ii] * ProjectorBase.fp_angle(self, temp_vol[ii], angle_ind)
                for ii in range(len(self.angles_det_rad))]
        sino_line = np.stack(sino_line, axis=0)

        if self.psf is not None:
            sino_line = spsig.convolve(sino_line, self.psf, mode='same')

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
        :rtype: (numpy.array_like)
        """
        if single_line:
            sino_line = sino
        else:
            sino_line = sino[angle_ind, ...]

        if self.psf is not None:
            sino_line = spsig.convolve(sino_line, self.psf, mode='same')

        sino_line = np.reshape(sino_line, [len(self.weights_det), *sino_line.shape[-(len(self.vol_shape)-1):]])
        weights = self.weights_det * self.weights_angles[angle_ind, :]
        vol = [
                weights[ii] * ProjectorBase.bp_angle(self, sino_line[ii, ...], angle_ind)
                for ii in range(len(self.angles_det_rad))]
        vol = np.stack(vol, axis=0)

        if self.is_symmetric:
            if self.precompute_attenuation:
                vol *= self.att_vol_angles[angle_ind, ...]
            else:
                a = self.angles_rot_rad[angle_ind]
                if self.att_in is not None:
                    vol *= self._compute_attenuation_angle_in(a)
                if self.vol_att_out is not None:
                    vol *= self._compute_attenuation_angle_out(a)

        return np.sum(vol, axis=0)

    def fp(self, vol):
        return np.stack(
                [self.fp_angle(vol, ii) for ii in range(len(self.angles_rot_rad))],
                axis=0)

    def bp(self, sino):
        return np.sum(
                [self.bp_angle(sino, ii) for ii in range(len(self.angles_rot_rad))],
                axis=0)

