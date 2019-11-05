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

class Projector2D(object):
    """Basic projection class, which implements the forward and back projection
    of the single lines of a sinogram.
    It takes care of initializing and disposing the ASTRA projectors when used
    in a *with* statement.
    It includes the computation of the attenuation volumes.
    """

    def __init__(self, vol_shape, angles_rot_rad):
        if len(vol_shape) > 2:
            raise ValueError("Only 2D volumes")
        if not vol_shape[0] == vol_shape[1]:
            raise ValueError("Only square volumes")

        self.proj_id = []
        self.dispose_projectors()

        self.vol_shape = vol_shape
        self.angles_rot_rad = angles_rot_rad

        self.vol_geom = astra.create_vol_geom(vol_shape)
        # The duplication of angles is used to work around an error in the
        # projection geometry creation of ASTRA. It adds a performance penalty.
        self.proj_geom = [
                astra.create_proj_geom('parallel', 1, vol_shape[0], (a, a)) for a in self.angles_rot_rad]

    def initialize_projectors(self):
        """Initialization of the ASTRA projectors.
        """
        self.proj_id = [astra.create_projector('linear', pg, self.vol_geom) for pg in self.proj_geom]
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
        return self.W[ii].FP(vol)[0, :]

    def bp_angle(self, sino_line, ii):
        """Back-projection of a single sinogram line to the volume.
        """
        sino_line = np.reshape(sino_line, (1, -1))
        sino_line = np.tile(sino_line, (2, 1))
        return self.W[ii].BP(sino_line) / 2

    @staticmethod
    def compute_attenuation(vol, direction, sampling=1, invert=False):
        """Computes the attenuation experienced by the photons emitted in every
        point of the volume, along a certain direction.
        """
        vol = np.array(vol)
        if len(vol.shape) > 2:
            raise ValueError("Only 2D Arrays")

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
        cum_arr = np.cumsum(cum_arr, axis=-1)

        cum_arr = spimg.interpolation.rotate(cum_arr, -rot_angle, reshape=False, order=1)
        cum_arr = cum_arr[edges[0]:-edges[0], edges[1]:-edges[1]]

        cum_arr = np.exp(- cum_arr)

        return cum_arr


class AttenuationProjector(Projector2D):
    """Attenuation corrected projection class.
    """

    def __init__(
            self, vol_shape, angles_rot_rad, att_in=None, att_out=None,
            detector_angle=(np.pi/2), psf=None, precompute_attenuation=True,
            is_symmetric=False, data_type=np.float32 ):
        Projector2D.__init__(self, vol_shape, angles_rot_rad)

        if precompute_attenuation:
            if att_in is None and att_out is None:
                print('Turning off precomputation of attenuation.')
                precompute_attenuation = False

        self.data_type = data_type

        self.att_in = att_in
        self.att_out = att_out
        self.detector_angle = detector_angle
        self.precompute_attenuation = precompute_attenuation
        self.is_symmetric = is_symmetric

        self.psf = psf

        if self.precompute_attenuation:
            self.compute_attenuation_volumes()
        else:
            self.att_vol_angles = None

    def _compute_attenuation_angle_in(self, a):
        direction_in = [np.sin(a), np.cos(a)]
        return Projector2D.compute_attenuation(self.att_in, direction_in)

    def _compute_attenuation_angle_out(self, a):
        a_det = a + self.detector_angle
        direction_out = [np.sin(a_det), np.cos(a_det)]
        return Projector2D.compute_attenuation(self.att_out, direction_out, invert=True)

    def compute_attenuation_volumes(self):
        """Computes the corrections for each angle.
        """
        if self.att_in is None and self.att_out is None:
            raise ValueError('No attenuation volumes were given')

        self.att_vol_angles = np.ones(
                [len(self.angles_rot_rad), *self.vol_shape], dtype=self.data_type)

        if self.att_in is not None:
            for ii, a in enumerate(self.angles_rot_rad):
                self.att_vol_angles[ii, ...] *= self._compute_attenuation_angle_in(a)

        if self.att_out is not None:
            for ii, a in enumerate(self.angles_rot_rad):
                self.att_vol_angles[ii, ...] *= self._compute_attenuation_angle_out(a)

    def fp_angle(self, vol, angle_ind):
        """Forward-projection of the volume to a single sinogram line. It
        applies the attenuation corrections.

        :param vol: The volume to forward-project (numpy.array_like)
        :param angle_ind: The angle index to foward project (int)

        :returns: The forward-projected sinogram line
        :rtype: (numpy.array_like)
        """
        temp_vol = copy.deepcopy(vol)

        if self.precompute_attenuation:
            temp_vol *= self.att_vol_angles[angle_ind, ...]
        else:
            a = self.angles_rot_rad[angle_ind]
            if self.att_in is not None:
                temp_vol *= self._compute_attenuation_angle_in(a)
            if self.vol_att_out is not None:
                temp_vol *= self._compute_attenuation_angle_out(a)

        sino_line = Projector2D.fp_angle(self, temp_vol, angle_ind)

        if self.psf is not None:
            sino_line = spsig.convolve(sino_line, self.psf, mode='same')

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
            sino_line = sino[angle_ind, :]

        if self.psf is not None:
            sino_line = spsig.convolve(sino_line, self.psf, mode='same')

        vol = Projector2D.bp_angle(self, sino_line, angle_ind)

        if self.is_symmetric:
            if self.precompute_attenuation:
                vol *= self.att_vol_angles[angle_ind, ...]
            else:
                a = self.angles_rot_rad[angle_ind]
                if self.att_in is not None:
                    vol *= self._compute_attenuation_angle_in(a)
                if self.vol_att_out is not None:
                    vol *= self._compute_attenuation_angle_out(a)

        return vol

    def fp(self, vol):
        return np.stack(
                [self.fp_angle(vol, ii) for ii in range(len(self.angles_rot_rad))],
                axis=0)

    def bp(self, sino):
        return np.sum(
                [self.bp_angle(sino, ii) for ii in range(len(self.angles_rot_rad))],
                axis=0)

