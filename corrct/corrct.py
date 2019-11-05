#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:37:13 2017

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands
"""

import numpy as np

from . import projectors
from . import solvers


def create_sino(
        vol, angles_rad, vol_att_in=None, vol_att_out=None, psf=None,
        data_type=np.float32):
    """Creates a synthetic sinogram, from the given volume, attenuations and
    PSF.

    :param vol: Volume containing the elemental concentrations or other quantities (numpy.array_like)
    :param angles_rad: Angles in radians of each sinogram line (numpy.array_like)
    :param vol_att_in: Volume containing the local attenuation coefficients for the incoming photons (numpy.array_like)
    :param vol_att_out: Volume containing the local attenuation coefficients for the emitted photons (numpy.array_like)
    :param psf: Detector point spread function (PSF) (numpy.array_like)
    :param data_type: Volume data type (numpy.dtype)

    :returns: The simulated sinogram
    :rtype: (numpy.array_like)
    """
    sino = np.empty([len(angles_rad), vol.shape[0]], dtype=np.float32)
    with projectors.AttenuationProjector(
            vol.shape, angles_rad, att_in=vol_att_in, att_out=vol_att_out,
            psf=psf, data_type=data_type) as p:
        for ii in range(len(angles_rad)):
            sino[ii, :] = p.fp_angle(vol, ii)
    return sino


def reconstruct(
        algo, sino, angles_rad, iterations=None,
        vol_att_in=None, vol_att_out=None,
        lower_limit=None, upper_limit=None, apply_circ_mask=True,
        symm=True, lambda_reg=1e-2, data_term='kl', psf=None,
        data_type=np.float32):
    """Reconstructs the given sinogram, with the requested algorithm.

    :param algo: Reconstruction algorithms to use. Options: 'SART' | 'SIRT' | 'CP' | 'CPTV' | 'CPL1' (string)
    :param sino: The sinogram to recosntruct (numpy.array_like)
    :param angles_rad: Angles in radians of each sinogram line (numpy.array_like)
    :param iterations: Number of iterations (int)
    :param vol_att_in: Volume containing the local attenuation coefficients for the incoming photons (numpy.array_like)
    :param vol_att_out: Volume containing the local attenuation coefficients for the emitted photons (numpy.array_like)
    :param lower_limit: Lower clipping limit (float)
    :param upper_limit: Upper clipping limit (float)
    :param apply_circ_mask: Switch to activate a circular volume mask (boolean)
    :param symm: Switch to define whether the projectors should be symmetric (boolean)
    :param lambda_reg: Regularizer weight (float)
    :param data_term: Data fidelity term. Options: 'l2' | 'kl' (string)
    :param psf: Detector point spread function (PSF) (numpy.array_like)
    :param data_type: Volume data type (numpy.dtype)

    :returns: The reconstructed volume
    :rtype: (numpy.array_like)
    """
    vol_shape = [sino.shape[1], sino.shape[1]]

    if apply_circ_mask:
        xx = np.arange(-(vol_shape[0]-1)/2, (vol_shape[0]-1)/2+0.001, 1, dtype=data_type)
        yy = np.arange(-(vol_shape[1]-1)/2, (vol_shape[1]-1)/2+0.001, 1, dtype=data_type)
        (xx, yy) = np.meshgrid(xx, yy, indexing='ij')
        circ_mask = np.sqrt(xx ** 2 + yy ** 2) <= (vol_shape[0]-1)/2

    det_pos = np.pi / 2

    with projectors.AttenuationProjector(
            vol_shape, angles_rad, att_in=vol_att_in, att_out=vol_att_out,
            detector_angle=det_pos, psf=psf, is_symmetric=symm,
            data_type=data_type) as p:

        if algo.upper() == 'SART':
            A = lambda x, ii : p.fp_angle(x, ii)
            At = lambda y, ii : p.bp_angle(y, ii, single_line=True)
        else:
            A = lambda x : p.fp(x)
            At = lambda y : p.bp(y)

        if iterations is None:
            if algo.upper() in ('SIRT', 'CPTV'):
                iterations = 50
            elif algo.upper() in ('CP'):
                iterations = 25
            else:
                iterations = 5

        # Algorithms
        if algo.upper() == 'SART':
            algo = solvers.Sart(verbose=True)
            (vol, _) = algo(A, sino, iterations, len(angles_rad), At=At, x_mask=circ_mask)
        elif algo.upper() == 'SIRT':
            algo = solvers.Sirt(verbose=True)
            (vol, _) = algo(A, sino, iterations, At=At, x_mask=circ_mask)
        elif algo.upper() == 'CP':
            algo = solvers.CP(verbose=True, data_term=data_term)
            (vol, _) = algo(A, sino, iterations, At=At, x_mask=circ_mask)
        elif algo.upper() == 'CPTV':
            regularizer = solvers.Regularizer_TV2D(weight=lambda_reg)
            algo = solvers.CP(
                    verbose=True, data_term=data_term, regularizer=regularizer)
            (vol, _) = algo(A, sino, iterations, At=At, x_mask=circ_mask)
        elif algo.upper() == 'CPL1':
            regularizer = solvers.Regularizer_l1(weight=lambda_reg)
            algo = solvers.CP(
                    verbose=True, data_term=data_term, regularizer=regularizer)
            (vol, _) = algo(A, sino, iterations, At=At, x_mask=circ_mask)
        else:
            raise ValueError('Unknown algorithm: %s' % algo)

    return vol

