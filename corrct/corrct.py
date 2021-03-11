#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:37:13 2017

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np

from . import projectors
from . import solvers
from . import utils_proc


def create_sino(
    vol,
    angles_rad,
    vol_att_in=None,
    vol_att_out=None,
    psf=None,
    angles_detectors_rad=(np.pi / 2),
    weights_detectors=None,
    data_type=np.float32,
):
    """Creates a synthetic sinogram, from the given volume, attenuations and PSF.

    :param vol: Volume containing the elemental concentrations or other quantities.
    :type vol: numpy.array_like
    :param angles_rad: Angles in radians of each sinogram line
    :type angles_rad: numpy.array_like
    :param vol_att_in: Volume containing the local attenuation coefficients for the incoming photons
    :type vol_att_in: numpy.array_like
    :param vol_att_out: Volume containing the local attenuation coefficients for the emitted photons
    :type vol_att_out: numpy.array_like
    :param angles_detectors_rad: Detectors' position angles in radians
    :type angles_detectors_rad: numpy.array_like or float, optional. Deafult: numpy.pi / 2
    :param weights_detectors: Detectors' weights
    :type weights_detectors: numpy.array_like or float, optional. Deafult: None
    :param psf: Detector point spread function (PSF)
    :type psf: numpy.array_like, optional. Deafult: None
    :param data_type: Volume data type
    :type data_type: `numpy.dtype`, optional. Default: `numpy.float32`

    :returns: The simulated sinogram
    :rtype: numpy.array_like
    """
    with projectors.ProjectorAttenuationXRF(
        vol.shape,
        angles_rad,
        att_in=vol_att_in,
        att_out=vol_att_out,
        angles_detectors_rad=angles_detectors_rad,
        weights_detectors=weights_detectors,
        psf=psf,
        data_type=data_type,
    ) as p:
        return p.fp(vol)


def reconstruct(  # noqa: C901
    algo,
    sino,
    angles_rad,
    iterations=None,
    vol_att_in=None,
    vol_att_out=None,
    angles_detectors_rad=(np.pi / 2),
    weights_detectors=None,
    lower_limit=None,
    upper_limit=None,
    apply_circ_mask=True,
    symm=True,
    lambda_reg=1e-2,
    data_term="l2",
    psf=None,
    data_type=np.float32,
):
    """Reconstructs the given sinogram, with the requested algorithm.

    :param algo: Reconstruction algorithms to use.
    :type algo: string. Options: 'SART' | 'SIRT' | 'CP' | 'CPTV' | 'CPL1' | 'CPWL'
    :param sino: The sinogram to recosntruct
    :type sino: numpy.array_like
    :param angles_rad: Angles in radians of each sinogram line
    :type angles_rad: numpy.array_like
    :param iterations: Number of iterations
    :type iterations: int, optional. Default: None
    :param vol_att_in: Volume containing the local attenuation coefficients for the incoming photons
    :type vol_att_in: numpy.array_like, optional. Default: None
    :param vol_att_out: Volume containing the local attenuation coefficients for the emitted photons
    :type vol_att_out: numpy.array_like, optional. Default: None
    :param angles_detectors_rad: Detectors' position angles in radians
    :type angles_detectors_rad: numpy.array_like or float, optional. Default: numpy.pi / 2
    :param weights_detectors: Detectors' weights
    :type weights_detectors: numpy.array_like or float, optional. Default: None
    :param lower_limit: Lower clipping limit
    :type lower_limit: float, optional. Default: None
    :param upper_limit: Upper clipping limit
    :type upper_limit: float, optional. Default: None
    :param apply_circ_mask: Switch to activate a circular volume mask
    :type apply_circ_mask: boolean, optional. Default: True
    :param symm: Switch to define whether the projectors should be symmetric
    :type symm: boolean, optional. Default: True
    :param lambda_reg: Regularizer weight
    :type lambda_reg: float, optional. Default: 1e-2
    :param data_term: Data fidelity term
    :type data_term: string, optional. Options: 'l2' | 'kl'. Default: 'l2'
    :param psf: Detector point spread function (PSF)
    :type psf: numpy.array_like, optional. Default: None
    :param data_type: Volume data type
    :type data_type: `numpy.dtype`, optional. Default: `numpy.float32`

    :raises ValueError: Raises an error if the algorithm is not known.

    :returns: The reconstructed volume
    :rtype: numpy.array_like
    """
    vol_shape = [sino.shape[-1], sino.shape[-1]]

    if apply_circ_mask:
        x_mask = utils_proc.get_circular_mask(vol_shape, radius_offset=-1)

    with projectors.ProjectorAttenuationXRF(
        vol_shape,
        angles_rad,
        att_in=vol_att_in,
        att_out=vol_att_out,
        angles_detectors_rad=angles_detectors_rad,
        weights_detectors=weights_detectors,
        psf=psf,
        is_symmetric=symm,
        data_type=data_type,
    ) as p:

        if algo.upper() == "SART":
            A = lambda x, ii: p.fp_angle(x, ii)  # noqa: E731
            At = lambda y, ii: p.bp_angle(y, ii, single_line=True)  # noqa: E731
        else:
            A = p
            At = p.T

        if iterations is None:
            if algo.upper() in ("SIRT", "CPTV", "CPL1", "CPWL"):
                iterations = 50
            elif algo.upper() in ("CP"):
                iterations = 25
            else:
                iterations = 5

        precondition = True

        # Algorithms
        if algo.upper() == "SART":
            algo = solvers.Sart(verbose=True)
            (vol, _) = algo(A, sino, iterations, len(angles_rad), At=At, x_mask=x_mask)
        elif algo.upper() == "SIRT":
            algo = solvers.Sirt(verbose=True)
            (vol, _) = algo(A, sino, iterations, At=At, x_mask=x_mask)
        elif algo.upper() == "CP":
            algo = solvers.CP(verbose=True, data_term=data_term)
            (vol, _) = algo(A, sino, iterations, At=At, x_mask=x_mask, precondition=precondition)
        elif algo.upper() == "CPTV":
            regularizer = solvers.Regularizer_TV2D(weight=lambda_reg)
            algo = solvers.CP(verbose=True, data_term=data_term, regularizer=regularizer)
            (vol, _) = algo(A, sino, iterations, At=At, x_mask=x_mask, precondition=precondition)
        elif algo.upper() == "CPL1":
            regularizer = solvers.Regularizer_l1(weight=lambda_reg)
            algo = solvers.CP(verbose=True, data_term=data_term, regularizer=regularizer)
            (vol, _) = algo(A, sino, iterations, At=At, x_mask=x_mask, precondition=precondition)
        elif algo.upper() == "CPWL":
            regularizer = solvers.Regularizer_l1swl(weight=lambda_reg, pad_on_demand=True)
            algo = solvers.CP(verbose=True, data_term=data_term, regularizer=regularizer)
            (vol, _) = algo(A, sino, iterations, At=At, x_mask=x_mask, precondition=precondition)
        else:
            raise ValueError("Unknown algorithm: %s" % algo)

    return vol
