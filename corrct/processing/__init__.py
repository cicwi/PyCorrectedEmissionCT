# -*- coding: utf-8 -*-

"""Package for data pre-processing and post-processing."""

__author__ = """Nicola VIGANÒ"""
__email__ = "N.R.Vigano@cwi.nl"

from . import pre  # noqa: F401, F402
from . import post  # noqa: F401, F402
from . import misc  # noqa: F401, F402

import numpy as np

from typing import Sequence, Optional, Union
from numpy.typing import DTypeLike, NDArray


eps = np.finfo(np.float32).eps


circular_mask = misc.circular_mask


def compute_variance_poisson(
    Is: NDArray, I0: Optional[NDArray] = None, var_I0: Optional[NDArray] = None, normalized: bool = True
) -> NDArray:
    """
    Compute the variance of a signal subject to Poisson noise, against a reference intensity.

    The reference intensity can also be subject to Poisson and Gaussian noise.
    If the variance of the reference intensity is not passed, it will be assumed to be Poisson.

    Parameters
    ----------
    Is : NDArray
        The signal intensity.
    I0 : Optional[NDArray], optional
        The reference intensity. The default is None.
    var_I0 : Optional[NDArray], optional
        The variance of the reference intensity. The default is None.
        If not given, it will be assumed to be equal to I0.
    normalized : bool, optional
        Whether to renormalize by the mean of the reference intensity.

    Returns
    -------
    NDArray
        The computed variance.
    """
    var_Is = np.abs(Is)
    Is = np.fmax(Is, eps)

    if I0 is not None:
        if var_I0 is None:
            var_I0 = np.abs(I0)
        I0 = np.fmax(I0, eps)

        Is2 = Is**2
        I02 = I0**2
        variance = (Is2 / I02) * (var_Is / Is2 + var_I0 / I02)
        if normalized:
            variance *= np.mean(I0)
        return variance
    else:
        return var_Is


def compute_variance_transmission(
    Is: NDArray, I0: NDArray, var_I0: Optional[NDArray] = None, normalized: bool = True
) -> NDArray:
    """
    Compute the variance of a linearized attenuation (transmission) signal, against a reference intensity.

    Parameters
    ----------
    Is : NDArray
        The transmitted signal.
    I0 : NDArray
        The reference intensity.
    var_I0 : Optional[NDArray], optional
        The variance of the reference intensity. The default is None.
        If not given, it will be assumed to be equal to I0.
    normalized : bool, optional
        Whether to renormalize by the mean of the reference intensity.

    Returns
    -------
    NDArray
        The computed variance.
    """
    var_Is = np.abs(Is)
    Is = np.fmax(Is, eps)

    if var_I0 is None:
        var_I0 = np.abs(I0)
    I0 = np.fmax(I0, eps)

    Is2 = Is**2
    I02 = I0**2
    variance = (Is / I0) * (var_Is / Is2 + var_I0 / I02)
    if normalized:
        variance *= np.mean(I0)
    return variance


def compute_variance_weight(
    variance: NDArray, *, percentile: float = 0.001, normalized: bool = False, use_std: bool = False, semilog: bool = False
) -> NDArray:
    """
    Compute the weight associated to the given variance, in a weighted least-squares context.

    Parameters
    ----------
    variance : NDArray
        The variance of the signal.
    percentile : float
        Minimum percentile to discard. The default is 0.001 (0.1%)
    normalized : bool, optional
        Scale the largest weight to 1. The default is False.
    use_std : bool, optional
        Use the standard deviation instead of the variance.
    semilog : bool, optional
        Scale the variance over a logarithmic curve. It can be beneficial with
        high dynamic range data. The default is False.

    Returns
    -------
    NDArray
        The computed weights.
    """
    variance = np.abs(variance)

    sorted_variances = np.sort(variance.flatten())
    percentiles_variances = np.cumsum(sorted_variances) / np.sum(sorted_variances)
    ind_threshold = np.fmin(np.fmax(0, np.sum(percentiles_variances < percentile)), percentiles_variances.size)
    min_val = np.fmax(sorted_variances[ind_threshold], eps)

    min_nonzero_variance = np.fmax(np.min(variance[variance > 0]), min_val)
    weight = np.fmax(variance, min_nonzero_variance)

    if normalized:
        weight /= min_nonzero_variance
    if use_std:
        weight = np.sqrt(weight)
    if semilog:
        weight = np.log(weight + float(normalized is False)) + 1
    return 1 / weight
