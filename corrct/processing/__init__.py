# -*- coding: utf-8 -*-

"""Package for data pre-processing and post-processing."""

__author__ = """Nicola VIGANÒ"""
__email__ = "N.R.Vigano@cwi.nl"

from . import pre  # noqa: F401, F402
from . import post  # noqa: F401, F402
from . import misc  # noqa: F401, F402
from . import noise  # noqa: F401, F402

import numpy as np

circular_mask = misc.circular_mask

# Functions to compute the variance of signals
compute_variance_poisson = noise.compute_variance_poisson
compute_variance_transmission = noise.compute_variance_transmission

# Function to compute the weights associated to the given variance
compute_variance_weight = noise.compute_variance_weight
