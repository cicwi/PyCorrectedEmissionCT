# -*- coding: utf-8 -*-

"""Top-level package for PyCorrectedEmissionCT."""

from importlib import metadata

from . import models  # noqa: F401, F402

from . import operators  # noqa: F401, F402
from . import projectors  # noqa: F401, F402

from . import filters  # noqa: F401, F402
from . import data_terms  # noqa: F401, F402
from . import regularizers  # noqa: F401, F402
from . import solvers  # noqa: F401, F402

from . import param_tuning  # noqa: F401, F402
from . import processing  # noqa: F401, F402
from . import denoisers  # noqa: F401, F402

from . import struct_illum  # noqa: F401, F402
from . import alignment  # noqa: F401, F402

from . import physics  # noqa: F401, F402

from . import testing  # noqa: F401, F402


def get_version(dist: str = "corrct") -> str:
    """Get version of the given distribution.

    Parameters:
        dist: A distribution name.

    Returns:
        A version number.
    """
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "0.0.0"


__author__ = """Nicola VIGANÒ"""
__email__ = "N.R.Vigano@cwi.nl"
__version__ = get_version()
