# -*- coding: utf-8 -*-

"""Top-level package for PyCorrectedEmissionCT."""

__author__ = """Nicola VIGANÒ"""
__email__ = "N.R.Vigano@cwi.nl"


def __get_version():
    import os.path

    version_filename = os.path.join(os.path.dirname(__file__), "VERSION")
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

from . import models  # noqa: F401, F402

from . import operators  # noqa: F401, F402
from . import projectors  # noqa: F401, F402
from . import attenuation  # noqa: F401, F402

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

# Import all definitions from main module.
from ._corrct import create_sino, reconstruct  # noqa: F401, F402, F403
