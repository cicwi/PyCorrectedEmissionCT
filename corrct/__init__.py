# -*- coding: utf-8 -*-

"""Top-level package for PyCorrectedEmissionCT."""

__author__ = """Nicola VIGANÒ"""
__email__ = 'N.R.Vigano@cwi.nl'


def __get_version():
    import os.path
    version_filename = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

# Import all definitions from main module.
from .corrct import *  # noqa: F401, F402, F403
from . import projectors  # noqa: F401, F402
from . import solvers  # noqa: F401, F402
from . import utils_proc  # noqa: F401, F402
