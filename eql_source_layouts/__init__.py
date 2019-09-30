# pylint: disable=missing-docstring
# Import functions/classes to make the public API
from ._version import get_versions
from .layouts import source_bellow_data, block_reduced_sources, grid_sources
from .synthetic_model import airborne_survey, ground_survey, synthetic_model

# Get the version number through versioneer
__version__ = get_versions()["version"]
del get_versions
