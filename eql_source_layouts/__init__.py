# pylint: disable=missing-docstring
# Import functions/classes to make the public API
from ._version import get_versions
from .layouts import block_reduced_points
from .synthetic_model import airborne_survey, synthetic_model

# Get the version number through versioneer
__version__ = get_versions()["version"]
del get_versions
