# pylint: disable=missing-docstring
# Import functions/classes to make the public API
from ._version import get_versions
from .layouts import block_reduced_points
from .fetch_data import fetch_airborne_gravity

# Get the version number through versioneer
__version__ = get_versions()["version"]
del get_versions
