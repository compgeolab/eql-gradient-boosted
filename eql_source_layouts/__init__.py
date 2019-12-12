# pylint: disable=missing-docstring
# Import functions/classes to make the public API
from ._version import get_versions
from .layouts import source_bellow_data, block_reduced_sources, grid_sources
from .synthetic_model import synthetic_model
from .plot import plot_prediction
from .utils import (
    get_best_prediction,
    grid_data,
    parameters_scores_to_df,
    grid_to_dataarray,
    predictions_to_datasets,
)

# Get the version number through versioneer
__version__ = get_versions()["version"]
del get_versions
