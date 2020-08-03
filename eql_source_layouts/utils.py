import itertools
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import harmonica as hm
from sklearn.metrics import r2_score

from . import layouts


def combine_parameters(**kwargs):
    """
    Create a combination of given parameters using itertools
    """
    values = [np.atleast_1d(v) for v in kwargs.values()]
    parameters = [
        {key: combo[i] for i, key in enumerate(kwargs.keys())}
        for combo in itertools.product(*values)
    ]
    return parameters


def grid_data(coordinates, data, region, shape, height, layout, parameters):
    """
    Interpolate data on a regular grid using EQL

    Parameters
    ----------
    coordinates
    data
    grid
    layout
    depth_type
    parameters
    """
    # Get function to build point sources
    source_builder = getattr(layouts, layout)
    # Create the source points
    points = source_builder(coordinates, **parameters)
    # Initialize the gridder passing the points and the damping
    eql = hm.EQLHarmonic(points=points, damping=parameters["damping"])
    # Fit the gridder giving the survey data
    eql.fit(coordinates, data)
    # Predict the field on the regular grid
    eql.extra_coords_name = "upward"
    grid = eql.grid(region=region, shape=shape, extra_coords=height)
    # Transform the xr.Dataset to xr.DataArray
    grid = grid.scalars
    # Append parameters to the attrs
    grid.attrs.update(parameters)
    grid.attrs["layout"] = layout
    return grid, points


def get_best_prediction(coordinates, data, target, layout, parameters_set):
    """
    Score interpolations with different parameters and get the best prediction

    Performs several predictions using the same source layout (but with different
    parameters) and score each of them agains the target grid.
    """
    # Get shape, region and height of the target grid
    region = vd.get_region((target.easting.values, target.northing.values))
    shape = target.shape
    height = target.height
    # Score the predictions
    scores = []
    for parameters in parameters_set:
        prediction, _ = grid_data(
            coordinates, data, region, shape, height, layout, parameters
        )
        # Score the prediction against target data
        scores.append(r2_score(target.values, prediction.values))
    # Get best prediction
    best = np.nanargmax(scores)
    best_parameters = parameters_set[best]
    best_score = scores[best]
    best_prediction, points = grid_data(
        coordinates, data, region, shape, height, layout, best_parameters
    )
    # Convert parameters and scores to a pandas.DataFrame
    parameters_and_scores = pd.DataFrame.from_dict(parameters_set)
    parameters_and_scores["score"] = scores
    # Add score and number of sources to the grid attributes
    best_prediction.attrs["score"] = best_score
    best_prediction.attrs["n_points"] = points[0].size
    return best_prediction, parameters_and_scores


def predictions_to_datasets(predictions):
    """
    Group all predictions in xarray.Datasets by source layout

    Create one xarray.Dataset for each source layout, containing the best predictions
    for each depth type.
    """
    datasets = []
    layouts_ = list(set([prediction.layout for prediction in predictions]))
    for layout in layouts_:
        predictions_same_layout = [p for p in predictions if p.layout == layout]
        for p in predictions_same_layout:
            p.name = p.depth_type
        ds = xr.merge(predictions_same_layout)
        ds.attrs["layout"] = layout
        datasets.append(ds)
    return datasets
