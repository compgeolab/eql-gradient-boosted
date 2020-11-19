import datetime
import json
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import harmonica as hm
from sklearn.metrics import mean_squared_error

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


def save_to_json(dictionary, json_file):
    """
    Save dictionary to json file
    """
    with open(json_file, "w") as f:
        json.dump(dictionary, f, default=_convert_numpy_types)


def _convert_numpy_types(obj):
    """
    Convert numpy variables to Python variables

    This prevents invalid types errors being raised by json
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()


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
    grid = eql.grid(upward=height, region=region, shape=shape)
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
    parameters) and score each of them against the target grid through RMS.
    """
    # Get shape, region and height of the target grid
    region = vd.get_region((target.easting.values, target.northing.values))
    shape = target.shape
    height = target.height
    # Score the predictions (with RMS)
    rms = []
    for parameters in parameters_set:
        prediction, _ = grid_data(
            coordinates, data, region, shape, height, layout, parameters
        )
        # Score the prediction against target data
        rms.append(np.sqrt(mean_squared_error(target.values, prediction.values)))
    # Get best prediction
    best = np.nanargmin(rms)
    best_parameters = parameters_set[best]
    best_rms = rms[best]
    best_prediction, points = grid_data(
        coordinates, data, region, shape, height, layout, best_parameters
    )
    # Convert parameters and RMS to a pandas.DataFrame
    parameters_and_rms = pd.DataFrame.from_dict(parameters_set)
    parameters_and_rms["rms"] = rms
    # Add RMS and number of sources to the grid attributes
    best_prediction.attrs["rms"] = best_rms
    best_prediction.attrs["n_points"] = points[0].size
    return best_prediction, parameters_and_rms


def predictions_to_datasets(predictions):
    """
    Group all predictions in xarray.Datasets by source layout

    Create one xarray.Dataset for each source layout, containing the best predictions
    for each depth type.
    """
    datasets = []
    layouts = []
    for prediction in predictions:
        if prediction.layout not in layouts:
            layouts.append(prediction.layout)
    for layout in layouts:
        predictions_same_layout = [p for p in predictions if p.layout == layout]
        for p in predictions_same_layout:
            p.name = p.depth_type
        ds = xr.merge(predictions_same_layout)
        ds.attrs["layout"] = layout
        datasets.append(ds)
    return datasets
