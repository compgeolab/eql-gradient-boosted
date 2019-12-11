import numpy as np
import pandas as pd
import xarray as xr
import harmonica as hm
from sklearn.metrics import r2_score


def parameters_scores_to_df(parameters_set, scores):
    """
    Convert scores and parameters into a pandas.DataFrame

    Parameters
    ----------
    parameters_set : list
    scores : list
    """
    df = {}
    for keys in parameters_set[0]:
        df[keys] = []
    df["score"] = []
    for parameters, score in zip(parameters_set, scores):
        for key, param in parameters.items():
            df[key].append(param)
        df["score"].append(score)
    df = pd.DataFrame(df)
    return df


def grid_to_dataarray(prediction, grid, **kwargs):
    """
    Convert a 2D grid to a xarray.DataArray
    """
    dims = ("northing", "easting")
    coords = {"northing": grid[1][:, 0], "easting": grid[0][0, :]}
    da = xr.DataArray(prediction, dims=dims, coords=coords, **kwargs)
    return da


def grid_data(coordinates, data, grid, source_builder, depth_type, parameters):
    """
    Interpolate data on a regular grid using EQL

    Parameters
    ----------
    coordinates
    data
    grid
    source_builder
    depth_type
    parameters
    """
    # Create the source points
    points = source_builder(coordinates, depth_type=depth_type, **parameters)
    # Initialize the gridder passing the points and the damping
    eql = hm.EQLHarmonic(points=points, damping=parameters["damping"])
    # Fit the gridder giving the survey data
    eql.fit(coordinates, data)
    # Predict the field on the regular grid
    prediction = eql.predict(grid)
    return prediction


def get_best_prediction(
    coordinates, data, grid, target, source_builder, depth_type, parameters_set
):
    """
    Score interpolations with different parameters and get the best prediction

    Performs several predictions using the same source layout (but with different
    parameters) and score each of them agains the target grid.
    """
    scores = []
    for parameters in parameters_set:
        prediction = grid_data(
            coordinates, data, grid, source_builder, depth_type, parameters
        )
        # Score the prediction against target data
        scores.append(r2_score(target, prediction))
    # Get best prediction
    best = np.nanargmax(scores)
    best_parameters = parameters_set[best]
    best_score = scores[best]
    best_prediction = grid_data(
        coordinates, data, grid, source_builder, depth_type, best_parameters
    )
    # Convert parameters and scores to a pandas.DataFrame
    params_and_scores = parameters_scores_to_df(parameters_set, scores)
    # Convert prediction to a xarray.DataArray
    da = target.copy()
    da.values = best_prediction
    da.attrs["score"] = best_score
    for key, value in best_parameters.items():
        da.attrs[key] = value
    best_prediction = da
    return best_prediction, params_and_scores
