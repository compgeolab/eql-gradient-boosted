# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Import packages

import os
import itertools
import pyproj
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import harmonica as hm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import eql_source_layouts


def scores_to_dataframe(parameters_sets, scores):
    """
    Convert scores and parameters into a pandas.DataFrame
    """
    df = {}
    for keys in parameters_sets[0]:
        df[keys] = []
    df["score"] = []
    for parameters, score in zip(parameters_sets, scores):
        for key, param in parameters.items():
            df[key].append(param)
        df["score"].append(score)
    df = pd.DataFrame(df)
    return df


def prediction_to_dataarray(prediction, grid, score=None, parameters=None):
    """
    Convert a prediction to a xarray.DataArray
    """
    dims = ("northing", "easting")
    coords = {"northing": grid[1][:, 0], "easting": grid[0][0, :]}
    if parameters is not None:
        attrs = parameters.copy()
    else:
        attrs = {}
    if score is not None:
        attrs["score"] = score
    da = xr.DataArray(prediction, dims=dims, coords=coords, attrs=attrs)
    return da


def get_best_prediction(coordinates, grid, target, parameters_set):
    """
    Perform interpolations and get the best one

    Performs several predictions using the same source layout (but with different
    parameters) and score each of them agains the target grid.
    """
    scores = []
    for parameters in parameters_set:
        # Create the source points
        points = getattr(eql_source_layouts, parameters["layout"])(
            coordinates, **parameters
        )
        # Initialize the gridder passing the points and the damping
        eql = hm.EQLHarmonic(points=points, damping=parameters["damping"])
        # Fit the gridder giving the survey data
        eql.fit(coordinates, getattr(survey, field))
        # Predict the field on the regular grid
        prediction = eql.predict(grid)
        # Score the prediction against target data
        scores.append(r2_score(target, prediction))

    # Get best prediction
    best = np.nanargmax(scores)
    best_params = parameters_set[best]
    best_score = scores[best]
    points = getattr(eql_source_layouts, best_params["layout"])(
        coordinates, **best_params
    )

    eql = hm.EQLHarmonic(points=points, damping=best_params["damping"])
    eql.fit(coordinates, getattr(survey, field))
    best_prediction = eql.predict(grid)

    # Convert scores to a pandas.DataFrame
    params_and_scores = scores_to_dataframe(parameters_set, scores)

    # Convert prediction to a xarray.DataArray
    best_prediction = prediction_to_dataarray(
        best_prediction, grid, score=best_score, parameters=best_params
    )
    return best_prediction, params_and_scores


#
# ## Define parameters used on the gridding

# +
# Define location of results directory
results = os.path.join("..", "results", "ground_survey")

# Define a survey region of 1 x 1 degrees (~ 100km x 100km)
region_degrees = (-0.5, 0.5, -0.5, 0.5)

# Define bottom and top of the synthetic model
model_bottom, model_top = -10e3, 0

# Define which field will be meassured
field = "g_z"
field_units = "mGal"

# Define standard deviation for the Gaussian noise that
# will be added to the synthetic survey (in mGal)
noise_std = 1

# Define a seed to reproduce the same results on each run
np.random.seed(12345)

# Define the spacing of the target regular grid
# and its observation height
grid_spacing = 2e3
grid_height = 2000

# Define set of interpolation parameters
# ======================================
# Define dampings used on every fitting of the gridder
dampings = [None, 1e-4, 1e-3, 1e-2]
# Define values of constant depth
constant_depths = [1e3, 2e3, 5e3, 10e3, 15e3]
# Define values of relative depth
relative_depths = [1e3, 2e3, 5e3, 10e3, 15e3]
# Define parameters for the grid layout:
#    spacing, depth and padding
source_grid_spacings = [0.5e3, 1e3, 2e3]
source_grid_depths = [1e3, 2e3, 5e3]
source_grid_paddings = [0, 0.1, 0.2]
# Define parameters for variable relative depth layouts:
#    depth factor, depth shift and k_values
depth_factors = [0.5, 1, 5, 10]
depth_shifts = [-100, -1000, -5000]
k_values = [1, 10]
# We will set the block spacing for the block reduced
# layouts equal to the target grid spacing
block_spacing = grid_spacing

# Finally, define a dict where the best predictions will be
# stored to plot them altogether at the end of the notebook.
best_predictions = {
    "constant_depth": {},
    "relative_depth": {},
    "variable_relative_depth": {},
}
# -

# ## Create the synthetic ground survey

# Get coordinates of observation points

survey = hm.synthetic.ground_survey(region=region_degrees)

# Project survey points into Cartesian coordinates

projection = pyproj.Proj(proj="merc", lat_ts=0)
survey["easting"], survey["northing"] = projection(
    survey.longitude.values, survey.latitude.values
)

# Define region boundaries in projected coordinates

region = (
    survey.easting.values.min(),
    survey.easting.values.max(),
    survey.northing.min(),
    survey.northing.max(),
)

# Plot the survey points

fig, ax = plt.subplots(figsize=(6, 6))
tmp = ax.scatter(survey.easting, survey.northing, c=survey.height, s=6)
plt.colorbar(tmp, ax=ax, label="m")
ax.set_aspect("equal")
ax.set_title("Height of ground survey points")
plt.show()

# Get the synthetic model

# Define model region

model_region = tuple(list(region) + [model_bottom, model_top])

# Create synthetic model

model = eql_source_layouts.synthetic_model(model_region)

# Plot the synthetic model made out of prisms

fig, ax = plt.subplots(figsize=(6, 6))
ax.add_collection(PatchCollection(model["rectangles"], match_original=True))
ax.set_aspect("equal")
ax.set_title("Synthetic model made out of prisms")
ax.set_xlim(region[:2])
ax.set_ylim(region[2:4])
plt.show()

# Compute the gravity field (g_z) on the observation points and add Gaussian noise

coordinates = (survey.easting, survey.northing, survey.height)
survey[field] = hm.prism_gravity(
    coordinates, model["prisms"], model["densities"], field=field
) + np.random.normal(scale=noise_std, size=survey.easting.size)

# Plot the observed field

fig, ax = plt.subplots(figsize=(6, 6))
tmp = ax.scatter(survey.easting, survey.northing, c=getattr(survey, field), s=6)
plt.colorbar(tmp, ax=ax, label=field_units)
ax.set_aspect("equal")
ax.set_title("Synthetic ground survey")
plt.show()


# ## Compute the field of the synthetic model on a grid

# Build the regular grid

grid = vd.grid_coordinates(
    region=region, spacing=grid_spacing, extra_coords=grid_height
)

# Compute the synthetic gravity field on the grid

target = hm.prism_gravity(grid, model["prisms"], model["densities"], field=field)
target = xr.DataArray(
    target,
    dims=("northing", "easting"),
    coords={"northing": grid[1][:, 0], "easting": grid[0][0, :]},
)

# Save target grid

target.to_netcdf("../results/ground_survey/target.nc")

# Plot target gravity field

fig, ax = plt.subplots(figsize=(6, 6))
tmp = target.plot.pcolormesh(ax=ax, center=False, add_colorbar=False)
plt.colorbar(tmp, ax=ax, shrink=0.7, label=field_units)
ax.set_aspect("equal")
ax.set_title("Target grid values")
plt.show()


# ## Grid data using different source layouts

depth_types = ["constant_depth", "relative_depth", "variable_relative_depth"]
layouts = ["source_bellow_data", "block_reduced_sources", "grid_sources"]


# Define set of parameters that will be used on each interpolation

# +
parameters = {"constant_depth": {}, "relative_depth": {}, "variable_relative_depth": {}}

depth_type = "constant_depth"
layout = "source_bellow_data"
parameters[depth_type][layout] = [
    dict(
        depth_type=depth_type, layout=layout, damping=combo[0], constant_depth=combo[1]
    )
    for combo in itertools.product(dampings, constant_depths)
]

layout = "block_reduced_sources"
parameters[depth_type][layout] = [
    dict(
        depth_type=depth_type,
        layout=layout,
        damping=combo[0],
        constant_depth=combo[1],
        spacing=block_spacing,
    )
    for combo in itertools.product(dampings, constant_depths)
]

layout = "grid_sources"
parameters[depth_type][layout] = [
    dict(
        depth_type=depth_type,
        layout=layout,
        damping=combo[0],
        constant_depth=combo[1],
        pad=combo[2],
        spacing=combo[3],
    )
    for combo in itertools.product(
        dampings, source_grid_depths, source_grid_paddings, source_grid_spacings
    )
]

depth_type = "relative_depth"
layout = "source_bellow_data"
parameters[depth_type][layout] = [
    dict(
        depth_type=depth_type, layout=layout, damping=combo[0], relative_depth=combo[1]
    )
    for combo in itertools.product(dampings, relative_depths)
]

layout = "block_reduced_sources"
parameters[depth_type][layout] = [
    dict(
        depth_type=depth_type,
        layout=layout,
        damping=combo[0],
        relative_depth=combo[1],
        spacing=block_spacing,
    )
    for combo in itertools.product(dampings, relative_depths)
]

depth_type = "variable_relative_depth"
layout = "source_bellow_data"
parameters[depth_type][layout] = [
    dict(
        depth_type=depth_type,
        layout=layout,
        damping=combo[0],
        depth_factor=combo[1],
        depth_shift=combo[2],
        k_nearest=combo[3],
    )
    for combo in itertools.product(dampings, depth_factors, depth_shifts, k_values)
]
layout = "block_reduced_sources"
parameters[depth_type][layout] = [
    dict(
        depth_type=depth_type,
        layout=layout,
        damping=combo[0],
        spacing=block_spacing,
        depth_factor=combo[1],
        depth_shift=combo[2],
        k_nearest=combo[3],
    )
    for combo in itertools.product(dampings, depth_factors, depth_shifts, k_values)
]
# -

# ## Score each interpolation


# +
scores = {"constant_depth": {}, "relative_depth": {}, "variable_relative_depth": {}}
best_predictions = {
    "constant_depth": {},
    "relative_depth": {},
    "variable_relative_depth": {},
}

for depth_type in parameters.keys():
    for layout in parameters[depth_type]:
        print(depth_type, layout)
        best_prediction, params_and_scores = get_best_prediction(
            coordinates, grid, target, parameters[depth_type][layout]
        )
        best_predictions[depth_type][layout] = best_prediction
        scores[depth_type][layout] = params_and_scores

        filename = "best_prediction-{}-{}.nc".format(
            best_prediction.depth_type, best_prediction.layout
        )
        best_prediction.to_netcdf(os.path.join(results, filename))

        filename = "scores-{}-{}.nc".format(
            best_prediction.depth_type, best_prediction.layout
        )
        params_and_scores.to_csv(os.path.join(results, filename), index=False)
# -
