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

# # Gridding a synthetic ground survey
#
# Let's perform a synthetic ground survey of the synthetic model made out of prisms and
# try to interpolate the observed data on a regular grid. Because the model is
# synthetic, we can compute the true value of the field on this regular grid. Therefore,
# we have an objective way to score the interpolation. This allow us to objectively
# compare the different source layouts.
#
# Firstly, we want to import useful packages and define some minor functions that will
# help us to visualize better the notebook cells and reduce ammount of code on them.

# **Import useful packages**

# +
from IPython.display import display
import os
import itertools
import pyproj
import numpy as np
import xarray as xr
import verde as vd
import harmonica as hm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from eql_source_layouts import (
    synthetic_model,
    source_bellow_data,
    block_reduced_sources,
    grid_sources,
    grid_to_dataarray,
    plot_prediction,
    get_best_prediction,
)

# -

# ## 1. Define parameters used on the gridding
#
# Let's define all the parameters that will be used on this notebook in the following
# cells. These control the results that will be obtain on the rest of the notebook. If
# you want to change something (like the computation height of the grid, the survey
# region, interpolation parameters, etc), you can just do it here.
#
# We will avoid hardcoding parameters outside of these few cells in order to facilitate
# reproducibility and keep things more organized.

# +
# Define results directory
results_dir = os.path.join("..", "results", "ground_survey")

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
# -

# ## 2. Create the synthetic ground survey
#
# To create the ground survey we need to load the synthetic model made out of prisms,
# the location of the observation points and then compute the field that the prisms
# generate on those points.

# Get coordinates of observation points

survey = hm.synthetic.ground_survey(region=region_degrees)
display(survey)

# Project survey points into Cartesian coordinates

# +
projection = pyproj.Proj(proj="merc", lat_ts=0)
survey["easting"], survey["northing"] = projection(
    survey.longitude.values, survey.latitude.values
)
display(survey)

# Define region boundaries in projected coordinates
region = (
    survey.easting.values.min(),
    survey.easting.values.max(),
    survey.northing.min(),
    survey.northing.max(),
)
# -

# Plot the survey points

fig, ax = plt.subplots(figsize=(6, 6))
tmp = ax.scatter(survey.easting, survey.northing, c=survey.height, s=6)
plt.colorbar(tmp, ax=ax, label="m")
ax.set_aspect("equal")
ax.set_title("Height of ground survey points")
plt.show()

# Get the synthetic model

# +
# Define model region
model_region = tuple(list(region) + [model_bottom, model_top])

# Create synthetic model
model = synthetic_model(model_region)
print(model.keys())
print("Number of prisms: {}".format(len(model["densities"])))
# -

fig, ax = plt.subplots(figsize=(6, 6))
ax.add_collection(PatchCollection(model["rectangles"], match_original=True))
ax.set_aspect("equal")
ax.set_title("Synthetic model made out of prisms")
ax.set_xlim(region[:2])
ax.set_ylim(region[2:4])
plt.show()

# Compute the gravity field (g_z) on the observation points in mGal and add Gaussian
# noise

coordinates = (survey.easting, survey.northing, survey.height)
survey[field] = hm.prism_gravity(
    coordinates, model["prisms"], model["densities"], field=field
) + np.random.normal(scale=noise_std, size=survey.easting.size)
display(survey)

# Plot the observed field

fig, ax = plt.subplots(figsize=(6, 6))
tmp = ax.scatter(survey.easting, survey.northing, c=getattr(survey, field), s=6)
plt.colorbar(tmp, ax=ax, label=field_units)
ax.set_aspect("equal")
ax.set_title("Synthetic ground survey")
plt.show()

# ## 3. Compute the field of the synthetic model on a grid
#
# Now, let's compute the gravity field that the synthetic model generates on the regular
# grid. These results will serve as a target for the interpolations using different
# source layouts.

# Build the regular grid

grid = vd.grid_coordinates(
    region=region, spacing=grid_spacing, extra_coords=grid_height
)

# Compute the synthetic gravity field on the grid

target = hm.prism_gravity(grid, model["prisms"], model["densities"], field=field)
target = grid_to_dataarray(target, grid, attrs={"height": grid_height})

# Plot target gravity field

fig, ax = plt.subplots(figsize=(6, 6))
tmp = target.plot.pcolormesh(center=False, add_colorbar=False)
plt.colorbar(tmp, ax=ax, shrink=0.7, label=field_units)
ax.set_aspect("equal")
ax.set_title("Target grid values")
plt.show()

# ## 4. Grid data using different source layouts
#
# Let's grid the synthetic data using the Equivalent Layer method using different source
# layouts. For each layout we will perform several interpolations, one for each set of
# parameters, score each prediction against the target data and get the best one.

# We will finally compare the performance of each source layout based on the best
# prediction produce by each of them.

# ### Define set of parameters for each source layout

# +
parameters = {"constant_depth": {}, "relative_depth": {}, "variable_relative_depth": {}}

# Constant depth
depth_type = "constant_depth"
layout = "source_bellow_data"
parameters[depth_type][layout] = [
    dict(damping=combo[0], constant_depth=combo[1])
    for combo in itertools.product(dampings, constant_depths)
]

layout = "block_reduced_sources"
parameters[depth_type][layout] = [
    dict(damping=combo[0], constant_depth=combo[1], spacing=block_spacing)
    for combo in itertools.product(dampings, constant_depths)
]

layout = "grid_sources"
parameters[depth_type][layout] = [
    dict(damping=combo[0], constant_depth=combo[1], pad=combo[2], spacing=combo[3])
    for combo in itertools.product(
        dampings, source_grid_depths, source_grid_paddings, source_grid_spacings
    )
]

# Relative depth
depth_type = "relative_depth"
layout = "source_bellow_data"
parameters[depth_type][layout] = [
    dict(damping=combo[0], relative_depth=combo[1])
    for combo in itertools.product(dampings, relative_depths)
]

layout = "block_reduced_sources"
parameters[depth_type][layout] = [
    dict(damping=combo[0], relative_depth=combo[1], spacing=block_spacing)
    for combo in itertools.product(dampings, relative_depths)
]

# Variable relative depth
depth_type = "variable_relative_depth"
layout = "source_bellow_data"
parameters[depth_type][layout] = [
    dict(
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
source_builders = {
    "source_bellow_data": source_bellow_data,
    "block_reduced_sources": block_reduced_sources,
    "grid_sources": grid_sources,
}

for depth_type in parameters:
    for layout in parameters[depth_type]:
        print("Processing: {} with {}".format(layout, depth_type))
        best_prediction, params_and_scores = get_best_prediction(
            coordinates,
            getattr(survey, field),
            grid,
            target,
            source_builders[layout],
            depth_type,
            parameters[depth_type][layout],
        )
        best_predictions[depth_type][layout] = best_prediction
        scores[depth_type][layout] = params_and_scores
# -

# ### Save best predictions and scores

for depth_type in best_predictions:
    for layout in best_predictions[depth_type]:
        prediction = best_predictions[depth_type][layout]
        score = scores[depth_type][layout]
        prediction.to_netcdf(
            os.path.join(
                results_dir, "best_prediction-{}-{}.nc".format(depth_type, layout)
            )
        )
        score.to_csv(
            os.path.join(results_dir, "scores-{}-{}.nc".format(depth_type, layout)),
            index=False,
        )

# ## 5. Plot best predictions

# Read best predictions from saved files

# +
best_predictions = {
    "constant_depth": {
        "source_bellow_data": None,
        "block_reduced_sources": None,
        "grid_sources": None,
    },
    "relative_depth": {"source_bellow_data": None, "block_reduced_sources": None},
    "variable_relative_depth": {
        "source_bellow_data": None,
        "block_reduced_sources": None,
    },
}

predictions_fnames = [
    os.path.join(results_dir, f)
    for f in os.listdir(results_dir)
    if "best_prediction" in f
]
for fname in predictions_fnames:
    _, depth_type, layout = os.path.basename(fname).replace(".nc", "").split("-")
    best_predictions[depth_type][layout] = xr.open_dataarray(fname)
# -

# Plot best predictions

for depth_type in best_predictions:
    for layout in best_predictions[depth_type]:
        prediction = best_predictions[depth_type][layout]
        print("{} with {}".format(layout, depth_type))
        print("Score: {}".format(prediction.score))
        print("Parameters: {}".format(prediction.attrs))
        plot_prediction(prediction, target, units=field_units)


# ## 6. Plot and compare all best predictions

# +
# Get the boundary values of the colorbar as the 99 percentile of
# the difference between target an all predictions.
# We will use the same boundary value for each plot in order to
# show them with the same color scale.
vmax = np.percentile(
    tuple(target - i for subset in best_predictions.values() for i in subset.values()),
    q=99.9,
)

# Initialize figure
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey=True)

# Plot the differences between the target and the best prediction for each layout
axes = axes.T
for ax_col, depth_type in zip(axes, best_predictions):
    for ax, layout in zip(ax_col, best_predictions[depth_type]):
        prediction = best_predictions[depth_type][layout]
        difference = target - prediction
        tmp = difference.plot.pcolormesh(
            ax=ax, vmin=-vmax, vmax=vmax, cmap="seismic", add_colorbar=False
        )
        ax.scatter(survey.easting, survey.northing, s=1, alpha=0.2, color="k")
        ax.set_aspect("equal")
        ax.set_title("{} (r2: {:.3f})".format(layout, r2_score(target, prediction)))

# Hide the last two axes because they are not used
axes[-1][-1].set_visible(False)
axes[-2][-1].set_visible(False)

# Annotate the columns of the figure
for i, depth_type in enumerate(list(best_predictions.keys())):
    axes[i][0].text(
        0.5,
        1.1,
        depth_type,
        fontsize="large",
        fontweight="bold",
        horizontalalignment="center",
        transform=axes[i][0].transAxes,
    )

# Add colorbar
fig.subplots_adjust(bottom=0.1, wspace=0.05)
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
fig.colorbar(tmp, cax=cbar_ax, orientation="horizontal", label=field_units)

plt.show()
