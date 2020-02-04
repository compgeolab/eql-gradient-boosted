# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python [conda env:eql_source_layouts]
#     language: python
#     name: conda-env-eql_source_layouts-py
# ---

# # Grid ground survey data using different source distributions

# **Import useful packages**

# +
import os
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import matplotlib.pyplot as plt

from eql_source_layouts import (
    combine_parameters,
    plot_prediction,
    get_best_prediction,
    predictions_to_datasets,
)

# -

# ## Define parameters used on the gridding
#
# Let's define all the parameters that will be used on this notebook in the following
# cells. These control the results that will be obtain on the rest of the notebook. If
# you want to change something (like the interpolation parameters), you can just do it
# here.
#
# We will avoid hardcoding parameters outside of these few cells in order to facilitate
# reproducibility and keep things more organized.

# +
# Define results directory
results_dir = os.path.join("..", "results")
ground_results_dir = os.path.join(results_dir, "ground_survey")

# Define which field will be meassured
field = "g_z"
field_units = "mGal"

# Define set of interpolation parameters
# ======================================
# Define a list of source layouts
layouts = ["source_bellow_data", "block_median_sources", "grid_sources"]
# Define dampings used on every fitting of the gridder
dampings = [None, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
# Define values of constant depth
constant_depths = [1e3, 2e3, 5e3, 10e3, 15e3]
# Define values of relative depth
relative_depths = [1e3, 2e3, 5e3, 10e3, 15e3]
# Define parameters for the grid layout:
#    spacing, depth and padding
source_grid_spacings = [1e3, 2e3, 3e3, 4e3]
source_grid_depths = [100, 500, 1e3, 2e3, 5e3, 7e3, 10e3]
source_grid_paddings = [0, 0.1, 0.2]
# Define parameters for variable relative depth layouts:
#    depth factor, depth shift and k_values
depth_factors = [0.5, 1, 5, 10]
depth_shifts = [0, -100, -500, -1000, -2000, -5000]
k_values = [1, 3, 5, 10]
# We will set the block spacing for the block-median
# layouts equal to the target grid spacing
block_spacing = 2000
# -

# ## Read synthetic ground survey and target grid

# Read ground survey

survey = pd.read_csv(os.path.join(ground_results_dir, "survey.csv"))

survey

# Read target grid

target = xr.open_dataarray(os.path.join(results_dir, "target.nc"))

target

# Define coordinates and grid arrays

# +
coordinates = (survey.easting.values, survey.northing.values, survey.height.values)

grid = np.meshgrid(target.easting, target.northing)
grid.extend(np.full_like(grid[0], target.height))
# -

# ## Grid data using different source layouts
#
# Let's grid the synthetic data using the Equivalent Layer method using different source
# layouts. For each layout we will perform several interpolations, one for each set of
# parameters, score each prediction against the target data and get the best one.

# We will finally compare the performance of each source layout based on the best
# prediction produce by each of them.

# ### Define set of parameters for each source layout

# +
parameters = {layout: {} for layout in layouts}

# Source bellow data
layout = "source_bellow_data"
depth_type = "constant_depth"
parameters[layout][depth_type] = combine_parameters(
    depth_type=depth_type, damping=dampings, constant_depth=constant_depths
)

depth_type = "relative_depth"
parameters[layout][depth_type] = combine_parameters(
    depth_type=depth_type, damping=dampings, relative_depth=relative_depths
)

depth_type = "variable_relative_depth"
parameters[layout][depth_type] = combine_parameters(
    depth_type=depth_type,
    damping=dampings,
    depth_factor=depth_factors,
    depth_shift=depth_shifts,
    k_nearest=k_values,
)

# Block-median sources
layout = "block_median_sources"
depth_type = "constant_depth"
parameters[layout][depth_type] = combine_parameters(
    depth_type=depth_type,
    damping=dampings,
    constant_depth=constant_depths,
    spacing=block_spacing,
)

depth_type = "relative_depth"
parameters[layout][depth_type] = combine_parameters(
    depth_type=depth_type,
    damping=dampings,
    relative_depth=relative_depths,
    spacing=block_spacing,
)

depth_type = "variable_relative_depth"
parameters[layout][depth_type] = combine_parameters(
    depth_type=depth_type,
    damping=dampings,
    spacing=block_spacing,
    depth_factor=depth_factors,
    depth_shift=depth_shifts,
    k_nearest=k_values,
)

# Grid sources
depth_type = "constant_depth"
layout = "grid_sources"
parameters[layout][depth_type] = combine_parameters(
    depth_type=depth_type,
    damping=dampings,
    constant_depth=source_grid_depths,
    pad=source_grid_paddings,
    spacing=source_grid_spacings,
)

# -

# ## Score each interpolation

# + jupyter={"outputs_hidden": true}
scores = {layout: {} for layout in layouts}
best_predictions = []

for layout in parameters:
    for depth_type in parameters[layout]:
        print("Processing: {} with {}".format(layout, depth_type))
        best_prediction, params_and_scores = get_best_prediction(
            coordinates,
            getattr(survey, field),
            grid,
            target,
            layout,
            parameters[layout][depth_type],
        )
        best_predictions.append(best_prediction)
        scores[layout][depth_type] = params_and_scores
# -

# ### Save best predictions and scores

# +
datasets = predictions_to_datasets(best_predictions)
for dataset in datasets:
    dataset.to_netcdf(
        os.path.join(
            ground_results_dir, "best_predictions-{}.nc".format(dataset.layout)
        )
    )

for layout in scores:
    for depth_type in scores[layout]:
        score = scores[layout][depth_type]
        score.to_csv(
            os.path.join(
                ground_results_dir, "scores-{}-{}.nc".format(depth_type, layout)
            ),
            index=False,
        )
# -

# ## 5. Plot best predictions

# Read best predictions from saved files

best_predictions = []
for layout in layouts:
    best_predictions.append(
        xr.open_dataset(
            os.path.join(ground_results_dir, "best_predictions-{}.nc".format(layout))
        )
    )

# Plot best predictions

for dataset in best_predictions:
    for depth_type in dataset:
        layout = dataset.layout
        prediction = dataset[depth_type]
        print("{} with {}".format(layout, depth_type))
        print("Score: {}".format(prediction.score))
        print("Parameters: {}".format(prediction.attrs))
        plot_prediction(prediction, target, units=field_units)


# ## 6. Plot and compare all best predictions

# +
# We will use the same boundary value for each plot in order to
# show them with the same color scale.
vmax = vd.maxabs(
    *list(
        target - dataset[depth_type]
        for dataset in best_predictions
        for depth_type in dataset
    )
)

# Initialize figure
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey=True)

# Plot the differences between the target and the best prediction for each layout
for i, (ax_row, dataset) in enumerate(zip(axes, best_predictions)):
    for ax, depth_type in zip(ax_row, dataset):
        prediction = dataset[depth_type]
        difference = target - prediction
        tmp = difference.plot.pcolormesh(
            ax=ax, vmin=-vmax, vmax=vmax, cmap="seismic", add_colorbar=False
        )
        ax.scatter(survey.easting, survey.northing, s=1, alpha=0.2, color="k")
        ax.set_aspect("equal")
        ax.set_title("{} (r2: {:.3f})".format(dataset.layout, prediction.score))
        # Annotate the columns of the figure
        if i == 0:
            ax.text(
                0.5,
                1.1,
                depth_type,
                fontsize="large",
                fontweight="bold",
                horizontalalignment="center",
                transform=ax.transAxes,
            )

# Hide the last two axes because they are not used
axes[-1][-1].set_visible(False)
axes[-1][-2].set_visible(False)

# Add colorbar
fig.subplots_adjust(bottom=0.1, wspace=0.05)
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
fig.colorbar(tmp, cax=cbar_ax, orientation="horizontal", label=field_units)

plt.show()
