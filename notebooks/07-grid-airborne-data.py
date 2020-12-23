# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python [conda env:eql-gradient-boosted]
#     language: python
#     name: conda-env-eql-gradient-boosted-py
# ---

# # Grid airborne survey data using different source distributions

# **Import useful packages**

# +
from IPython.display import display
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import matplotlib.pyplot as plt

from boost_and_layouts import (
    combine_parameters,
    save_to_json,
    plot_prediction,
    get_best_prediction,
    predictions_to_datasets,
)

# -

# ## Define parameters used on the gridding
#
# Let's define all the parameters that will be used on this notebook in the following
# cells. These control the results that will be obtain on the rest of the notebook. If
# you want to change something (like the interpolation parameters), you can just do it here.
#
# We will avoid hardcoding parameters outside of these few cells in order to facilitate
# reproducibility and keep things more organized.

# +
# Define results directory
results_dir = Path("..") / "results"
airborne_results_dir = results_dir / "airborne_survey"

# Define which field will be meassured
field = "g_z"
field_units = "mGal"

# Define set of interpolation parameters
# ======================================
# Define a list of source layouts
layouts = ["source_below_data", "block_averaged_sources", "grid_sources"]
# Define dampings used on every fitting of the gridder
dampings = np.logspace(-4, 2, 7)
# Define depht values
depths = np.arange(1e3, 18e3, 2e3)
# Define parameters for the grid sources:
#    spacing, depth and damping
grid_sources_spacings = [1e3, 2e3, 3e3]
grid_sources_depths = np.arange(1e3, 11e3, 2e3)
grid_sources_dampings = np.logspace(1, 4, 4)
# Define parameters for variable relative depth layouts:
#    depth factor, depth shift and k_values
depth_factors = [1, 2, 3, 4, 5, 6]
variable_depths = np.arange(50, 1500, 200)
k_values = [1, 5, 10, 15]
# Define block spacing for block averaged sources
block_spacings = np.arange(1e3, 5e3, 1e3)
# -

# ## Create dictionary with the parameter values for each source distribution

# +
parameters = {layout: {} for layout in layouts}

# Source below data
layout = "source_below_data"
depth_type = "constant_depth"
parameters[layout][depth_type] = dict(
    depth_type=depth_type, damping=dampings, depth=depths
)

depth_type = "relative_depth"
parameters[layout][depth_type] = dict(
    depth_type=depth_type, damping=dampings, depth=depths
)

depth_type = "variable_depth"
parameters[layout][depth_type] = dict(
    depth_type=depth_type,
    damping=dampings,
    depth_factor=depth_factors,
    depth=variable_depths,
    k_nearest=k_values,
)

# Block-averaged sources
layout = "block_averaged_sources"
depth_type = "constant_depth"
parameters[layout][depth_type] = dict(
    depth_type=depth_type,
    damping=dampings,
    depth=depths,
    spacing=block_spacings,
)

depth_type = "relative_depth"
parameters[layout][depth_type] = dict(
    depth_type=depth_type,
    damping=dampings,
    depth=depths,
    spacing=block_spacings,
)

depth_type = "variable_depth"
parameters[layout][depth_type] = dict(
    depth_type=depth_type,
    damping=dampings,
    spacing=block_spacings,
    depth_factor=depth_factors,
    depth=variable_depths,
    k_nearest=k_values,
)

# Grid sources
depth_type = "constant_depth"
layout = "grid_sources"
parameters[layout][depth_type] = dict(
    depth_type=depth_type,
    damping=grid_sources_dampings,
    depth=grid_sources_depths,
    spacing=grid_sources_spacings,
)
# -

# ### Dump parameters to a JSON file

json_file = results_dir / "parameters-airborne-survey.json"
save_to_json(parameters, json_file)

# ### Combine parameter values for each source distribution
#
# Lets create combinations of parameter values for each source distribution

# +
parameters_combined = {layout: {} for layout in layouts}

for layout in parameters:
    for depth_type in parameters[layout]:
        parameters_combined[layout][depth_type] = combine_parameters(
            **parameters[layout][depth_type]
        )
# -

# ## Read synthetic airborne survey and target grid

# Read airborne survey

survey = pd.read_csv(airborne_results_dir / "survey.csv")
survey

# Read target grid

target = xr.open_dataarray(results_dir / "target.nc")
target

# Define coordiantes tuple with the location of the survey points

coordinates = (survey.easting.values, survey.northing.values, survey.height.values)

# ## Grid data using different source layouts
#
# Let's grid the synthetic data using the Equivalent Layer method using different source
# layouts. For each layout we will perform several interpolations, one for each set of
# parameters, score each prediction against the target data and get the best one.

# We will finally compare the performance of each source layout based on the best
# prediction produce by each of them.

# ## Score each interpolation

# +
rms = {layout: {} for layout in layouts}
best_predictions = []

for layout in layouts:
    for depth_type in parameters_combined[layout]:
        with warnings.catch_warnings():
            # Disable warnings
            # (we expect some warnings due to
            # underdetermined problems on grid layouts)
            warnings.simplefilter("ignore")
            print("Processing: {} with {}".format(layout, depth_type))
            best_prediction, params_and_rms = get_best_prediction(
                coordinates,
                getattr(survey, field).values,
                target,
                layout,
                parameters_combined[layout][depth_type],
            )
            best_predictions.append(best_prediction)
            rms[layout][depth_type] = params_and_rms

# Group best predictions into datasets
best_predictions = predictions_to_datasets(best_predictions)
# -

for prediction in best_predictions:
    display(prediction)

# ### Save best predictions

for dataset in best_predictions:
    dataset.to_netcdf(
        airborne_results_dir / "best_predictions-{}.nc".format(dataset.layout)
    )


# ## Plot best predictions

for dataset in best_predictions:
    for depth_type in dataset:
        layout = dataset.layout
        prediction = dataset[depth_type]
        print("{} with {}".format(layout, depth_type))
        print("RMS: {}".format(prediction.rms))
        print("Number of sources: {}".format(prediction.n_points))
        print("Parameters: {}".format(prediction.attrs))
        plot_prediction(prediction, target, units=field_units)


# ## Plot and compare all best predictions

# +
vmax = vd.maxabs(
    *list(
        target - dataset[depth_type]
        for dataset in best_predictions
        for depth_type in dataset
    )
)

# Initialize figure
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), sharex=True, sharey=True)

for i, (ax_row, dataset) in enumerate(zip(axes, best_predictions)):
    for j, (ax, depth_type) in enumerate(zip(ax_row, dataset)):
        prediction = dataset[depth_type]
        difference = target - prediction
        tmp = difference.plot.pcolormesh(
            ax=ax,
            vmin=-vmax,
            vmax=vmax,
            cmap="seismic",
            add_colorbar=False,
        )
        ax.scatter(survey.easting, survey.northing, s=0.3, alpha=0.2, color="k")
        ax.set_aspect("equal")
        # Set scientific notation on axis labels (and change offset text position)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        # Set title with RMS and number of points
        ax.set_title(
            "RMS: {:.3f}, #sources: {}".format(prediction.rms, prediction.n_points),
            horizontalalignment="center",
        )

        # Annotate the columns of the figure
        if i == 0:
            ax.text(
                0.5,
                1.16,
                depth_type.replace("_", " ").title(),
                fontsize="large",
                fontweight="bold",
                horizontalalignment="center",
                transform=ax.transAxes,
            )
        # Annotate the rows of the figure
        if j == 0:
            ax.text(
                -0.38,
                0.5,
                dataset.layout.replace("_", " ").title(),
                fontsize="large",
                fontweight="bold",
                verticalalignment="center",
                rotation="vertical",
                transform=ax.transAxes,
            )
        # Remove xlabels and ylabels from inner axes
        if i != 2:
            ax.set_xlabel("")
        if j != 0:
            ax.set_ylabel("")

# Hide the last two axes because they are not used
axes[-1][-1].set_visible(False)
axes[-1][-2].set_visible(False)

# Add colorbar
cbar_ax = fig.add_axes([0.38, 0.075, 0.015, 0.24])
fig.colorbar(tmp, cax=cbar_ax, orientation="vertical", label=field_units)

plt.show()
