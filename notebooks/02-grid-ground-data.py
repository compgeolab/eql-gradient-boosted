# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
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
    latex_parameters,
    latex_variables,
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
layouts = ["source_below_data", "block_median_sources", "grid_sources"]
# Define dampings used on every fitting of the gridder
dampings = [None, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
# Define depht values
depths = np.arange(1e3, 18e3, 2e3)
# Define parameters for the grid sources:
#    spacing, depth, padding and damping
grid_sources_spacings = [1e3, 2e3, 3e3, 4e3]
grid_sources_depths = np.arange(3e3, 15e3, 2e3)
grid_sources_paddings = [0, 0.1]
grid_sources_dampings = [1e1, 1e2, 1e3, 1e4]
# Define parameters for variable relative depth layouts:
#    depth_factor, depth and k_nearest
depth_factors = [0.1, 0.5, 1, 2, 3, 4, 5, 6]
variable_depths = np.arange(0, 1500, 200)
k_values = [1, 5, 10, 15]
# Define block spacing for block median sources
block_spacings = [1_000, 2_000, 3_000, 4_000]
# -

# ## Create dictionary with the parameter values for each source distribution

# +
parameters = {layout: {} for layout in layouts}

# Source below data
layout = "source_below_data"
depth_type = "constant_depth"
parameters[layout][depth_type] = dict(
    depth_type=depth_type, damping=dampings, depth=depths.tolist()
)

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

# Block-median sources
layout = "block_median_sources"
depth_type = "constant_depth"
parameters[layout][depth_type] = dict(
    depth_type=depth_type, damping=dampings, depth=depths, spacing=block_spacings,
)

depth_type = "relative_depth"
parameters[layout][depth_type] = dict(
    depth_type=depth_type, damping=dampings, depth=depths, spacing=block_spacings,
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
    pad=grid_sources_paddings,
    spacing=grid_sources_spacings,
)
# -

# ### Save parameters to LaTeX variables file

with open(os.path.join("..", "manuscript", "parameters_ground_survey.tex"), "w") as f:
    f.write("\n".join(latex_parameters(parameters, "ground")))

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

# +
scores = {layout: {} for layout in layouts}
best_predictions = []

for layout in parameters_combined:
    for depth_type in parameters_combined[layout]:
        print("Processing: {} with {}".format(layout, depth_type))
        best_prediction, params_and_scores = get_best_prediction(
            coordinates,
            getattr(survey, field),
            grid,
            target,
            layout,
            parameters_combined[layout][depth_type],
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

# ### Read best predictions from saved files

best_predictions = []
for layout in layouts:
    best_predictions.append(
        xr.open_dataset(
            os.path.join(ground_results_dir, "best_predictions-{}.nc".format(layout))
        )
    )

# ## Plot best predictions

for dataset in best_predictions:
    for depth_type in dataset:
        layout = dataset.layout
        prediction = dataset[depth_type]
        print("{} with {}".format(layout, depth_type))
        print("Score: {}".format(prediction.score))
        print("Number of sources: {}".format(prediction.n_points))
        print("Parameters: {}".format(prediction.attrs))
        plot_prediction(prediction, target, units=field_units)


# ## Plot and compare all best predictions

# +
# Load matplotlib configuration
plt.style.use(os.path.join("..", "matplotlib.rc"))

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
fig, axes = plt.subplots(
    nrows=3, ncols=3, figsize=(6.66, 6.66), sharex=True, sharey=True
)

# Plot the differences between the target and the best prediction for each layout
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
            rasterized=True,
        )
        ax.scatter(survey.easting, survey.northing, s=0.3, alpha=0.2, color="k")
        ax.set_aspect("equal")
        # Set scientific notation on axis labels (and change offset text position)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        ax.yaxis.offsetText.set_x(-0.13)
        ax.set_xlabel(ax.get_xlabel() + " [m]")
        ax.set_ylabel(ax.get_ylabel() + " [m]")
        # Set title with r2 score and number of points
        ax.set_title(
            r"R$^2$: {:.3f}, \#sources: {}".format(
                prediction.score, prediction.n_points
            ),
            fontsize="small",
            horizontalalignment="center",
        )

        # Annotate the columns of the figure
        if i == 0:
            ax.text(
                0.5,
                1.16,
                r"\textbf{{" + depth_type.replace("_", " ").title() + r"}}",
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
                r"\textbf{{" + dataset.layout.replace("_", " ").title() + r"}}",
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

plt.tight_layout()
plt.savefig(
    os.path.join("..", "manuscript", "figs", "ground_survey_differences.pdf"), dpi=300
)
plt.show()
# -


# ## Save best predictions parameters
#
# Save best predictions parameters as LaTeX variables

# +
tex_variables = []
for dataset in best_predictions:
    for depth_type in dataset:
        parameters = dataset[depth_type].attrs
        layout = parameters["layout"]
        for parameter, value in parameters.items():
            if parameter in ("depth_type", "layout", "height"):
                continue
            fmt = ""
            variable_name = (
                "_".join(["best", "ground", layout, depth_type, parameter])
                .replace("_", " ")
                .title()
                .replace(" ", "")
            )
            if parameter in ("spacing", "depth"):
                fmt = ".1f"
            if parameter == "score":
                fmt = ".3f"
            if value == 0:
                fmt = ".0f"
            tex_variables.append(latex_variables(variable_name, value, fmt=fmt))


with open(
    os.path.join("..", "manuscript", "best_parameters_ground_survey.tex"), "w"
) as f:
    f.write("\n".join(tex_variables,))
