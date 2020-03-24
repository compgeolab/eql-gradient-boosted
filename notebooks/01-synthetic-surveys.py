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

# # Create synthetic ground and airborne surveys from a synthetic model

# **Import useful packages**

# +
from IPython.display import display
import os
import pyproj
import numpy as np
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from eql_source_layouts import (
    synthetic_model,
    grid_to_dataarray,
    latex_variables,
)

# -

# **Define parameters**

# +
# Define results directories
results_dir = os.path.join("..", "results")
ground_results_dir = os.path.join(results_dir, "ground_survey")
airborne_results_dir = os.path.join(results_dir, "airborne_survey")

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
target_grid_spacing = 2e3
target_grid_height = 2000
# -

# **Initialize latex_lines list**
#
# On this list we will store all the variables we want to mention on the LaTeX manuscript.

latex_lines = []

# ## Synthetic model made out of prisms

# Project region coordinates to get synthetic model boundaries

# +
projection = pyproj.Proj(proj="merc", lat_ts=0)
easting, northing = projection(region_degrees[:2], region_degrees[2:])

# Define region and model_region
region = (min(easting), max(easting), min(northing), max(northing))
model_region = tuple(list(region) + [model_bottom, model_top])
# -

# Create synthetic model

model = synthetic_model(model_region)

fig, ax = plt.subplots(figsize=(6, 6))
ax.add_collection(PatchCollection(model["rectangles"], match_original=True))
ax.set_aspect("equal")
ax.set_title("Synthetic model made out of prisms")
ax.set_xlim(region[:2])
ax.set_ylim(region[2:4])
plt.show()

# **Save model related quantities to LaTeX file**

# +
assert model_region[5] == 0

latex_lines.extend(
    [
        latex_variables("NPrisms", len(model["densities"])),
        latex_variables(
            "ModelEasting",
            (model_region[1] - model_region[0]) * 1e-3,
            r"\km",
            fmt=".0f",
        ),
        latex_variables(
            "ModelNorthing",
            (model_region[1] - model_region[0]) * 1e-3,
            r"\km",
            fmt=".0f",
        ),
        latex_variables("ModelDepth", abs(model_region[4]) * 1e-3, r"\km", fmt=".0f"),
        latex_variables(
            "ModelMinDensity", np.min(model["densities"]), r"\kg\per\cubic\m", fmt=".0f"
        ),
        latex_variables(
            "ModelMaxDensity", np.max(model["densities"]), r"kg\per\cubic\m", fmt=".0f"
        ),
    ]
)
# -

# ## Synthetic ground survey

# Get coordinates of observation points

survey = hm.synthetic.ground_survey(region=region_degrees)
display(survey)

# Project observation points

survey["easting"], survey["northing"] = projection(
    survey.longitude.values, survey.latitude.values
)
display(survey)

# Plot the survey points

fig, ax = plt.subplots(figsize=(6, 6))
tmp = ax.scatter(survey.easting, survey.northing, c=survey.height, s=6)
plt.colorbar(tmp, ax=ax, label="m")
ax.set_aspect("equal")
ax.set_title("Height of ground survey points")
plt.show()

# Compute gravitational field of synthetic model on ground survey

coordinates = (survey.easting, survey.northing, survey.height)
survey[field] = hm.prism_gravity(
    coordinates, model["prisms"], model["densities"], field=field
) + np.random.normal(scale=noise_std, size=survey.easting.size)
display(survey)

# +
# Load matplotlib configuration
plt.style.use(os.path.join("..", "matplotlib.rc"))

# Define useful parameters
width = 3.33
figsize = (width, width * 1.7)
cbar_shrink = 0.95
cbar_pad = 0.03
cbar_aspect = 30
size = 2
labels = "a b".split()

# Initialize figure and axes
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=figsize)

# Plot survey points
tmp = ax1.scatter(
    survey.easting, survey.northing, c=survey.height, cmap="cividis", s=size
)
clb = plt.colorbar(
    tmp,
    ax=ax1,
    shrink=cbar_shrink,
    orientation="vertical",
    pad=cbar_pad,
    aspect=cbar_aspect,
)
clb.set_label("m", labelpad=-15, y=1.05, rotation=0)

# Plot measured values
tmp = ax2.scatter(survey.easting, survey.northing, c=survey.g_z, cmap="viridis", s=size)
clb = plt.colorbar(
    tmp,
    ax=ax2,
    shrink=cbar_shrink,
    orientation="vertical",
    pad=cbar_pad,
    aspect=cbar_aspect,
)
clb.set_label("mGal", labelpad=-15, y=1.05, rotation=0)


ax2.set_xlabel("easting [m]")
ax1.tick_params(
    axis="x", which="both", bottom=False, top=False, labelbottom=False,
)

for ax, label in zip((ax1, ax2), labels):
    ax.set_aspect("equal")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.set_ylabel("northing [m]")
    ax.yaxis.offsetText.set_x(-0.2)
    ax.annotate(
        label,
        xy=(0.04, 0.94),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )

ax1.set_title("Ground survey points", pad=3)
ax2.set_title("Observed gravity acceleration", pad=3)


plt.tight_layout(h_pad=0.2)
plt.savefig(
    os.path.join("..", "manuscript", "figs", "ground-survey.pdf"),
    bbox_inches="tight",
    dpi=300,
)
plt.show()
# -

# Save ground survey for future usage

survey.to_csv(os.path.join(ground_results_dir, "survey.csv"), index=False)

# Save ground survey quantities to LaTeX file

latex_lines.extend(
    [
        latex_variables(
            "SurveyEasting", (region[1] - region[0]) * 1e-3, r"\km", fmt=".0f"
        ),
        latex_variables(
            "SurveyNorthing", (region[1] - region[0]) * 1e-3, r"\km", fmt=".0f"
        ),
        latex_variables("SurveyNoise", noise_std, r"\milli Gal", fmt=".0f"),
        latex_variables("GroundSurveyPoints", int(survey.height.size)),
        latex_variables("GroundSurveyMinHeight", survey.height.min(), r"\m", fmt=".0f"),
        latex_variables("GroundSurveyMaxHeight", survey.height.max(), r"\m", fmt=".0f"),
    ]
)

# ## Synthetic airborne survey

# Get coordinates of observation points

survey = hm.synthetic.airborne_survey(region=region_degrees)
display(survey)

# Project observation points

survey["easting"], survey["northing"] = projection(
    survey.longitude.values, survey.latitude.values
)
display(survey)

# Plot the survey points

fig, ax = plt.subplots(figsize=(6, 6))
tmp = ax.scatter(survey.easting, survey.northing, c=survey.height, s=6)
plt.colorbar(tmp, ax=ax, label="m")
ax.set_aspect("equal")
ax.set_title("Height of airborne survey points")
plt.show()

# Compute gravitational field of synthetic model on airborne survey

coordinates = (survey.easting, survey.northing, survey.height)
survey[field] = hm.prism_gravity(
    coordinates, model["prisms"], model["densities"], field=field
) + np.random.normal(scale=noise_std, size=survey.easting.size)
display(survey)

# +
# Load matplotlib configuration
plt.style.use(os.path.join("..", "matplotlib.rc"))

# Define useful parameters
width = 3.33
figsize = (width, width * 1.7)
cbar_shrink = 0.95
cbar_pad = 0.03
cbar_aspect = 30
size = 2
labels = "a b".split()

# Initialize figure and axes
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=figsize)

# Plot survey points
tmp = ax1.scatter(
    survey.easting, survey.northing, c=survey.height, cmap="cividis", s=size
)
clb = plt.colorbar(
    tmp, ax=ax1, shrink=cbar_shrink, orientation="vertical", pad=0.03, aspect=30
)
clb.set_label("m", labelpad=-15, y=1.05, rotation=0)

# Plot measured values
tmp = ax2.scatter(survey.easting, survey.northing, c=survey.g_z, cmap="viridis", s=size)
clb = plt.colorbar(
    tmp, ax=ax2, shrink=cbar_shrink, orientation="vertical", pad=0.03, aspect=30
)
clb.set_label("mGal", labelpad=-15, y=1.05, rotation=0)

ax2.set_xlabel("easting [m]")
ax1.tick_params(
    axis="x", which="both", bottom=False, top=False, labelbottom=False,
)

for ax, label in zip((ax1, ax2), labels):
    ax.set_aspect("equal")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.set_ylabel("northing [m]")
    ax.yaxis.offsetText.set_x(-0.2)
    ax.annotate(
        label,
        xy=(0.04, 0.94),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )

ax1.set_title("Airborne survey points", pad=3)
ax2.set_title("Observed gravity acceleration", pad=3)

plt.tight_layout(h_pad=0.2)
plt.savefig(
    os.path.join("..", "manuscript", "figs", "airborne-survey.pdf"),
    bbox_inches="tight",
    dpi=300,
)
plt.show()
# -

# Save ground survey for future usage

survey.to_csv(os.path.join(airborne_results_dir, "survey.csv"), index=False)

# Save airborne survey quantities to LaTeX file

latex_lines.extend(
    [
        latex_variables("AirborneSurveyPoints", int(survey.height.size)),
        latex_variables(
            "AirborneSurveyMinHeight", survey.height.min(), r"\m", fmt=".0f"
        ),
        latex_variables(
            "AirborneSurveyMaxHeight", survey.height.max(), r"\m", fmt=".0f"
        ),
    ]
)

# ## Compute gravity field on target grid

# We want to compute the true gravitational effect generated by the synthetic model on
# a regular grid at a constant height.

grid = vd.grid_coordinates(
    region=region,
    spacing=target_grid_spacing,
    adjust="region",
    extra_coords=target_grid_height,
)

target = hm.prism_gravity(grid, model["prisms"], model["densities"], field=field)
target = grid_to_dataarray(target, grid, attrs={"height": target_grid_height})

# Save target grid to disk for future usage

target.to_netcdf(os.path.join(results_dir, "target.nc"))

# Save target grid quantities to LaTeX file

latex_lines.extend(
    [
        latex_variables("TargetHeight", target_grid_height, r"\m", fmt=".0f"),
        latex_variables("TargetSpacing", target_grid_spacing * 1e-3, r"\km", fmt=".0f"),
        latex_variables("TargetEastingSize", target.easting.size),
        latex_variables("TargetNorthingSize", target.northing.size),
    ]
)

# +
# Load matplotlib configuration
plt.style.use(os.path.join("..", "matplotlib.rc"))

width = 3.33
figsize = (width, width * 0.85)
fig, ax = plt.subplots(figsize=figsize)

tmp = target.plot.pcolormesh(
    ax=ax, add_colorbar=False, cmap="viridis", center=False, rasterized=True
)
ax.set_aspect("equal")
ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
ax.set_xlabel(ax.get_xlabel() + " [m]")
ax.set_ylabel(ax.get_ylabel() + " [m]")
clb = plt.colorbar(tmp, ax=ax, shrink=1, orientation="vertical", pad=0.03, aspect=30)
clb.set_label("mGal", labelpad=-15, y=1.05, rotation=0)

ax.set_title("Target grid")
plt.tight_layout()
plt.savefig(
    os.path.join("..", "manuscript", "figs", "target-grid.pdf"),
    bbox_inches="tight",
    dpi=300,
)
plt.show()
# -

# ## Dump LaTeX variables to file

for i in latex_lines:
    print(i)

with open(os.path.join("..", "manuscript", "synthetic_model_surveys.tex"), "w") as f:
    f.write("\n".join(latex_lines))
