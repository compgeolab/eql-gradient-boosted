# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python [conda env:eql_source_layouts]
#     language: python
#     name: conda-env-eql_source_layouts-py
# ---

# # Create synthetic ground and airborne surveys from a synthetic model

# **Import useful packages**

# +
from IPython.display import display
from pathlib import Path
import json
import pyproj
import numpy as np
import xarray as xr
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from source_layouts import (
    synthetic_model,
)

# -

# **Define parameters**

# +
# Define results directories
results_dir = Path("..") / "results"
ground_results_dir = results_dir / "ground_survey"
airborne_results_dir = results_dir / "airborne_survey"

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

# **Save model related quantities on dictionary**

np.min(model["densities"])

variables = {
    "n_prisms": len(model["densities"]),
    "model_easting": (model_region[1] - model_region[0]),
    "model_northing": (model_region[1] - model_region[0]),
    "model_depth": abs(model_region[4]),
    "model_min_density": np.min(model["densities"]),
    "model_max_density": np.max(model["densities"]),
}

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

# Plot survey points and observed gravity field

# +
size = 6

plt.scatter(survey.easting, survey.northing, c=survey.height, cmap="cividis", s=size)
plt.colorbar(label="m")
plt.gca().set_aspect("equal")
plt.xlabel("easting")
plt.ylabel("northing")
plt.title("Ground survey points")
plt.show()

plt.scatter(survey.easting, survey.northing, c=survey.g_z, cmap="viridis", s=size)
plt.colorbar(label="mGal")
plt.gca().set_aspect("equal")
plt.xlabel("easting")
plt.ylabel("northing")
plt.title("Observed gravity acceleration")
plt.show()
# -

# Save ground survey for future usage

survey.to_csv(ground_results_dir / "survey.csv", index=False)

# Save ground survey quantities to LaTeX file

variables["survey_easting"] = region[1] - region[0]
variables["survey_northing"] = region[3] - region[2]
variables["survey_noise"] = noise_std
variables["ground_survey_points"] = int(survey.height.size)
variables["ground_survey_min_height"] = survey.height.min()
variables["ground_survey_max_height"] = survey.height.max()

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

# Plot survey points and observed gravity field

# +
size = 6

plt.scatter(survey.easting, survey.northing, c=survey.height, cmap="cividis", s=size)
plt.colorbar(label="m")
plt.gca().set_aspect("equal")
plt.xlabel("easting")
plt.ylabel("northing")
plt.title("Airborne survey points")
plt.show()

plt.scatter(survey.easting, survey.northing, c=survey.g_z, cmap="viridis", s=size)
plt.colorbar(label="mGal")
plt.gca().set_aspect("equal")
plt.xlabel("easting")
plt.ylabel("northing")
plt.title("Observed gravity acceleration")
plt.show()
# -

# Save ground survey for future usage

survey.to_csv(airborne_results_dir / "survey.csv", index=False)

# Save airborne survey quantities to variables dictionary

variables["airborne_survey_points"] = int(survey.height.size)
variables["airborne_survey_min_height"] = survey.height.min()
variables["airborne_survey_max_height"] = survey.height.max()

# ## Compute gravity field on target grid

# We want to compute the true gravitational effect generated by the synthetic model on
# a regular grid at a constant height.

grid = vd.grid_coordinates(
    region=region,
    spacing=target_grid_spacing,
    adjust="region",
    extra_coords=target_grid_height,
)

# Compute gravity field on the grid

target = hm.prism_gravity(grid, model["prisms"], model["densities"], field=field)

# Create a xarray.DataArray for the grid

dims = ("northing", "easting")
coords = {"northing": grid[1][:, 0], "easting": grid[0][0, :]}
target = xr.DataArray(
    target, dims=dims, coords=coords, attrs={"height": target_grid_height}
)

# Save target grid to disk for future usage

target.to_netcdf(results_dir / "target.nc")

# Save target grid quantities to variables dictionary

variables["target_height"] = target_grid_height
variables["target_spacing"] = target_grid_spacing * 1e-3
variables["target_easting_size"] = target.easting.size
variables["target_northing_size"] = target.northing.size

# Plot target grid

target.plot(center=False, cbar_kwargs={"label": "mGal"})
plt.gca().set_aspect("equal")
plt.title("Target grid")
plt.show()

# ## Dump variables dictionary to a JSON file

json_file = results_dir / "synthetic-surveys.json"
with open(json_file, "w") as f:
    json.dump(variables, f)
