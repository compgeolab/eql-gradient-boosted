# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python [conda env:eql_source_layouts]
#     language: python
#     name: conda-env-eql_source_layouts-py
# ---

import os
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import matplotlib
import matplotlib.pyplot as plt

# +
white = "#BECFC7"
light_blue = "#2A576A"
dark_blue = "#002D40"
light_orange = "#FABD4A"
dark_orange = "#FA9600"

matplotlib.rcParams["axes.edgecolor"] = light_blue
matplotlib.rcParams["axes.labelcolor"] = light_blue
matplotlib.rcParams["ytick.color"] = light_blue
matplotlib.rcParams["xtick.color"] = light_blue

# +
results_dir = os.path.join("..", "results")
ground_results_dir = os.path.join(results_dir, "ground_survey")
airborne_results_dir = os.path.join(results_dir, "airborne_survey")

# Define which field will be meassured
field = "g_z"
field_units = "mGal"

layout = "block_averaged_sources"

figsize = (4, 3.5)

cbar_kwargs = dict(shrink=0.82, pad=0.04)
cbar_label_kwargs = dict(label="mGal", rotation=0, labelpad=-25, y=1.06)
# -

dirname = "egu_figures"
if not os.path.isdir("egu_figures"):
    os.makedirs(dirname)

ground_survey = pd.read_csv(os.path.join(ground_results_dir, "survey.csv"))
airborne_survey = pd.read_csv(os.path.join(airborne_results_dir, "survey.csv"))

target = xr.open_dataarray(os.path.join(results_dir, "target.nc"))

# +
ground_prediction = xr.open_dataset(
    os.path.join(ground_results_dir, "best_predictions-{}.nc".format(layout))
).variable_depth

airborne_prediction = xr.open_dataset(
    os.path.join(airborne_results_dir, "best_predictions-{}.nc".format(layout))
).variable_depth
# -

fig, ax = plt.subplots(figsize=figsize)
tmp = plt.pcolormesh(target.easting, target.northing, target.values, rasterized=True)
ax.set_aspect("equal")
ax.set_yticks([])
ax.set_xticks([])
clb = plt.colorbar(tmp, ax=ax, **cbar_kwargs)
clb.set_label(**cbar_label_kwargs)
plt.tight_layout()
plt.savefig(os.path.join(dirname, "target_grid.svg"))
plt.show()

fig, ax = plt.subplots(figsize=figsize)
tmp = ax.scatter(ground_survey.easting, ground_survey.northing, c=ground_survey.g_z, s=10)
ax.set_aspect("equal")
ax.set_yticks([])
ax.set_xticks([])
clb = plt.colorbar(tmp, ax=ax, **cbar_kwargs)
clb.set_label(**cbar_label_kwargs)
plt.tight_layout()
plt.savefig(os.path.join(dirname, "ground_survey.svg"))
plt.show()

fig, ax = plt.subplots(figsize=figsize)
tmp = ax.scatter(airborne_survey.easting, airborne_survey.northing, c=airborne_survey.g_z, s=10)
ax.set_aspect("equal")
ax.set_yticks([])
ax.set_xticks([])
clb = plt.colorbar(tmp, ax=ax, **cbar_kwargs)
clb.set_label(**cbar_label_kwargs)
plt.tight_layout()
plt.savefig(os.path.join(dirname, "airborne_survey.svg"))
plt.show()

# +
ground_difference = target - ground_prediction
airborne_difference = target - airborne_prediction

vmin = min(ground_prediction.min(), airborne_prediction.min())
vmax = max(ground_prediction.max(), airborne_prediction.max())
difference_maxabs = vd.maxabs(ground_difference, airborne_difference)

# +
fig, ax = plt.subplots(figsize=figsize)
tmp = plt.pcolormesh(
    ground_prediction.easting,
    ground_prediction.northing,
    ground_prediction.values,
    vmin=vmin,
    vmax=vmax,
    rasterized=True,
)
ax.set_aspect("equal")
ax.set_yticks([])
ax.set_xticks([])
clb = plt.colorbar(tmp, ax=ax, **cbar_kwargs)
clb.set_label(**cbar_label_kwargs)
plt.tight_layout()
plt.savefig(os.path.join(dirname, "ground_prediction.svg"))
plt.show()

fig, ax = plt.subplots(figsize=figsize)
tmp = plt.pcolormesh(
    ground_difference.easting,
    ground_difference.northing,
    ground_difference.values,
    cmap="seismic",
    vmin=-difference_maxabs,
    vmax=difference_maxabs,
    rasterized=True,
)
ax.set_aspect("equal")
ax.set_yticks([])
ax.set_xticks([])
clb = plt.colorbar(tmp, ax=ax, **cbar_kwargs)
clb.set_label(**cbar_label_kwargs)
plt.tight_layout()
plt.savefig(os.path.join(dirname, "ground_difference.svg"))
plt.show()

# +
fig, ax = plt.subplots(figsize=figsize)
tmp = plt.pcolormesh(
    airborne_prediction.easting,
    airborne_prediction.northing,
    airborne_prediction.values,
    vmin=vmin,
    vmax=vmax,
    rasterized=True,
)
ax.set_aspect("equal")
ax.set_yticks([])
ax.set_xticks([])
clb = plt.colorbar(tmp, ax=ax, **cbar_kwargs)
clb.set_label(**cbar_label_kwargs)
plt.tight_layout()
plt.savefig(os.path.join(dirname, "airborne_prediction.svg"))
plt.show()

fig, ax = plt.subplots(figsize=figsize)
tmp = plt.pcolormesh(
    airborne_difference.easting,
    airborne_difference.northing,
    airborne_difference.values,
    cmap="seismic",
    vmin=-difference_maxabs,
    vmax=difference_maxabs,
    rasterized=True,
)
ax.set_aspect("equal")
ax.set_yticks([])
ax.set_xticks([])
clb = plt.colorbar(tmp, ax=ax, **cbar_kwargs)
clb.set_label(**cbar_label_kwargs)
plt.tight_layout()
plt.savefig(os.path.join(dirname, "airborne_difference.svg"))
plt.show()
