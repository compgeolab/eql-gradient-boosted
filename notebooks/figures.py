# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python [conda env:eql_source_layouts]
#     language: python
#     name: conda-env-eql_source_layouts-py
# ---

# # Generate manuscript figures

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# ## Load matplotlib configuration

plt.style.use(os.path.join("..", "matplotlib.rc"))

# ## Define results directory

results_dir = os.path.join("..", "results")
ground_results_dir = os.path.join(results_dir, "ground_survey")
airborne_results_dir = os.path.join(results_dir, "airborne_survey")

# ## Ground survey

survey = pd.read_csv(os.path.join(ground_results_dir, "survey.csv"))

# +
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
# -

# ## Airborne survey

survey = pd.read_csv(os.path.join(airborne_results_dir, "survey.csv"))

# +
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
# -

# ## Target grid

target = xr.open_dataarray(os.path.join(results_dir, "target.nc"))

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
