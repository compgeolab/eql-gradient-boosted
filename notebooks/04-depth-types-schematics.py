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

# # Schematic figure of depth types

# Import packages

# +
from IPython.display import display
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from boost_and_layouts import source_below_data

# -

# Read a synthetic 1d survey

survey = pd.read_csv(Path("..") / "data" / "survey_1d.csv")
display(survey)

# ## Create source distributions

points = {}

# Source below data with constant depth

# +
depth_type = "constant_depth"
coordinates = (survey.easting, np.zeros_like(survey.easting), survey.height)
points[depth_type] = source_below_data(coordinates, depth_type=depth_type, depth=150)

plt.scatter(coordinates[0], coordinates[2])
plt.scatter(points[depth_type][0], points[depth_type][2])
plt.show()
# -

# Source below data with relative depth

# +
depth_type = "relative_depth"
coordinates = (survey.easting, np.zeros_like(survey.easting), survey.height)
points[depth_type] = source_below_data(coordinates, depth_type=depth_type, depth=150)

plt.scatter(coordinates[0], coordinates[2])
plt.scatter(points[depth_type][0], points[depth_type][2])
plt.show()
# -

# Source below data with variable depth

# +
depth_type = "variable_depth"
coordinates = (survey.easting, np.zeros_like(survey.easting), survey.height)
points[depth_type] = source_below_data(
    coordinates, depth_type=depth_type, depth_factor=1, depth=100, k_nearest=3
)

plt.scatter(coordinates[0], coordinates[2])
plt.scatter(points[depth_type][0], points[depth_type][2])
plt.show()
# +
# Load matplotlib configuration
plt.style.use(Path(".") / "matplotlib.rc")

# Initialize figure and axes
width = 6.66
height = 1.5
fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(width, height))

# Define styles and axes labels and titles
size = 8
titles = ["Constant Depth", "Relative Depth", "Variable Depth"]
labels = "a b c".split()

# Plot
for ax, depth_type, title, label in zip(axes, points, titles, labels):
    observations = ax.scatter(coordinates[0], coordinates[2], s=size, marker="^")
    sources = ax.scatter(points[depth_type][0], points[depth_type][2], s=size)
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.annotate(
        label,
        xy=(0.045, 0.98),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )

plt.figlegend(
    handles=(observations, sources),
    labels=("observation points", "equivalent sources"),
    loc=(0.62, 0.02),
    borderpad=0,
    handletextpad=0,
    frameon=False,
    ncol=2,
    columnspacing=0,
)
axes[0].set_ylim(-200, 130)

plt.tight_layout(w_pad=0)
plt.savefig(Path("..") / "manuscript" / "figs" / "depth_types.pdf", bbox_inches="tight")
plt.show()
