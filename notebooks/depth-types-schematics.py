# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python [conda env:eql_source_layouts]
#     language: python
#     name: conda-env-eql_source_layouts-py
# ---

# # Schematic figure of depth types

# Import packages

# +
from IPython.display import display
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eql_source_layouts.layouts import source_bellow_data

# -

# Read a synthetic 1d survey

survey = pd.read_csv(os.path.join("..", "data", "survey_1d.csv"))
display(survey)

# ## Create source distributions

points = {}

# Source bellow data with constant depth

# +
depth_type = "constant_depth"
coordinates = (survey.easting, np.zeros_like(survey.easting), survey.height)
points[depth_type] = source_bellow_data(coordinates, depth_type=depth_type, depth=150)

plt.scatter(coordinates[0], coordinates[2])
plt.scatter(points[depth_type][0], points[depth_type][2])
plt.show()
# -

# Source bellow data with relative depth

# +
depth_type = "relative_depth"
coordinates = (survey.easting, np.zeros_like(survey.easting), survey.height)
points[depth_type] = source_bellow_data(coordinates, depth_type=depth_type, depth=150)

plt.scatter(coordinates[0], coordinates[2])
plt.scatter(points[depth_type][0], points[depth_type][2])
plt.show()
# -

# Source bellow data with variable depth

# +
depth_type = "variable_depth"
coordinates = (survey.easting, np.zeros_like(survey.easting), survey.height)
points[depth_type] = source_bellow_data(
    coordinates, depth_type=depth_type, depth_factor=1, depth=100, k_nearest=3
)

plt.scatter(coordinates[0], coordinates[2])
plt.scatter(points[depth_type][0], points[depth_type][2])
plt.show()
# +
# Load matplotlib configuration
plt.style.use(os.path.join("..", "matplotlib.rc"))

# Initialize figure and axes
width = 6.66
height = 1.5
fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(width, height))

# Define styles and axes labels and titles
size = 8
titles = ["Constant Depth", "Relative Depth", "Variable Depth"]
labels = "a b c".split()

# Plot
for i, (ax, depth_type, title, label) in enumerate(zip(axes, points, titles, labels)):
    ax.scatter(coordinates[0], coordinates[2], s=size, label="stations", marker="^")
    ax.scatter(points[depth_type][0], points[depth_type][2], s=size, label="sources")
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.tick_params(
        axis="y",  # changes apply to the y-axis
        which="both",  # both major and minor ticks are affected
        left=False,  # ticks along the left edge are off
        right=False,  # ticks along the right edge are off
        labelleft=False,  # labels along the bottom edge are off
    )
    ax.tick_params(
        axis="x",  # changes apply to the y-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the left edge are off
        top=False,  # ticks along the right edge are off
        labelbottom=False,  # labels along the bottom edge are off
    )
    ax.annotate(
        label,
        xy=(0.045, 0.88),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )

axes[0].legend(loc=(0.6, 0.32), borderpad=0.2, labelspacing=0.3)
axes[0].set_ylim(-200, 130)

plt.tight_layout(w_pad=0)
plt.savefig(os.path.join("..", "manuscript", "figs", "depth_types.pdf"))
plt.show()
