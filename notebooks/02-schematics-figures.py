# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python [conda env:eql-gradient-boosted]
#     language: python
#     name: conda-env-eql-gradient-boosted-py
# ---

# # Source layouts schematics

# +
from IPython.display import display  # noqa: F401  # ignore used but not imported
from pathlib import Path
import numpy as np
import pandas as pd
import verde as vd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import boost_and_layouts
from boost_and_layouts import save_to_json

# -

# ## Define parameters for building the source distributions

# Define results directory to read synthetic ground survey
results_dir = Path("..") / "results"
ground_results_dir = results_dir / "ground_survey"

# ## Read synthetic ground survey
#

# Get coordinates of observation points from a synthetic ground survey

survey = pd.read_csv(ground_results_dir / "survey.csv")

inside = np.logical_and(
    np.logical_and(
        survey.easting > 0,
        survey.easting < 40e3,
    ),
    np.logical_and(
        survey.northing > -60e3,
        survey.northing < -20e3,
    ),
)
survey = survey.loc[inside]

survey

fig, ax = plt.subplots(figsize=(6, 6))
tmp = ax.scatter(survey.easting, survey.northing)
ax.set_aspect("equal")
ax.set_title("Height of ground survey points")
plt.show()

coordinates = (survey.easting, survey.northing, survey.height)

# ### Generate the source distributions

# +
block_spacing = 3000
grid_spacing = 2000

layouts = ["source_below_data", "grid_sources", "block_averaged_sources"]
depth_type = "constant_depth"

parameters = {}

layout = "source_below_data"
parameters[layout] = dict(
    depth_type=depth_type,
    depth=500,
)

layout = "grid_sources"
parameters[layout] = dict(depth_type=depth_type, depth=500, spacing=grid_spacing)

layout = "block_averaged_sources"
parameters[layout] = dict(depth_type=depth_type, depth=500, spacing=block_spacing)
# -

source_distributions = {}
for layout in parameters:
    source_distributions[layout] = getattr(boost_and_layouts, layout)(
        coordinates, **parameters[layout]
    )

# Create lines for plotting the boundaries of the blocks

# +
region = vd.get_region(coordinates)
grid_nodes = vd.grid_coordinates(region, spacing=block_spacing)

grid_lines = (np.unique(grid_nodes[0]), np.unique(grid_nodes[1]))
for nodes in grid_lines:
    nodes.sort()
# -

# ## Plot observation points and source layouts

# +
# Load matplotlib configuration
plt.style.use(Path(".") / "matplotlib.rc")

titles = {
    "source_below_data": "Sources below data",
    "block_averaged_sources": "Block-averaged sources",
    "grid_sources": "Regular grid",
}

fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(7, 1.7), dpi=300)
size = 3
labels = "a b c d".split()

for ax, label in zip(axes, labels):
    ax.set_aspect("equal")
    ax.annotate(
        label,
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )
    ax.axis("off")

# Plot observation points
ax = axes[0]
ax.scatter(survey.easting, survey.northing, s=size, c="C0", marker="^")
ax.set_title("Observation points")

# Plot location of sources for each source layout
for ax, layout in zip(axes[1:], layouts):
    ax.scatter(*source_distributions[layout][:2], s=size, c="C1")
    ax.set_title(titles[layout])

# Add blocks boundaries to Block Averaged Sources plot
ax = axes[3]
grid_style = dict(color="grey", linewidth=0.5, linestyle="--")
xmin, xmax, ymin, ymax = region[:]
for x in grid_lines[0]:
    ax.plot((x, x), (ymin, ymax), **grid_style)
for y in grid_lines[1]:
    ax.plot((xmin, xmax), (y, y), **grid_style)

plt.tight_layout(w_pad=0)
plt.savefig(
    Path("..") / "manuscript" / "figs" / "source-layouts-schematics.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# -

# ## Dump number of observation points and sources to JSON file

# +
variables = {
    "source_layouts_schematics_observations": survey.easting.size,
}
for layout in layouts:
    variables["source_layouts_schematics_{}".format(layout)] = source_distributions[
        layout
    ][0].size

json_file = results_dir / "source-layouts-schematics.json"
save_to_json(variables, json_file)
# -

# # Gradient boosting schematics

sources = source_distributions["source_below_data"]
region = vd.get_region(sources)

# +
overlapping = 0.5
window_size = 18e3
spacing = window_size * (1 - overlapping)

centers, indices = vd.rolling_window(sources, size=window_size, spacing=spacing)
spacing_easting = centers[0][0, 1] - centers[0][0, 0]
spacing_northing = centers[1][1, 0] - centers[1][0, 0]

print("Desired spacing:", spacing)
print("Actual spacing:", (spacing_easting, spacing_northing))

# +
indices = [i[0] for i in indices.ravel()]
centers = [i.ravel() for i in centers]
n_windows = centers[0].size

print("Number of windows:", n_windows)

# +
ncols = 10
figsize = (1.7 * ncols, 1.7)
size = 3

fig, axes = plt.subplots(ncols=ncols, nrows=1, figsize=figsize, dpi=300, sharex=True, sharey=True)

for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

# Observation points
axes[0].scatter(survey.easting, survey.northing, s=size, c="C0", marker="^")

# Sources
axes[1].scatter(*sources[:2], s=size, c="C1")

# First fit
# ---------
window_i = 0
window = indices[window_i]
not_window = [i for i in np.arange(sources[0].size) if i not in window]
w_center_easting, w_center_northing = centers[0][window_i], centers[1][window_i]
rectangle_kwargs = dict(
    xy=(w_center_easting - window_size / 2, w_center_northing - window_size / 2),
    width=window_size,
    height=window_size,
    fill=False,
    linewidth=0.5,
    linestyle="--",
    color="#444444",
)

# Observation points
axes[2].scatter(
    survey.easting.values[window], survey.northing.values[window], s=size, c="C0", marker="^"
)
axes[2].scatter(
    survey.easting.values[not_window], survey.northing.values[not_window], s=size, c="C7", marker="^"
)
rectangle = Rectangle(**rectangle_kwargs)
axes[2].add_patch(rectangle)

# Sources
axes[3].scatter(sources[0][window], sources[1][window], s=size, c="C1")
axes[3].scatter(sources[0][not_window], sources[1][not_window], s=size, c="C7")
rectangle = Rectangle(**rectangle_kwargs)
axes[3].add_patch(rectangle)

# First Prediction
# ----------------
axes[4].scatter(
    survey.easting, survey.northing, s=size, c="C3", marker="v"
)
axes[5].scatter(sources[0][window], sources[1][window], s=size, c="C1")
rectangle = Rectangle(**rectangle_kwargs)
axes[5].add_patch(rectangle)

# Second fit
# ----------
window_i = 3
window = indices[window_i]
not_window = [i for i in np.arange(sources[0].size) if i not in window]
w_center_easting, w_center_northing = centers[0][window_i], centers[1][window_i]
rectangle_kwargs = dict(
    xy=(w_center_easting - window_size / 2, w_center_northing - window_size / 2),
    width=window_size,
    height=window_size,
    fill=False,
    linewidth=0.5,
    linestyle="--",
    color="#444444",
)

# Residue
axes[6].scatter(
    survey.easting.values[window], survey.northing.values[window], s=size, c="C3", marker="v"
)
axes[6].scatter(
    survey.easting.values[not_window], survey.northing.values[not_window], s=size, c="C7", marker="^"
)
rectangle = Rectangle(**rectangle_kwargs)
axes[6].add_patch(rectangle)

# Sources
axes[7].scatter(sources[0][window], sources[1][window], s=size, c="C1")
axes[7].scatter(sources[0][not_window], sources[1][not_window], s=size, c="C7")
rectangle = Rectangle(**rectangle_kwargs)
axes[7].add_patch(rectangle)

# Second Prediction
# -----------------
axes[8].scatter(
    survey.easting, survey.northing, s=size, c="C3", marker="v"
)
axes[9].scatter(sources[0][window], sources[1][window], s=size, c="C1")
rectangle = Rectangle(**rectangle_kwargs)
axes[9].add_patch(rectangle)


plt.savefig(
    Path("..") / "manuscript" / "figs" / "svg" / "gradient-boosting-raw.svg"
)
plt.show()
# -


