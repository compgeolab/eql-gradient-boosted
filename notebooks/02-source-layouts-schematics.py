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

# +
from IPython.display import display  # noqa: F401  # ignore used but not imported
from pathlib import Path
import numpy as np
import pandas as pd
import verde as vd
import matplotlib.pyplot as plt

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
block_spacing = 4000
grid_spacing = 4000

layouts = ["source_below_data", "block_averaged_sources", "grid_sources"]
depth_type = "constant_depth"

parameters = {}

layout = "source_below_data"
parameters[layout] = dict(
    depth_type=depth_type,
    depth=500,
)

layout = "block_averaged_sources"
parameters[layout] = dict(depth_type=depth_type, depth=500, spacing=block_spacing)

layout = "grid_sources"
parameters[layout] = dict(depth_type=depth_type, depth=500, spacing=grid_spacing)
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

fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(6.66, 1.65), dpi=300)
size = 3
labels = "a b c d".split()

for ax, label in zip(axes, labels):
    ax.set_aspect("equal")
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
    )
    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )
    ax.annotate(
        label,
        xy=(0.05, 0.9),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )

# Plot observation points
ax = axes[0]
ax.scatter(survey.easting, survey.northing, s=size, c="C0")
ax.set_title("Observation points")

# Plot location of sources for each source layout
for ax, layout in zip(axes[1:], layouts):
    ax.scatter(*source_distributions[layout][:2], s=size, c="C1")
    ax.set_title(layout.replace("_", " ").title())

# Add blocks boundaries to Block Averaged Sources plot
ax = axes[2]
grid_style = dict(color="grey", linewidth=0.3, linestyle="--")
xmin, xmax, ymin, ymax = region[:]
for x in grid_lines[0]:
    ax.plot((x, x), (ymin, ymax), **grid_style)
for y in grid_lines[1]:
    ax.plot((xmin, xmax), (y, y), **grid_style)


plt.tight_layout(w_pad=0)
plt.savefig(
    Path("..") / "manuscript" / "figs" / "source-layouts-schematics.pdf", dpi=300
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
