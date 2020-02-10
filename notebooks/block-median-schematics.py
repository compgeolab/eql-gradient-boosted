# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python [conda env:eql_source_layouts]
#     language: python
#     name: conda-env-eql_source_layouts-py
# ---

# +
import os
import numpy as np
import verde as vd
import matplotlib.pyplot as plt

from eql_source_layouts import block_median_sources

# +
region = (-50e3, 50e3, -50e3, 50e3)

n_data = 30
spacing = 25000
seed = 1234
height = 1000
depth = 100
# -

coordinates = vd.scatter_points(
    region, size=n_data, random_state=seed, extra_coords=height
)

points = block_median_sources(
    coordinates, constant_depth=depth, spacing=spacing, depth_type="constant_depth"
)

# +
grid_nodes = vd.grid_coordinates(vd.get_region(coordinates), spacing=spacing)

grid_lines = (np.unique(grid_nodes[0]), np.unique(grid_nodes[1]))
for nodes in grid_lines:
    nodes.sort()

# +
xmin, xmax, ymin, ymax = vd.get_region(coordinates)
grid_style = dict(color="grey", linewidth=0.7, linestyle="--")
scatter_style = dict(s=12)
labels = "a b c".split()

plt.style.use(os.path.join("..", "matplotlib.rc"))

fig, axes = plt.subplots(
    ncols=3, nrows=1, sharey=True, figsize=(6.66, 6.66 / 3), dpi=300
)

ax = axes[0]
ax.scatter(*coordinates[:2], label="data points", **scatter_style)


ax = axes[1]
for x in grid_lines[0]:
    ax.plot((x, x), (ymin, ymax), **grid_style)
for y in grid_lines[1]:
    ax.plot((xmin, xmax), (y, y), **grid_style)
ax.scatter(*coordinates[:2], **scatter_style)

ax = axes[2]
for x in grid_lines[0]:
    ax.plot((x, x), (ymin, ymax), **grid_style)
for y in grid_lines[1]:
    ax.plot((xmin, xmax), (y, y), **grid_style)
ax.scatter(*points[:2], color="C1", label="block-median points", **scatter_style)


for ax, label in zip(axes, labels):
    ax.annotate(
        label,
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )
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

# axes[0].legend(loc=3)
# axes[2].legend(loc=3)
plt.tight_layout(w_pad=0)
plt.show()
# -
