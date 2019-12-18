# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Source Distributions
#
#
# We will compare different source distributions for gridding potential field data
# throught the Equivalent Layer method. Each source distributions is created as
# a combination of a source layout and a depth type.
# Lets define three layouts of point sources:
#
# 1. **Source beneath data**: One source point beneath each data point
# 2. **Block-median sources**: Split the region in blocks of equal size, compute the
#    median coordinate of the data points per block, and put one point source beneath
#    this block-median coordinate.
# 3. **Grid of sources**: A regular grid of source points
#
# And the depth types, i.e. how deep do we put the point sources:
#
# 1. **Constant depth**: Source points located all at the same depth, which can be
#    computed as the mean height of the data points minus a constant depth.
# 2. **Relative depth**: Source points located at a constant relative depth bellow the
#    data points. If the data points are not at the same height (which is the usual
#    case), the point sources will be located at different depths, but at the same
#    relative depth from its corresponding data point -or block reduced data point.
# 3. **Variable relative depth**: Source points located at a variable relative depth
#    from the data points. We can define a relative depth for each source point as
#    a quantity proportional to the median distance to its k nearest source points.
#    Also, this relative depth can be static shifted by a constant depth in order to
#    avoid very shallow sources in case of too close source points.
#
# The first two layouts can be setted with any of these three types of depth, although
# the grid of sources can only be defined with the constant depth.
# Therefore we get a total of 7 possible combinations of layouts and depths:
#
# |    | Constant depth | Relative depth | Variable relative depth |
# | -- |----------------|----------------|-------------------------|
# | **Source beneath data** | ✅ | ✅ | ✅ |
# | **Block reduced sources** | ✅ | ✅ | ✅ |
# | **Grid of sources** | ✅ | ❌ | ❌ |
#
#
# Here we will show some examples on how each source distribution look like for the same
# airborne survey.

# **Import useful packages**

# +
from IPython.display import display
import pyproj
import numpy as np
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt

import eql_source_layouts

# -

# ## Define parameters for building the source distributions

# +
# Define dictionaries where the source distributions will be stored
layouts = ["source_bellow_data", "block_median_sources", "grid_sources"]
source_distributions = {layout: {} for layout in layouts}

# Define a region for the synthetic survey
region_degrees = [-0.5, 0.5, -0.5, 0.5]  # given in degrees

# Set a constant depth of 2km
constant_depth = 2000

# Set a relative depth of 2km
relative_depth = 2000

# Define a block size of 2km for block-median layouts
spacing = 4000

# Define set of parameters for each source distribution
# =====================================================
parameters = {layout: {} for layout in layouts}

# ... for source beneath data layout
layout = "source_bellow_data"
depth_type = "constant_depth"
parameters[layout][depth_type] = {
    "depth_type": depth_type,
    "constant_depth": constant_depth,
}
depth_type = "relative_depth"
parameters[layout][depth_type] = {
    "depth_type": depth_type,
    "relative_depth": relative_depth,
}
depth_type = "variable_relative_depth"
parameters[layout][depth_type] = {
    "depth_type": depth_type,
    "depth_factor": 0.5,
    "depth_shift": -500,
    "k_nearest": 15,
}

# ... for block-median sources layout
layout = "block_median_sources"
depth_type = "constant_depth"
parameters[layout][depth_type] = {
    "depth_type": depth_type,
    "spacing": spacing,
    "constant_depth": constant_depth,
}
depth_type = "relative_depth"
parameters[layout][depth_type] = {
    "depth_type": depth_type,
    "spacing": spacing,
    "relative_depth": relative_depth,
}
depth_type = "variable_relative_depth"
parameters[layout][depth_type] = {
    "depth_type": depth_type,
    "depth_factor": 0.2,
    "depth_shift": -500,
    "k_nearest": 2,
    "spacing": spacing,
}

layout = "grid_sources"
depth_type = "constant_depth"
parameters[layout][depth_type] = {
    "depth_type": depth_type,
    "spacing": spacing,
    "constant_depth": constant_depth,
}
# -

# ## 1. Create a synthetic airborne survey
#

# Get coordinates of observation points from a synthetic airborne survey

survey = hm.synthetic.airborne_survey(region=region_degrees)
display(survey)

# Project survey points into Cartesian coordinates

# +
projection = pyproj.Proj(proj="merc", lat_ts=0)
survey["easting"], survey["northing"] = projection(
    survey.longitude.values, survey.latitude.values
)
display(survey)

# Define region boundaries in projected coordinates
region = (
    survey.easting.values.min(),
    survey.easting.values.max(),
    survey.northing.min(),
    survey.northing.max(),
)
# -

# Plot the survey points

fig, ax = plt.subplots(figsize=(6, 6))
tmp = ax.scatter(survey.easting, survey.northing, c=survey.height, s=6)
plt.colorbar(tmp, ax=ax, label="m")
ax.set_aspect("equal")
ax.set_title("Height of airborne survey points")
plt.show()

# Define coordinates tuple and the projected region

coordinates = (survey.easting, survey.northing, survey.height)
region = vd.get_region(coordinates)

# ## Generate the source distributions

for layout in parameters:
    for depth_type in parameters[layout]:
        source_distributions[layout][depth_type] = getattr(eql_source_layouts, layout)(
            coordinates, **parameters[layout][depth_type]
        )


# Plot source distributions

# +
heights = tuple(
    source_distributions[layout][depth_type][2]
    for layout in source_distributions
    for depth_type in source_distributions[layout]
)
vmin = np.min([h.min() for h in heights])
vmax = np.max([h.max() for h in heights])

fig, axes = plt.subplots(
    figsize=(12, 12), nrows=3, ncols=3, sharex=True, sharey=True,
)
for i, (ax_row, layout) in enumerate(zip(axes, source_distributions)):
    for j, (ax, depth_type) in enumerate(zip(ax_row, source_distributions[layout])):
        points = source_distributions[layout][depth_type]
        tmp = ax.scatter(*points[:2], c=points[2], s=10, vmin=vmin, vmax=vmax)
        ax.set_title("n_points: {}".format(points[0].size))
        ax.set_aspect("equal")
        # Annotate the columns of the figure
        if i == 0:
            ax.text(
                0.5,
                1.15,
                depth_type,
                fontsize="large",
                fontweight="bold",
                horizontalalignment="center",
                transform=ax.transAxes,
            )
        # Annotate the rows of the figure
        if j == 0:
            ax.text(
                -0.3,
                0.5,
                layout,
                fontsize="large",
                fontweight="bold",
                verticalalignment="center",
                transform=ax.transAxes,
                rotation="vertical",
            )


# Hide the last two axes because they are not used
axes[-1][-1].set_visible(False)
axes[-1][-2].set_visible(False)

# Add colorbar
fig.subplots_adjust(bottom=0.1, wspace=0.05)
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
fig.colorbar(tmp, cax=cbar_ax, orientation="horizontal", label="m")

plt.show()
