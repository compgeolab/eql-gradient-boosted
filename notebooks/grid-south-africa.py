# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python [conda env:eql_source_layouts]
#     language: python
#     name: conda-env-eql_source_layouts-py
# ---

# # Grid South Africa Gravity Survey

# +
from IPython.display import display
import os
import itertools
import pyproj
import dask
import numpy as np
import pandas as pd
import boule as bl
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from eql_source_layouts import (
    block_averaged_sources,
    combine_parameters,
    EQLIterative,
)

# -

# Define results directory

# +
results_dir = os.path.join("..", "results", "south_africa")

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
# -

# Fetch South Africa gravity data

data = hm.datasets.fetch_south_africa_gravity()
display(data)

# +
ax = plt.axes(projection=ccrs.Mercator())

fig = plt.gcf()
fig.set_size_inches(6.66, 6.66)

ax.coastlines()
ax.gridlines(draw_labels=True)

tmp = ax.scatter(
    data.longitude, data.latitude, c=data.gravity, s=2, transform=ccrs.PlateCarree()
)

plt.colorbar(tmp, ax=ax, pad=0.1, shrink=0.75)
plt.show()
# -

# Project data into Cartesian coordinates

# +
lat_ts = (data.latitude.min() + data.latitude.max()) / 2
projection = pyproj.Proj(proj="merc", lat_ts=lat_ts, ellps="WGS84")
easting, northing = projection(data.longitude.values, data.latitude.values)

coordinates = (easting, northing, data.elevation.values)

# +
ax = plt.axes(projection=ccrs.Mercator())

fig = plt.gcf()
fig.set_size_inches(6.66, 6.66)

ax.coastlines()
ax.gridlines(draw_labels=True)

tmp = ax.scatter(
    data.longitude, data.latitude, c=data.gravity, s=2, transform=ccrs.PlateCarree()
)

plt.colorbar(tmp, ax=ax, pad=0.1, shrink=0.75)
plt.show()
# -

# Compute gravity disturbance by removing normal gravity

ellipsoid = bl.WGS84
gravity_disturbance = data.gravity.values - ellipsoid.normal_gravity(
    data.latitude, data.elevation
)

# +
q = 99.9
vmax = np.percentile(gravity_disturbance, q)
vmin = -np.percentile(-gravity_disturbance, q)
maxabs = max(abs(vmin), abs(vmax))

fig, ax = plt.subplots(figsize=(6.66, 6.66))
tmp = ax.scatter(
    easting,
    northing,
    c=gravity_disturbance,
    vmin=-maxabs,
    vmax=maxabs,
    s=2,
    cmap="seismic",
)
plt.colorbar(tmp, ax=ax, shrink=0.75)
ax.grid()
ax.set_aspect("equal")
plt.show()
# -

# ## Grid gravity disturbance using block-averaged sources with variable depth

# Define sets of parameters for the equivalent layer

# +
depth_type = "variable_depth"
spacing = 15e3
k_nearest = 10
depth_factors = [0.1, 0.5, 1]
dampings = [1e1, 1e2, 1e3, 1e4, 1e5]
depths = [1e3, 2e3, 5e3, 7e3, 10e3]

# Combine these parameters
parameter_sets = combine_parameters(
    **dict(
        depth_type=depth_type,
        depth=depths,
        depth_factor=depth_factors,
        damping=dampings,
        spacing=spacing,
        k_nearest=k_nearest,
    )
)
print("Number of combinations:", len(parameter_sets))
# -

# Check how many sources are created with the chosen spacing

points = block_averaged_sources(coordinates, **parameter_sets[0])
print("Number of data points: {}".format(coordinates[0].size))
print("Number of sources: {}".format(points[0].size))

# Score the prediction made by each set of parameters through cross-validation

# +
cv = vd.BlockKFold(spacing=100e3, shuffle=True, random_state=0)

scores_delayed = []
for parameters in parameter_sets:
    points = block_averaged_sources(coordinates, **parameters)
    eql = hm.EQLHarmonic(damping=parameters["damping"], points=points)
    score = np.mean(
        vd.cross_val_score(eql, coordinates, gravity_disturbance, cv=cv, delayed=True)
    )
    scores_delayed.append(score)

scores = dask.compute(*scores_delayed)
# -

# Get the set of parameters that achieve the best score

# +
best_score = np.max(scores)
best_parameters = parameter_sets[np.argmax(scores)]

print(best_score)
print(best_parameters)
# -

# Grid the data using the best set of parameters

# +
points = block_averaged_sources(coordinates, **best_parameters)
eql = hm.EQLHarmonic(damping=best_parameters["damping"], points=points)
eql.fit(coordinates, gravity_disturbance)

eql.extra_coords_name = "upward"
grid = eql.grid(
    region=vd.get_region((data.longitude, data.latitude)),
    spacing=0.05,
    extra_coords=3000,
    data_names=["gravity_disturbance"],
    projection=projection,
)

# +
ax = plt.axes(projection=ccrs.Mercator())

fig = plt.gcf()
fig.set_size_inches(6.66, 6.66)

ax.coastlines()
ax.gridlines(draw_labels=True)

tmp = grid.gravity_disturbance.plot.pcolormesh(
    ax=ax,
    vmin=-maxabs,  # use the same colorscale used for the data
    vmax=maxabs,
    cmap="seismic",
    add_colorbar=False,
    transform=ccrs.PlateCarree(),
)

plt.colorbar(tmp, ax=ax, pad=0.11, shrink=0.75)
plt.show()
# -

# Mask values outside the convex hull

grid = vd.distance_mask(
    data_coordinates=(data.longitude, data.latitude),
    maxdist=50e3,
    grid=grid,
    projection=projection,
)

# Plot gridded data

# +
ax = plt.axes(projection=ccrs.Mercator())

fig = plt.gcf()
fig.set_size_inches(6.66, 6.66)

ax.coastlines()
ax.gridlines(draw_labels=True)

tmp = grid.gravity_disturbance.plot.pcolormesh(
    ax=ax,
    vmin=-maxabs,  # use the same colorscale used for the data
    vmax=maxabs,
    cmap="seismic",
    add_colorbar=False,
    transform=ccrs.PlateCarree(),
)

plt.colorbar(tmp, ax=ax, pad=0.11, shrink=0.75)
plt.show()
# -

# Save grid to disk

grid.to_netcdf(os.path.join(results_dir, "south_africa_gravity_grid.nc"))

# ## Grid gravity disturbance with EQLIterative

# Define sets of parameters for the equivalent layer

# +
depth_type = "variable_depth"
spacing = 15e3
k_nearest = 10
depth_factors = [0.1, 0.5, 1]
dampings = [1e1, 1e2, 1e3, 1e4, 1e5]
depths = [1e3, 2e3, 5e3, 7e3, 10e3]
window_size = 500e3
random_state = 0

# Combine these parameters
parameter_sets = combine_parameters(
    **dict(
        depth_type=depth_type,
        depth=depths,
        depth_factor=depth_factors,
        damping=dampings,
        spacing=spacing,
        k_nearest=k_nearest,
        window_size=window_size,
        random_state=random_state,
    )
)
print("Number of combinations:", len(parameter_sets))
# -

# Check how many sources are created with the chosen spacing

points = block_averaged_sources(coordinates, **parameter_sets[0])
print("Number of data points: {}".format(coordinates[0].size))
print("Number of sources: {}".format(points[0].size))

# Score the prediction made by each set of parameters through cross-validation

# + jupyter={"outputs_hidden": true}
cv = vd.BlockKFold(spacing=100e3, shuffle=True, random_state=0)

scores_delayed = []
for parameters in parameter_sets:
    points = block_averaged_sources(coordinates, **parameters)
    eql = EQLIterative(
        points=points,
        window_size=parameters["window_size"],
        damping=parameters["damping"],
        random_state=parameters["random_state"],
    )
    score = np.mean(
        vd.cross_val_score(eql, coordinates, gravity_disturbance, cv=cv, delayed=True)
    )
    scores_delayed.append(score)

scores = dask.compute(*scores_delayed)
# -

# Get the set of parameters that achieve the best score

# +
best_score = np.max(scores)
best_parameters = parameter_sets[np.argmax(scores)]

print(best_score)
print(best_parameters)
# -

# Grid the data using the best set of parameters

# +
points = block_averaged_sources(coordinates, **best_parameters)
eql = hm.EQLHarmonic(damping=best_parameters["damping"], points=points)
eql.fit(coordinates, gravity_disturbance)

eql.extra_coords_name = "upward"
grid = eql.grid(
    region=vd.get_region((data.longitude, data.latitude)),
    spacing=0.05,
    extra_coords=3000,
    data_names=["gravity_disturbance"],
    projection=projection,
)

# +
ax = plt.axes(projection=ccrs.Mercator())

fig = plt.gcf()
fig.set_size_inches(6.66, 6.66)

ax.coastlines()
ax.gridlines(draw_labels=True)

tmp = grid.gravity_disturbance.plot.pcolormesh(
    ax=ax,
    vmin=-maxabs,  # use the same colorscale used for the data
    vmax=maxabs,
    cmap="seismic",
    add_colorbar=False,
    transform=ccrs.PlateCarree(),
)

plt.colorbar(tmp, ax=ax, pad=0.11, shrink=0.75)
plt.show()
# -

# Mask values outside the convex hull

grid = vd.distance_mask(
    data_coordinates=(data.longitude, data.latitude),
    maxdist=50e3,
    grid=grid,
    projection=projection,
)

# Plot gridded data

# +
ax = plt.axes(projection=ccrs.Mercator())

fig = plt.gcf()
fig.set_size_inches(6.66, 6.66)

ax.coastlines()
ax.gridlines(draw_labels=True)

tmp = grid.gravity_disturbance.plot.pcolormesh(
    ax=ax,
    vmin=-maxabs,  # use the same colorscale used for the data
    vmax=maxabs,
    cmap="seismic",
    add_colorbar=False,
    transform=ccrs.PlateCarree(),
)

plt.colorbar(tmp, ax=ax, pad=0.11, shrink=0.75)
plt.show()
# -

# Save grid to disk

grid.to_netcdf(os.path.join(results_dir, "south_africa_gravity_grid_iterative.nc"))
