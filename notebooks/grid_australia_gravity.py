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
import time
import pooch
import pyproj
import numpy as np
import xarray as xr
import boule as bl
import verde as vd
import matplotlib.pyplot as plt

from boost_and_layouts import (
    EQLHarmonicBoost,
    block_averaged_sources,
    combine_parameters,
)

# -

# ## Download Australia gravity data

# +
fname = pooch.retrieve(
    url="https://github.com/compgeolab/australia-gravity-data/releases/download/v1.0/australia-ground-gravity.nc",
    known_hash="sha256:50f2fa53c5dc2c66dd3358b8e50024d21074fcc77c96191c549a10a37075bc7e",
    downloader=pooch.HTTPDownloader(progressbar=True),
)

# Load the data with xarray
data = xr.load_dataset(fname)
# -

data

plt.figure(figsize=(12, 12))
tmp = plt.scatter(data.longitude, data.latitude, c=data.gravity, s=0.01)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="mGal", shrink=0.7)
plt.show()

# ## Compute gravity disturbance

ell = bl.WGS84
disturbance = data.gravity - ell.normal_gravity(data.latitude, data.height)
data["disturbance"] = ("point", disturbance)

data

plt.figure(figsize=(12, 12))
tmp = plt.scatter(data.longitude, data.latitude, c=data.disturbance, s=0.01)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="mGal", shrink=0.7)
plt.show()

# ## Keep only points close to the continent

# +
coordinates = (data.longitude.values, data.latitude.values, data.height.values)
disturbance = data.disturbance.values

vd.get_region(coordinates)
# -

inside = vd.inside(coordinates, region=(111, 154, -44, -7))
coordinates = tuple(c[inside] for c in coordinates)
disturbance = disturbance[inside]

vd.get_region(coordinates)

plt.figure(figsize=(12, 12))
tmp = plt.scatter(*coordinates[:2], c=disturbance, s=0.01)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="mGal", shrink=0.7)
plt.show()

# ## Project coordinates

# +
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.values.mean())

easting, northing = projection(*coordinates[:2])
proj_coordinates = (easting, northing, coordinates[-1])
# -

plt.figure(figsize=(12, 12))
tmp = plt.scatter(*proj_coordinates[:2], c=disturbance, s=0.01)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="mGal", shrink=0.7)
plt.show()

# ## Estimate parameters: spacing for block-averaged sources and window size for gradient boosting

# Estimate the block spacing that we will use for block-averaged sources

# Get number of data points
n_data = proj_coordinates[0].size
print("Number of data points: {}".format(n_data))

# +
spacings = np.linspace(4e3, 10e3, 7)

n_data_per_sources = []
for spacing in spacings:
    sources = block_averaged_sources(
        proj_coordinates, spacing=spacing, depth_type="relative_depth", depth=0
    )
    n_data_per_sources.append(n_data / sources[0].size)
# -

plt.plot(spacings, n_data_per_sources, "o")
plt.xlabel("Block spacing [m]")
plt.ylabel("# data points / # sources")
plt.title("Number of data points per sources")
plt.grid()
plt.show()

# Lets choose a block spacing of 9000m so we obtain ~20 data points per source

spacing = 9000

# Estimate the window size for gradient boosting

# +
window_sizes = np.linspace(100e3, 1000e3, 10)

# Create sources with the spacing obtained before
sources = block_averaged_sources(
    proj_coordinates, spacing=spacing, depth_type="relative_depth", depth=0
)

memory_gb = []
for window_size in window_sizes:
    eql = EQLHarmonicBoost(window_size=window_size)
    eql.points_ = sources
    source_windows, data_windows = eql._create_rolling_windows(proj_coordinates)
    # Get the size of each source and data windows
    source_sizes = np.array([w.size for w in source_windows])
    data_sizes = np.array([w.size for w in data_windows])
    # Compute the size of the Jacobian matrix for each window
    jacobian_sizes = source_sizes * data_sizes
    # Register the amount of memory to store the Jacobian matrix (double precision)
    memory_gb.append(jacobian_sizes.max() * (64 / 8) / 1024 ** 3)
# -

plt.plot(window_sizes * 1e-3, memory_gb, "o")
plt.xlabel("Window size [km]")
plt.ylabel("Memory [GB]")
plt.title("Memory needed to store the larger Jacobian matrix")
plt.grid()
plt.show()

# ### Conclusions:
#
# - Choose a spacing of 9000m so we obtain ~20 data points per source
# - Choose a window size of 600km so we don't exceed 10GB of RAM.

window_size = 600e3
spacing = 9e3

# ## Cross-validate gridder for estimating parameters

# Choose only a portion of the data to apply CV to speed up things

# +
easting_0, northing_0 = 13783825.0, -3661038.0
easting_size, northing_size = 1000e3, 1000e3
smaller_region = (
    easting_0,
    easting_0 + easting_size,
    northing_0,
    northing_0 + northing_size,
)

inside = vd.inside(proj_coordinates, region=smaller_region)
proj_coords_cv = tuple(c[inside] for c in proj_coordinates)
disturbance_cv = disturbance[inside]

print(f"Number of data points for CV: {proj_coords_cv[0].size}")
print(f"Small region: {vd.get_region(proj_coords_cv)}")
# -

plt.figure(figsize=(12, 12))
tmp = plt.scatter(*proj_coords_cv[:2], c=disturbance_cv, s=0.01)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="mGal", shrink=0.8)
plt.show()

# Define parameters space

# +
depth_type = "relative_depth"
random_state = 0
dampings = np.logspace(-2, 3, 6)
depths = [1e3, 2e3, 5e3, 10e3, 15e3]

# Combine these parameters
parameter_sets = combine_parameters(
    **dict(
        depth_type=depth_type,
        depth=depths,
        damping=dampings,
        spacing=spacing,
        window_size=window_size,
        random_state=random_state,
    )
)
print("Number of combinations:", len(parameter_sets))
# -

# Apply cross validation

# +
# %%time
cv = vd.BlockKFold(spacing=50e3, n_splits=3, shuffle=True, random_state=0)

scores = []
for parameters in parameter_sets:
    points = block_averaged_sources(proj_coords_cv, **parameters)
    eql = EQLHarmonicBoost(
        points=points,
        damping=parameters["damping"],
        window_size=parameters["window_size"],
        random_state=parameters["random_state"],
        line_search=True,
    )
    start = time.time()
    score = np.mean(
        vd.cross_val_score(
            eql,
            proj_coords_cv,
            disturbance_cv,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
    )
    end = time.time()
    print("Last CV took: {:.0f}s".format(end - start))
    scores.append(score)

# +
best_score = np.max(scores)
best_parameters = parameter_sets[np.argmax(scores)]

print(best_score)
print(best_parameters)
# -

# ## Cross validate using the entire dataset

# %%time
points = block_averaged_sources(proj_coordinates, **best_parameters)
eql = EQLHarmonicBoost(
    points=points,
    damping=best_parameters["damping"],
    window_size=window_size,
    random_state=best_parameters["random_state"],
    line_search=True,
)
scores = vd.cross_val_score(
    eql,
    proj_coordinates,
    disturbance,
    cv=cv,
    scoring="neg_root_mean_squared_error",
)

np.mean(scores)

# ## Grid gravity disturbance

points = block_averaged_sources(proj_coordinates, **best_parameters)

# +
memory_gb = proj_coordinates[0].size * points[0].size * (64 / 8) / 1024 ** 3

print("Number of data points:", proj_coordinates[0].size)
print("Number of sources:", points[0].size)
print("Memory needed to store the full Jacobian matrix: {:.2f} GB".format(memory_gb))
# -

# %%time
eql = EQLHarmonicBoost(
    points=points,
    damping=best_parameters["damping"],
    window_size=best_parameters["window_size"],
    random_state=best_parameters["random_state"],
    line_search=True,
)
eql.fit(proj_coordinates, disturbance)

plt.plot(eql.errors_)
plt.show()

# %%time
# Get region of longitude, latitude coordinates (in degrees)
region = vd.get_region(coordinates)
# Interpolate on a regular grid on geographic coordinates
grid = eql.grid(
    upward=data.height.values.max(),
    region=region,
    spacing=0.02,
    projection=projection,
)

grid

grid_masked = vd.distance_mask(
    coordinates, maxdist=80e3, grid=grid, projection=projection
)

plt.figure(figsize=(12, 12))
grid_masked.scalars.plot()
plt.gca().set_aspect("equal")
plt.show()
