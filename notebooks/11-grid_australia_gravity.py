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

# +
from pathlib import Path
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
    save_to_json,
)

# -

results_dir = Path("..") / "results" / "australia"

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

plt.figure(figsize=(12, 12))
tmp = plt.scatter(*coordinates[:2], c=coordinates[-1], s=0.01)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="m", shrink=0.7)
plt.show()

# Save gravity disturbance points on a netCDF file to save disk space

# +
coords = {
    "longitude": ("points", coordinates[0]),
    "latitude": ("points", coordinates[1]),
    "height": ("points", coordinates[2]),
}
data_vars = {
    "gravity": ("points", data.gravity[inside]),
    "disturbance": ("points", np.array(disturbance, dtype="float32")),
}

ds = xr.Dataset(data_vars, coords=coords)
# -

ds

ds.to_netcdf(results_dir / "australia-data.nc")

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

# ## Estimate window size for gradient boosting

# We will choose the spacing for the block-averaged sources equal to the spacing for the ultimate grid.
# The final grid will have a spacing of 1 arc-minute, which can be approximated by ~1.8km.

ell.mean_radius * np.radians(1 / 60)

spacing = 1800.0

# Estimate the window size for gradient boosting

# +
window_sizes = np.arange(50e3, 350e3, 25e3)

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

# Choose a window size of 250km so we use around of ~10GB of RAM.

window_size = 225e3

# ## Cross-validate gridder for estimating parameters

# Choose only a portion of the data to apply CV to speed up things

# +
easting_0, northing_0 = 14053825.0, -3451038.0
easting_size, northing_size = 300e3, 300e3
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
tmp = plt.scatter(*proj_coords_cv[:2], c=disturbance_cv, s=0.2)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="mGal", shrink=0.8)
plt.show()

# Define parameters space

# +
depth_type = "relative_depth"
random_state = 0
dampings = np.logspace(-3, 3, 7)
depths = np.linspace(1e3, 10e3, 10)

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
cv = vd.BlockKFold(spacing=5e3, n_splits=6, shuffle=True, random_state=0)

scores, scores_std = [], []
for parameters in parameter_sets:
    points = block_averaged_sources(proj_coords_cv, **parameters)
    eql = EQLHarmonicBoost(
        points=points,
        damping=parameters["damping"],
        window_size=parameters["window_size"],
        random_state=parameters["random_state"],
    )
    start = time.time()
    scores_i = vd.cross_val_score(
        eql,
        proj_coords_cv,
        disturbance_cv,
        cv=cv,
    )
    end = time.time()
    score = np.mean(scores_i)
    score_std = np.std(scores_i)
    print(
        "Last CV took: {:.0f}s. Score: {:.3f}. Score std: {:.6f}".format(
            end - start, score, score_std
        )
    )
    scores.append(score)
    scores_std.append(score_std)
# -

plt.hist(scores)
plt.show()

for score, param in zip(scores, parameter_sets):
    print(score, param)

# +
dampings_m, depths_m = np.meshgrid(dampings, depths)
scores_2d = np.array(scores).reshape(depths_m.shape)

plt.scatter(depths_m, dampings_m, c=scores_2d, s=200, vmax=1, vmin=0.7)
plt.yscale("log")
plt.colorbar()
plt.show()
# -

scores_2d

# +
best_score = np.max(scores)
best_parameters = parameter_sets[np.argmax(scores)]

print(best_score)
print(best_parameters)
# -

# ## Use the subset for gridding on this subregion

points = block_averaged_sources(proj_coords_cv, **best_parameters)

# %%time
eql = EQLHarmonicBoost(
    points=points,
    damping=best_parameters["damping"],
    window_size=best_parameters["window_size"],
    random_state=best_parameters["random_state"],
)
eql.fit(proj_coords_cv, disturbance_cv)

# %%time
# Interpolate on a regular grid
grid = eql.grid(
    upward=data.height.values.max(),
    spacing=spacing,
    data_names=["disturbance"],
)

# +
maxabs = vd.maxabs(grid.disturbance, disturbance_cv)

fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2, figsize=(18, 12), sharex=True, sharey=True
)
grid.disturbance.plot(ax=ax1, add_colorbar=False, vmin=-maxabs, vmax=maxabs)

easting, northing = np.meshgrid(grid.easting, grid.northing)
# ax1.scatter(easting, northing, c="k", s=0.05)
# ax1.scatter(*points[:2], c="k", s=0.05)

tmp = ax2.scatter(
    *proj_coords_cv[:2], c=disturbance_cv, s=20, vmin=-maxabs, vmax=maxabs
)
for ax in (ax1, ax2):
    ax.set_aspect("equal")
plt.show()
# -

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
    spacing=1 / 60,
    projection=projection,
    dims=("latitude", "longitude"),
    data_names=["disturbance"],
)

grid

grid_masked = vd.distance_mask(
    coordinates, maxdist=50e3, grid=grid, projection=projection
)

plt.figure(figsize=(12, 12))
grid_masked.disturbance.plot()
plt.gca().set_aspect("equal")
plt.show()

# +
region = (128, 135, -35, -25)

subgrid = grid_masked.sel(longitude=slice(*region[:2]), latitude=slice(*region[2:]))

inside = vd.inside(coordinates, region)
scatter = [c[inside] for c in coordinates]
disturbance_portion = disturbance[inside]
# +
maxabs = vd.maxabs(subgrid.disturbance.values)

fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2, figsize=(18, 12), sharey=True, sharex=True
)
subgrid.disturbance.plot(ax=ax1, add_colorbar=False)
ax1.set_aspect("equal")

tmp = ax2.scatter(
    *scatter[:2], c=disturbance_portion, s=2, vmin=-maxabs, vmax=maxabs, cmap="RdBu_r"
)
ax2.set_aspect("equal")

cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
fig.colorbar(tmp, cax=cbar_ax, orientation="horizontal", label="mGal")

plt.show()
# -
# Save grid on a netCDF file. First transform some arrays into float32 to save disk space.

grid_masked["upward"] = (
    ("latitude", "longitude"),
    np.array(grid_masked.upward, dtype="float32"),
)
grid_masked["disturbance"] = (
    ("latitude", "longitude"),
    np.array(grid_masked.disturbance, dtype="float32"),
)

grid_masked

grid_masked.to_netcdf(results_dir / "australia-grid.nc")

# Save parameters to a JSON file

# +
variables = {
    "australia_eql_depth": best_parameters["depth"],
    "australia_eql_damping": best_parameters["damping"],
    "australia_eql_spacing": best_parameters["spacing"],
    "australia_eql_window_size": best_parameters["window_size"],
    "australia_eql_n_sources": points[0].size,
    "australia_eql_grid_n_longitude": grid_masked.longitude.size,
    "australia_eql_grid_n_latitude": grid_masked.latitude.size,
    "australia_eql_grid_height": grid_masked.upward.values.ravel()[0],
}

json_file = Path("..") / "results" / "australia.json"
save_to_json(variables, json_file)
