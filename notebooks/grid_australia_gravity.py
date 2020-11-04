# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python [conda env:eql_source_layouts]
#     language: python
#     name: conda-env-eql_source_layouts-py
# ---

# +
import pooch
import pyproj
import numpy as np
import xarray as xr
import boule as bl
import verde as vd
import matplotlib.pyplot as plt

from source_layouts import EQLIterative, block_averaged_sources
# -

# ## Download Australia gravity data

# +
fname = pooch.retrieve(
    url="https://github.com/compgeolab/australia-gravity-data/releases/download/v1.0/australia-ground-gravity.nc",
    known_hash="sha256:50f2fa53c5dc2c66dd3358b8e50024d21074fcc77c96191c549a10a37075bc7e",
    downloader=pooch.HTTPDownloader(progressbar=True)
)

# Load the data with xarray
data = xr.load_dataset(fname)
# -

data

plt.figure(figsize=(12, 12))
tmp = plt.scatter(data.longitude, data.latitude, c=data.gravity, s=0.01)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="mGal")
plt.show()

# ## Compute gravity disturbance

ell = bl.WGS84
disturbance = data.gravity - ell.normal_gravity(data.latitude, data.height)
data["disturbance"] = ("point", disturbance)

data

plt.figure(figsize=(12, 12))
tmp = plt.scatter(data.longitude, data.latitude, c=data.disturbance, s=0.01)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="mGal")
plt.show()

# ## Block reduce the data

reducer = vd.BlockReduce(np.median, spacing=0.02, drop_coords=False)
coordinates, disturbance = reducer.filter(
    (data.longitude.values, data.latitude.values, data.height.values),
    data=data.disturbance.values,
)
coordinates[0].size

# ## Keep only points close to the continent

vd.get_region(coordinates)

inside = vd.inside(coordinates, region=(111, 154, -44, -7))
coordinates = tuple(c[inside] for c in coordinates)
disturbance = disturbance[inside]

vd.get_region(coordinates)

plt.figure(figsize=(12, 12))
tmp = plt.scatter(*coordinates[:2], c=disturbance, s=0.01)
plt.gca().set_aspect("equal")
plt.colorbar(tmp, label="mGal")
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
plt.colorbar(tmp, label="mGal")
plt.show()

# ## Grid gravity disturbance

depth_type = "relative_depth"
random_state = 0
block_spacing = 8e3
damping = 1e-1
depth = 2e3
window_size = 500e3

points = block_averaged_sources(proj_coordinates, depth_type=depth_type, spacing=block_spacing, depth=depth)

# +
memory_gb = proj_coordinates[0].size * points[0].size * (64 / 8) / 1024 ** 3

print("Number of data points:", proj_coordinates[0].size)
print("Number of sources:", points[0].size)
print("Memory needed to store the full Jacobian matrix: {:.2f} GB".format(memory_gb))
# -

# %%time
eql = EQLIterative(damping=damping, points=points, window_size=window_size, random_state=random_state)
eql.fit(proj_coordinates, disturbance)

# %%time
# Get region of longitude, latitude coordinates (in degrees)
region = vd.get_region(coordinates)
# Interpolate on a regular grid on geographic coordinates
grid = eql.grid(region=region, spacing=0.02, extra_coords=data.height.values.max(), projection=projection)

grid

grid_masked = vd.distance_mask(coordinates, maxdist=80e3, grid=grid, projection=projection)

plt.figure(figsize=(12, 12))
grid_masked.scalars.plot()
plt.gca().set_aspect("equal")
plt.show()


