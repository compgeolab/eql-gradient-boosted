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

# **Import useful packages**

# +
import os
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from eql_source_layouts import (
    block_averaged_sources,
    combine_parameters,
    EQLIterative,
)

# -

# **Define results directory**

results_dir = os.path.join("..", "results")
airborne_results_dir = os.path.join(results_dir, "airborne_survey")

# **Define which field will be meassured**

field = "g_z"
field_units = "mGal"

# ## Read synthetic airborne survey and target grid

# Read airborne survey

survey = pd.read_csv(os.path.join(airborne_results_dir, "survey.csv"))
survey

# Read target grid

target = xr.open_dataarray(os.path.join(results_dir, "target.nc"))
target

# Define coordiantes tuple with the location of the survey points

coordinates = (survey.easting.values, survey.northing.values, survey.height.values)

# Get region of the target grid

region = (
    target.easting.min().values,
    target.easting.max().values,
    target.northing.min().values,
    target.northing.max().values,
)

# ## Grid data with EQLIterative
#
#
# ### Use a window size of 20km
#
# Define gridding parameters

depth_type = "relative_depth"
random_state = 0
block_spacing = 2e3
dampings = np.logspace(-6, 1, 8)
depths = np.arange(1e3, 20e3, 2e3)
window_size = 20e3

# Combine parameters values

parameters = combine_parameters(
    **dict(
        depth_type=depth_type,
        depth=depths,
        damping=dampings,
        spacing=block_spacing,
        window_size=window_size,
        random_state=random_state,
    )
)

# Grid and score the prediction with each set of parameters

scores = []
for params in parameters:
    points = block_averaged_sources(coordinates, **params)
    eql = EQLIterative(
        points=points,
        damping=params["damping"],
        window_size=params["window_size"],
        random_state=params["random_state"],
    )
    eql.fit(coordinates, getattr(survey, field).values)
    grid = eql.grid(
        region=region, shape=target.shape, extra_coords=target.height
    ).scalars
    scores.append(r2_score(grid.values, target.values))

# Get maximum score and the corresponding set of parameters

# +
max_score = np.max(scores)
best_params = parameters[np.argmax(scores)]

print("Max R2 score: {}".format(max_score))
print("Best parameters: {}".format(best_params))
# -

# Obtain grid with the best set of parameters

points = block_averaged_sources(coordinates, **best_params)
eql = EQLIterative(
    points=points,
    damping=best_params["damping"],
    window_size=best_params["window_size"],
    random_state=best_params["random_state"],
)

# +
eql.fit(coordinates, getattr(survey, field).values)
grid = eql.grid(region=region, shape=target.shape, extra_coords=target.height).scalars

print("R2 Score: {}".format(r2_score(grid.values, target.values)))
print("RMS: {}".format(mean_squared_error(grid.values, target.values)))
grid.plot(center=False)
plt.gca().set_aspect("equal")
plt.show()


maxabs = vd.maxabs(grid - target)
(grid - target).plot(cmap="seismic", vmin=-maxabs, vmax=maxabs)
plt.gca().set_aspect("equal")
plt.show()
# -

# Save grid

grid.to_netcdf(os.path.join(results_dir, "gravity_grid_iterative_20km.nc"))

# ### Use a window size of 50km

dampings = np.logspace(-4, 3, 8)
depths = [100, 500, 1e3, 2e3, 5e3, 10e3]
window_size = 50e3

# Combine parameters values

parameters = combine_parameters(
    **dict(
        depth_type=depth_type,
        depth=depths,
        damping=dampings,
        spacing=block_spacing,
        window_size=window_size,
        random_state=random_state,
    )
)

# Grid and score the prediction with each set of parameters

scores = []
for params in parameters:
    points = block_averaged_sources(coordinates, **params)
    eql = EQLIterative(
        points=points,
        damping=params["damping"],
        window_size=params["window_size"],
        random_state=params["random_state"],
    )
    eql.fit(coordinates, getattr(survey, field).values)
    grid = eql.grid(
        region=region, shape=target.shape, extra_coords=target.height
    ).scalars
    scores.append(r2_score(grid.values, target.values))

# Get maximum score and the corresponding set of parameters

# +
max_score = np.max(scores)
best_params = parameters[np.argmax(scores)]

print("Max R2 score: {}".format(max_score))
print("Best parameters: {}".format(best_params))
# -

# Obtain grid with the best set of parameters

points = block_averaged_sources(coordinates, **best_params)
eql = EQLIterative(
    points=points,
    damping=best_params["damping"],
    window_size=best_params["window_size"],
    random_state=best_params["random_state"],
)

# +
eql.fit(coordinates, getattr(survey, field).values)
grid = eql.grid(region=region, shape=target.shape, extra_coords=target.height).scalars

print("R2 Score: {}".format(r2_score(grid.values, target.values)))
print("RMS: {}".format(mean_squared_error(grid.values, target.values)))
grid.plot(center=False)
plt.gca().set_aspect("equal")
plt.show()


maxabs = vd.maxabs(grid - target)
(grid - target).plot(cmap="seismic", vmin=-maxabs, vmax=maxabs)
plt.gca().set_aspect("equal")
plt.show()
# -
# Save grid

grid.to_netcdf(os.path.join(results_dir, "gravity_grid_iterative_50km.nc"))
