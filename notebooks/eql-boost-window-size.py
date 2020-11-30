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

# # Compare performance of EQLHarmonicBoost with different window size

# **Import useful packages**

# +
from pathlib import Path
import time
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from boost_and_layouts import (
    block_averaged_sources,
    combine_parameters,
    EQLHarmonicBoost,
)

# -

# **Define results directory**

results_dir = Path("..") / "results"
airborne_results_dir = results_dir / "airborne_survey"
eql_boost_results_dir = results_dir / "eql-boost"

# **Define which field will be meassured**

field = "g_z"
field_units = "mGal"

# ## Read synthetic airborne survey and target grid

# Read airborne survey

survey = pd.read_csv(airborne_results_dir / "survey.csv")
survey

# Read target grid

target = xr.open_dataarray(results_dir / "target.nc")
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

# # Grid data with EQLHarmonic for reference on performance
#
# Use the best of parameters for block-averaged sources with relative depth, which was obtained on a previous notebook.

depth_type = "relative_depth"
block_spacing = 2e3
damping = 1e-3
depth = 9e3

# Grid the data using `hm.EQLHarmonic` and track time of the fitting process

# +
points = block_averaged_sources(
    coordinates, depth_type=depth_type, spacing=block_spacing, depth=depth
)
eql = hm.EQLHarmonic(
    points=points,
    damping=damping,
)

n_runs = 10
times = np.empty(n_runs)
eql.fit(coordinates, getattr(survey, field).values)  # fit to compile Numba functions
for i in range(n_runs):
    start = time.time()
    eql.fit(coordinates, getattr(survey, field).values)
    end = time.time()
    times[i] = end - start

eql_fitting_time = times.mean()

grid = eql.grid(upward=target.height, region=region, shape=target.shape).scalars
# -

# Compute RMS of the grid against the target grid

# +
eql_rms = np.sqrt(mean_squared_error(grid.values, target.values))

print("RMS score: {} mGal".format(eql_rms))
print("Fitting time: {} +/- {} s".format(eql_fitting_time, times.std()))
# -

# ## Grid data with EQLHarmonicBoost using different window sizes
#
# Define gridding parameters

depth_type = "relative_depth"
random_state = int(0)
block_spacing = 2e3
dampings = np.logspace(-6, 1, 8)
depths = np.arange(1e3, 20e3, 2e3)

# Combine parameters values

parameters = combine_parameters(
    **dict(
        depth_type=depth_type,
        depth=depths,
        damping=dampings,
        spacing=block_spacing,
        random_state=random_state,
    )
)

# Find the best set of parameters for each window size

# +
# Define window sizes
window_sizes = (2e3, 5e3, 10e3, 20e3, 30e3, 40e3, 50e3, 60e3, 70e3)

best_parameters = {}
for window_size in window_sizes:

    # Grid and score the gridders for each combination of parameters
    rms = []
    residual_rms = []
    for params in parameters:
        points = block_averaged_sources(coordinates, **params)
        eql = EQLHarmonicBoost(
            points=points,
            damping=params["damping"],
            window_size=window_size,
            random_state=params["random_state"],
        )
        eql.fit(coordinates, getattr(survey, field).values)
        residual_rms.append(eql.errors_[-1])
        grid = eql.grid(upward=target.height, region=region, shape=target.shape).scalars
        rms.append(np.sqrt(mean_squared_error(grid.values, target.values)))

    # Keep only the set of parameters that achieve the best score
    best_rms = np.min(rms)
    best_params = parameters[np.argmin(rms)]
    residual_rms = residual_rms[np.argmin(rms)]
    best_parameters[window_size] = {
        "params": best_params, "rms": best_rms, "residual_rms": residual_rms
    }
# -

best_parameters

# +
rms = [best_parameters[w]["rms"] for w in window_sizes]

plt.plot(window_sizes, rms, "o", label="RMS of EQLHarmonicBoost")
plt.axhline(eql_rms, linestyle="--", color="C1", label="RMS of EQLHarmonic")
plt.xlabel("Window size [m]")
plt.ylabel("RMS [mGal]")
plt.legend()
plt.show()

# +
residuals = [best_parameters[w]["residual_rms"] for w in window_sizes]

plt.plot(window_sizes, residuals, "o")
plt.xlabel("Window size [m]")
plt.ylabel("RMS of residuals [mGal]")
plt.grid()
plt.show()
# -

# Grid data with the best set of parameters per window size and register the fitting time for each one.

# +
# Define how many times each gridder will be fitted to get a statistic of fitting times
n_runs = 10
times = np.empty(n_runs)

grids = []
fitting_times = []
fitting_times_std = []
for window_size in window_sizes:
    params = best_parameters[window_size]["params"]
    points = block_averaged_sources(coordinates, **params)
    eql = EQLHarmonicBoost(
        points=points,
        damping=params["damping"],
        window_size=window_size,
        random_state=params["random_state"],
    )

    # Register mean fitting time and its std
    for i in range(n_runs):
        start = time.time()
        eql.fit(coordinates, getattr(survey, field).values)
        end = time.time()
        times[i] = end - start

    fitting_times.append(times.mean())
    fitting_times_std.append(times.std())

    # Grid data
    grids.append(
        eql.grid(
            upward=target.height,
            region=region,
            shape=target.shape,
            data_names=["{:.0f}".format(window_size)],
        )
    )
# -

fitting_times

fitting_times_std

plt.errorbar(
    window_sizes,
    np.array(fitting_times) / eql_fitting_time,
    yerr=np.array(fitting_times_std) / eql_fitting_time,
    fmt="o",
    capsize=3,
)
plt.axhline(1, linestyle="--", color="C1", label="Fitting time of EQLHarmonic")
plt.xlabel("Window size [m]")
plt.ylabel("Fitting time ratio")
plt.yscale("log")
plt.title("Fitting time of gradient boosted eqls over fitting time of regular eql")
plt.show()

ds = xr.merge(grids)

ds

# +
maxabs = max([vd.maxabs(ds[grid] - target) for grid in ds])

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), sharex=True, sharey=True)
axes = axes.ravel()

for grid, ax in zip(ds, axes):
    diff = ds[grid] - target
    tmp = diff.plot(
        ax=ax, vmin=-maxabs, vmax=maxabs, cmap="seismic", add_colorbar=False
    )
    ax.set_aspect("equal")
    ax.set_title(f"Window size: {grid} m")


cbar_ax = fig.add_axes([0.15, 0.06, 0.73, 0.02])
fig.colorbar(tmp, cax=cbar_ax, orientation="horizontal", label="mGal")
plt.show()
# -

ds.to_netcdf(eql_boost_results_dir / "airborne_grid_boost_grids.nc")
