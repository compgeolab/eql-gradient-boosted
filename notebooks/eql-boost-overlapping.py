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

# # Compare performance of EQLHarmonicBoost with different overlapping

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

# ## Grid data with EQLHarmonicBoost using different overlappings
#
# Define gridding parameters

depth_type = "relative_depth"
random_state = int(0)
block_spacing = 2e3
window_size = 30e3
dampings = np.logspace(-6, 1, 8)
depths = np.arange(1e3, 20e3, 2e3)

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

# Find the best set of parameters for each window size

# +
# Define overlappings
overlaps = np.arange(0, 1, 0.05)

best_parameters = {}
for overlapping in overlaps:

    # Grid and score the gridders for each combination of parameters
    rms = []
    for params in parameters:
        points = block_averaged_sources(coordinates, **params)
        eql = EQLHarmonicBoost(
            points=points,
            damping=params["damping"],
            window_size=params["window_size"],
            random_state=params["random_state"],
        )
        eql.overlapping = overlapping
        eql.fit(coordinates, getattr(survey, field).values)
        grid = eql.grid(upward=target.height, region=region, shape=target.shape).scalars
        rms.append(np.sqrt(mean_squared_error(grid.values, target.values)))

    # Keep only the set of parameters that achieve the best score
    best_rms = np.min(rms)
    best_params = parameters[np.argmin(rms)]
    best_parameters[overlapping] = {"params": best_params, "rms": best_rms}
# -

best_parameters

# +
rms = [best_parameters[o]["rms"] for o in overlaps]

plt.plot(overlaps, rms, "o", label="RMS of EQLHarmonicBoost")
plt.axhline(eql_rms, linestyle="--", color="C1", label="RMS of EQLHarmonic")
plt.xlabel("Overlapping")
plt.ylabel("RMS [mGal]")
plt.legend()
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
for overlapping in overlaps:
    params = best_parameters[overlapping]["params"]
    points = block_averaged_sources(coordinates, **params)
    eql = EQLHarmonicBoost(
        points=points,
        damping=params["damping"],
        window_size=params["window_size"],
        random_state=params["random_state"],
    )
    eql.overlapping = overlapping

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
            data_names=["{:.2f}".format(overlapping)],
        )
    )
# -

fitting_times

fitting_times_std

plt.errorbar(
    overlaps,
    np.array(fitting_times) / eql_fitting_time,
    yerr=np.array(fitting_times_std) / eql_fitting_time,
    fmt="o",
    capsize=3,
)
plt.axhline(1, linestyle="--", color="C1", label="Fitting time of EQLHarmonic")
plt.xlabel("Overlapping")
plt.ylabel("Fitting time ratio")
plt.yscale("log")
plt.title("Fitting time of gradient boosted eqls over fitting time of regular eql")
plt.show()

rms_relative = rms / eql_rms
time_relative = np.array(fitting_times) / eql_fitting_time

# +


plt.plot(overlaps, rms_relative, 'o', label="Relative RMS")
plt.plot(overlaps, time_relative, 'o', label="Relative fitting time")
plt.legend()
# plt.yscale("log")
plt.ylim(-1, 5)
plt.show()

# +
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(6, 6))

ax1.plot(overlaps, rms_relative, 'o', label="Relative RMS", c="C0")
ax2.plot(overlaps, time_relative, 'o', label="Relative fitting time", c="C1")
ax1.legend()
ax2.legend()
ax2.set_ylim(-0.5, 2)
for ax in (ax1, ax2):
    ax.axhline(1, linestyle="--", color="black")
    ax.grid()
# plt.yscale("log")
plt.tight_layout()
plt.show()

# +
objective = rms_relative + time_relative

plt.plot(overlaps, objective, 'o')
plt.grid()
plt.ylim(1, 5)
plt.title("Relative fitting time + relative RMS")
plt.show()
# -


