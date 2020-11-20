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
    save_to_json,
)

# -

# **Define results directory**

results_dir = Path("..") / "results"
airborne_results_dir = results_dir / "airborne_survey"

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
# Define gridding parameters

depth_type = "relative_depth"
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
    )
)

# Grid and score the gridder with each set of parameters

rms = []
for params in parameters:
    points = block_averaged_sources(coordinates, **params)
    eql = hm.EQLHarmonic(
        points=points,
        damping=params["damping"],
    )
    eql.fit(coordinates, getattr(survey, field).values)
    grid = eql.grid(upward=target.height, region=region, shape=target.shape).scalars
    rms.append(np.sqrt(mean_squared_error(grid.values, target.values)))

# Get maximum score and the corresponding set of parameters

# +
eql_rms = np.min(rms)
best_params = parameters[np.argmin(rms)]

print("Best RMS score: {}".format(eql_rms))
print("Best parameters: {}".format(best_params))
# -

# Track time of the fitting process

# +
points = block_averaged_sources(coordinates, **best_params)
eql = hm.EQLHarmonic(
    points=points,
    damping=best_params["damping"],
)

start = time.time()
eql.fit(coordinates, getattr(survey, field).values)
end = time.time()

eql_fitting_time = end - start
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
    for params in parameters:
        points = block_averaged_sources(coordinates, **params)
        eql = EQLHarmonicBoost(
            points=points,
            damping=params["damping"],
            window_size=window_size,
            random_state=params["random_state"],
        )
        eql.fit(coordinates, getattr(survey, field).values)
        grid = eql.grid(upward=target.height, region=region, shape=target.shape).scalars
        rms.append(np.sqrt(mean_squared_error(grid.values, target.values)))

    # Keep only the set of parameters that achieve the best score
    best_rms = np.min(rms)
    best_params = parameters[np.argmin(rms)]
    best_parameters[window_size] = {"params": best_params, "rms": best_rms}
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
# -

# Register fitting time for each window size

fitting_times = {}
for window_size in window_sizes:
    params = best_parameters[window_size]["params"]
    points = block_averaged_sources(coordinates, **params)
    eql = EQLHarmonicBoost(
        points=points,
        damping=params["damping"],
        window_size=window_size,
        random_state=params["random_state"],
    )
    start = time.time()
    eql.fit(coordinates, getattr(survey, field).values)
    end = time.time()
    fitting_times[window_size] = end - start

fitting_times

plt.plot(
    fitting_times.keys(),
    fitting_times.values(),
    "o",
    label="Fitting time of EQLHarmonicBoost",
)
plt.axhline(
    eql_fitting_time, linestyle="--", color="C1", label="Fitting time of EQLHarmonic"
)
plt.xlabel("Window size [m]")
plt.ylabel("Fitting time [s]")
plt.yscale("log")
plt.legend()
plt.show()
