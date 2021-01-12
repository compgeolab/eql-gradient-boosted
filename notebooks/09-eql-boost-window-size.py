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

# Compute RMS of the grid against the target grid and the residue of the gridder (difference between data and predictions on the same observation points)

# +
eql_rms = np.sqrt(mean_squared_error(grid.values, target.values))
diff = survey.g_z - eql.predict((survey.easting, survey.northing, survey.height))
eql_residue = np.sqrt(np.mean(diff ** 2))

print("RMS score: {} mGal".format(eql_rms))
print("Residue: {} mGal".format(eql_residue))
print("Fitting time: {} +/- {} s".format(eql_fitting_time, times.std()))
# -

# Dump results to a csv file

eql_harmonic_results = pd.DataFrame(
    {
        "rms": [eql_rms],
        "fitting_time": [eql_fitting_time],
        "fitting_time_std": [times.std()],
        "residue": [eql_residue],
    }
)
eql_harmonic_results.to_csv(
    results_dir / "gradient-boosted" / "eql_harmonic.csv", index=False
)

# # Grid data with EQLHarmonicBoost using different window sizes

# Define gridding parameters. Use the same depth obtained for EQLHarmonic. The damping might be changed to produce similar quality results.

# +
dampings = np.logspace(-3, 1, 5)

parameters = combine_parameters(
    **dict(
        depth_type=depth_type,
        depth=depth,
        damping=dampings,
        spacing=block_spacing,
    )
)
# -

# Define window sizes and different random states

window_sizes = (2e3, 5e3, 10e3, 20e3, 30e3, 40e3, 50e3, 60e3, 70e3)
random_states = np.arange(10)

# Get the best set of parameters for each window size. Use different random states for determining the RMS of each set of parameters to reduce the effects of random shuffling of the windows.

# +
best_parameters = {}

for window_size in window_sizes:

    rms_mean, rms_std = [], []
    residue_mean, residue_std = [], []
    for params in parameters:
        rms = []
        residue = []
        for random_state in random_states:
            points = block_averaged_sources(coordinates, **params)
            eql = EQLHarmonicBoost(
                points=points,
                damping=params["damping"],
                window_size=window_size,
                random_state=random_state,
            )
            eql.fit(coordinates, getattr(survey, field).values)
            grid = eql.grid(
                upward=target.height, region=region, shape=target.shape
            ).scalars
            rms.append(np.sqrt(mean_squared_error(grid.values, target.values)))
            residue.append(eql.errors_[-1])

        # Compute mean RMS and its std for the current set of parameters
        rms_mean.append(np.mean(rms))
        rms_std.append(np.std(rms))

        # Compute mean residue and its std for the current set of parameters
        residue_mean.append(np.mean(residue))
        residue_std.append(np.std(residue))

    # Get best set of parameters for each window size
    best_rms = np.min(rms_mean)
    argmin = np.argmin(rms_mean)
    best_rms_std = rms_std[argmin]
    best_residue = residue_mean[argmin]
    best_residue_std = residue_std[argmin]
    best_params = parameters[argmin]
    best_parameters[window_size] = {
        "params": best_params,
        "rms_mean": best_rms,
        "rms_std": best_rms_std,
        "residue_mean": best_residue,
        "residue_std": best_residue_std,
    }


# +
rms_mean = [best_parameters[w]["rms_mean"] for w in window_sizes]
rms_std = [best_parameters[w]["rms_std"] for w in window_sizes]

# Get window sizes as fractions of the survey area
region_size = region[1] - region[0]
window_sizes_ratio = window_sizes / region_size

plt.errorbar(
    window_sizes_ratio,
    rms_mean,
    yerr=rms_std,
    fmt="o",
    capsize=3,
    label="RMS of EQLHarmonicBoost",
)
plt.axhline(eql_rms, linestyle="--", color="C1", label="RMS of EQLHarmonic")
plt.xlabel("Window size as a fraction of the survey area")
plt.ylabel("RMS [mGal]")
plt.legend()
plt.show()

# +
residue_mean = [best_parameters[w]["residue_mean"] for w in window_sizes]
residue_std = [best_parameters[w]["residue_std"] for w in window_sizes]

plt.errorbar(
    window_sizes_ratio,
    residue_mean,
    yerr=residue_std,
    fmt="o",
    capsize=3,
    label="Residue of EQLHarmonicBoost",
)
plt.axhline(eql_residue, linestyle="--", color="C1", label="Residue of EQLHarmonic")
plt.xlabel("Window size as a fraction of the survey area")
plt.ylabel("Residue [mGal]")
plt.legend()
plt.show()
# -

# Register fitting time for each window size

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
        random_state=0,
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

plt.errorbar(
    window_sizes_ratio,
    np.array(fitting_times) / eql_fitting_time,
    yerr=np.array(fitting_times_std) / eql_fitting_time,
    fmt="o",
    capsize=3,
)
plt.axhline(1, linestyle="--", color="C1", label="Fitting time of EQLHarmonic")
plt.xlabel("Window size as a fraction of the survey area")
plt.ylabel("Fitting time ratio")
plt.yscale("log")
plt.title("Fitting time of gradient boosted eqls over fitting time of regular eql")
plt.show()

# Dump results to a csv file

gradient_boosted_results = pd.DataFrame(
    {
        "window_size": window_sizes,
        "window_size_ratio": window_sizes_ratio,
        "rms": rms_mean,
        "rms_std": rms_std,
        "fitting_time": fitting_times,
        "fitting_time_std": fitting_times_std,
    }
)
gradient_boosted_results.to_csv(
    results_dir / "gradient-boosted" / "gradient-boosted-window-size.csv",
    index=False,
)

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
