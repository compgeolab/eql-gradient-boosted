# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.0
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

# # Read EQLHarmonic results for reference

eql_harmonic_results = pd.read_csv(
    results_dir / "gradient-boosted" / "eql_harmonic.csv"
)

eql_rms = eql_harmonic_results.rms.values[0]
eql_residue = eql_harmonic_results.residue.values[0]
eql_fitting_time = eql_harmonic_results.fitting_time.values[0]

# # Grid data with EQLHarmonicBoost using different overlappings

# Define gridding parameters. Use the same depth obtained for EQLHarmonic and a window size of 30km. The damping might be changed to produce similar quality results.

# +
depth_type = "relative_depth"
block_spacing = 2e3
depth = 9e3
dampings = np.logspace(-3, 1, 5)
window_size = 30e3

# Save window size on a dictionary to save its value to JSON file
variables = {"boost_overlapping_window_size": window_size}

parameters = combine_parameters(
    **dict(
        depth_type=depth_type,
        depth=depth,
        damping=dampings,
        spacing=block_spacing,
        window_size=window_size,
    )
)
# -

# Define overlappings and different random states

overlaps = np.arange(0, 1, 0.05)
random_states = np.arange(10)

# Find the best set of parameters for each window size

# +
best_parameters = {}

for overlapping in overlaps:

    # Grid and score the gridders for each combination of parameters
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
                window_size=params["window_size"],
                random_state=random_state,
            )
            eql.overlapping = overlapping
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
    best_parameters[overlapping] = {
        "params": best_params,
        "rms_mean": best_rms,
        "rms_std": best_rms_std,
        "residue_mean": best_residue,
        "residue_std": best_residue_std,
    }

# +
rms_mean = [best_parameters[o]["rms_mean"] for o in overlaps]
rms_std = [best_parameters[o]["rms_std"] for o in overlaps]

plt.errorbar(
    overlaps,
    rms_mean,
    yerr=rms_std,
    fmt="o",
    capsize=3,
    label="RMS of EQLHarmonicBoost",
)
plt.axhline(eql_rms, linestyle="--", color="C1", label="RMS of EQLHarmonic")
plt.xlabel("Overlapping")
plt.ylabel("RMS [mGal]")
plt.legend()
plt.show()

# +
residue_mean = [best_parameters[o]["residue_mean"] for o in overlaps]
residue_std = [best_parameters[o]["residue_std"] for o in overlaps]

plt.errorbar(
    overlaps,
    residue_mean,
    yerr=residue_std,
    fmt="o",
    capsize=3,
    label="Residue of EQLHarmonicBoost",
)
plt.axhline(eql_residue, linestyle="--", color="C1", label="Residue of EQLHarmonic")
plt.xlabel("Overlapping")
plt.ylabel("Residue [mGal]")
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
        random_state=0,
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

rms_relative = rms_mean / eql_rms
time_relative = np.array(fitting_times) / eql_fitting_time

# +
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(6, 6))

ax1.plot(overlaps, rms_relative, "o", label="Relative RMS", c="C0")
ax2.plot(overlaps, time_relative, "o", label="Relative fitting time", c="C1")
ax1.legend()
ax2.legend()
# ax2.set_ylim(-0.5, 2)
for ax in (ax1, ax2):
    ax.axhline(1, linestyle="--", color="black")
    ax.grid()
ax2.set_yscale("log")
plt.tight_layout()
plt.show()

# +
objective = (rms_relative + time_relative) / 2

plt.plot(overlaps, objective, "o")
plt.grid()
plt.ylim(0.5, 3)
plt.title("0.5 (Relative fitting time + relative RMS)")
plt.show()
# -

tmp = plt.scatter(rms_relative, time_relative, c=overlaps, s=50)
plt.grid()
plt.xlabel("RMS Relative")
plt.ylabel("Relative fitting time")
plt.colorbar(tmp, label="Overlapping")
for rms_i, fitting_i, overlapping_i in zip(rms_relative, time_relative, overlaps):
    plt.annotate(
        "{:.2f}".format(overlapping_i),
        (rms_i, fitting_i),
        xytext=(3, 3),
        xycoords="data",
        textcoords="offset points",
    )
plt.show()

# +
xlim = (1, 2)
ylim = (0, 10)

tmp = plt.scatter(rms_relative, time_relative, c=overlaps, s=50)
plt.grid()
plt.xlabel("RMS Relative")
plt.ylabel("Relative fitting time")
plt.colorbar(tmp, label="Overlapping")
plt.xlim(xlim)
plt.ylim(ylim)
for rms_i, fitting_i, overlapping_i in zip(rms_relative, time_relative, overlaps):
    if rms_i < xlim[1] and fitting_i < ylim[1]:
        plt.annotate(
            "{:.2f}".format(overlapping_i),
            (rms_i, fitting_i),
            xytext=(3, 3),
            xycoords="data",
            textcoords="offset points",
        )
plt.show()
# -

# ## Dump variables dictionary to a JSON file

json_file = results_dir / "boost-overlapping.json"
save_to_json(variables, json_file)

# ## Dump results to a CSV file

gradient_boosted_results = pd.DataFrame(
    {
        "overlaps": overlaps,
        "rms": rms_mean,
        "rms_std": rms_std,
        "fitting_time": fitting_times,
        "fitting_time_std": fitting_times_std,
    }
)
gradient_boosted_results.to_csv(
    results_dir / "gradient-boosted" / "gradient-boosted-overlapping.csv",
    index=False,
)
