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

# # Generate manuscript figures

from pathlib import Path
import xarray as xr
import pandas as pd
import verde as vd
import pygmt
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# ## Load custom matplotlib configuration

plt.style.use(Path(".") / "matplotlib.rc")

# ## Define results directory

results_dir = Path("..") / "results"
ground_results_dir = results_dir / "ground_survey"
airborne_results_dir = results_dir / "airborne_survey"

figs_dir = Path("..") / "manuscript" / "figs"

# ## Ground survey

survey = pd.read_csv(ground_results_dir / "survey.csv")

# +
# Define useful parameters
width = 3.33
figsize = (width, width * 1.7)
cbar_shrink = 0.95
cbar_pad = 0.03
cbar_aspect = 30
size = 2
labels = "a b".split()

# Initialize figure and axes
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=figsize)

# Plot survey points
tmp = ax1.scatter(
    survey.easting, survey.northing, c=survey.height, cmap="cividis", s=size
)
clb = plt.colorbar(
    tmp,
    ax=ax1,
    shrink=cbar_shrink,
    orientation="vertical",
    pad=cbar_pad,
    aspect=cbar_aspect,
)
clb.set_label("m", labelpad=-15, y=1.05, rotation=0)

# Plot measured values
tmp = ax2.scatter(survey.easting, survey.northing, c=survey.g_z, cmap="viridis", s=size)
clb = plt.colorbar(
    tmp,
    ax=ax2,
    shrink=cbar_shrink,
    orientation="vertical",
    pad=cbar_pad,
    aspect=cbar_aspect,
)
clb.set_label("mGal", labelpad=-15, y=1.05, rotation=0)


ax2.set_xlabel("easting [m]")
ax1.tick_params(
    axis="x",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
)

for ax, label in zip((ax1, ax2), labels):
    ax.set_aspect("equal")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.set_ylabel("northing [m]")
    ax.yaxis.offsetText.set_x(-0.2)
    ax.annotate(
        label,
        xy=(0.04, 0.94),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )

ax1.set_title("Ground survey points", pad=3)
ax2.set_title("Observed gravity acceleration", pad=3)


plt.tight_layout(h_pad=0.2)
plt.savefig(
    figs_dir / "ground-survey.pdf",
    bbox_inches="tight",
    dpi=300,
)
# -

# ## Airborne survey

survey = pd.read_csv(airborne_results_dir / "survey.csv")

# +
# Define useful parameters
width = 3.33
figsize = (width, width * 1.7)
cbar_shrink = 0.95
cbar_pad = 0.03
cbar_aspect = 30
size = 2
labels = "a b".split()

# Initialize figure and axes
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=figsize)

# Plot survey points
tmp = ax1.scatter(
    survey.easting, survey.northing, c=survey.height, cmap="cividis", s=size
)
clb = plt.colorbar(
    tmp, ax=ax1, shrink=cbar_shrink, orientation="vertical", pad=0.03, aspect=30
)
clb.set_label("m", labelpad=-15, y=1.05, rotation=0)

# Plot measured values
tmp = ax2.scatter(survey.easting, survey.northing, c=survey.g_z, cmap="viridis", s=size)
clb = plt.colorbar(
    tmp, ax=ax2, shrink=cbar_shrink, orientation="vertical", pad=0.03, aspect=30
)
clb.set_label("mGal", labelpad=-15, y=1.05, rotation=0)

ax2.set_xlabel("easting [m]")
ax1.tick_params(
    axis="x",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
)

for ax, label in zip((ax1, ax2), labels):
    ax.set_aspect("equal")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.set_ylabel("northing [m]")
    ax.yaxis.offsetText.set_x(-0.2)
    ax.annotate(
        label,
        xy=(0.04, 0.94),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )

ax1.set_title("Airborne survey points", pad=3)
ax2.set_title("Observed gravity acceleration", pad=3)

plt.tight_layout(h_pad=0.2)
plt.savefig(
    figs_dir / "airborne-survey.pdf",
    bbox_inches="tight",
    dpi=300,
)
# -

# ## Target grid

target = xr.open_dataarray(results_dir / "target.nc")

# +
width = 3.33
figsize = (width, width * 0.85)
fig, ax = plt.subplots(figsize=figsize)

tmp = target.plot.pcolormesh(
    ax=ax, add_colorbar=False, cmap="viridis", center=False, rasterized=True
)
ax.set_aspect("equal")
ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
ax.set_xlabel(ax.get_xlabel() + " [m]")
ax.set_ylabel(ax.get_ylabel() + " [m]")
clb = plt.colorbar(tmp, ax=ax, shrink=1, orientation="vertical", pad=0.03, aspect=30)
clb.set_label("mGal", labelpad=-15, y=1.05, rotation=0)

ax.set_title("Target grid")
plt.tight_layout()
plt.savefig(
    figs_dir / "target-grid.pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.show()
# -

# ## Gridding ground survey

# +
layouts = ["source_below_data", "block_averaged_sources", "grid_sources"]
field_units = "mGal"

best_predictions = []
for layout in layouts:
    best_predictions.append(
        xr.open_dataset(ground_results_dir / "best_predictions-{}.nc".format(layout))
    )

# +
# We will use the same boundary value for each plot in order to
# show them with the same color scale.
vmax = vd.maxabs(
    *list(
        target - dataset[depth_type]
        for dataset in best_predictions
        for depth_type in dataset
    )
)

# Initialize figure
fig, axes = plt.subplots(
    nrows=3, ncols=3, figsize=(6.66, 6.66), sharex=True, sharey=True
)

# Plot the differences between the target and the best prediction for each layout
for i, (ax_row, dataset) in enumerate(zip(axes, best_predictions)):
    for j, (ax, depth_type) in enumerate(zip(ax_row, dataset)):
        prediction = dataset[depth_type]
        difference = target - prediction
        tmp = difference.plot.pcolormesh(
            ax=ax,
            vmin=-vmax,
            vmax=vmax,
            cmap="seismic",
            add_colorbar=False,
            rasterized=True,
        )
        ax.scatter(survey.easting, survey.northing, s=0.3, alpha=0.2, color="k")
        ax.set_aspect("equal")
        # Set scientific notation on axis labels (and change offset text position)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        ax.yaxis.offsetText.set_x(-0.16)
        ax.set_xlabel(ax.get_xlabel() + " [m]")
        ax.set_ylabel(ax.get_ylabel() + " [m]")
        # Set title with RMS and number of points
        ax.set_title(
            r"RMS: {:.2f} mGal, \#sources: {}".format(
                prediction.rms, prediction.n_points
            ),
            fontsize="small",
            horizontalalignment="center",
        )

        # Annotate the columns of the figure
        if i == 0:
            ax.text(
                0.5,
                1.16,
                r"\textbf{{" + depth_type.replace("_", " ").title() + r"}}",
                fontsize="large",
                fontweight="bold",
                horizontalalignment="center",
                transform=ax.transAxes,
            )
        # Annotate the rows of the figure
        if j == 0:
            ax.text(
                -0.38,
                0.5,
                r"\textbf{{" + dataset.layout.replace("_", " ").title() + r"}}",
                fontsize="large",
                fontweight="bold",
                verticalalignment="center",
                rotation="vertical",
                transform=ax.transAxes,
            )
        # Remove xlabels and ylabels from inner axes
        if i != 2:
            ax.set_xlabel("")
        if j != 0:
            ax.set_ylabel("")

# Hide the last two axes because they are not used
axes[-1][-1].set_visible(False)
axes[-1][-2].set_visible(False)

# Add colorbar
cbar_ax = fig.add_axes([0.38, 0.075, 0.015, 0.24])
fig.colorbar(tmp, cax=cbar_ax, orientation="vertical", label=field_units)

plt.tight_layout()
plt.savefig(figs_dir / "ground_survey_differences.pdf", dpi=300)
plt.show()
# -

# ## Gridding airborne survey

# +
layouts = ["source_below_data", "block_averaged_sources", "grid_sources"]
field_units = "mGal"

best_predictions = []
for layout in layouts:
    best_predictions.append(
        xr.open_dataset(airborne_results_dir / "best_predictions-{}.nc".format(layout))
    )

# +
# We will use the same boundary value for each plot in order to
# show them with the same color scale.
vmax = vd.maxabs(
    *list(
        target - dataset[depth_type]
        for dataset in best_predictions
        for depth_type in dataset
    )
)

# Initialize figure
fig, axes = plt.subplots(
    nrows=3, ncols=3, figsize=(6.66, 6.66), sharex=True, sharey=True
)

# Plot the differences between the target and the best prediction for each layout
for i, (ax_row, dataset) in enumerate(zip(axes, best_predictions)):
    for j, (ax, depth_type) in enumerate(zip(ax_row, dataset)):
        prediction = dataset[depth_type]
        difference = target - prediction
        tmp = difference.plot.pcolormesh(
            ax=ax,
            vmin=-vmax,
            vmax=vmax,
            cmap="seismic",
            add_colorbar=False,
            rasterized=True,
        )
        ax.scatter(survey.easting, survey.northing, s=0.1, alpha=0.2, color="k")
        ax.set_aspect("equal")
        # Set scientific notation on axis labels (and change offset text position)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        ax.yaxis.offsetText.set_x(-0.16)
        ax.set_xlabel(ax.get_xlabel() + " [m]")
        ax.set_ylabel(ax.get_ylabel() + " [m]")
        # Set title with RMS and number of points
        ax.set_title(
            r"RMS: {:.2f} mGal, \#sources: {}".format(
                prediction.rms, prediction.n_points
            ),
            fontsize="small",
            horizontalalignment="center",
        )

        # Annotate the columns of the figure
        if i == 0:
            ax.text(
                0.5,
                1.16,
                r"\textbf{{" + depth_type.replace("_", " ").title() + r"}}",
                fontsize="large",
                fontweight="bold",
                horizontalalignment="center",
                transform=ax.transAxes,
            )
        # Annotate the rows of the figure
        if j == 0:
            ax.text(
                -0.38,
                0.5,
                r"\textbf{{" + dataset.layout.replace("_", " ").title() + r"}}",
                fontsize="large",
                fontweight="bold",
                verticalalignment="center",
                rotation="vertical",
                transform=ax.transAxes,
            )
        # Remove xlabels and ylabels from inner axes
        if i != 2:
            ax.set_xlabel("")
        if j != 0:
            ax.set_ylabel("")

# Hide the last two axes because they are not used
axes[-1][-1].set_visible(False)
axes[-1][-2].set_visible(False)

# Add colorbar
cbar_ax = fig.add_axes([0.38, 0.075, 0.015, 0.24])
fig.colorbar(tmp, cax=cbar_ax, orientation="vertical", label=field_units)

plt.tight_layout()
plt.savefig(figs_dir / "airborne_survey_differences.pdf", dpi=300)
plt.show()
# -

# # Gradient boosted eqls: window size

# +
eql_harmonic_results = pd.read_csv(
    results_dir / "gradient-boosted" / "eql_harmonic.csv"
)

eql_rms = eql_harmonic_results.rms.values[0]
eql_residue = eql_harmonic_results.residue.values[0]
eql_fitting_time = eql_harmonic_results.fitting_time.values[0]
# -

boost_window_size = pd.read_csv(
    results_dir / "gradient-boosted" / "gradient-boosted-window-size.csv",
)

boost_window_size

# +
width = 3.33
figsize = (width, width * 0.85 * 1.5)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
ax1.errorbar(
    boost_window_size.window_size_ratio,
    boost_window_size.rms,
    yerr=boost_window_size.rms_std,
    fmt="o",
    capsize=2,
    label="Gradient-boosted sources",
)
ax1.axhline(eql_rms, linestyle="--", color="C1", label="Regular sources")
ax1.set_ylabel("RMS [mGal]")
ax1.grid()
ax1.legend()

ax2.errorbar(
    boost_window_size.window_size_ratio,
    boost_window_size.fitting_time / eql_fitting_time,
    yerr=boost_window_size.fitting_time_std / eql_fitting_time,
    fmt="o",
    capsize=3,
)
ax2.axhline(1, linestyle="--", color="C1", label="Fitting time of EQLHarmonic")
ax2.set_xlabel("Window size as a fraction of the survey area")
ax2.set_ylabel("Fitting time ratio")
ax2.set_yscale("log")
ax2.set_xlim(0, 0.7)
ax2.grid()
ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
plt.tight_layout()
plt.savefig(figs_dir / "gradient-boosted-window-size.pdf", dpi=300)
plt.show()
# -

# # Gradient boosted eqls: overlapping

boost_overlapping = pd.read_csv(
    results_dir / "gradient-boosted" / "gradient-boosted-overlapping.csv"
)

# +
width = 3.33
figsize = (width, width * 0.85 * 1.5)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)

ax1.errorbar(
    boost_overlapping.overlaps,
    boost_overlapping.rms,
    yerr=boost_overlapping.rms_std,
    fmt="o",
    capsize=2,
    label="Gradient-boosted sources",
)
ax1.axhline(eql_rms, linestyle="--", color="C1", label="Regular sources")
ax1.set_ylabel("RMS [mGal]")
ax1.grid()
ax1.legend()

ax2.errorbar(
    boost_overlapping.overlaps,
    boost_overlapping.fitting_time / eql_fitting_time,
    yerr=boost_overlapping.fitting_time_std / eql_fitting_time,
    fmt="o",
    capsize=3,
)
ax2.axhline(1, linestyle="--", color="C1", label="Fitting time of EQLHarmonic")
ax2.set_xlabel("Overlap")
ax2.set_ylabel("Fitting time ratio")
ax2.set_yscale("log")
ax2.set_xlim(-0.05, 1)
ax2.grid()
ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
plt.tight_layout()
plt.savefig(figs_dir / "gradient-boosted-overlap.pdf", dpi=300)
plt.show()
# -

# ## Australia gravity

australia_data = xr.open_dataset(results_dir / "australia" / "australia-data.nc")
australia_grid = xr.open_dataset(results_dir / "australia" / "australia-grid.nc")

# +
region = vd.get_region(
    (australia_data.longitude.values, australia_data.latitude.values)
)
lat_ts = australia_data.latitude.mean().values
lon_ts = australia_data.longitude.mean().values

proj_gmt = "M{:.0f}/{:.0f}/6.66i".format(lon_ts, lat_ts)

# +
vmin, vmax = australia_data.gravity.values.min(), australia_data.gravity.values.max()

fig = pygmt.Figure()
fig.basemap(region=region, projection=proj_gmt, frame=["afg"])
pygmt.makecpt(cmap="viridis", series=(vmin, vmax))
fig.plot(
    x=australia_data.longitude,
    y=australia_data.latitude,
    color=australia_data.gravity,
    style="c1p",
    cmap=True,
)
fig.colorbar(frame='af+l"Observed gravity [mGal]"')
fig.coast(shorelines=True)
fig.savefig(figs_dir / "australia-data-gravity.png")
fig.show()

# +
maxabs = vd.maxabs(australia_data.disturbance.values)

fig = pygmt.Figure()
fig.basemap(region=region, projection=proj_gmt, frame=["afg"])
pygmt.makecpt(cmap="polar", series=(-maxabs, maxabs))
fig.plot(
    x=australia_data.longitude,
    y=australia_data.latitude,
    color=australia_data.disturbance,
    style="c1p",
    cmap=True,
)
fig.colorbar(frame='af+l"Gravity Disturbance[mGal]"')
fig.coast(shorelines=True)
fig.savefig(figs_dir / "australia-data-disturbance.png")
fig.show()

# +
maxabs = vd.maxabs(australia_grid.disturbance.values)

fig = pygmt.Figure()
fig.basemap(region=region, projection=proj_gmt, frame=["afg"])
pygmt.makecpt(cmap="polar", series=(-maxabs, maxabs))
fig.grdimage(
    australia_grid.disturbance,
    nan_transparent=True,
)
fig.colorbar(frame='af+l"Gravity Disturbance [mGal]"')
fig.coast(shorelines=True)
fig.savefig(figs_dir / "australia-grid.png")
fig.show()
