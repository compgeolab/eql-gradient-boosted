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

# ## Ground and airborne synthetic surveys

survey_ground = pd.read_csv(ground_results_dir / "survey.csv")
survey_airborne = pd.read_csv(airborne_results_dir / "survey.csv")

# +
# Define useful parameters
figsize = (6.66, 2.9)
cbar_args = dict(
    shrink=0.95,
    pad=0.16,
    aspect=40,
    orientation="horizontal",
)
size = 0.5
labels = "a b c d".split()
coords_scale = 1e-3

# Initialize figure and axes
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, sharey=True, figsize=figsize)

tmp = ax1.scatter(
    survey_ground.easting * coords_scale,
    survey_ground.northing * coords_scale,
    c=survey_ground.height,
    cmap="cividis",
    s=size,
)
clb = plt.colorbar(tmp, ax=ax1, **cbar_args)
clb.set_label("meters")

tmp = ax2.scatter(
    survey_ground.easting * coords_scale,
    survey_ground.northing * coords_scale,
    c=survey_ground.g_z,
    cmap="viridis",
    s=size,
)
clb = plt.colorbar(tmp, ax=ax2, **cbar_args)
clb.set_label("mGal")

tmp = ax3.scatter(
    survey_airborne.easting * coords_scale,
    survey_airborne.northing * coords_scale,
    c=survey_airborne.height,
    cmap="cividis",
    s=size,
)
clb = plt.colorbar(tmp, ax=ax3, **cbar_args)
clb.set_label("meters")


tmp = ax4.scatter(
    survey_airborne.easting * coords_scale,
    survey_airborne.northing * coords_scale,
    c=survey_airborne.g_z,
    cmap="viridis",
    s=size,
)
clb = plt.colorbar(tmp, ax=ax4, **cbar_args)
clb.set_label("mGal")

ax1.set_ylabel("northing [km]")
for ax, label in zip((ax1, ax2, ax3, ax4), labels):
    ax.set_aspect("equal")
    ax.set_xlabel("easting [km]")
    ax.annotate(
        label,
        xy=(0.04, 0.94),
        xycoords="axes fraction",
        bbox=dict(boxstyle="circle", fc="white", lw=0.2),
    )
    ax.grid(linestyle="--", linewidth=0.1)

title_args = dict(
    pad=4,
    fontsize="medium",
)
ax1.set_title("Observation points", **title_args)
ax2.set_title("Synthetic gravity", **title_args)
ax3.set_title("Observation points", **title_args)
ax4.set_title("Synthetic gravity", **title_args)

plt.figtext(0.3, 0.9, "Ground survey", horizontalalignment="center", fontsize="large")
plt.figtext(
    0.78, 0.9, "Airborne survey", horizontalalignment="center", fontsize="large"
)

plt.tight_layout(w_pad=0, pad=0)
plt.savefig(
    figs_dir / "synthetic-survey-layouts.pdf",
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
layout_names = ["Sources below data", "Block-averaged sources", "Regular grid sources"]
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
    nrows=3, ncols=3, figsize=(6.66, 6.9), sharex=True, sharey=True
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
        ax.scatter(
            survey_ground.easting,
            survey_ground.northing,
            s=2,
            alpha=0.3,
            color="k",
            linewidths=0,
        )
        ax.set_aspect("equal")
        # Set scientific notation on axis labels (and change offset text position)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        ax.yaxis.offsetText.set_x(-0.16)
        ax.set_xlabel(ax.get_xlabel() + " [m]")
        ax.set_ylabel(ax.get_ylabel() + " [m]")
        # Set title with RMS and number of points
        ax.set_title(
            r"RMS = {:.2f} mGal, sources = {}".format(
                prediction.rms, prediction.n_points
            ),
            fontsize="medium",
            horizontalalignment="center",
            pad=5,
        )

        # Annotate the columns of the figure
        if i == 0:
            ax.text(
                0.5,
                1.13,
                # r"\textbf{{" + depth_type.replace("_", " ").title() + r"}}",
                depth_type.replace("_", " ").capitalize(),
                fontsize="large",
                fontweight="bold",
                horizontalalignment="center",
                transform=ax.transAxes,
            )
        # Annotate the rows of the figure
        if j == 0:
            ax.text(
                -0.33,
                0.5,
                # r"\textbf{{" + layout_names[i] + r"}}",
                layout_names[i],
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
# cbar_ax = fig.add_axes([0.39, 0.075, 0.01, 0.25])
# fig.colorbar(tmp, cax=cbar_ax, orientation="vertical", label=f"Difference between\ntarget and interpolation\n[{field_units}]")
cbar_ax = fig.add_axes([0.49, 0.3, 0.4, 0.01])
cbl = fig.colorbar(tmp, cax=cbar_ax, orientation="horizontal", label=f"{field_units}")
cbl.ax.set_title("Difference between target and interpolated", fontsize="medium")

plt.tight_layout(w_pad=0)
plt.savefig(figs_dir / "ground_survey_differences.pdf", dpi=300)
plt.show()
# -

# ## Gridding airborne survey

# +
layouts = ["source_below_data", "block_averaged_sources", "grid_sources"]
layout_names = ["Sources below data", "Block-averaged sources", "Regular grid sources"]
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
    nrows=3, ncols=3, figsize=(6.66, 6.9), sharex=True, sharey=True
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
        ax.scatter(
            survey_airborne.easting,
            survey_airborne.northing,
            s=1,
            alpha=0.3,
            color="k",
            linewidths=0,
        )
        ax.set_aspect("equal")
        # Set scientific notation on axis labels (and change offset text position)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        ax.yaxis.offsetText.set_x(-0.16)
        ax.set_xlabel(ax.get_xlabel() + " [m]")
        ax.set_ylabel(ax.get_ylabel() + " [m]")
        # Set title with RMS and number of points
        ax.set_title(
            r"RMS = {:.2f} mGal, sources = {}".format(
                prediction.rms, prediction.n_points
            ),
            fontsize="medium",
            horizontalalignment="center",
            pad=5,
        )

        # Annotate the columns of the figure
        if i == 0:
            ax.text(
                0.5,
                1.13,
                # r"\textbf{{" + depth_type.replace("_", " ").title() + r"}}",
                depth_type.replace("_", " ").capitalize(),
                fontsize="large",
                fontweight="bold",
                horizontalalignment="center",
                transform=ax.transAxes,
            )
        # Annotate the rows of the figure
        if j == 0:
            ax.text(
                -0.33,
                0.5,
                # r"\textbf{{" + layout_names[i] + r"}}",
                layout_names[i],
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
# cbar_ax = fig.add_axes([0.39, 0.075, 0.01, 0.25])
# fig.colorbar(tmp, cax=cbar_ax, orientation="vertical", label=f"Difference between\ntarget and interpolation\n[{field_units}]")
cbar_ax = fig.add_axes([0.49, 0.3, 0.4, 0.01])
cbl = fig.colorbar(tmp, cax=cbar_ax, orientation="horizontal", label=f"{field_units}")
cbl.ax.set_title("Difference between target and interpolated", fontsize="medium")

plt.tight_layout(w_pad=0)
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
region = vd.pad_region(
    vd.get_region((australia_data.longitude.values, australia_data.latitude.values)),
    1,
)
lat_ts = australia_data.latitude.mean().values
lon_ts = australia_data.longitude.mean().values

proj_gmt = "M{:.0f}/{:.0f}/5i".format(lon_ts, lat_ts)
# -

maxabs = vd.maxabs(
    australia_data.disturbance.values,
    australia_grid.disturbance.values,
)

# +
fig = pygmt.Figure()

pygmt.config(
    FONT_ANNOT="11p,Helvetica,black",
    FONT_TITLE="15p,Helvetica,black",
    FONT_LABEL="11p,Helvetica,black",
    MAP_FRAME_WIDTH="2p",
)

fig.grdimage(
    "@earth_relief_01m",
    region=region,
    projection=proj_gmt,
    shading="+a45+nt0.7",
    cmap="gray",
)
fig.coast(
    land="#333333",
)
pygmt.makecpt(cmap="polar", series=(-maxabs, maxabs))
fig.plot(
    x=australia_data.longitude,
    y=australia_data.latitude,
    color=australia_data.disturbance,
    style="c0.5p",
    cmap=True,
)
fig.coast(shorelines=True)
fig.basemap(frame=["af", 'WeSN+t"Gravity disturbance observations"'])
with pygmt.config(FONT_ANNOT="9p,Helvetica,black"):
    fig.colorbar(
        box="+gwhite+c-0.1c/0.2c+r0.1c",
        position="jBL+h+w2.6i/0.07i+o0.2i/0.65i",
        frame=['xa50+l"mGal"'],
    )

fig.shift_origin("5.1i", 0)

fig.grdimage(
    "@earth_relief_01m",
    region=region,
    projection=proj_gmt,
    shading="+a45+nt0.7",
    cmap="gray",
)
fig.coast(
    land="#333333",
)
pygmt.makecpt(cmap="polar", series=(-maxabs, maxabs))
fig.grdimage(
    australia_grid.disturbance,
    nan_transparent=True,
)
fig.coast(shorelines=True)
fig.basemap(frame=["af", 'wESN+t"Interpolated grid of gravity disturbances"'])

fig.savefig(figs_dir / "australia.png")
fig.show(width=900)
