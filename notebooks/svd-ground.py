# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python [conda env:eql_source_layouts]
#     language: python
#     name: conda-env-eql_source_layouts-py
# ---

# # Singular Value Decomposition on Ground Survey

# +
from IPython.display import display
import os
import numpy as np
import pandas as pd
import xarray as xr
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt

import eql_source_layouts

# +
# Define results directory
results_dir = os.path.join("..", "results", "ground_survey")

# Define a list of source layouts
layouts = ["source_bellow_data", "block_median_sources", "grid_sources"]
# -

# ## Read the synthetic ground survey

# +
survey = pd.read_csv(os.path.join(results_dir, "survey.csv"))
display(survey)

coordinates = (survey.easting.values, survey.northing.values, survey.height.values)
# -

fig, ax = plt.subplots(figsize=(6, 6))
tmp = ax.scatter(survey.easting, survey.northing, c=survey.height, s=6)
plt.colorbar(tmp, ax=ax, label="m")
ax.set_aspect("equal")
ax.set_title("Height of ground survey points")
plt.show()

# ## Read best predictions

best_predictions = []
for layout in layouts:
    best_predictions.append(
        xr.open_dataset(
            os.path.join(results_dir, "best_predictions-{}.nc".format(layout))
        )
    )

# ## Compute singular values for best predictions

# +
singular_values = {layout: {} for layout in layouts}

for dataset in best_predictions:
    for depth_type in dataset:
        prediction = dataset[depth_type]
        print("Processing {} with {}".format(prediction.layout, depth_type))
        points = getattr(eql_source_layouts.layouts, prediction.layout)(
            coordinates, **prediction.attrs
        )
        points = vd.base.n_1d_arrays(points, 3)
        eql = hm.EQLHarmonic(points=points)
        jac = np.matrix(eql.jacobian(coordinates, eql.points))
        _, sv, _ = np.linalg.svd(jac)
        singular_values[prediction.layout][depth_type] = sv
# -

# Save singular values arrays to files

for layout in singular_values:
    for depth_type in singular_values[layout]:
        sv = singular_values[layout][depth_type]
        np.savetxt(
            os.path.join(results_dir, "svd", "{}-{}.txt".format(layout, depth_type)), sv
        )

# Read singular values arrays

# +
singular_values = {layout: {} for layout in layouts}

for dataset in best_predictions:
    for depth_type in dataset:
        prediction = dataset[depth_type]
        layout = prediction.layout
        singular_values[layout][depth_type] = np.loadtxt(
            os.path.join(results_dir, "svd", "{}-{}.txt".format(layout, depth_type))
        )

# +
fig, axes = plt.subplots(
    ncols=3, figsize=(2 * 3.33 * 1.618, 3.33), sharey=True, sharex=True
)

for ax, layout in zip(axes, singular_values):
    for depth_type in singular_values[layout]:
        sv = singular_values[layout][depth_type]
        ax.plot(sv, label=depth_type)
        ax.set_yscale("log")
        ax.grid()
        ax.set_title(layout)
        ax.legend()

plt.tight_layout()
plt.show()

# +
for layout in singular_values:
    for depth_type in singular_values[layout]:
        sv = singular_values[layout][depth_type]
        plt.plot(sv, label="{} {}".format(layout, depth_type))

plt.yscale("log")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
