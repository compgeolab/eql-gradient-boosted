# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import pyproj
import numpy as np
import pandas as pd
import verde as vd
import harmonica as hm
import matplotlib.pyplot as plt

from eql_source_layouts.layouts import (
    source_bellow_data
)
# -

survey = pd.read_csv(os.path.join("..", "data", "survey_1d.csv"))

survey

# +
coordinates = (survey.easting, np.zeros_like(survey.easting), survey.height)
points = source_bellow_data(coordinates, depth_type="constant_depth", constant_depth=100)

plt.scatter(coordinates[0], coordinates[2])
plt.scatter(points[0], points[2])
plt.show()

# +
coordinates = (survey.easting, np.zeros_like(survey.easting), survey.height)
points = source_bellow_data(coordinates, depth_type="relative_depth", relative_depth=100)

plt.scatter(coordinates[0], coordinates[2])
plt.scatter(points[0], points[2])
plt.show()

# +
coordinates = (survey.easting, np.zeros_like(survey.easting), survey.height)
points = source_bellow_data(coordinates, depth_type="variable_relative_depth", depth_factor=1, depth_shift=-100, k_nearest=3)

plt.scatter(coordinates[0], coordinates[2])
plt.scatter(points[0], points[2])
plt.show()
# -


