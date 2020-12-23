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

# # Convert variables stored in JSON files to LaTeX variables

# +
from IPython.display import display
from pathlib import Path
import json
import numpy as np
import xarray as xr

from boost_and_layouts import (
    create_latex_variable,
    create_loglist,
    list_to_latex,
    format_variable_name,
)


# +
def parameters_to_latex(parameters, survey):
    """
    Generate a list of latex variables from a set of parameters
    """
    parameters_to_latex = []

    for layout in parameters:
        for depth_type in parameters[layout]:
            for parameter in parameters[layout][depth_type]:

                if parameter == "depth_type":
                    continue
                elif parameter == "damping":
                    values = create_loglist(parameters[layout][depth_type][parameter])
                else:
                    values = list_to_latex(parameters[layout][depth_type][parameter])
                variable_name = format_variable_name(
                    "_".join([survey, layout, depth_type, parameter])
                )
                parameters_to_latex.append(
                    r"\newcommand{{\{variable_name}}}{{{values}}}".format(
                        variable_name=variable_name, values=values
                    )
                )
    return parameters_to_latex


def format_damping(variable_name, value):
    """
    Convert damping to a LaTeX variable
    """
    variable_name = format_variable_name(variable_name)
    value = "e{:.0f}".format(np.log10(value))
    return r"\newcommand{{\{variable_name}}}{{\num{{{value}}}}}".format(
        variable_name=variable_name, value=value
    )


def best_parameters_to_latex(parameters, survey):
    """
    Convert best parameters to LaTeX variables

    Parameters
    ----------
    parameters : dict
        Dictionary containing the parameters of the best prediction.
    survey : str
        Name of the gridded survey. Eg. ``"ground"``, ``"airborne"``.
    """
    latex_variables = []
    layout = parameters["layout"]
    depth_type = parameters["depth_type"]
    for key, value in parameters.items():
        if key in ["metadata", "depth_type", "layout"]:
            continue

        variable_name = "_".join(["best", survey, layout, depth_type, key])
        if key == "damping":
            variable = format_damping(variable_name, value)
        elif key == "rms":
            variable = create_latex_variable(variable_name, value, fmt=".2f")
        else:
            variable = create_latex_variable(variable_name, value)
        latex_variables.append(variable)
    return latex_variables


# -

# ## Define results and manuscript directories

results_dir = Path("..") / "results"
manuscript_dir = Path("..") / "manuscript"
ground_results_dir = results_dir / "ground_survey"
airborne_results_dir = results_dir / "airborne_survey"

# ## Source layouts schematics

# +
json_file = results_dir / "source-layouts-schematics.json"
tex_file = manuscript_dir / "source-layouts-schematics.tex"

with open(json_file, "r") as f:
    variables = json.loads(f.read())
# -

latex_variables = [
    create_latex_variable(key, value) for key, value in variables.items()
]
with open(tex_file, "w") as f:
    f.write("\n".join(latex_variables))

# ## Synthetic survey and target grid

# +
json_file = results_dir / "synthetic-surveys.json"
tex_file = manuscript_dir / "synthetic-surveys.tex"

with open(json_file, "r") as f:
    variables = json.loads(f.read())

# +
units = variables.copy()

units["n_prisms"] = None
units["model_easting"] = "meter"
units["model_northing"] = "meter"
units["model_depth"] = "meter"
units["model_min_density"] = "kg per cubic meter"
units["model_max_density"] = "kg per cubic meter"
units["survey_easting"] = "meter"
units["survey_northing"] = "meter"
units["survey_noise"] = "milli Gal"
units["ground_survey_points"] = None
units["ground_survey_min_height"] = "meter"
units["ground_survey_max_height"] = "meter"
units["airborne_survey_points"] = None
units["airborne_survey_min_height"] = "meter"
units["airborne_survey_max_height"] = "meter"
units["target_height"] = "meter"
units["target_spacing"] = "meter"
units["target_easting_size"] = None
units["target_northing_size"] = None
# -

latex_variables = [
    create_latex_variable(key, value, unit=units[key])
    for key, value in variables.items()
]
display(latex_variables)

with open(tex_file, "w") as f:
    f.write("\n".join(latex_variables))

# ## Gridding ground survey

# +
json_file = results_dir / "parameters-ground-survey.json"
tex_file = manuscript_dir / "parameters-ground-survey.tex"

with open(json_file, "r") as f:
    parameters = json.loads(f.read())
# -

latex_variables = parameters_to_latex(parameters, survey="ground")
display(latex_variables)

with open(tex_file, "w") as f:
    f.write("\n".join(latex_variables))

# ### Ground survey: best predictions

# +
best_predictions = [
    xr.open_dataset(f)
    for f in ground_results_dir.iterdir()
    if "best_predictions" in f.name
]

latex_variables = []
for dataset in best_predictions:
    for array in dataset:
        latex_variables.extend(
            best_parameters_to_latex(dataset[array].attrs, survey="ground")
        )

tex_file = manuscript_dir / "best-parameters-ground-survey.tex"
with open(tex_file, "w") as f:
    f.write("\n".join(latex_variables))
# -

# ## Gridding airborne survey

# +
json_file = results_dir / "parameters-airborne-survey.json"
tex_file = manuscript_dir / "parameters-airborne-survey.tex"

with open(json_file, "r") as f:
    parameters = json.loads(f.read())
# -

latex_variables = parameters_to_latex(parameters, survey="airborne")
display(latex_variables)

with open(tex_file, "w") as f:
    f.write("\n".join(latex_variables))

# ### Airborne survey: best predictions

# +
best_predictions = [
    xr.open_dataset(f)
    for f in airborne_results_dir.iterdir()
    if "best_predictions" in f.name
]

latex_variables = []
for dataset in best_predictions:
    for array in dataset:
        latex_variables.extend(
            best_parameters_to_latex(dataset[array].attrs, survey="airborne")
        )

tex_file = manuscript_dir / "best-parameters-airborne-survey.tex"
with open(tex_file, "w") as f:
    f.write("\n".join(latex_variables))
# -

# ## Gradient boosted sources

# +
json_file = results_dir / "boost-overlapping.json"
tex_file = manuscript_dir / "boost-overlapping.tex"

with open(json_file, "r") as f:
    variables = json.loads(f.read())

units["boost_overlapping_window_size"] = "meter"
# -

latex_variables = [
    create_latex_variable(key, value, unit=units[key])
    for key, value in variables.items()
]
display(latex_variables)

with open(tex_file, "w") as f:
    f.write("\n".join(latex_variables))
