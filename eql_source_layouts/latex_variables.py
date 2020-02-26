"""
Function to define LaTeX variables
"""
import numpy as np


def latex_variables(variable, number, unit=None, fmt=None):
    """
    Create string to define a new LaTeX variable
    """
    if fmt is not None:
        number = "{number:>{fmt}}".format(number=number, fmt=fmt)
    if unit:
        tex_line = r"\newcommand{{\{variable}}}{{\SI{{{number}}}{{{unit}}}}}".format(
            variable=variable, number=number, unit=unit
        )
    else:
        tex_line = r"\newcommand{{\{variable}}}{{{number}}}".format(
            variable=variable, number=number
        )
    return tex_line


def latex_parameters_table(parameters):
    """
    Create a LaTeX table for source distributions parameters
    """
    # Initialize tabular
    tabular_tex = []
    tabular_tex.append(
        r"\begin{{tabular}}{{{table_layout}}}".format(
            table_layout=" ".join(["c"] * (len(parameters.keys()) + 2))
        )
    )
    # Create table titles
    column_titles = list(p.replace("_", " ").title() for p in parameters)
    tabular_tex.append("& & " + " & ".join(column_titles) + r" \\")
    tabular_tex.append(r"\hline")

    # Append values for each depth type
    for depth_type in get_depth_types(parameters):
        # Get parameters names that involve this depth type
        params_names = get_parameters_names(parameters, depth_type)
        tabular_tex.append(
            r"\multirow{{{n_params}}}{{*}}{{\parbox{{0.05\linewidth}}".format(
                n_params=len(params_names)
            )
            + r"{{\centering {depth_type}}}}}".format(
                depth_type=depth_type.replace("_", " ").title()
            )
        )
        # Fill rows with parameter names and values for each source layout
        for param in params_names:
            line = r"    & {parameter_name} ".format(
                parameter_name=param.replace("_", " ").title()
            )
            for layout in parameters:
                if depth_type in parameters[layout]:
                    if param in parameters[layout][depth_type]:
                        values = format_parameters(
                            parameters[layout][depth_type][param], param
                        )
                        line += "& {} ".format(values)
                    else:
                        line += "& - "
            line += r"\\"
            tabular_tex.append(line)
        tabular_tex.append(r"\hline")
    tabular_tex.append(r"\end{tabular}")
    tabular_tex = "\n".join(tabular_tex)
    return tabular_tex


def get_depth_types(parameters):
    """
    Return a list with available depth types without repetition
    """
    depth_types = []
    for layout in parameters:
        for depth_type in parameters[layout]:
            if depth_type not in depth_types:
                depth_types.append(depth_type)
    return depth_types


def get_parameters_names(parameters, depth_type):
    """
    Return a list with available parameters names for a specific depth type
    """
    params_names = []
    for layout in parameters:
        if depth_type in parameters[layout]:
            for p in parameters[layout][depth_type]:
                if p not in params_names and p not in ("damping", "depth_type"):
                    params_names.append(p)
    return params_names


def format_parameters(parameters_list, parameter_name):
    """
    Format a list of parameters to appear on the table
    """
    fmt = ""
    if parameter_name in ("spacing", "depth"):
        parameters_list = np.array(parameters_list) * 1e-3
        if min(parameters_list) < 1:  # asign formats if min value is lower than 1 km
            fmt = ".1f"
        else:
            fmt = ".0f"
    values = ["{p:>{fmt}}".format(p=p, fmt=fmt) for p in parameters_list]
    values = ", ".join(values)
    return values
