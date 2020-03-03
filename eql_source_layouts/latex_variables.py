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


def latex_parameters(parameters, survey):
    """
    Create list of parameters for each source distribution as LaTeX variables
    """
    tex_variables = []
    for layout in parameters:
        for depth_type in parameters[layout]:
            for parameter in parameters[layout][depth_type]:
                if parameter != "depth_type":
                    values = format_parameters(
                        parameters[layout][depth_type][parameter], parameter
                    )
                    variable_name = format_variable_name(
                        "_".join([survey, layout, depth_type, parameter])
                    )
                    tex_variables.append(latex_variables(variable_name, values))
    return tex_variables


def format_variable_name(name):
    """
    Format Python variable name to LaTeX variable name

    Remove underscores and case the first letter of each word.
    """
    return name.replace("_", " ").title().replace(" ", "")


def format_parameters(parameters_list, parameter_name):
    """
    Format a list of parameters to appear on the table
    """
    fmt = ""
    # Convert spacings and depths from m to km and configure string format
    if parameter_name in ("spacing", "depth"):
        parameters_list = np.array(parameters_list) * 1e-3
        if min(parameters_list) < 1:  # asign formats if min value is lower than 1 km
            fmt = ".2f"
        else:
            fmt = ".0f"
    # Create a string of parameters separated by commas
    values = []
    for p in parameters_list:
        if p is None:
            values.append("{}".format(p))
        elif p == 0:
            values.append("0")
        else:
            values.append("{p:>{fmt}}".format(p=p, fmt=fmt))
    values = ", ".join(values)
    return values
