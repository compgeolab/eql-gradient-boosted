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
    Create list or ranges of parameters for each source distribution as LaTeX variables
    """
    tex_variables = []
    for layout in parameters:
        for depth_type in parameters[layout]:
            for parameter in parameters[layout][depth_type]:
                if parameter != "depth_type":
                    values, interval = format_parameters(
                        parameters[layout][depth_type][parameter], parameter
                    )
                    variable_name = format_variable_name(
                        "_".join([survey, layout, depth_type, parameter])
                    )
                    tex_variables.append(latex_variables(variable_name, values))
                    tex_variables.append(
                        latex_variables(variable_name + "Interval", interval)
                    )
    return tex_variables


def format_variable_name(name):
    """
    Format Python variable name to LaTeX variable name

    Remove underscores and case the first letter of each word.
    """
    return name.replace("_", " ").title().replace(" ", "")


def format_parameters(parameters, parameter_name):
    """
    Format a list or range of parameters to appear on the table
    """
    parameters = np.array(parameters, dtype=float)

    # Format damping
    if parameter_name == "damping":
        return _format_damping(parameters)

    # Convert depths and spacings to km
    if parameter_name in ("depth", "spacing"):
        parameters *= 1e-3

    # Check if parameters are given in ranges
    differences = parameters[1:] - parameters[:-1]
    if np.allclose(differences, differences[0]):
        values, interval = _create_numrange(parameters)
    else:
        values, interval = _create_numlist(parameters)
    return values, interval


def _create_numrange(parameters):
    """
    Create range of values using LaTeX numrange
    """
    interval = parameters[1] - parameters[0]
    # Set format
    if parameters.min() < 1:
        fmt = ".1f"
    else:
        fmt = ".0f"
    values = r"\numrange{{{min:>{fmt}}}}{{{max:>{fmt}}}}".format(
        min=parameters.min(), max=parameters.max(), fmt=fmt
    )
    interval = r"\num{{{interval:>{fmt}}}}".format(interval=interval, fmt=fmt)
    return values, interval


def _create_numlist(parameters):
    """
    Create list of values using LaTeX numlist
    """
    values = []
    for value in parameters:
        if value == 0:
            fmt = ".0f"
        if value < 1:
            fmt = ".1f"
        else:
            fmt = ".0f"
        values.append("{v:>{fmt}}".format(v=value, fmt=fmt))
    values = ";".join(values)
    values = r"\numlist{{{values}}}".format(values=values)
    interval = "-"
    return values, interval


def _format_damping(parameters):
    """
    Create range of values for damping parameters
    """
    fmt = ".0f"
    parameters = np.log10(parameters)
    differences = parameters[1:] - parameters[:-1]
    interval = differences[1] - differences[0]
    assert np.allclose(differences[0], differences)
    values = r"\numrange{{10$^{min:>{fmt}}$}}{{10$^{max:>{fmt}}$}}".format(
        min=parameters.min(), max=parameters.max(), fmt=fmt
    )
    interval = r"\num{{10$^{interval:>{fmt}}$}}".format(interval=interval, fmt=fmt)
    return values, interval
