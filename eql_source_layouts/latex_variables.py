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


def latex_best_parameters(best_parameters, survey, layout, depth_type):
    """
    Create LaTeX variables for best parameters
    """
    tex_lines = []
    for parameter, value in best_parameters.items():
        if parameter in ("depth_type", "layout", "height"):
            continue
        variable_name = format_variable_name(
            "_".join(["best", survey, layout, depth_type, parameter])
        )
        if parameter == "damping":
            value = r"\num{{e{:.0f}}}".format(np.log10(value))
            fmt = None
        elif parameter == "score":
            fmt = ".3f"
        else:
            fmt = _determine_fmt(value)
        tex_lines.append(latex_variables(variable_name, value, fmt=fmt))
    return tex_lines


def latex_parameters(parameters, survey):
    """
    Create list or ranges of parameters for each source distribution as LaTeX variables
    """
    tex_variables = []
    for layout in parameters:
        for depth_type in parameters[layout]:
            for parameter in parameters[layout][depth_type]:
                if parameter != "depth_type":
                    values, increment = format_parameters(
                        parameters[layout][depth_type][parameter], parameter
                    )
                    variable_name = format_variable_name(
                        "_".join([survey, layout, depth_type, parameter])
                    )
                    tex_variables.append(latex_variables(variable_name, values))
                    tex_variables.append(
                        latex_variables(variable_name + "Increment", increment)
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

    # Check if parameters are given in ranges and create numrages if # parameter > 4
    differences = parameters[1:] - parameters[:-1]
    if np.allclose(differences, differences[0]) and len(parameters) > 4:
        values, increment = _create_numrange(parameters)
    else:
        values, increment = _create_numlist(parameters)
    return values, increment


def _create_numrange(parameters):
    """
    Create range of values using LaTeX numrange
    """
    increment = parameters[1] - parameters[0]
    # Set format
    params_min_max = []
    for value in (parameters.min(), parameters.max()):
        fmt = _determine_fmt(value)
        params_min_max.append("{v:>{fmt}}".format(v=value, fmt=fmt))
    values = r"\numrange{{{min}}}{{{max}}}".format(
        min=params_min_max[0], max=params_min_max[1]
    )
    fmt = _determine_fmt(increment)
    increment = r", step size \num{{{increment:>{fmt}}}}".format(
        increment=increment, fmt=fmt
    )
    return values, increment


def _determine_fmt(value, decimal_digits=1):
    """
    Determine the proper format for float numbers

    We assume that value is > 0.
    """
    if value == 0:
        fmt = ".0f"
    else:
        if value < 1:
            fmt = ".{:d}f".format(decimal_digits)
        else:
            fmt = ".0f"
    return fmt


def _create_numlist(parameters):
    """
    Create list of values using LaTeX numlist
    """
    values = []
    for value in parameters:
        fmt = _determine_fmt(value)
        values.append("{v:>{fmt}}".format(v=value, fmt=fmt))
    values = ";".join(values)
    values = r"\numlist{{{values}}}".format(values=values)
    increment = ""
    return values, increment


def _format_damping(parameters):
    """
    Create range of values for damping parameters
    """
    parameters = np.log10(parameters)
    differences = parameters[1:] - parameters[:-1]
    increment = differences[1] - differences[0]
    assert np.allclose(differences[0], differences)
    values = ["e{:.0f}".format(p) for p in parameters]
    if len(values) > 4:
        values = [r"\num{{{v}}}".format(v=v) for v in values]
        values = r"{}, {},$\dots$, {}".format(values[0], values[1], values[-1])
    else:
        values = ";".join(values)
        values = r"\numlist{{{values}}}".format(values=values)
    increment = ""
    return values, increment
