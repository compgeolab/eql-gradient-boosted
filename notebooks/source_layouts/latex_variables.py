"""
Function to define LaTeX variables
"""
import numpy as np


def create_latex_variable(variable_name, value, unit=None, fmt="5g"):
    """
    Create string to define a new LaTeX variable
    """
    variable_name = format_variable_name(variable_name)
    if fmt is not None:
        value = value_to_string(value, fmt)
    if unit:
        unit = format_unit(unit)
        tex_string = (
            r"\newcommand{{\{variable_name}}}{{\SI{{{value}}}{{{unit}}}}}".format(
                variable_name=variable_name, value=value, unit=unit
            )
        )
    else:
        tex_string = r"\newcommand{{\{variable_name}}}{{{value}}}".format(
            variable_name=variable_name, value=value
        )
    return tex_string


def value_to_string(value, fmt):
    """
    Convert numerical value to string with a specific format
    """
    return "{value:>{fmt}}".format(value=value, fmt=fmt).strip()


def format_variable_name(name):
    """
    Format Python variable name to LaTeX variable name

    Remove underscores and case the first letter of each word.
    """
    return name.replace("_", " ").title().replace(" ", "")


def format_unit(unit):
    """
    Format unit string to work well with SIunits LaTeX packcage
    """
    return " ".join(["\\" + u for u in unit.strip().split()])


def list_to_latex(values, max_list_items=4, fmt="5g"):
    """
    Generate a numrange or numlist from a list of values
    """
    # Convert list to np.array
    values = np.array(values)
    # Sort values
    values.sort()
    # Check if we must create a numrange or numlist
    diffs = values[1:] - values[:-1]
    if np.allclose(diffs, diffs[0]) and values.size > max_list_items:
        return create_numrange(values, fmt)
    else:
        return create_numlist(values, fmt)


def create_numrange(values, fmt):
    """
    Create a numrange out of a list of values

    Intended to represent a linspace.
    """
    increment = value_to_string(values[1] - values[0], fmt=fmt)
    vmin = value_to_string(values.min(), fmt=fmt)
    vmax = value_to_string(values.max(), fmt=fmt)
    numrange = r"\numrange{{{min}}}{{{max}}}".format(min=vmin, max=vmax)
    numrange += r", step size \num{{{increment}}}".format(increment=increment)
    return numrange


def create_loglist(values):
    """
    Create a list or a range out of a set of logscaled values
    """
    values = np.log10(values)
    differences = values[1:] - values[:-1]
    assert np.allclose(differences[0], differences)
    values = ["e{:.0f}".format(v) for v in values]
    if len(values) > 4:
        values = [r"\num{{{v}}}".format(v=v) for v in values]
        values = r"{}, {},$\dots$, {}".format(values[0], values[1], values[-1])
    else:
        values = ";".join(values)
        values = r"\numlist{{{values}}}".format(values=values)
    return values


def create_numlist(values, fmt):
    """
    Create a numlist out of a list of values

    Intended to represent a list of random values.
    """
    numlist = ";".join([value_to_string(v, fmt) for v in values])
    numlist = r"\numlist{{{v}}}".format(v=numlist)
    return numlist
