"""
Function to define LaTeX variables
"""
import re
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
        tex_string = r"\newcommand{{\{variable_name}}}{{${value} \, {unit}$}}".format(
            variable_name=variable_name, value=value, unit=unit
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
    Format unit string to LaTeX
    """
    # Define regex pattern for units as alphabetic characters followed by
    # a positive or negative int.
    unit_pattern = r"^[a-zA-Z]+-?(\d+)?$"
    # Get each unit from the passed string
    splits = unit.strip().split()
    # Generate the LaTeX units
    units = []
    for split in splits:
        # Check if the passed unit has the valid format
        if not re.match(unit_pattern, split):
            raise ValueError("Invalid unit '{}'.".format(split))
        # Take the alphabetic characters of the unit and its power (if exists)
        alpha = re.findall("[a-zA-Z]+", split)[0]
        power = re.findall(r"-?\d+", split)
        # Build LaTeX unit
        unit_tex = r"\text{{{}}}".format(alpha)
        if power:
            unit_tex += "^{{{}}}".format(power[0])
        units.append(unit_tex)
    return r" \, ".join(units)


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


def create_numlist(values, fmt):
    """
    Create a numlist out of a list of values

    Intended to represent a list of random values.
    """
    values = [value_to_string(v, fmt) for v in values]
    return ", ".join(values[:-1]) + " and {}".format(values[-1])


def create_numrange(values, fmt):
    """
    Create a numrange out of a list of values

    Intended to represent a linspace.
    """
    increment = value_to_string(values[1] - values[0], fmt=fmt)
    vmin = value_to_string(values.min(), fmt=fmt)
    vmax = value_to_string(values.max(), fmt=fmt)
    numrange = "{vmin} to {vmax}".format(vmin=vmin, vmax=vmax)
    numrange += ", step size {increment}".format(increment=increment)
    return numrange


def create_loglist(values):
    """
    Create a list or a range out of a set of logscaled values
    """
    #  Check if values are actually an equispaced logarithmic list
    values_log = np.log10(values)
    differences = values_log[1:] - values_log[:-1]
    assert np.allclose(differences[0], differences)
    # Create loglist
    values = [format_power_of_ten(v) for v in values]
    if len(values) > 4:
        return r"{}, {},$\dots$, {}".format(values[0], values[1], values[-1])
    else:
        return ", ".join(values[:-1]) + " and {}".format(values[-1])


def format_power_of_ten(value):
    """
    Create a LaTeX string for any power of ten

    Examples
    --------
    >>> format_power_of_ten(1e-5)
    ... "10$^{-5}$"

    """
    power = np.log10(value)
    if power != int(power):
        raise ValueError("Passed value '{}' is not a power of 10".format(value))
    if power == 0:
        return "1"
    else:
        return r"10$^{{{value}}}$".format(value=int(power))
