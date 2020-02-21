"""
Function to define LaTeX variables
"""


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
