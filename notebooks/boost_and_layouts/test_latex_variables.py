"""
Test functions to create LaTeX variables
"""
import pytest
from .latex_variables import (
    format_power_of_ten,
    create_latex_variable,
    list_to_latex,
    create_loglist,
)


def test_power_of_ten():
    """
    Test format_power_of_ten
    """
    assert format_power_of_ten(1e2) == "10$^{2}$"
    assert format_power_of_ten(1e-5) == "10$^{-5}$"
    assert format_power_of_ten(1e10) == "10$^{10}$"
    assert format_power_of_ten(1e-10) == "10$^{-10}$"
    assert format_power_of_ten(1e0) == "1"
    with pytest.raises(ValueError):
        format_power_of_ten(3.14)


def test_latex_variable():
    """
    Test create_latex_variable
    """
    assert (
        create_latex_variable("my_variable", 3.1416, unit=None)
        == r"\newcommand{\MyVariable}{3.1416}"
    )
    assert (
        create_latex_variable("my_other_variable", 3.141592653589793, unit=None)
        == r"\newcommand{\MyOtherVariable}{3.14159}"
    )
    assert (
        create_latex_variable("my_length", 5400, unit="m")
        == r"\newcommand{\MyLength}{5400~m}"
    )
    assert (
        create_latex_variable("my_area", 200, unit="m$^2$")
        == r"\newcommand{\MyArea}{200~m$^2$}"
    )


def test_list_to_latex():
    """
    Test list_to_latex
    """
    # Check if passing a equidistant list works as expected
    values = [1, 2, 3, 4]  # a short list
    assert list_to_latex(values) == "1, 2, 3 and 4"
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # a long list
    assert list_to_latex(values) == "1 to 10, step size 1"
    # Check if passing a random list works as expected
    values = [1, 3, 8, 12]
    assert list_to_latex(values) == "1, 3, 8 and 12"


def test_loglist():
    """
    Test create_loglist
    """
    values = [1e-5, 1e-4, 1e-3, 1e-2]
    assert create_loglist(values) == r"10$^{-5}$, 10$^{-4}$, 10$^{-3}$ and 10$^{-2}$"
    values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    assert create_loglist(values) == r"10$^{-5}$, 10$^{-4}$,$\dots$, 1"
    values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    assert create_loglist(values) == r"10$^{-5}$, 10$^{-4}$,$\dots$, 10$^{2}$"
