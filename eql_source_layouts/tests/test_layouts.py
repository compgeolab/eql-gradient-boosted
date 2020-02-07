"""
Test functions for layouts constructors
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..layouts import constant_depth, relative_depth, variable_depth, source_bellow_data


@pytest.fixture
def coordinates():
    """
    Return a random set of scattered coordinates
    """
    n_points = 200
    region = (-50e3, 50e3, -50e3, 50e3, 0, 2000)
    np.random.seed(12345)
    easting = np.random.uniform(*region[:2], n_points)
    northing = np.random.uniform(*region[2:4], n_points)
    upward = np.random.uniform(*region[4:], n_points)
    return easting, northing, upward


# -----------
# Depth Types
# -----------


def test_constant_depth(coordinates):
    """
    Test constant depth
    """
    depth = 100
    points = constant_depth(coordinates, depth=depth)
    npt.assert_allclose(coordinates[0], points[0])
    npt.assert_allclose(coordinates[1], points[1])
    npt.assert_allclose(coordinates[2].min() - depth, points[2])


def test_relative_depth(coordinates):
    """
    Test relative depth
    """
    depth = 100
    points = relative_depth(coordinates, depth=depth)
    npt.assert_allclose(coordinates[0], points[0])
    npt.assert_allclose(coordinates[1], points[1])
    npt.assert_allclose(coordinates[2] - depth, points[2])


def test_variable_depth(coordinates):
    """
    Test variable depth
    """
    depth = 100
    depth_factor = 1
    k_nearest = 3
    points = variable_depth(
        coordinates, depth=depth, depth_factor=depth_factor, k_nearest=k_nearest,
    )
    npt.assert_allclose(coordinates[0], points[0])
    npt.assert_allclose(coordinates[1], points[1])
    # Check if the variable term is summed up
    assert not np.allclose(coordinates[2] - depth, points[2])
    # Check if depths are not all equal
    assert not np.allclose(points[2][0], points[2])
    # Check if all masses are bellow relative depth
    assert np.all(points[2] <= coordinates[2] - depth)
    # Set depth_factor equal to zero and check if relative depth is recovered
    points = variable_depth(
        coordinates, depth=depth, depth_factor=0, k_nearest=k_nearest,
    )
    npt.assert_allclose(coordinates[2] - depth, points[2])


# --------------------
# Source distributions
# --------------------

# Sources bellow data
# -------------------
def test_source_bellow_data(coordinates):
    """
    Check if source_bellow_data puts sources beneath data points
    """
    depth = 100
    depth_factor = 1
    k_nearest = 3
    parameters = {
        "constant_depth": {"depth": depth},
        "relative_depth": {"depth": depth},
        "variable_depth": {
            "depth": depth,
            "depth_factor": depth_factor,
            "k_nearest": k_nearest,
        },
    }
    for depth_type, params in parameters.items():
        points = source_bellow_data(coordinates, depth_type=depth_type, **params)
        npt.assert_allclose(points[0], coordinates[0])
        npt.assert_allclose(points[1], coordinates[1])


def test_source_bellow_data_kwargs(coordinates):
    """
    Check if extra kwargs on source_bellow_data are ignored
    """
    source_bellow_data(
        coordinates, depth_type="constant_depth", depth=100, blabla=3.1415
    )
