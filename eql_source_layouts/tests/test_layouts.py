"""
Test functions for layouts constructors
"""
import pytest
import numpy as np
import numpy.testing as npt
import verde as vd

from ..layouts import (
    constant_depth,
    relative_depth,
    variable_depth,
    source_bellow_data,
    block_median_sources,
    grid_sources,
)


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


def test_variable_depth_analytical(coordinates):
    """
    Test variable depth against analytical solution
    """
    distance = 1.21e3
    easting = np.array([-distance, 0, distance])
    northing = np.array([150, 150, 150], dtype=float)
    upward = np.array([100, 130, 120], dtype=float)
    coordinates = (easting, northing, upward)
    depth = 100
    depth_factor = 1
    k_nearest = 1
    points = variable_depth(
        coordinates, depth=depth, depth_factor=depth_factor, k_nearest=k_nearest
    )
    upward_expected = upward - depth - depth_factor * distance
    npt.assert_allclose(easting, points[0])
    npt.assert_allclose(northing, points[1])
    npt.assert_allclose(upward_expected, points[2])


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
        "constant_depth": {},
        "relative_depth": {},
        "variable_depth": {"depth_factor": depth_factor, "k_nearest": k_nearest},
    }
    for depth_type, params in parameters.items():
        points = source_bellow_data(
            coordinates, depth_type=depth_type, depth=depth, **params
        )
        npt.assert_allclose(points[0], coordinates[0])
        npt.assert_allclose(points[1], coordinates[1])


def test_source_bellow_data_kwargs(coordinates):
    """
    Check if extra kwargs on source_bellow_data are ignored
    """
    depth_type = "constant_depth"
    depth = 100
    npt.assert_allclose(
        source_bellow_data(coordinates, depth_type=depth_type, depth=depth,),
        source_bellow_data(
            coordinates, depth_type=depth_type, depth=depth, blabla=3.1415
        ),
    )


# Block median sources
# --------------------
def test_block_median_sources(coordinates):
    """
    Check if block_median_sources block average coordinates
    """
    spacing = 4000
    depth = 100
    depth_factor = 1
    k_nearest = 3
    parameters = {
        "constant_depth": {},
        "relative_depth": {},
        "variable_depth": {"depth_factor": depth_factor, "k_nearest": k_nearest},
    }
    for depth_type, params in parameters.items():
        points = block_median_sources(
            coordinates, depth_type=depth_type, spacing=spacing, depth=depth, **params
        )
        # Check if there's one source per block
        # We do so by checking if every averaged coordinate is close enough to
        # the center of the block
        block_coords, labels = vd.block_split(
            points, spacing=spacing, region=vd.get_region(coordinates)
        )
        npt.assert_allclose(points[0], block_coords[0][labels], atol=spacing / 2)
        npt.assert_allclose(points[1], block_coords[1][labels], atol=spacing / 2)


def test_block_median_sources_kwargs(coordinates):
    """
    Check if extra kwargs on block_median_sources are ignored
    """
    depth_type = "constant_depth"
    depth = 100
    spacing = 4000
    npt.assert_allclose(
        block_median_sources(
            coordinates, depth_type=depth_type, depth=depth, spacing=spacing,
        ),
        block_median_sources(
            coordinates,
            depth_type=depth_type,
            depth=depth,
            spacing=spacing,
            blabla=3.1415,
        ),
    )


# Grid sources
# ------------
def test_grid_sources(coordinates):
    """
    Check if grid_sources creates a regular grid of sources
    """
    depth = 100
    spacing = 4000
    # Check behaviour with zero padding
    pad = 0
    points = grid_sources(coordinates, spacing=spacing, depth=depth, pad=pad)
    grid = vd.grid_coordinates(vd.get_region(coordinates), spacing=spacing)
    npt.assert_allclose(points[0], grid[0])
    npt.assert_allclose(points[1], grid[1])
    npt.assert_allclose(points[2], coordinates[2].min() - depth)
    # Check behaviour with non zero padding
    pad = 0.1
    points = grid_sources(coordinates, spacing=spacing, depth=depth, pad=pad)
    region = vd.get_region(coordinates)
    w = region[0] - (region[1] - region[0]) * pad
    e = region[1] + (region[1] - region[0]) * pad
    s = region[2] - (region[3] - region[2]) * pad
    n = region[3] + (region[3] - region[2]) * pad
    region = (w, e, s, n)
    grid = vd.grid_coordinates(region, spacing=spacing)
    npt.assert_allclose(points[0], grid[0])
    npt.assert_allclose(points[1], grid[1])
    npt.assert_allclose(points[2], coordinates[2].min() - depth)


def test_grid_sources_kwargs(coordinates):
    """
    Check if extra kwargs on grid_sources are ignored
    """
    depth = 100
    spacing = 4000
    npt.assert_allclose(
        grid_sources(coordinates, depth=depth, spacing=spacing),
        grid_sources(coordinates, depth=depth, spacing=spacing, blabla=3.1415),
    )


# Invalid depth type
# ------------------
def test_invalid_depth_type(coordinates):
    """
    Test invalid depth_type
    """
    depth_type = "not a valid depth_type"
    with pytest.raises(ValueError):
        source_bellow_data(coordinates, depth_type=depth_type)
    with pytest.raises(ValueError):
        block_median_sources(coordinates, depth_type=depth_type, spacing=4000)
