"""
Test EQLIterative class
"""
import numpy as np
import numpy.testing as npt
import verde as vd
import harmonica as hm
from sklearn.metrics import mean_squared_error

from ..eql_iterative import EQLIterative
from .. import block_averaged_sources


def test_eql_iterative_single_window():
    """
    Check if EQLIterative works with a single window that covers the whole region
    """
    # Define a squared region
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(40, 40), extra_coords=0)
    # Get synthetic data
    data = hm.point_mass_gravity(coordinates, points, masses, field="g_z")

    # The interpolation should be perfect on the data points
    eql = EQLIterative(window_size=region[1] - region[0])
    eql.fit(coordinates, data)
    npt.assert_allclose(data, eql.predict(coordinates), rtol=1e-5)

    # Gridding onto a denser grid should be reasonably accurate when compared
    # to synthetic values
    grid = vd.grid_coordinates(region=region, shape=(60, 60), extra_coords=0)
    true = hm.point_mass_gravity(grid, points, masses, field="g_z")
    npt.assert_allclose(true, eql.predict(grid), rtol=1e-3)


def test_eql_iterative_large_region():
    """
    Check if EQLIterative works on a large region

    The iterative process ignores the effect of sources on far observation
    points. If the region is very large, this error should be diminished.
    """
    # Define a squared region
    region = (-1000e3, 1000e3, -1000e3, 1000e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(40, 40), extra_coords=0)
    # Get synthetic data
    data = hm.point_mass_gravity(coordinates, points, masses, field="g_z")

    # The interpolation should be sufficiently accurate on the data points
    eql = EQLIterative(window_size=100e3)
    eql.fit(coordinates, data)
    assert mean_squared_error(data, eql.predict(coordinates)) < 1e-5 * vd.maxabs(data)

    # Gridding onto a denser grid should be reasonably accurate when compared
    # to synthetic values
    grid = vd.grid_coordinates(region=region, shape=(60, 60), extra_coords=0)
    true = hm.point_mass_gravity(grid, points, masses, field="g_z")
    assert mean_squared_error(true, eql.predict(grid)) < 1e-3 * vd.maxabs(data)


def test_eql_iterative_random_state():
    """
    Check if EQLIterative produces same result by setting random_state
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=0)
    # Get synthetic data
    data = hm.point_mass_gravity(coordinates, points, masses, field="g_z")

    # Initialize two EQLIterative with the same random_state
    eql_a = EQLIterative(window_size=500, random_state=0)
    eql_a.fit(coordinates, data)
    eql_b = EQLIterative(window_size=500, random_state=0)
    eql_b.fit(coordinates, data)

    # Check if fitted coefficients are the same
    npt.assert_allclose(eql_a.coefs_, eql_b.coefs_)


def test_same_number_of_windows_data_and_sources():
    """
    Check if it creates the same number of windows for data and sources
    """
    spacing = 1
    # Create data points on a large region
    region = (1, 3, 1, 3)
    coordinates = vd.grid_coordinates(region=region, spacing=spacing, extra_coords=0)
    # Create source points on a smaller region
    sources_region = (1.5, 2.5, 1.5, 2.5)
    points = vd.grid_coordinates(
        region=sources_region, spacing=spacing, extra_coords=-10
    )
    # Create EQLIterative
    eql = EQLIterative(window_size=spacing)
    # Make EQL believe that it has already created the points
    eql.points_ = points
    # Create windows for data points and sources
    source_windows, data_windows = eql._create_rolling_windows(coordinates)
    # Check if number of windows are the same
    assert len(source_windows) == len(data_windows)


def test_eql_iterative_warm_start():
    """
    Check if EQLIterative can be fitted with warm_start
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=0)
    # Get synthetic data
    data = hm.point_mass_gravity(coordinates, points, masses, field="g_z")

    # Check if refitting with warm_start=True changes its coefficients
    eql = EQLIterative(window_size=500, warm_start=True)
    eql.fit(coordinates, data)
    coefs = eql.coefs_.copy()
    eql.fit(coordinates, data)
    assert not np.allclose(coefs, eql.coefs_)

    # Check if refitting with warm_start=False doesn't change its coefficients
    # (need to set random_state, otherwise coefficients might be different due
    # to another random shuffling of the windows).
    eql = EQLIterative(window_size=500, warm_start=False, random_state=0)
    eql.fit(coordinates, data)
    coefs = eql.coefs_.copy()
    eql.fit(coordinates, data)
    npt.assert_allclose(coefs, eql.coefs_)


def test_eql_iterative_warm_start_new_coords():
    """
    Check if sources are not changed when warm_start is True

    If the gridder is already fitted (and has warm_start=True), the location of
    the sources must not be modified, even if the new fitting process is
    carried out with another set of observation points.
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=0)
    # Get synthetic data
    data = hm.point_mass_gravity(coordinates, points, masses, field="g_z")

    # Capture the location of sources created by EQLIterative on the first fit
    eql = EQLIterative(window_size=500, warm_start=True)
    eql.fit(coordinates, data)
    sources = tuple(c.copy() for c in eql.points_)

    # Fit the gridder with a new set of observation data
    coordinates_new = (coordinates[0] + 100, coordinates[1] - 100, coordinates[2])
    data_new = hm.point_mass_gravity(coordinates_new, points, masses, field="g_z")
    eql.fit(coordinates_new, data_new)
    # Check if sources remain on the same location
    for i in range(3):
        npt.assert_allclose(sources[i], eql.points_[i])


def test_eql_iterative_custom_points():
    """
    Check EQLIterative with custom points

    Check if the iterative gridder works well with a custom set of sources.
    By default, the gridder puts one source beneath each data point, therefore
    the indices of data and sources windows are identical. When passing
    a custom set of sources, these indices may differ.
    """
    region = (-3e3, -1e3, 5e3, 7e3)
    # Build synthetic point masses
    points = vd.grid_coordinates(region=region, shape=(6, 6), extra_coords=-1e3)
    masses = vd.datasets.CheckerBoard(amplitude=1e13, region=region).predict(points)
    # Define a set of observation points
    coordinates = vd.grid_coordinates(region=region, shape=(20, 20), extra_coords=0)
    # Get synthetic data
    data = hm.point_mass_gravity(coordinates, points, masses, field="g_z")

    # Define a set of block-averaged equivalent sources
    sources = block_averaged_sources(
        coordinates, 100, depth_type="relative_depth", depth=500
    )

    # Fit EQLIterative with the block-averaged sources
    eql = EQLIterative(window_size=500, points=sources)
    eql.fit(coordinates, data)

    # Check if sources are located on the same points
    for i in range(3):
        npt.assert_allclose(sources[i], eql.points_[i])

    # The interpolation should be sufficiently accurate on the data points
    assert mean_squared_error(data, eql.predict(coordinates)) < 1e-3 * vd.maxabs(data)
