"""
Create different layouts of point sources
"""
import numpy as np
from verde import median_distance, BlockReduce, get_region, pad_region, grid_coordinates


DEPTH_TYPES = ["constant_depth", "relative_depth", "variable_depth"]


def _dispatcher(points, depth_type, **kwargs):
    """
    Dispatch points to functions that change its upward component based on depth_type
    """
    if depth_type not in DEPTH_TYPES:
        raise ValueError("Invalid depth type '{}'.".format(depth_type))
    if depth_type == "constant_depth":
        return set_constant_depth(points, **kwargs)
    elif depth_type == "relative_depth":
        return set_relative_depth(points, **kwargs)
    elif depth_type == "variable_depth":
        return set_variable_depth(points, **kwargs)


def source_bellow_data(
    coordinates, depth_type, **kwargs,
):
    """
    Put one point source beneath each observation point

    The depth of the point sources will be the upward coordinate of the corresponding
    observation point minus a relative depth. This relative depth can either be constant
    or variable. In case it's variable, it will be equal to a quantity proportional to
    the median distance to the k nearest neighbours. The variable depth can
    also be static shifted by a constant depth.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    depth_type : str
        Type of depth distribution for source points.
        Availables types: ``"constant_depth"``, ``"relative_depth"``,
        ``"variable_depth"``.
    kwargs

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    points = tuple(np.atleast_1d(i).copy() for i in coordinates)
    points = _dispatcher(points, depth_type, **kwargs)
    return points


def block_median_sources(
    coordinates, spacing, depth_type, **kwargs,
):
    """
    Put one point source beneath the block-median observation points

    The depth of the point sources will be the upward coordinate of the corresponding
    block reduced observation point minus a relative depth. This relative depth can
    either be constant or variable. In case it's variable, it will be equal to
    a quantity proportional to the median distance to the k nearest neighbours. The
    variable relative depth can also be static shifted by a constant depth.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    spacing : float, tuple = (s_north, s_east)
        The block size in the South-North and West-East directions, respectively.
        A single value means that the size is equal in both directions.
    depth_type : str
        Type of depth distribution for source points.
        Availables types: ``"constant_depth"``, ``"relative_depth"``,
        ``"variable_relative_depth"``.
    kwargs

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    reducer = BlockReduce(spacing=spacing, reduction=np.median)
    (easting, northing), upward = reducer.filter(coordinates[:2], coordinates[2])
    points = (easting, northing, upward)
    points = _dispatcher(points, depth_type, **kwargs)
    return points


def grid_sources(coordinates, spacing=None, constant_depth=None, pad=None, **kwargs):
    """
    Create a regular grid of point sources

    All point sources will be located at the same depth, computed as the mean height of
    data points minus the ``constant_depth``.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    spacing : float, tuple = (s_north, s_east)
        The block size in the South-North and West-East directions, respectively.
        A single value means that the size is equal in both directions.
    constant_depth : float
        Depth at which the sources will be located, relative to the mean height of the
        data points.
    pad : float or None
        Ratio of region padding. Controls the ammount of padding that will be added to
        the coordinates region. It's useful to remove boundary artifacts. The pad will
        be computed as the product of the ``pad`` and the dimension of the region along
        each direction.
    kwargs
        Additional keyword arguments that won't be taken into account on the generation
        of point sources. These keyword arguments are taken in case the arguments for
        this function are passed as a dictionary with additional keys that aren't meant
        for building the sources (eg, the ``damping`` argument for the gridders).

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    # Generate grid sources (with or without padding)
    region = get_region(coordinates)
    if pad:
        w, e, s, n = region[:]
        padding = (pad * (n - s), pad * (e - w))
        region = pad_region(region, padding)
    easting, northing = grid_coordinates(region=region, spacing=spacing)
    upward = np.full_like(easting, coordinates[2].min()) - constant_depth
    points = (easting, northing, upward)
    return points


def set_constant_depth(points, constant_depth, **kwargs):
    """
    Put all source points at a constant depth
    """
    easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in points)
    upward = np.full_like(upward, upward.min()) - constant_depth
    points = (easting, northing, upward)
    return points


def set_relative_depth(points, relative_depth, **kwargs):
    """
    Put sources at a relative depth beneath computation points

    The depth of sources is equal to the elevation of passed points minus
    a relative_depth. So each source point will be located at a different depth if the
    original points have different elevations.
    """
    easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in points)
    points = (easting, northing, upward - relative_depth)
    return points


def set_variable_depth(points, depth_factor, relative_depth, k_nearest, **kwargs):
    """
    Change depth of points based on the distance to nearest neighbours
    """
    easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in points)
    upward -= relative_depth
    upward -= depth_factor * median_distance(points, k_nearest=k_nearest)
    points = (easting, northing, upward)
    return points
