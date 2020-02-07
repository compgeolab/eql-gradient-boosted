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
        return constant_depth(points, **kwargs)
    elif depth_type == "relative_depth":
        return relative_depth(points, **kwargs)
    elif depth_type == "variable_depth":
        return variable_depth(points, **kwargs)


def source_bellow_data(
    coordinates, depth_type, **kwargs,
):
    """
    Put one point source beneath each observation point

    The depth of the point sources can be set according to the following methods:
    *constant*, *relative* or *variable* depth.

    The *constant depth* locates all sources at the same depth. It can be computed as
    the difference between the minimum elevation of observation points and the ``depth``
    argument.

    The *relative depth* locates all sources at a constant _relative_ ``depth`` beneath
    its corresponding observation point. Each source point will be located at the same
    distance from its corresponding observation point. Sources points will _copy_ the
    elevations of the coordinates at a ``depth`` bellow.

    The *variable depth* locates the sources in the same way the _relative depth_ does
    but also adds a term that can be computed as the product of the ``depth_factor`` and
    the median distance between the ``k_nearest`` nearest neighbor sources.

    The depth type can be chosen through the ``depth_type`` argument, while the
    ``depth``, ``depth_factor`` and ``k_nearest`` arguments can be passed as ``kwargs``.


    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the observation points in the following
        order: (``easting``, ``northing``, ``upward``).
    depth_type : str
        Type of depth distribution for source points.
        Available types: ``"constant_depth"``, ``"relative_depth"``,
        ``"variable_depth"``.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the source points in the following order:
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

    The depth of the point sources can be set according to the following methods:
    *constant*, *relative* or *variable* depth.

    The *constant depth* locates all sources at the same depth. It can be computed as
    the difference between the minimum elevation of block-median coordinates and the
    ``depth`` argument.

    The *relative depth* locates all sources at a constant _relative_ ``depth`` beneath
    its corresponding block-median point. Each source point will be located at the same
    distance from its corresponding block-median point. Sources points will _copy_ the
    elevations of the block-median coordinates at a ``depth`` bellow.

    The *variable depth* locates the sources in the same way the _relative depth_ does
    but also adds a term that can be computed as the product of the ``depth_factor`` and
    the median distance between the ``k_nearest`` nearest neighbor sources.

    The depth type can be chosen through the ``depth_type`` argument, while the
    ``depth``, ``depth_factor`` and ``k_nearest`` arguments can be passed as ``kwargs``.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the observation points in the following
        order: (``easting``, ``northing``, ``upward``).
    spacing : float, tuple = (s_north, s_east)
        The block size in the South-North and West-East directions, respectively.
        A single value means that the size is equal in both directions.
    depth_type : str
        Type of depth distribution for source points.
        Available types: ``"constant_depth"``, ``"relative_depth"``,
        ``"variable_relative_depth"``.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the source points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    reducer = BlockReduce(spacing=spacing, reduction=np.median, drop_coords=False)
    points, _ = reducer.filter(coordinates, np.zeros_like(coordinates[0]))
    points = _dispatcher(points, depth_type, **kwargs)
    return points


def grid_sources(coordinates, spacing=None, depth=None, pad=None, **kwargs):
    """
    Create a regular grid of point sources

    All point sources will be located at the same depth, equal to the difference between
    the minimum elevation of observation points and the ``depth`` argument.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the observation points in the following
        order: (``easting``, ``northing``, ``upward``).
    spacing : float, tuple = (s_north, s_east)
        The block size in the South-North and West-East directions, respectively.
        A single value means that the size is equal in both directions.
    depth : float
        Depth shift used to compute the constant depth at which point sources will be
        located.
    pad : float or None
        Ratio of region padding. Controls the amount of padding that will be added to
        the coordinates region. It's useful to remove boundary artifacts. The pad will
        be computed as the product of the ``pad`` and the dimension of the region along
        each direction.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the source points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    # Generate grid sources (with or without padding)
    region = get_region(coordinates)
    if pad:
        w, e, s, n = region[:]
        padding = (pad * (n - s), pad * (e - w))
        region = pad_region(region, padding)
    easting, northing = grid_coordinates(region=region, spacing=spacing)
    upward = np.full_like(easting, coordinates[2].min()) - depth
    return easting, northing, upward


def constant_depth(coordinates, depth, **kwargs):
    """
    Put all source points at the same depth

    The depth at which the point sources will be located is computed as the difference
    between the minimum height of ``coordinates`` and ``depth``.

    Any extra keyword argument passed will be ignored.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the observation points or block-median
        points in the following order: (``easting``, ``northing``, ``upward``).
    depth : float
        Depth shift used to compute the constant depth at which point sources will be
        located.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the source points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in coordinates)
    upward = np.full_like(upward, upward.min()) - depth
    return easting, northing, upward


def relative_depth(coordinates, depth, **kwargs):
    """
    Put sources at a relative depth beneath coordinates

    The depth of sources is equal to difference between the height of ``coordinates``
    and ``depth``. So each source point will be located at different depths if the
    original points have different elevations. The point sources will _copy_ the
    topography of ``coordinates``.

    Any extra keyword argument passed will be ignored.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the observation points or block-median
        points in the following order: (``easting``, ``northing``, ``upward``).
    depth : float
        Depth shift used to compute the relative depth at which point sources will be
        located.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the source points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in coordinates)
    upward -= depth
    return easting, northing, upward


def variable_depth(coordinates, depth, depth_factor, k_nearest, **kwargs):
    """
    Put sources at a depth based on the distance to nearest neighbors

    Depth of sources will be set by applying the relative depth strategy plus a term
    equal to the product of ``depth_factor`` and the median distance to the
    ``k_nearest`` nearest neighbor sources. Sources beneath clustered ``coordinates``
    points will be shallower than sources bellow scattered ``coordinates``.

    Any extra keyword argument passed will be ignored.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the observation points or block-median
        points in the following order: (``easting``, ``northing``, ``upward``).
    depth : float
        Depth shift used to compute the relative depth at which point sources will be
        located.
    depth_factor : float
        Factor used on the variable depth term.
    k_nearest : int
        Number of nearest neighbor sources used to compute the median distance.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the source points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in coordinates)
    upward -= depth
    upward -= depth_factor * median_distance(coordinates, k_nearest=k_nearest)
    return easting, northing, upward
