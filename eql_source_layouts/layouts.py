"""
Create different layouts of point sources
"""
import numpy as np
from verde import median_distance, BlockReduce, get_region, grid_coordinates


DEPTH_TYPES = ["constant_depth", "relative_depth", "variable_relative_depth"]


def source_bellow_data(
    coordinates,
    depth_type,
    constant_depth=None,
    relative_depth=None,
    depth_factor=None,
    depth_shift=None,
    k_nearest=None,
    **kwargs,
):
    """
    Put one point source beneath each observation point

    The depth of the point sources will be the upward coordinate of the corresponding
    observation point minus a relative depth. This relative depth can either be constant
    or variable. In case it's variable, it will be equal to a quantity proportional to
    the median distance to the k nearest neighbours. The variable relative depth can
    also be static shifted by a constant depth.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    depth_type : str
        Type of depth distribution for source points.
        Availables types: ``"constant_depth"``, ``"relative_depth"``,
        ``"variable_relative_depth"``.
    constant_depth : float
        Depth at which the sources will be located, relative to the mean height of the
        data points. This parameter is ignored if ``depth_type`` is not
        ``constant_depth``.
    relative_depth : float
        Constant relative depth. If it's not None, then each source will be located at
        a depth equal to ``relative_depth`` beneath its corresponding data point.
        This parameter is ignored if ``depth_type`` is not ``relative_depth``.
    depth_factor : float
        Adimensional factor to set the depth of each point source for the variable
        relative depth approach. The upward component of the source points will be
        lowered to a relative depth given by the product of the ``depth_factor`` and the
        mean distance to the nearest ``k_nearest`` source points plus a ``depth_shift``.
        A greater ``depth_factor`` will increase the depth of the point source.
        This parameter is ignored if ``depth_type`` is not ``variable_relative_depth``.
    depth_shift : float
        Constant shift for the upward component of the source points for the variable
        relative depth approach. A negative value will make the ``upward`` component
        deeper, while a positive one will make it shallower.
        This parameter is ignored if ``depth_type`` is not ``variable_relative_depth``.
    k_nearest : int
        Number of source points used to compute the median distance to its nearest
        neighbours in the variable relative depth approach. This argument is passed to
        :func:`verde.mean_distance`.
        This parameter is ignored if ``depth_type`` is not ``variable_relative_depth``.
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
    if depth_type not in DEPTH_TYPES:
        raise ValueError("Invalid depth type '{}'.")
    if depth_type == "constant_depth":
        easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in coordinates)
        upward = np.full_like(upward, np.mean(upward) - constant_depth)
        points = (easting, northing, upward)
    elif depth_type == "relative_depth":
        easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in coordinates)
        points = (easting, northing, upward - relative_depth)
    elif depth_type == "variable_relative_depth":
        points = variable_relative_depth(
            coordinates,
            depth_factor=depth_factor,
            depth_shift=depth_shift,
            k_nearest=k_nearest,
        )
    return points


def block_reduced_sources(
    coordinates,
    spacing,
    depth_type,
    center_coordinates=False,
    constant_depth=None,
    relative_depth=None,
    depth_factor=None,
    depth_shift=None,
    k_nearest=None,
    **kwargs,
):
    """
    Put one point source beneath the block reduced observation points

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
    center_coordinates : bool
        If True, then the point sources will be put bellow the center of the populated
        blocks. Otherwise, they will be located beneath the reduced observation points.
        Default False.
    constant_depth : float
        Depth at which the sources will be located, relative to the mean height of the
        data points. This parameter is ignored if ``depth_type`` is not
        ``constant_depth``.
    relative_depth : float
        Constant relative depth. If it's not None, then each source will be located at
        a depth equal to ``relative_depth`` beneath its corresponding block reduced data
        point. This parameter is ignored if ``depth_type`` is not ``relative_depth``.
    depth_factor : float
        Adimensional factor to set the depth of each point source for the variable
        relative depth approach. The upward component of the source points will be
        lowered to a relative depth given by the product of the ``depth_factor`` and the
        mean distance to the nearest ``k_nearest`` source points plus a ``depth_shift``.
        A greater ``depth_factor`` will increase the depth of the point source.
        This parameter is ignored if ``depth_type`` is not ``variable_relative_depth``.
    depth_shift : float
        Constant shift for the upward component of the source points for the variable
        relative depth approach. A negative value will make the ``upward`` component
        deeper, while a positive one will make it shallower.
        This parameter is ignored if ``depth_type`` is not ``variable_relative_depth``.
    k_nearest : int
        Number of source points used to compute the median distance to its nearest
        neighbours in the variable relative depth approach. This argument is passed to
        :func:`verde.mean_distance`.
        This parameter is ignored if ``depth_type`` is not ``variable_relative_depth``.
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
    if depth_type not in DEPTH_TYPES:
        raise ValueError("Invalid depth type '{}'.")
    points = block_reduce_points(
        coordinates, spacing=spacing, center_coordinates=center_coordinates
    )
    if depth_type == "constant_depth":
        easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in points)
        upward = np.full_like(upward, np.mean(upward) - constant_depth)
        points = (easting, northing, upward)
    elif depth_type == "relative_depth":
        easting, northing, upward = tuple(np.atleast_1d(i).copy() for i in points)
        points = (easting, northing, upward - relative_depth)
    elif depth_type == "variable_relative_depth":
        points = variable_relative_depth(
            points,
            depth_factor=depth_factor,
            depth_shift=depth_shift,
            k_nearest=k_nearest,
        )
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
    region = get_region(coordinates)
    if pad:
        w, e, s, n = region[:]
        w_padded, e_padded = w - pad * (e - w), e + pad * (e - w)
        s_padded, n_padded = s - pad * (n - s), n + pad * (n - s)
        region = (w_padded, e_padded, s_padded, n_padded)
    easting, northing = grid_coordinates(region=region, spacing=spacing)
    upward = np.full_like(easting, np.mean(coordinates[2])) - relative_depth
    points = (easting, northing, upward)
    return points


def block_reduce_points(
    coordinates, spacing=None, center_coordinates=False, reduction=np.median
):
    """
    Block reduce points to create one point per populated block

    The upward component of the points is also reduced. The block reduction is
    performed by :class:`vd.BlockReduce`.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    spacing : float, tuple = (s_north, s_east)
        The block size in the South-North and West-East directions, respectively.
        A single value means that the size is equal in both directions.
    center_coordinates : bool
        If True, then the returned coordinates correspond to the center of each block.
        Otherwise, the coordinates are calculated by applying the same reduction
        operation to the input coordinates. Default False.
    reduction : function
        A reduction function that takes an array and returns a single value (e.g.,
        :func:`numpy.median`, :func:`numpy.mean`, etc). Default to :func:`numpy.median`.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    reducer = BlockReduce(
        spacing=spacing, center_coordinates=center_coordinates, reduction=reduction
    )
    (easting, northing), upward = reducer.filter(coordinates[:2], coordinates[2])
    points = (easting, northing, upward)
    return points


def variable_relative_depth(points, depth_factor, depth_shift, k_nearest):
    """
    Change depth of points based on the distance to nearest neighbours

    Modify the upward component of the points setting it a relative depth proportional
    to the median distance to the nearest k points. This proportionality is given by the
    ``depth_factor`` argument. Also, a static shift can be added throght the
    ``depth_shift`` argument.

    Parameters
    ----------
    points : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    depth_factor : float
        Adimensional factor to set the depth of each point source. The upward component
        of the source points will be lowered to a relative depth given by the product of
        the ``depth_factor`` and the mean distance to the nearest ``k_nearest`` source
        points plus a ``depth_shift``. A greater ``depth_factor`` will increase the
        depth of the point source.
    depth_shift : float
        Constant shift for the upward component of the source points. A negative value
        will make the ``upward`` component deeper, while a positive one will make it
        shallower.
    k_nearest : int
        Number of source points used to compute the median distance to its nearest
        neighbours. This argument is passed to :func:`verde.mean_distance`.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the modified point sources in the following
        order: (``easting``, ``northing``, ``upward``).
    """
    easting, northing, upward = tuple(np.atleast_1d(i).ravel().copy() for i in points)
    upward -= depth_factor * median_distance(points, k_nearest=k_nearest)
    upward += depth_shift
    return (easting, northing, upward)
