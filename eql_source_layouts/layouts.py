"""
Create different layouts of point sources
"""
import numpy as np
from verde import BlockReduce, median_distance


def source_beneath_data(coordinates, depth_factor=3, static_shift=0, k_nearest=1):
    """
    Create a set of source points beneath the data points

    Place one source point beneath each data point. The depth of each point source is
    determined throught the :func:`_adaptive_points_depth`: it will be placed at the
    reduced upward coordinate minus a relative depth proportional to the median distance
    of the nearest point sources. This upward component can also be static shifted by
    a constant value.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the data points in the following order:
        (``easting``, ``northing``, ``upward``).
    depth_factor : float
        Adimensional factor to set the depth of each point source. The upward component
        of the source points will be lowered to a relative depth given by the product of
        the ``depth_factor`` and the mean distance to the nearest ``k_nearest`` source
        points plus a ``static_shift``. A greater ``depth_factor`` will increase the
        depth of the point source. This argument is passed to
        :func:`_adaptive_points_depth`. Default 3 (following [Cooper2000]_).
    static_shift : float
        Constant shift for the upward component of the source points. A negative value
        will make the ``upward`` component deeper, while a positive one will make it
        shallower. This argument is passed to :func:`_adaptive_points_depth`. Default 0.
    k_nearest : int
        Number of source points used to compute the median distance to its nearest
        neighbours. This argument is passed to :func:`verde.mean_distance`. This
        argument is passed to :func:`_adaptive_points_depth`. Default 1.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the point sources in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    points = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    return _adaptive_points_depth(points, depth_factor, static_shift, k_nearest)


def block_reduced_points(
    coordinates, depth_factor=3, static_shift=0, k_nearest=1, **kwargs
):
    """
    Block reduce data points to create one source point per populated block

    Place one source point beneath the block reduced data points. The depth of each
    point source is determined throught the :func:`_adaptive_points_depth`: it will be
    placed at the reduced upward coordinate minus a relative depth proportional to the
    median distance of the nearest point sources. This upward component can also be
    static shifted by a constant value.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the data points in the following order:
        (``easting``, ``northing``, ``upward``).
    depth_factor : float
        Adimensional factor to set the depth of each point source. The upward component
        of the source points will be lowered to a relative depth given by the product of
        the ``depth_factor`` and the mean distance to the nearest ``k_nearest`` source
        points plus a ``static_shift``. A greater ``depth_factor`` will increase the
        depth of the point source. This argument is passed to
        :func:`_adaptive_points_depth`. Default 3 (following [Cooper2000]_).
    static_shift : float
        Constant shift for the upward component of the source points. A negative value
        will make the ``upward`` component deeper, while a positive one will make it
        shallower. This argument is passed to :func:`_adaptive_points_depth`. Default 0.
    k_nearest : int
        Number of source points used to compute the median distance to its nearest
        neighbours. This argument is passed to :func:`verde.mean_distance`. This
        argument is passed to :func:`_adaptive_points_depth`. Default 1.
    kwargs
        Keyword arguments passed to :class:`verde.BlockReduce` like ``reduction``,
        ``spacing``, ``region``, ``adjust``, ``shape``. If ``reduction`` is not passed,
        it will be set by default to :func:`numpy.median`.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the point sources in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    if "reduction" not in kwargs:
        kwargs["reduction"] = np.median
    reducer = BlockReduce(**kwargs)
    (easting, northing), upward = reducer.filter(coordinates[:2], coordinates[2])
    points = (easting, northing, upward)
    points = _adaptive_points_depth(
        points,
        depth_factor=depth_factor,
        static_shift=static_shift,
        k_nearest=k_nearest,
    )
    return points


def _adaptive_points_depth(points, depth_factor, static_shift, k_nearest):
    """
    Set upward component of source points proportional to distance of nearest neighbours

    Modify the upward component of the source points setting it a relative depth
    proportional to the median distance to the nearest k source points.
    This proportionality is given by the ``depth_factor`` argument.
    Also, a static shift can be added throght the ``static_shift`` argument.

    Parameters
    ----------
    points : tuple of arrays
        Tuple containing the coordinates of the point sources in the following order:
        (``easting``, ``northing``, ``upward``).
    depth_factor : float
        Adimensional factor to set the depth of each point source. The upward component
        of the source points will be lowered to a relative depth given by the product of
        the ``depth_factor`` and the mean distance to the nearest ``k_nearest`` source
        points plus a ``static_shift``. A greater ``depth_factor`` will increase the
        depth of the point source. This parameter is ignored if ``points`` is not None.
    static_shift : float
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
    easting, northing, upward = tuple(np.atleast_1d(i).ravel() for i in points)
    upward -= depth_factor * median_distance(points, k_nearest=k_nearest)
    upward += static_shift
    return (easting, northing, upward)
