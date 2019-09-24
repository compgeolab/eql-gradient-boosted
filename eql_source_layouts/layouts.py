"""
Create different layouts of point sources
"""
import numpy as np
from verde import median_distance, BlockReduce


def block_reduce_points(coordinates, **kwargs):
    """
    Block reduce points to create one point per populated block

    The upward component of the points is also reduced. The block reduction is
    performed by :class:`vd.BlockReduce` and it can be controlled by the ``kwargs``.

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    kwargs
        Keyword arguments passed to :class:`verde.BlockReduce` like ``reduction``,
        ``spacing``, ``region``, ``adjust``, ``shape``, ``center_coordinates``.
        If ``reduction`` is not passed, it will be set by default to
        :func:`numpy.median`.

    Returns
    -------
    points : tuple of arrays
        Tuple containing the coordinates of the points in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    if "reduction" not in kwargs:
        kwargs["reduction"] = np.median
    reducer = BlockReduce(**kwargs)
    (easting, northing), upward = reducer.filter(coordinates[:2], coordinates[2])
    points = (easting, northing, upward)
    return points


def adaptive_depth(points, depth_factor, depth_shift, k_nearest):
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
        depth of the point source. This parameter is ignored if ``points`` is not None.
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
