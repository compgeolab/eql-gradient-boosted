"""
Create different layouts of point sources
"""
import numpy as np
import verde as vd


def point_per_block(coordinates, upward, **kwargs):
    """
    Create a set of point sources beneath the center of data blocks

    Split the data points region into blocks and put one point source beneath the center
    of each block. All source points will be located at the same depth given by
    ``upward``.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the following
        order: (easting, northing, upward, ...). Only easting and northing will be used,
        all subsequent coordinates will be ignored.
    upward : float
        Vertical coordinate of the point masses.
    kwargs
        Keyword arguments passed to :class:`verde.block_split` like ``spacing``,
        ``region``, ``adjust``, etc.

    Returns
    -------
    points : list of arrays
        List containing the coordinates of the point sources in the following order:
        (``easting``, ``northing``, ``upward``).
    """
    (easting, northing), _ = vd.block_split(coordinates, **kwargs)
    upward = upward * np.ones_like(easting)
    points = (easting, northing, upward)
    return points
