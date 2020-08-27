"""
Iterative equivalent layer for generic harmonic functions
"""
import numpy as np
import verde as vd
import verde.base as vdb
from sklearn.utils import shuffle
from harmonica import EQLHarmonic


class EQLIterative(EQLHarmonic):
    r"""
    Iterative equivalent-layer for generic harmonic functions

    This equivalent layer fits the sources coefficients through an iterative
    strategy. On each iteration, it fits the coefficients of the sources that
    fall inside one window out of a set of overlapping windows.

    Parameters
    ----------
    damping : None or float
        The positive damping regularization parameter. Controls how much
        smoothness is imposed on the estimated coefficients. If None, no
        regularization is used.
    points : None or list of arrays (optional)
        List containing the coordinates of the point sources used as the
        equivalent layer. Coordinates are assumed to be in the following order:
        (``easting``, ``northing``, ``upward``). If None, will place one point
        source bellow each observation point at a fixed relative depth bellow
        the observation point [Cooper2000]_. Defaults to None.
    relative_depth : float
        Relative depth at which the point sources are placed beneath the
        observation points. Each source point will be set beneath each data
        point at a depth calculated as the elevation of the data point minus
        this constant *relative_depth*. Use positive numbers (negative numbers
        would mean point sources are above the data points). Ignored if
        *points* is specified.
    window_size : float
        Size of the square windows used to choose point sources that will be
        fitted on each iteration.
    warm_start : bool
        If False, coefficient values are set to zero on every call of the
        ``fit`` method. If True, the fitting process will start with the
        previous fitted coefficients (if existent) and using the residue
        between the data and the field the fitted sources predict on the same
        points. Default True.
    random_state : int, RandomState instance or None
        Random state for shuffling the rolling windows. If int,
        ``random_state`` is the seed used by the random number generator. If
        ``np.random.RandomState`` instance, ``random_state`` is the random
        number generator. If None, the random number generator is the
        ``RandomState`` instance used by `np.random`. Default None.

    Attributes
    ----------
    points_ : 2d-array
        Coordinates of the point sources used to build the equivalent layer.
    coefs_ : array
        Estimated coefficients of every point source.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~harmonica.EQLHarmonic.grid` and
        :meth:`~harmonica.EQLHarmonic.scatter` methods.
    """

    # Define amount of overlapping between adjacent windows to 50%.
    overlapping = 0.5

    def __init__(
        self,
        damping=None,
        points=None,
        relative_depth=500,
        window_size=10e3,
        warm_start=False,
        random_state=None,
    ):
        super().__init__(damping=damping, points=points, relative_depth=relative_depth)
        self.window_size = window_size
        self.warm_start = warm_start
        self.random_state = random_state

    def fit(self, coordinates, data, weights=None):
        """
        Fit the coefficients of the equivalent layer.

        The source coefficients are iteratively fitted.
        A regular set of rolling windows with a 50% of overlap is defined along
        the entire data region. On each iteration, one window is randomly
        selected and then all the coefficients of the sources that fall inside
        that window are fitted using the data points that also fall inside it.
        Then the field produced by these sources is computed and removed from
        the data to obtain a residue. The next iteration follows the same way,
        randomly choosing another window, but now the fit is done on the
        residue.

        The data region is captured and used as default for the
        :meth:`~harmonica.HarmonicEQL.grid` and
        :meth:`~harmonica.HarmonicEQL.scatter` methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, upward, ...).
            Only easting, northing, and upward will be used, all subsequent
            coordinates will be ignored.
        data : array
            The data values of each data point.
        weights : None or array
            If not None, then the weights assigned to each data point.
            Typically, this should be 1 over the data uncertainty squared.

        Returns
        -------
        self
            Returns this estimator instance for chaining operations.
        """
        coordinates, data, weights = vdb.check_fit_input(coordinates, data, weights)
        # Capture the data region to use as a default when gridding.
        self.region_ = vd.get_region(coordinates[:2])
        # Ravel coordinates, data and weights to 1d-arrays
        coordinates = vdb.n_1d_arrays(coordinates, 3)
        data = data.ravel()
        if weights is not None:
            weights = weights.ravel()
        # Define self.points_ if warm_start is False and gridder is not
        # already fitted
        if not self.warm_start or not hasattr(self, "coefs_"):
            if self.points is None:
                self.points_ = (
                    coordinates[0],
                    coordinates[1],
                    coordinates[2] - self.relative_depth,
                )
            else:
                self.points_ = vdb.n_1d_arrays(self.points, 3)
        # Initialize coefficients and residue arrays
        if self.warm_start and hasattr(self, "coefs_"):
            residue = data - self.predict(coordinates)
        else:
            self.coefs_ = np.zeros(self.points_[0].size)
            residue = data.copy()
        # Fit coefficients iteratively
        self._fit_iteratively(coordinates, residue, weights)
        return self

    def _fit_iteratively(self, coordinates, residue, weights):
        """
        Iteratively fit the coefficients of the sources
        """
        # Create rolling windows
        point_windows, data_windows = self._create_rolling_windows(coordinates)
        # Get number of windows
        n_windows = len(point_windows)
        # Initialize errors array
        self.errors_ = np.zeros(n_windows)
        # Set weights_chunk to None (will be changed unless weights is None)
        weights_chunk = None
        # Iterate over the windows
        for window_i in range(n_windows):
            # Get source and data points indices for current window
            point_window, data_window = point_windows[window_i], data_windows[window_i]
            # Choose source and data points that fall inside the window
            points_chunk = tuple(p[point_window] for p in self.points_)
            coords_chunk = tuple(c[data_window] for c in coordinates)
            # Skip the window if no sources or data points fall inside it
            if points_chunk[0].size == 0 or coords_chunk[0].size == 0:
                continue
            # Choose weights for data points inside the window (if not None)
            if weights is not None:
                weights_chunk = weights[data_window]
            # Compute jacobian (for sources in windows and all data points)
            jacobian = self.jacobian(coordinates, points_chunk)
            # Fit coefficients of sources with data points inside window
            # (we need to copy the jacobian so it doesn't get overwritten)
            coeffs_chunk = vdb.least_squares(
                jacobian[data_window, :],
                residue[data_window],
                weights_chunk,
                self.damping,
                copy_jacobian=True,
            )
            self.coefs_[point_window] += coeffs_chunk
            # Update residue (on every point)
            residue -= np.dot(jacobian, coeffs_chunk)
            self.errors_[window_i] = np.sqrt(np.mean(residue ** 2))

    def _create_rolling_windows(self, coordinates):
        """
        Create indices of sources and data points for each rolling window
        """
        # Compute window spacing based on overlapping
        window_spacing = self.window_size * (1 - self.overlapping)
        # Get the largest region between data points and sources
        data_region = vd.get_region(coordinates)
        sources_region = vd.get_region(self.points_)
        region = (
            min(data_region[0], sources_region[0]),
            max(data_region[1], sources_region[1]),
            min(data_region[2], sources_region[2]),
            max(data_region[3], sources_region[3]),
        )
        # The windows for sources and data points are the same, but the
        # verde.rolling_window function creates indices for the given
        # coordinates. That's why we need to create two set of window indices:
        # one for the sources and one for the data points
        # We pass the same region, size and spacing to be sure that both set of
        # windows are the same
        _, source_windows = vd.rolling_window(
            self.points_, region=region, size=self.window_size, spacing=window_spacing
        )
        _, data_windows = vd.rolling_window(
            coordinates, region=region, size=self.window_size, spacing=window_spacing
        )
        # Ravel the indices
        source_windows = [i[0] for i in source_windows.ravel()]
        data_windows = [i[0] for i in data_windows.ravel()]
        # Shuffle windows
        source_windows, data_windows = shuffle(
            source_windows, data_windows, random_state=self.random_state
        )
        return source_windows, data_windows
