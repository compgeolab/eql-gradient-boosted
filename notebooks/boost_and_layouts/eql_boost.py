"""
Gradient boosted equivalent layer for generic harmonic functions
"""
import numpy as np
import verde as vd
import verde.base as vdb
from sklearn.utils import shuffle
from harmonica import EQLHarmonic

from harmonica.equivalent_layer.harmonic import greens_func_cartesian
from harmonica.equivalent_layer.utils import predict_numba


class EQLHarmonicBoost(EQLHarmonic):
    r"""
    Gradient boosted equivalent-layer for generic harmonic functions

    This equivalent layer fits the sources coefficients through a gradient
    boosting strategy: generates one set of equivalent sources per rolling
    window. On each iteration, the coefficients of each set of equivalent
    sources are fitted. A single equivalent-layer is generated at the end as
    a superposition of every set of equivalent sources.

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
    shuffle : bool
        If True, the rolling windows are randomly shuffled to prevent
        prioritinzing a portion of the sources (the ones that are fitted on the
        first iterations) over the rest. If False, windows won't be shuffled.
        Default True.
    random_state : int, RandomState instance or None
        Random state for shuffling the rolling windows. If int,
        ``random_state`` is the seed used by the random number generator. If
        ``np.random.RandomState`` instance, ``random_state`` is the random
        number generator. If None, the random number generator is the
        ``RandomState`` instance used by `np.random`. Ignored if ``shuffle`` is
        False. Default None.
    line_search : bool
        If True, the gradient boosting method fits the step-size parameter on
        each iteration in order to minimize the misfit between the effect of
        the fitted coefficients on that iteration on every observation point
        and the previous residue. This helps to stabilize the convergence and
        to take into account, to some extent, the effect of the fitted
        coefficients on observation points outside the current window. If
        False, only the source coefficients are fitted. Default to True.

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
        shuffle=True,
        random_state=None,
        line_search=True,
    ):
        super().__init__(damping=damping, points=points, relative_depth=relative_depth)
        self.window_size = window_size
        self.warm_start = warm_start
        self.shuffle = shuffle
        self.random_state = random_state
        self.line_search = line_search

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
        :meth:`~harmonica.EQLHarmonic.grid` and
        :meth:`~harmonica.EQLHarmonic.scatter` methods.

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
        # Fit coefficients through gradient boosting
        self._gradient_boosting(coordinates, residue, weights)
        return self

    def _gradient_boosting(self, coordinates, residue, weights):
        """
        Fit source coefficients through gradient boosting
        """
        # Create rolling windows
        point_windows, data_windows = self._create_rolling_windows(coordinates)
        # Get number of windows
        n_windows = len(point_windows)
        # Initialize errors array
        errors = [np.sqrt(np.mean(residue ** 2))]
        # Set weights_chunk to None (will be changed unless weights is None)
        weights_chunk = None
        predicted = np.empty_like(residue)
        # Iterate over the windows
        for window_i in range(n_windows):
            # Get source and data points indices for current window
            point_window, data_window = point_windows[window_i], data_windows[window_i]
            # Choose source and data points that fall inside the window
            points_chunk = tuple(p[point_window] for p in self.points_)
            coords_chunk = tuple(c[data_window] for c in coordinates)
            # Choose weights for data points inside the window (if not None)
            if weights is not None:
                weights_chunk = weights[data_window]
            # Compute jacobian (for sources and data points in current window)
            jacobian = self.jacobian(coords_chunk, points_chunk)
            # Fit coefficients of sources with data points inside window
            # (we need to copy the jacobian so it doesn't get overwritten)
            coeffs_chunk = vdb.least_squares(
                jacobian,
                residue[data_window],
                weights_chunk,
                self.damping,
                copy_jacobian=True,
            )
            predicted[:] = 0
            predict_numba(
                coordinates,
                points_chunk,
                coeffs_chunk,
                predicted,
                greens_func_cartesian,
            )
            if self.line_search:
                step = np.sum(residue * predicted) / np.sum(predicted ** 2)
                predicted *= step
                coeffs_chunk *= step
            residue -= predicted
            errors.append(np.sqrt(np.mean(residue ** 2)))
            self.coefs_[point_window] += coeffs_chunk
        self.errors_ = np.array(errors)

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
        # one for the sources and one for the data points.
        # We pass the same region, size and spacing to be sure that both set of
        # windows are the same.
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
        if self.shuffle:
            source_windows, data_windows = shuffle(
                source_windows, data_windows, random_state=self.random_state
            )
        # Remove empty windows
        source_windows_nonempty = []
        data_windows_nonempty = []
        for src, data in zip(source_windows, data_windows):
            if src.size > 0 and data.size > 0:
                source_windows_nonempty.append(src)
                data_windows_nonempty.append(data)
        return source_windows_nonempty, data_windows_nonempty
