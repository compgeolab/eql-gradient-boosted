"""
Create synthetic model made out of prisms and synthetic surveys
"""
import pyproj
import numpy as np
import verde as vd
import harmonica as hm


def synthetic_model(region):
    """
    Create a set of prisms in order to compute a gravity synthetic model

    Parameters
    ----------
    region : tuple or list
        Boundaries of the synthetic region where the synthetic model will be build.
        Should be in Cartesian coordinates, in meters and passed in the following order:
        (``east``, ``west``, ``south``, ``north``, ``bottom``, ``top``).

    Returns
    -------
    model : dict
        Dictionary containing the prisms and the density for each prism.
    """
    # Create two big prisms that resemble a fault
    prisms, densities = [], []
    prisms.append([0.1, 0.5, 0.1, 0.9, 0.05, 0.25])
    densities.append(200)
    prisms.append([0.5, 0.9, 0.1, 0.9, 0, 0.2])
    densities.append(200)

    # Add shallower prisms
    prisms.append([0.1, 0.4, 0.7, 0.9, 0.4, 0.5])
    densities.append(-300)
    prisms.append([0, 0.25, 0.4, 0.6, 0.6, 0.75])
    densities.append(-150)
    prisms.append([0.7, 0.8, 0.1, 0.4, 0.3, 0.8])
    densities.append(150)

    # Add dikes
    prisms.append([0.30, 0.32, 0.26, 0.28, 0.1, 1])
    densities.append(500)
    prisms.append([0.49, 0.51, 0.49, 0.51, 0.1, 1])
    densities.append(500)

    # Add diagonal dike
    n_blocks = 301
    dike_density = 500
    horizontal_size = 0.05
    easting_1, easting_2 = 0.6, 0.75
    northing_1, northing_2 = 0.52, 0.65
    upward_1, upward_2 = 0.3, 1

    t = np.linspace(0, 1, n_blocks)
    easting_center = (easting_2 - easting_1) * t + easting_1
    northing_center = (northing_2 - northing_1) * t + northing_1
    upward_center = (upward_2 - upward_1) * t + upward_1
    upward_size = (upward_2 - upward_1) / n_blocks
    for i in range(n_blocks):
        prisms.append(
            [
                easting_center[i] - horizontal_size / 2,
                easting_center[i] + horizontal_size / 2,
                northing_center[i] - horizontal_size / 2,
                northing_center[i] + horizontal_size / 2,
                upward_center[i] - upward_size / 2,
                upward_center[i] + upward_size / 2,
            ]
        )
        densities.append(dike_density)

    # Scale prisms to the passed region
    w, e, s, n, bottom, top = region[:]
    prisms = np.atleast_2d(prisms)
    prisms[:, :2] *= e - w
    prisms[:, :2] += w
    prisms[:, 2:4] *= n - s
    prisms[:, 2:4] += s
    prisms[:, 4:6] *= top - bottom
    prisms[:, 4:6] += bottom
    return {"prisms": prisms, "densities": densities}


def airborne_survey(region, center=(-42.25, -22.27)):
    """
    Create a synthetic airborne survey based on Rio de Janeiro magnetic measurements

    Parameters
    ----------
    region : tuple or list
        Boundaries of the synthetic region where the flight lines are going to be drawn
        in the following order: (``east``, ``west``, ``south``, ``north``, ...). All
        subsequent boundaries will be ignored. All boundaries should be in Cartesian
        coordinates and in meters.
    center : tuple (optional)
        Coordiantes of the center of the region in the original coordinates of the Rio
        de Janeiro magnetic survey. The fligh lines are chosen around this center point.
        The coordinates must be in degrees, defined on a geodetic coordinate system and
        passed in the following order: (``longitude``, ``latitude``).

    Returns
    -------
    data : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the flight line and its altitude on the
        Cartesian coordinats for the synthetic model. All coordinates and altitude are
        in meters.
    """
    # Fetch data
    data = hm.datasets.fetch_rio_magnetic()

    # Keep only the latitudinal, longitudinal coordinates and altitude of measurement
    data = data.filter(["longitude", "latitude", "altitude_m"])

    # Project coordinates
    projection = pyproj.Proj(proj="merc", lat_ts=center[1])
    data["easting"], data["northing"] = projection(
        data.longitude.values, data.latitude.values
    )
    center = projection(*center)

    # Cut the data into a region that has the same dimensions as the region argument
    w, e, s, n = region[:4]
    cut_region = (
        center[0] - (e - w) / 2,
        center[0] + (e - w) / 2,
        center[1] - (n - s) / 2,
        center[1] + (n - s) / 2,
    )
    inside = vd.inside((data.easting, data.northing), cut_region)
    data = data[inside]

    # Move projected coordinates to the boundaries of the region argument
    data["easting"] = (e - w) / (data.easting.max() - data.easting.min()) * (
        data.easting - data.easting.min()
    ) + w
    data["northing"] = (n - s) / (data.northing.max() - data.northing.min()) * (
        data.northing - data.northing.min()
    ) + s

    # Drop longitude and latitude from dataframe
    data = data.filter(["easting", "northing", "altitude_m"])
    return data
