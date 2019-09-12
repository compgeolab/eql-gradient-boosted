"""
Create synthetic model made out of prisms and synthetic surveys
"""
import pyproj
import numpy as np
import verde as vd
import harmonica as hm


def synthetic_model():
    """
    Create a set of prisms in order to compute a gravity synthetic model

    Returns
    -------
    model : dict
        Dictionary containing the prisms and the density for each prism.
    """
    # Create two big prisms that resemble a fault
    prisms, densities = [], []
    prisms.append([-40e3, 0, -40e3, 40e3, -13e3, -10e3])
    densities.append(200)
    prisms.append([-40e3, 0, -40e3, 40e3, -13.5e3, -10.5e3])
    densities.append(200)

    # Add shallower prisms
    prisms.append([-35e3, -15e3, 20e3, 30e3, -6e3, -5e3])
    densities.append(-300)
    prisms.append([-45e3, -20e3, -5e3, 15e3, -4e3, -2.5e3])
    densities.append(-150)
    prisms.append([26e3, 38e3, -30e3, -10e3, -7e3, -2e3])
    densities.append(150)

    # Add dikes
    prisms.append([-32e3, -30e3, -28e3, -26e3, -9e3, -1e3])
    densities.append(500)
    prisms.append([-2e3, 2e3, -2e3, 2e3, -9e3, -1e3])
    densities.append(300)

    # Add diagonal dike
    n_blocks = 301
    dike_density = 500
    horizontal_size = 2e3
    easting_1, easting_2 = 14e3, 25e3
    northing_1, northing_2 = 12e3, 22e3
    upward_1, upward_2 = -7e3, 0

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
    return {"prisms": prisms, "densities": densities}


def airborne_survey(region, center=(-42.25, -32.27)):
    """
    Create a synthetic airborne survey based on Rio de Janeiro magnetic measurements

    Parameters
    ----------
    region : tuple or list
        Boundaries of the synthetic region where the flight lines are going to be drawn.
        Should be in Cartesian coordinates and in meters.
    center : tuple (optional)
        Coordiantes of the center of the region in the original coordinates of the Rio
        de Janeiro magnetic survey. The fligh lines are chosen around this center point.
        The coordinates must be in degrees, defined on a geodetic coordinate system.

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
    projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
    data["easting"], data["northing"] = projection(
        data.longitude.values, data.latitude.values
    )
    center = projection(*center)

    # Cut the data into a region that has the same dimensions as the region argument
    w, e, s, n = region[:]
    cut_region = (
        center[0] - (e - w) / 2,
        center[0] + (e - w) / 2,
        center[1] - (n - s) / 2,
        center[1] + (n - s) / 2,
    )
    inside = vd.inside((data.longitude, data.latitude), cut_region)
    data = data[inside]
    return data
