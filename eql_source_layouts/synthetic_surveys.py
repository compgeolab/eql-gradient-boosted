"""
Create synthetic ground and airborne surveys
"""
import pyproj
import verde as vd
import harmonica as hm


def airborne_survey(region, cut_region=(-42.35, -42.10, -22.35, -22.15)):
    """
    Create a synthetic airborne survey based on Rio de Janeiro magnetic measurements

    Parameters
    ----------
    region : tuple or list
        Boundaries of the synthetic region where the flight lines are going to be drawn
        in the following order: (``east``, ``west``, ``south``, ``north``, ...). All
        subsequent boundaries will be ignored. All boundaries should be in Cartesian
        coordinates and in meters.
    cut_region : tuple (optional)
        Region to reduce the extension of the survey. Must be boundaries of the original
        survey, in degrees.

    Returns
    -------
    data : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the flight line and its altitude on the
        Cartesian coordinates for the synthetic model. All coordinates and altitude are
        in meters.
    """
    # Fetch data
    data = hm.datasets.fetch_rio_magnetic()
    # Keep only the latitudinal, longitudinal coordinates and elevation of measurement
    data["elevation"] = data["altitude_m"]
    data = data.filter(["longitude", "latitude", "elevation"])
    # Cut the data into a region that has the same dimensions as the region argument
    inside = vd.inside((data.longitude, data.latitude), cut_region)
    data = data[inside]
    # Project coordinates
    projection = pyproj.Proj(proj="merc", lat_ts=(cut_region[2] + cut_region[3]) / 2)
    data["easting"], data["northing"] = projection(
        data.longitude.values, data.latitude.values
    )
    # Move projected coordinates to the boundaries of the region argument
    w, e, s, n = region[:4]
    data["easting"] = (e - w) / (data.easting.max() - data.easting.min()) * (
        data.easting - data.easting.min()
    ) + w
    data["northing"] = (n - s) / (data.northing.max() - data.northing.min()) * (
        data.northing - data.northing.min()
    ) + s
    # Drop longitude and latitude from dataframe
    data = data.filter(["easting", "northing", "elevation"])
    return data


def ground_survey(region, cut_region=(13.60, 20.30, -24.20, -17.5)):
    """
    Create a synthetic ground survey based on South Africa gravity measurements

    Parameters
    ----------
    region : tuple or list
        Boundaries of the synthetic region where the observation points are going to be
        located. Must be passed in the following order: (``east``, ``west``, ``south``,
        ``north``, ...). All subsequent boundaries will be ignored. All boundaries
        should be in Cartesian coordinates and in meters.
    cut_region : tuple (optional)
        Region to reduce the extension of the survey. Must be boundaries of the original
        survey, in degrees.

    Returns
    -------
    data : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the flight line and its altitude on the
        Cartesian coordinates for the synthetic model. All coordinates and altitude are
        in meters.
    """
    # Fetch data
    data = hm.datasets.fetch_south_africa_gravity()
    # Keep only the latitudinal, longitudinal coordinates and altitude of measurement
    data = data.filter(["longitude", "latitude", "elevation"])
    # Cut the data into the cut_region
    inside = vd.inside((data.longitude, data.latitude), cut_region)
    data = data[inside]
    # Project coordinates
    projection = pyproj.Proj(proj="merc", lat_ts=(cut_region[2] + cut_region[3]) / 2)
    data["easting"], data["northing"] = projection(
        data.longitude.values, data.latitude.values
    )
    # Move projected coordinates to the boundaries of the region argument
    w, e, s, n = region[:4]
    data["easting"] = (e - w) / (data.easting.max() - data.easting.min()) * (
        data.easting - data.easting.min()
    ) + w
    data["northing"] = (n - s) / (data.northing.max() - data.northing.min()) * (
        data.northing - data.northing.min()
    ) + s
    # Drop longitude and latitude from dataframe
    data = data.filter(["easting", "northing", "elevation"])
    return data
