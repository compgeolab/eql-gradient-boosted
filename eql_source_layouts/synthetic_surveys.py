"""
Create synthetic ground and airborne surveys
"""
import pyproj
import verde as vd
import harmonica as hm


def airborne_survey(region, cut_region=(-5.0, -3.5, 55.5, 56.5)):
    """
    Create a synthetic airborne survey based on Great Britain magnetic measurements

    Parameters
    ----------
    region : tuple or list
        Boundaries of the synthetic region where the observation points will be located
        in the following order: (``east``, ``west``, ``south``, ``north``, ...). All
        subsequent boundaries will be ignored. All boundaries should be in Cartesian
        coordinates and in meters.
    cut_region : tuple (optional)
        Region to reduce the extension of the survey. Must be boundaries of the original
        survey, in degrees.

    Returns
    -------
    survey : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the observation points and their
        elevation Cartesian coordinates for the synthetic model. All coordinates and
        altitude are in meters.
    """
    # Fetch airborne magnetic survey from Great Britain
    survey = hm.datasets.fetch_britain_magnetic()
    # Rename the "altitude_m" column to "elevation"
    survey["elevation"] = survey["altitude_m"]
    # Cut the region into the cut_region, project it with a mercator projection to
    # convert the coordinates into Cartesian and move this Cartesian region into the
    # passed region
    survey = _synthetic_survey(survey, region, cut_region)
    return survey


def ground_survey(region, cut_region=(13.60, 20.30, -24.20, -17.5)):
    """
    Create a synthetic ground survey based on South Africa gravity measurements

    Parameters
    ----------
    region : tuple or list
        Boundaries of the synthetic region where the observation points will be located
        in the following order: (``east``, ``west``, ``south``, ``north``, ...). All
        subsequent boundaries will be ignored. All boundaries should be in Cartesian
        coordinates and in meters.
    cut_region : tuple (optional)
        Region to reduce the extension of the survey. Must be boundaries of the original
        survey, in degrees.

    Returns
    -------
    survey : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the observation points and their
        elevation Cartesian coordinates for the synthetic model. All coordinates and
        altitude are in meters.
    """
    # Fetch ground gravity survey from South Africa
    survey = hm.datasets.fetch_south_africa_gravity()
    # Cut the region into the cut_region, project it with a mercator projection to
    # convert the coordinates into Cartesian and move this Cartesian region into the
    # passed region
    survey = _synthetic_survey(survey, region, cut_region)
    return survey


def _synthetic_survey(survey, region, cut_region):
    """
    Cut, project and move the original survey to the passed region

    Parameters
    ----------
    survey : :class:`pandas.DataFrame`
        Original survey as a :class:`pandas.DataFrame` containing the following columns:
        ``longitude``, ``latitude`` and ``elevation``. The ``longitude`` and
        ``latitude`` must be in degrees and the ``elevation`` in meters.
    region : tuple or list
        Boundaries of the synthetic region where the observation points will be located
        in the following order: (``east``, ``west``, ``south``, ``north``, ...). All
        subsequent boundaries will be ignored. All boundaries should be in Cartesian
        coordinates and in meters.
    cut_region : tuple (optional)
        Region to reduce the extension of the survey. Must be boundaries of the original
        survey, in degrees.

    Returns
    -------
    survey : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the observation points and their
        elevation Cartesian coordinates for the synthetic model. All coordinates and
        altitude are in meters.
    """
    # Cut the data into the cut_region
    inside = vd.inside((survey.longitude, survey.latitude), cut_region)
    survey = survey[inside]
    # Project coordinates
    projection = pyproj.Proj(proj="merc", lat_ts=(cut_region[2] + cut_region[3]) / 2)
    survey["easting"], survey["northing"] = projection(
        survey.longitude.values, survey.latitude.values
    )
    # Move projected coordinates to the boundaries of the region argument
    w, e, s, n = region[:4]
    survey["easting"] = (e - w) / (survey.easting.max() - survey.easting.min()) * (
        survey.easting - survey.easting.min()
    ) + w
    survey["northing"] = (n - s) / (survey.northing.max() - survey.northing.min()) * (
        survey.northing - survey.northing.min()
    ) + s
    # Keep only the easting, northing and elevation on the DataFrame
    survey = survey.filter(["easting", "northing", "elevation"])
    return survey
