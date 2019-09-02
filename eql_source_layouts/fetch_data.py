"""
Fetch and load data
"""
import os
import pooch
import pandas as pd

POOCH = pooch.create(
    path=pooch.os_cache("eql_source_layouts"),
    base_url="https://github.com/pinga-lab/eql_source_layouts/raw/master/data/",
    registry=None,
)
POOCH.load_registry(os.path.join(os.path.dirname(__file__), "registry.txt"))


def fetch_airborne_gravity():
    """
    Fetch and load airborne gravity: GRAV-D Data Block PN02

    Returns
    -------
    df : :class:`pandas.DataFrame`
        :class:`pandas.DataFrame` containing the data read from GRAV-D Data Block PN02.
        The `latitude` and `longitude` are in degrees, `ellipsoidal_height` in meters
        and `full_gravity` in mGal.

    References
    ----------
    - GRAV-D Team (2018). "Gravity for the Redefinition of the American Vertical Datum
    (GRAV-D) Project, Airborne Gravity Data; Block PN02". Available August 2019. Online
    at: https://www.ngs.noaa.gov/GRAV-D/data_pn02.shtml
    """
    # Extract the file that contains the airborne gravity data
    unpack = pooch.Unzip(members=["NGS_GRAVD_Block_PN02_Gravity_Data_BETA1.txt"])
    # Pass in the processor to unzip the data file
    fnames = POOCH.fetch("NGS_GRAVD_Block_PN02_BETA1.zip", processor=unpack)
    # Returns the paths of all extract members (in our case, only one)
    fname = fnames[0]
    return _read_airborne_gravity_datafile(fname)


def _read_airborne_gravity_datafile(fname):
    """
    Read GRAV-D airborne gravity datafile

    Parameters
    ----------
    fname : str or Path
        Path to the unzipped NGS_GRAVD_Block_PN02_Gravity_Data_BETA1.txt file that
        contains the airborne gravity survey data.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        :class:`pandas.DataFrame` containing the data read from GRAV-D Data Block PN02.
        The `latitude` and `longitude` are in degrees, `ellipsoidal_height` in meters
        and `full_gravity` in mGal.
    """
    columns = [
        "block_line_number",
        "date",
        "latitude",
        "longitude",
        "ellipsoidal_height",
        "full_gravity",
    ]
    df = pd.read_csv(fname, sep=r"\s+", names=columns)
    # Convert the dates into numpy.datetime64
    # According to the NGS GRAV-D General Airborne Gravity Data User Manual the dates
    # are given in UTC time and following the format: yyyymmddHHMMSSFFF, where yyyy is
    # the year, mm the month, dd the day, HH the hour, MM the minutes, SS the seconds
    # and FFF the miliseconds.
    df.date = pd.to_datetime(df.date, format="%Y%m%d%H%M%S%f")
    return df
