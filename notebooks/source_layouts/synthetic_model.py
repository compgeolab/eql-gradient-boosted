"""
Create synthetic model made out of prisms
"""
import numpy as np
from matplotlib.patches import Rectangle


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
    prisms.append([0, 0.5, 0, 1, 0.1, 0.3])
    densities.append(150)
    prisms.append([0.5, 1, 0, 1, 0, 0.2])
    densities.append(150)

    # Create one big deep prism to simulate a long wavelength
    prisms.append([0.2, 0.8, 0.2, 0.8, 0.3, 0.5])
    densities.append(100)

    # Add a horseshoe
    horseshoe = _horseshoe(
        radius=0.15,
        center=(0.35, 0.35),
        prism_size=0.01,
        bottom=0.3,
        top=1,
        density=-900,
        min_angle=110,
        max_angle=260,
    )
    prisms += horseshoe["prisms"]
    densities += horseshoe["densities"]

    # Add an horizontal large dike
    prisms.append([0.05, 0.85, 0.82, 0.83, 0.3, 1])
    densities.append(300)

    # # Add dikes
    prisms.append([0.80, 0.82, 0.66, 0.68, 0.1, 1])
    densities.append(500)
    prisms.append([0.44, 0.46, 0.53, 0.55, 0.1, 1])
    densities.append(500)

    # Add basin
    basin = _basin(
        center=(0.65, 0.2),
        bottom=0.6,
        top=1,
        bottom_size=0.05,
        top_size=0.2,
        density=-200,
        n_prisms=5,
    )
    prisms += basin["prisms"]
    densities += basin["densities"]

    # # Add diagonal dike
    dike = _dipping_dike(
        n_prisms=20,
        dike_density=300,
        deep_point=(0.60, 0.35, 0.3),
        shallow_point=(0.78, 0.48, 1),
        size=0.04,
    )
    prisms += dike["prisms"]
    densities += dike["densities"]

    # Scale prisms to the passed region
    prisms = _scale_model(prisms, region)
    # Create a collection of Rectangle from the prisms
    rectangles = _create_rectangles(prisms)
    return {"prisms": prisms, "densities": densities, "rectangles": rectangles}


def _create_rectangles(prisms, fill=False):
    """
    Create a set of matplotlib.patches.Rectangle from prisms
    """
    rectangles = []
    for prism in prisms:
        w, e, s, n = prism[:4]
        rectangles.append(Rectangle((w, s), width=(e - w), height=(n - s), fill=fill))
    return rectangles


def _scale_model(prisms, region):
    """
    Scale the prisms model accoring to the passed region

    The prisms model is defined on a region that spans between 0 and 1 on every
    direction (easting, northing, upward). It is then rescaled to meet the boundaries of
    the passed region.
    """
    w, e, s, n, bottom, top = region[:]
    prisms = np.atleast_2d(prisms)
    prisms[:, :2] *= e - w
    prisms[:, :2] += w
    prisms[:, 2:4] *= n - s
    prisms[:, 2:4] += s
    prisms[:, 4:6] *= top - bottom
    prisms[:, 4:6] += bottom
    return prisms


def _dipping_dike(n_prisms, dike_density, deep_point, shallow_point, size):
    """
    Add a dipping dike made out of prisms
    """
    prisms = []
    densities = []
    t = np.linspace(0, 1, n_prisms)
    easting_center = (shallow_point[0] - deep_point[0]) * t + deep_point[0]
    northing_center = (shallow_point[1] - deep_point[1]) * t + deep_point[1]
    upward_center = (shallow_point[2] - deep_point[2]) * t + deep_point[2]
    thicknes = (shallow_point[2] - deep_point[2]) / n_prisms
    for easting, northing, upward in zip(
        easting_center, northing_center, upward_center
    ):
        prisms.append(
            [
                easting - size / 2,
                easting + size / 2,
                northing - size / 2,
                northing + size / 2,
                upward - thicknes / 2,
                upward + thicknes / 2,
            ]
        )
        densities.append(dike_density)
    return {"prisms": prisms, "densities": densities}


def _basin(center, bottom, top, bottom_size, top_size, density, n_prisms):
    """
    Add a basin made out of prisms
    """
    prisms = []
    densities = []
    thickness = (top - bottom) / n_prisms
    upward_centers = np.linspace(bottom + thickness / 2, top - thickness / 2, n_prisms)
    for upward in upward_centers:
        size = (top_size - bottom_size) / (top - bottom) * (upward - top) + top_size
        prisms.append(
            [
                center[0] - size / 2,
                center[0] + size / 2,
                center[1] - size / 2,
                center[1] + size / 2,
                upward - thickness / 2,
                upward + thickness / 2,
            ]
        )
        densities.append(density)
    return {"prisms": prisms, "densities": densities}


def _horseshoe(
    radius, center, prism_size, bottom, top, density, min_angle=90, max_angle=240
):
    """
    Add a horseshoe shaped body made out of prisms

    Parameters
    ----------
    radius : float
        Radius of the horseshoe in (0, 1) units.
    center : tuple
        Coordinates of the center of the horseshoe in (0, 1) units.
    prism_size : float
        Horizontal size of each one of the prisms that will construct the horseshoe.
        A smaller prism size will increase the number of prisms.
    bottom : float
        Bottom coordinate of every prism in the horseshoe in (0, 1) units.
    top : float
        Top coordinate of every prism in the horseshoe in (0, 1) units.
    density : float
        Density of the horseshoe in kg/m^3.
    min_angle : float
        Angle from which the horseshoe will start (in degrees). The angle is meassured
        from the easting axes on counterclockwise orientation.
    max_angle : float
        Angle to which the horseshoe will end (in degrees). The angle is meassured
        from the easting axes on counterclockwise orientation.

    Returns
    -------
    model : dict
        Dictionary containing the prisms and the density for each prism.
    """
    prisms, densities = [], []
    d_angle = 2 * np.arcsin(np.sqrt(2) / 2 * prism_size / radius)
    for angle in np.arange(np.radians(min_angle), 1.1 * np.radians(max_angle), d_angle):
        easting_center = radius * np.cos(angle)
        northing_center = radius * np.sin(angle)
        prisms.append(
            [
                easting_center - prism_size / 2 + center[0],
                easting_center + prism_size / 2 + center[0],
                northing_center - prism_size / 2 + center[1],
                northing_center + prism_size / 2 + center[1],
                bottom,
                top,
            ]
        )
        densities.append(density)
    return {"prisms": prisms, "densities": densities}
