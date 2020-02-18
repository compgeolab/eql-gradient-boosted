"""
Functions for plotting a prism on PyVista
"""
import vtk
import numpy as np
import pyvista as pv

# Define prism faces array ready to be passed to PyVista.
# The faces array must be a 2d array of integers. Each row must start with the number of
# vertices belonging to the face followed by the indices of the vertices that form the
# corresponding face. The returned faces arrays correspond with the following order of
# prism vertices: first changing upwards, then northing and lastly easting.
PRISM_FACES = np.array(
    [
        [4, 0, 1, 3, 2],
        [4, 2, 3, 7, 6],
        [4, 4, 5, 7, 6],
        [4, 0, 1, 5, 4],
        [4, 1, 3, 7, 5],
        [4, 0, 2, 6, 4],
    ],
    dtype=np.int8,
)
VERTICES_ORDER_FOR_GRID = [0, 4, 6, 2, 1, 5, 7, 3]


def plot_prisms(prisms, density):
    """
    Convert a set of prisms to a PyVista.UnstructuredGrid
    """
    n_prisms = len(prisms)
    n_vertices = 8 * n_prisms
    vertices = np.vstack(
        tuple(_prism_vertices(p)[VERTICES_ORDER_FOR_GRID] for p in prisms)
    )
    # Define cells array
    # Contains information on the points composing each cell (8 for prisms).
    # Each cell begins with the number of points in the cell (8) and then the points
    # composing the cell.
    cells = np.arange(n_vertices).reshape(n_prisms, 8)
    n_vertices_per_cell = np.full(n_prisms, 8)[
        :, np.newaxis
    ]  # vertical array full of 8
    cells = np.hstack((n_vertices_per_cell, cells)).ravel()
    # Define offset array
    # Identifies the start of each cell in the cells array
    offset = np.arange(0, n_vertices + 1, 8 + 1)
    # Define cell_type array
    cell_type = np.full(n_prisms, vtk.VTK_HEXAHEDRON)
    grid = pv.UnstructuredGrid(offset, cells, cell_type, vertices)
    grid.cell_arrays["density"] = np.array(density)
    return grid


def _prism_vertices(prism):
    """
    Return list of prism vertices

    Parameters
    ----------
    prism : list, tuple or array
        Coordinates of the prism boundaries.

    Returns
    -------
    vertices : array
        Coordinates of each prism vertex.
    """
    vertices = np.vstack(
        [
            i.ravel()
            for i in np.meshgrid(prism[:2], prism[2:4], prism[4:], indexing="ij")
        ]
    ).T
    return vertices
