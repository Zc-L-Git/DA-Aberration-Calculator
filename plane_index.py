import numpy as np
from scipy.spatial import cKDTree


def create_plane_trees(x, y, z, bx, by, bz, tolerance=1e-8):
    """
    Build a KD-tree for each unique z plane.

    Returns
    -------
    dict
        Mapping: plane_z -> (x_plane, y_plane, bx_plane, by_plane, bz_plane, tree)
    """
    unique_z = np.unique(z)
    plane_dict = {}

    for plane_z in unique_z:
        mask = np.abs(z - plane_z) < tolerance
        if not np.any(mask):
            continue

        x_plane = x[mask]
        y_plane = y[mask]
        bx_plane = bx[mask]
        by_plane = by[mask]
        bz_plane = bz[mask]

        points = np.column_stack((x_plane, y_plane))
        tree = cKDTree(points)

        plane_dict[plane_z] = (x_plane, y_plane, bx_plane, by_plane, bz_plane, tree)

    return plane_dict


def find_nearest_planes(plane_dict, query_z, num_planes=2):
    """
    Find the z planes closest to query_z.
    """
    if not plane_dict:
        raise ValueError("plane_dict is empty")

    plane_z_values = np.array(list(plane_dict.keys()))
    num_planes = min(num_planes, len(plane_z_values))
    distances = np.abs(plane_z_values - query_z)

    nearest_indices = np.argpartition(distances, num_planes - 1)[:num_planes]
    sorted_indices = nearest_indices[np.argsort(distances[nearest_indices])]
    return plane_z_values[sorted_indices]


def find_points_in_plane(plane_dict, plane_z, query_xy, num_points=16):
    """
    Find nearby points in a given z plane.
    """
    if plane_z not in plane_dict:
        return None

    x_plane, y_plane, bx_plane, by_plane, bz_plane, tree = plane_dict[plane_z]
    n_points = min(num_points, len(x_plane))
    distances, indices = tree.query(query_xy, k=n_points)

    return (
        x_plane[indices],
        y_plane[indices],
        bx_plane[indices],
        by_plane[indices],
        bz_plane[indices],
        distances,
    )
