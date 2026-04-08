from scipy.interpolate import lagrange


def lagrange_interpolation(z_points, values, query_z):
    """
    Perform 1D Lagrange interpolation along z.
    """
    poly = lagrange(z_points, values)
    return poly(query_z)
