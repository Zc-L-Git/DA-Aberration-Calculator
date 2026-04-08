import numpy as np
from sympy import symbols


def fit_polynomial_2d_symmetric(x, y, bx, by, bz, weights):
    """
    Fit symmetric 2D polynomial bases for Bx and By.

    Notes
    -----
    The basis is retained from the original implementation.
    Bz is currently returned by the caller as zero.
    """
    a = y
    b = y * (3 * x**2 + y**2)
    c = y * (5 * x**4 + 6 * x**2 * y**2 + y**4)
    d = y * (7 * x**6 + 15 * x**4 * y**2 + 9 * x**2 * y**4 + y**6)

    e = x
    f = x * (3 * y**2 + x**2)
    g = x * (5 * y**4 + 6 * x**2 * y**2 + x**4)
    h = x * (7 * y**6 + 15 * y**4 * x**2 + 9 * y**2 * x**4 + x**6)

    weights = weights / np.sum(weights)
    w_sqrt = np.sqrt(weights)

    x_bx = np.column_stack((a, b, c, d))
    x_bx_weighted = x_bx * w_sqrt[:, np.newaxis]
    bx_weighted = bx * w_sqrt
    coeffs_bx = np.linalg.lstsq(x_bx_weighted, bx_weighted, rcond=None)[0]

    x_by = np.column_stack((e, f, g, h))
    x_by_weighted = x_by * w_sqrt[:, np.newaxis]
    by_weighted = by * w_sqrt
    coeffs_by = np.linalg.lstsq(x_by_weighted, by_weighted, rcond=None)[0]

    avg_coeffs = (coeffs_bx + coeffs_by) / 2.0
    coeffs_bx = avg_coeffs
    coeffs_by = avg_coeffs
    return coeffs_bx, coeffs_by


def create_sympy_expr(coeffs_bx, coeffs_by):
    """
    Create symbolic expressions for Bx and By.
    """
    x, y = symbols("x y")

    a = y
    b = y * (3 * x**2 + y**2)
    c = y * (5 * x**4 + 6 * x**2 * y**2 + y**4)
    d = y * (7 * x**6 + 15 * x**4 * y**2 + 9 * x**2 * y**4 + y**6)

    e = x
    f = x * (3 * y**2 + x**2)
    g = x * (5 * y**4 + 6 * x**2 * y**2 + x**4)
    h = x * (7 * y**6 + 15 * y**4 * x**2 + 9 * y**2 * x**4 + x**6)

    a00, a01, a02, a03 = coeffs_bx

    poly_expr_bx = a00 * a + a01 * b + a02 * c + a03 * d
    poly_expr_by = a00 * e + a01 * f + a02 * g + a03 * h
    return poly_expr_bx, poly_expr_by
