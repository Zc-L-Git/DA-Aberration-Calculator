import traceback
import numpy as np
from sympy import symbols, lambdify

from plane_index import find_nearest_planes, find_points_in_plane
from interpolation import lagrange_interpolation
from field_fitting import fit_polynomial_2d_symmetric, create_sympy_expr


def analytic_expr_bx_by_bz(xi, yi, zi, plane_dict, z_range):
    """
    Build local analytic field expressions around a query point.

    Returns
    -------
    tuple
        f_bx, f_by, f_bz, poly_expr_bx, poly_expr_by, poly_expr_bz
    """
    if zi < z_range[0] or zi > z_range[1]:
        def zero_field(x_val, y_val, z_val=None):
            return 0.0
        return zero_field, zero_field, zero_field, None, None, None

    try:
        nearest_planes = find_nearest_planes(plane_dict, zi, num_planes=2)

        all_points = []
        for plane_z in nearest_planes:
            result = find_points_in_plane(plane_dict, plane_z, (xi, yi), num_points=16)
            if result is None:
                def zero_field(x_val, y_val, z_val=None):
                    return 0.0
                return zero_field, zero_field, zero_field, None, None, None
            x_pts, y_pts, bx_pts, by_pts, bz_pts, _ = result
            all_points.append((x_pts, y_pts, bx_pts, by_pts, bz_pts))

        n_points = min(len(p[0]) for p in all_points)
        x_interp = np.zeros(n_points)
        y_interp = np.zeros(n_points)
        bx_interp = np.zeros(n_points)
        by_interp = np.zeros(n_points)
        bz_interp = np.zeros(n_points)

        for i in range(n_points):
            z_points = nearest_planes
            bx_points = [plane[2][i] for plane in all_points]
            by_points = [plane[3][i] for plane in all_points]
            bz_points = [plane[4][i] for plane in all_points]

            x_interp[i] = all_points[0][0][i]
            y_interp[i] = all_points[0][1][i]
            bx_interp[i] = lagrange_interpolation(z_points, bx_points, zi)
            by_interp[i] = lagrange_interpolation(z_points, by_points, zi)
            bz_interp[i] = lagrange_interpolation(z_points, bz_points, zi)

        dx = x_interp - xi
        dy = y_interp - yi
        dist_xy = np.sqrt(dx**2 + dy**2)
        weights = 1.0 / (dist_xy**2 + 1e-8)
        weights = weights / np.sum(weights)

        coeffs_bx, coeffs_by = fit_polynomial_2d_symmetric(
            x_interp, y_interp, bx_interp, by_interp, bz_interp, weights
        )

        poly_expr_bx, poly_expr_by = create_sympy_expr(coeffs_bx, coeffs_by)
        poly_expr_bz = 0

        x_sym, y_sym = symbols("x y")
        f_bx = lambdify((x_sym, y_sym), poly_expr_bx, "numpy")
        f_by = lambdify((x_sym, y_sym), poly_expr_by, "numpy")
        f_bz = lambdify((x_sym, y_sym), poly_expr_bz, "numpy")

        def wrap_func(func):
            def wrapped(x_val, y_val, z_val=None):
                return func(x_val, y_val)
            return wrapped

        return (
            wrap_func(f_bx),
            wrap_func(f_by),
            wrap_func(f_bz),
            poly_expr_bx,
            poly_expr_by,
            poly_expr_bz,
        )

    except Exception as exc:
        print(f"Error calculating field at ({xi}, {yi}, {zi}): {exc}")
        traceback.print_exc()

        def error_field(x_val, y_val, z_val=None):
            return 0.0

        return error_field, error_field, error_field, None, None, None
