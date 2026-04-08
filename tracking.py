import numpy as np
from scipy.constants import e, m_e
from tqdm import tqdm

from field_evaluator import analytic_expr_bx_by_bz


ELECTRON_CHARGE = e
ELECTRON_MASS = m_e
LIGHT_SPEED = 299792458.0


def _a_to_x1(a, b, d):
    d_term = np.sqrt((1 + d) ** 2 - a**2 - b**2)
    x1 = a / d_term
    y1 = b / d_term
    return x1, y1


def _x1_to_a(x1, y1, d):
    t = np.sqrt(1 + x1**2 + y1**2)
    a = (1 + d) * x1 / t
    b = (1 + d) * y1 / t
    return a, b


def _compute_inv_brho(voltage, param):
    a0 = (ELECTRON_CHARGE * voltage) / (ELECTRON_MASS * LIGHT_SPEED**2)
    p0 = ELECTRON_MASS * LIGHT_SPEED * np.sqrt(a0 * (2.0 + a0))

    def eps_to_delta(eps):
        return np.sqrt(((1.0 + eps) * (2.0 + a0 * (1.0 + eps))) / (2.0 + a0)) - 1.0

    param[4] = eps_to_delta(param[4])
    inv_brho = (ELECTRON_CHARGE / p0) / (1.0 + param[4])
    return inv_brho, param


def _particle_tracking_step(param, bx_val, by_val, bz_val, inv_brho, dz):
    p_val = np.sqrt(1 + param[1] ** 2 + param[3] ** 2)
    bt_val = (bz_val + param[1] * bx_val + param[3] * by_val) / p_val
    ax_val = p_val**2 * inv_brho * (p_val * by_val - param[3] * bt_val)
    ay_val = p_val**2 * inv_brho * (-p_val * bx_val + param[1] * bt_val)

    param[1] += ax_val * dz
    param[3] += ay_val * dz
    param[0] += param[1] * dz
    param[2] += param[3] * dz
    return param


def euler_dz(z0, param, start_param, plane_dict, z_range, potential_func, dz, steps, show_progress=True):
    """
    Advance DA and real particles with an Euler integrator.
    """
    voltage_da = potential_func(param[0], param[2], z0)
    voltage_real = potential_func(start_param[0], start_param[2], z0)

    inv_brho_da, param = _compute_inv_brho(voltage_da, param)
    inv_brho_real, start_param = _compute_inv_brho(voltage_real, start_param)

    param[1], param[3] = _a_to_x1(param[1], param[3], param[4])

    iterator = range(1, steps)
    if show_progress:
        iterator = tqdm(iterator)

    for _ in iterator:
        f_bx, f_by, f_bz, _, _, _ = analytic_expr_bx_by_bz(
            start_param[0], start_param[2], z0, plane_dict, z_range
        )

        bx_da = f_bx(param[0], param[2], z0)
        by_da = f_by(param[0], param[2], z0)
        bz_da = f_bz(param[0], param[2], z0)

        bx_real = f_bx(start_param[0], start_param[2], z0)
        by_real = f_by(start_param[0], start_param[2], z0)
        bz_real = f_bz(start_param[0], start_param[2], z0)

        param = _particle_tracking_step(param, bx_da, by_da, bz_da, inv_brho_da, dz)
        start_param = _particle_tracking_step(start_param, bx_real, by_real, bz_real, inv_brho_real, dz)

        z0 += dz

    param[1], param[3] = _x1_to_a(param[1], param[3], param[4])
    return param
