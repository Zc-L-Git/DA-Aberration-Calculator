import sympy as sp


def no_electric_field(voltage):
    """
    Return a constant scalar potential U(x, y, z) with zero electric field.
    """
    x, y, z = sp.symbols("x y z")
    potential = 0 * x + 0 * y + 0 * z + voltage
    return sp.lambdify((x, y, z), potential)


def no_electric_field_axis_corrected(voltage):
    """
    Return a z-only scalar potential with the empirical correction factor
    used in the original script.
    """
    z = sp.symbols("z")
    potential = (0 * z + voltage) * (1 + 0.978e-6 * (0 * z + voltage))
    return sp.lambdify(z, potential)
