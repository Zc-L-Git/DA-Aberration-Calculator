"""
Paraxial Spatial Expansion Library for Electromagnetic Fields
Used for multipole expansion calculations of electromagnetic fields in electron optical systems
"""

import sympy as sp
import numba
import numpy as np
from sympy.functions.combinatorial.factorials import factorial

def compute_magnetic_field_expansion(axial_field_function, expansion_order):
    """
    Compute the Cartesian-coordinate spatial expansion of a rotationally symmetric magnetic field
    
    Parameters:
        axial_field_function: Axially symmetric magnetic field function B(z)
        expansion_order: Expansion order
        
    Returns:
        field_function: Compiled magnetic field computation function
        symbolic_function: Symbolic expression function
        z_derivative: Expression of the z-component of the magnetic field
    """
    
    # Define symbolic variables
    x, y, z = sp.symbols('x y z')
    summation_index, multipole_order = sp.symbols('k m', integer=True)
    series_start, series_end = sp.symbols('i j', integer=True)
    
    # Define axial field function
    axial_field = sp.Function('B')(z, multipole_order)
    
    # Compute radial trigonometric component
    radial_trig_component = sp.Sum(
        ((-1)**summation_index * factorial(multipole_order)) /
        (factorial(2**summation_index) * factorial(multipole_order - 2*summation_index)) *
        (x**(multipole_order - 2*summation_index) * y**(2*summation_index)),
        (summation_index, 0, multipole_order)
    )
    
    # Compute axial derivative series component
    axial_derivative_series = sp.Sum(
        ((-1)**summation_index * factorial(multipole_order)) /
        (4**summation_index * factorial(summation_index) * 
         factorial(multipole_order + summation_index)) *
        (x**2 + y**2)**summation_index *
        axial_field.diff((z, 2*summation_index)),
        (summation_index, series_start, series_end)
    )
    
    # Construct scalar potential function
    scalar_potential = axial_derivative_series * radial_trig_component
    
    # Set summation limits for the series
    axial_start, axial_end = 0, expansion_order
    transverse_start, transverse_end = 1, expansion_order + 1
    
    # Compute components in each direction
    potential_x = scalar_potential.subs(
        [[series_start, transverse_start], [series_end, transverse_end]]
    ).doit()
    
    potential_y = scalar_potential.subs(
        [[series_start, transverse_start], [series_end, transverse_end]]
    ).doit()
    
    potential_z = scalar_potential.subs(
        [[series_start, axial_start], [series_end, axial_end]]
    ).doit()
    
    # Compute magnetic field components (B = -∇φ, but magnetic scalar potential is defined differently)
    field_x_component = potential_x.subs(
        multipole_order, 0
    ).diff(x).subs(
        sp.diff(sp.Function('B')(z, 0), z), axial_field_function
    )
    
    field_y_component = potential_y.subs(
        multipole_order, 0
    ).diff(y).subs(
        sp.diff(sp.Function('B')(z, 0), z), axial_field_function
    )
    
    field_z_component = potential_z.subs(
        multipole_order, 0
    ).diff(z).subs(
        sp.diff(sp.Function('B')(z, 0), z), axial_field_function
    )
    
    # Create numerical computation function
    field_calculator = sp.lambdify(
        (x, y, z), 
        (field_x_component.doit(), field_y_component.doit(), field_z_component.doit()), 
        'math'
    )
    
    # Compile for optimization
    optimized_calculator = numba.jit(field_calculator)
    
    return optimized_calculator, field_calculator, field_z_component


def compute_electrostatic_field_expansion(electric_potential_function, expansion_order):
    """
    Compute the Cartesian-coordinate spatial expansion of a rotationally symmetric electrostatic field
    
    Parameters:
        electric_potential_function: Axially symmetric electric potential function U(z)
        expansion_order: Expansion order
        
    Returns:
        field_function: Compiled electric field computation function
        potential_function: Compiled electric potential computation function
        symbolic_field: Symbolic electric field function
        symbolic_potential: Symbolic electric potential function
    """
    
    # Define symbolic variables
    x, y, z = sp.symbols('x y z')
    summation_index, multipole_order = sp.symbols('k m', integer=True)
    series_start, series_end = sp.symbols('i j', integer=True)
    
    # Define electric potential function
    electric_potential = sp.Function('U')(z, multipole_order)
    
    # Compute radial trigonometric component
    radial_trig_component = sp.Sum(
        ((-1)**summation_index * factorial(multipole_order)) /
        (factorial(2**summation_index) * factorial(multipole_order - 2*summation_index)) *
        (x**(multipole_order - 2*summation_index) * y**(2*summation_index)),
        (summation_index, 0, multipole_order)
    )
    
    # Compute axial derivative series component
    axial_derivative_series = sp.Sum(
        ((-1)**summation_index * factorial(multipole_order)) /
        (4**summation_index * factorial(summation_index) * 
         factorial(multipole_order + summation_index)) *
        (x**2 + y**2)**summation_index *
        electric_potential.diff((z, 2*summation_index)),
        (summation_index, series_start, series_end)
    )
    
    # Construct electric potential function
    potential_function = axial_derivative_series * radial_trig_component
    
    # Set summation limits for the series
    axial_start, axial_end = 0, expansion_order
    transverse_start, transverse_end = 1, expansion_order + 1
    
    # Compute components in each direction
    potential_x = potential_function.subs(
        [[series_start, transverse_start], [series_end, transverse_end]]
    ).doit()
    
    potential_y = potential_function.subs(
        [[series_start, transverse_start], [series_end, transverse_end]]
    ).doit()
    
    potential_z = potential_function.subs(
        [[series_start, axial_start], [series_end, axial_end]]
    ).doit()
    
    # Compute scalar electric potential (for verification)
    scalar_potential = potential_z.subs(
        multipole_order, 0
    ).subs(sp.Function('U')(z, 0), electric_potential_function).doit()
    
    # Compute electric field components (E = -∇U)
    field_x_component = -1 * potential_x.subs(
        multipole_order, 0
    ).diff(x).subs(sp.Function('U')(z, 0), electric_potential_function)
    
    field_y_component = -1 * potential_y.subs(
        multipole_order, 0
    ).diff(y).subs(sp.Function('U')(z, 0), electric_potential_function)
    
    field_z_component = -1 * potential_z.subs(
        multipole_order, 0
    ).diff(z).subs(sp.Function('U')(z, 0), electric_potential_function)
    
    # Create numerical computation functions
    potential_calculator = sp.lambdify(
        (x, y, z), scalar_potential, 'numpy'
    )
    
    field_calculator = sp.lambdify(
        (x, y, z), 
        [field_x_component.doit(), field_y_component.doit(), field_z_component.doit()], 
        'numpy'
    )
    
    # Compile for optimization
    optimized_field_calculator = numba.jit(field_calculator)
    optimized_potential_calculator = numba.jit(potential_calculator)
    
    return (optimized_field_calculator, optimized_potential_calculator, 
            field_calculator, potential_calculator)