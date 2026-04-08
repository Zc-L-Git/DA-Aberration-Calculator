"""
On-axis Electromagnetic Field Distribution Model Library (Symbolic Expression Version)
Function: Provide symbolic-expression-based mathematical models for on-axis electromagnetic field distributions
"""

import sympy as sp

# Define the symbolic variable z
z = sp.symbols('z', real=True)

def glaser_magnetic_lens_symbolic(B0, z0, a):
    """
    Glaser magnetic lens model (symbolic expression)
    
    Mathematical model: B = B0 / (1 + ((z+z0) / a)**2)
    
    Parameters:
        B0: Maximum on-axis magnetic field strength, scalar
        z0: Magnetic field center offset, scalar
        a: Characteristic length parameter, scalar
    
    Returns:
        sympy expression: symbolic expression of magnetic field distribution
    """
    return B0 / (1 + ((z + z0) / a) ** 2)

def schiske_electric_lens_symbolic(U0, z0, a, k):
    """
    Schiske electrostatic lens model (symbolic expression)
    
    Mathematical model: U = U0 * (1 - (k**2 / (1 + ((z+z0)/a)**2)))
    
    Parameters:
        U0: Reference electric potential, scalar
        z0: Potential center offset, scalar
        a: Characteristic length parameter, scalar
        k: Lens strength coefficient, scalar
    
    Returns:
        sympy expression: symbolic expression of electric potential distribution
    """
    return U0 * (1 - (k ** 2 / (1 + ((z + z0) / a) ** 2)))