"""
Numerical Integration Methods Library
Provides various numerical integration algorithms for solving differential equations
"""

import numpy as np
from tqdm import tqdm

def runge_kutta_4th_order(differential_equation, initial_x, initial_y, final_x, num_steps, additional_args=()):
    """
    Fourth-order Runge–Kutta method for solving systems of ordinary differential equations
    
    Parameters:
        differential_equation: Differential equation function dy/dx = f(x, y, *args)
        initial_x: Initial value of the independent variable
        initial_y: Initial value vector of the dependent variable
        final_x: Final value of the independent variable
        num_steps: Number of integration steps
        additional_args: Additional arguments passed to the differential equation
        
    Returns:
        x_values: Array of discretized independent variable values
        final_y: Value of the dependent variable at the final integration point
    """
    
    # Initialize storage arrays
    x_array = [0] * (num_steps + 1)
    y_array = [0] * (num_steps + 1)
    
    # Compute step size
    step_size = (final_x - initial_x) / float(num_steps)
    
    # Set initial values
    current_x = initial_x
    current_y = initial_y
    
    x_array[0] = current_x
    y_array[0] = current_y
    
    # Perform iterative integration
    for step_index in tqdm(range(1, num_steps + 1)):
        # Compute four slopes
        k1 = step_size * differential_equation(current_x, current_y, *additional_args)
        k2 = step_size * differential_equation(
            current_x + 0.5 * step_size, 
            current_y + 0.5 * k1, 
            *additional_args
        )
        k3 = step_size * differential_equation(
            current_x + 0.5 * step_size, 
            current_y + 0.5 * k2, 
            *additional_args
        )
        k4 = step_size * differential_equation(
            current_x + step_size, 
            current_y + k3, 
            *additional_args
        )
        
        # Update variables
        current_x = initial_x + step_index * step_size
        current_y = current_y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Store current values
        x_array[step_index] = current_x
        y_array[step_index] = current_y
    
    return x_array, y_array[-1]