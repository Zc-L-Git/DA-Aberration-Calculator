"""
Electron Trajectory Solver Library
Used to compute electron trajectories in electromagnetic fields
"""

import numpy as np
from tqdm import tqdm

def linearized_electromagnetic_euler_cromer(
    position_array, 
    velocity_array, 
    z_coordinates, 
    magnetic_field_z, 
    step_size, 
    num_steps, 
    electric_potential_z,
    electric_potential_dz,
    electric_potential_ddz,
    charge,
    mass,
    field_scale_factor_v,
    field_scale_factor_b
):
    """
    Euler–Cromer method for solving the equations of electron motion in a linearized electromagnetic field
    
    Parameters:
        position_array: Radial position array
        velocity_array: Radial velocity array
        z_coordinates: Axial coordinate array
        magnetic_field_z: Axial magnetic field function
        step_size: Axial step size
        num_steps: Total number of steps
        electric_potential_z: Axial electric potential function
        electric_potential_dz: First derivative of axial electric potential
        electric_potential_ddz: Second derivative of axial electric potential
        charge: Electron charge
        mass: Electron mass
        field_scale_factor_v: Potential scaling factor
        field_scale_factor_b: Magnetic field scaling factor
        
    Returns:
        position_array: Updated position array
        velocity_array: Updated velocity array
    """
    
    current_index = 1
    
    for step in tqdm(range(1, num_steps)):
        # Get current z-coordinate
        current_z = z_coordinates[current_index - 1]
        
        # Compute velocity increment
        term1 = 0.5 * (electric_potential_ddz(current_z) / 
                      electric_potential_z(current_z)) * velocity_array[current_index - 1]
        
        term2 = 1.0 / (4.0 * (electric_potential_z(current_z) / field_scale_factor_v))
        term3 = (electric_potential_dz(current_z) / field_scale_factor_v) + \
                (charge / (2.0 * mass)) * (magnetic_field_z(current_z) / field_scale_factor_b)**2
        
        velocity_increment = -(term1 + term2 * term3 * position_array[current_index - 1]) * step_size
        
        # Update velocity
        velocity_array[current_index] = velocity_array[current_index - 1] + velocity_increment
        
        # Update position
        position_array[current_index] = position_array[current_index - 1] + \
                                        velocity_array[current_index] * step_size
        
        current_index += 1
    
    return position_array, velocity_array


def electromagnetic_motion_equation(
    axial_position,
    state_vector,
    field_constant,
    electric_field_function,
    electric_potential_function,
    axial_potential_function,
    axial_potential_derivative,
    magnetic_field_function,
    axial_magnetic_field_function,
    axial_magnetic_field_derivative
):
    """
    Equations of motion of an electron in an electromagnetic field in a rotating Cartesian coordinate system
    
    Parameters:
        axial_position: Axial position
        state_vector: State vector [x, x', y, y', d]
        field_constant: Field constant eta = sqrt(q/(2m))
        electric_field_function: Electric field computation function
        electric_potential_function: Electric potential computation function
        axial_potential_function: Axial electric potential function
        axial_potential_derivative: Derivative of axial electric potential
        magnetic_field_function: Magnetic field computation function
        axial_magnetic_field_function: Axial magnetic field function
        axial_magnetic_field_derivative: Derivative of axial magnetic field
        
    Returns:
        Derivative of state vector [x', x'', y', y'', 0]
    """
    
    # Extract state variables
    x_pos = state_vector[0]
    x_vel = state_vector[1]
    y_pos = state_vector[2]
    y_vel = state_vector[3]
    
    # Compute magnetic field components
    B_x, B_y, B_z = magnetic_field_function(x_pos, y_pos, axial_position)
    
    # Compute electric field components and potential
    E_x, E_y, E_z = electric_field_function(x_pos, y_pos, axial_position)
    potential = electric_potential_function(x_pos, y_pos, axial_position)
    potential = (1 - state_vector[4])*potential
    
    # Compute intermediate parameters
    axial_potential = axial_potential_function(axial_position)
    axial_potential_deriv = axial_potential_derivative(axial_position)
    
    # Compute rotation angle parameters
    theta_prime = (field_constant / 2.0) * \
                  (axial_magnetic_field_function(axial_position) / np.sqrt(axial_potential))
    
    theta_double_prime = (field_constant / 2.0) * (
        axial_magnetic_field_derivative(axial_position) / np.sqrt(axial_potential) - 
        0.5 * axial_magnetic_field_function(axial_position) * 
        (axial_potential**(-1.5)) * axial_potential_deriv
    )
    
    # Compute rotation terms
    rotation_x = 2.0 * theta_prime * y_vel + (theta_prime**2) * x_pos + theta_double_prime * y_pos
    rotation_y = -2.0 * theta_prime * x_vel + (theta_prime**2) * y_pos - theta_double_prime * x_pos
    
    # Compute momentum factor
    momentum_factor = np.sqrt(
        1.0 + (x_vel - theta_prime * y_pos)**2 + (y_vel + theta_prime * x_pos)**2
    )
    
    # Compute tangential magnetic field component
    tangential_B = (1.0 / momentum_factor) * (
        B_z + (x_vel - theta_prime * y_pos) * B_x + (y_vel + theta_prime * x_pos) * B_y
    )
    
    # Compute acceleration components
    acceleration_x = (
        -(momentum_factor**2) / (2.0 * potential) * 
        (E_x - (x_vel - theta_prime * y_pos) * E_z) + 
        (field_constant * (momentum_factor**2)) / np.sqrt(potential) * 
        (momentum_factor * B_y - (y_vel + theta_prime * x_pos) * tangential_B) + 
        rotation_x
    )
    
    acceleration_y = (
        -(momentum_factor**2) / (2.0 * potential) * 
        (E_y - (y_vel + theta_prime * x_pos) * E_z) + 
        (field_constant * (momentum_factor**2)) / np.sqrt(potential) * 
        (-momentum_factor * B_x + (x_vel - theta_prime * y_pos) * tangential_B) + 
        rotation_y
    )
    
    return np.array([x_vel, acceleration_x, y_vel, acceleration_y, 0])