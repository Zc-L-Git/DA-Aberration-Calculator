"""
Gaussian Wavelet Fitting Library
Used to fit complex curves using multiple Gaussian functions
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate

def gaussian_sum_function(x_coordinates, *parameters):
    """
    Superposition of multiple Gaussian functions
    
    Parameters:
        x_coordinates: Independent variable coordinate array
        parameters: Sequence of Gaussian parameters [A1, μ1, σ1, A2, μ2, σ2, ...]
        
    Returns:
        Array of function values after summing multiple Gaussian functions
    """
    
    # Determine the number of Gaussian functions
    num_gaussians = len(parameters) // 3
    
    # Initialize result array
    result = np.zeros_like(x_coordinates, dtype=float)
    
    # Superpose all Gaussian functions
    for gaussian_index in range(num_gaussians):
        # Extract parameters of the current Gaussian function
        amplitude = parameters[gaussian_index * 3]
        mean = parameters[gaussian_index * 3 + 1]
        std_dev = parameters[gaussian_index * 3 + 2]
        
        # Compute current Gaussian function and accumulate
        result += amplitude * np.exp(-((x_coordinates - mean) ** 2) / (2 * std_dev ** 2))
    
    return result


def adaptive_gaussian_fitting(x_data, y_data, max_gaussians=30, tolerance=1e-7):
    """
    Adaptive Gaussian wavelet fitting algorithm
    
    Parameters:
        x_data: Independent variable data
        y_data: Dependent variable data
        max_gaussians: Maximum number of Gaussian functions
        tolerance: Fitting tolerance
        
    Returns:
        fitted_curve: Fitted curve data
        optimal_parameters: Optimal parameter array
        optimal_gaussian_count: Optimal number of Gaussian functions
        integral_difference: Integral difference value
    """
    
    # Initialize optimal results
    best_fitted_curve = None
    best_parameters = None
    best_gaussian_count = 0
    best_integral_difference = float('inf')
    
    # Compute the integral (area) of the target curve
    target_integral = np.trapz(y_data, x_data)
    
    # Try different numbers of Gaussian functions
    for current_gaussian_count in range(2, max_gaussians + 1):
        # Initialize parameter guesses
        initial_parameter_guess = []
        
        for gaussian_index in range(current_gaussian_count):
            # Amplitude guess: evenly distribute the total amplitude range
            amplitude_guess = (np.max(y_data) - np.min(y_data)) / current_gaussian_count
            
            # Mean guess: uniformly distributed within the x range
            mean_guess = np.min(x_data) + (np.max(x_data) - np.min(x_data)) * gaussian_index / current_gaussian_count
            
            # Standard deviation guess: based on 1/(2n) of the x range
            std_dev_guess = (x_data[-1] - x_data[0]) / (current_gaussian_count * 2)
            
            initial_parameter_guess.extend([amplitude_guess, mean_guess, std_dev_guess])
        
        try:
            # Perform curve fitting
            fitted_parameters, _ = curve_fit(
                gaussian_sum_function, 
                x_data, 
                y_data, 
                p0=initial_parameter_guess, 
                maxfev=100000
            )
            
            # Compute fitted curve
            fitted_curve = gaussian_sum_function(x_data, *fitted_parameters)
            
            # Compute integral of fitted curve
            fitted_integral = np.trapz(fitted_curve, x_data)
            
            # Compute integral difference (new evaluation criterion)
            current_integral_difference = abs(fitted_integral - target_integral) / target_integral
            
            # Check for improvement
            if current_integral_difference < best_integral_difference:
                best_fitted_curve = fitted_curve
                best_parameters = fitted_parameters
                best_gaussian_count = current_gaussian_count
                best_integral_difference = current_integral_difference
            
            # Stop early if the integral difference is below tolerance
            if current_integral_difference < tolerance:
                break
                
        except (RuntimeError, ValueError):
            # If fitting fails, continue trying with more wavelets
            continue
    
    return best_fitted_curve, best_parameters, best_gaussian_count, best_integral_difference


def evaluate_fit_quality(x_data, y_original, y_fitted, method='integral'):
    """
    Multiple methods for evaluating fitting quality
    
    Parameters:
        x_data: Independent variable data
        y_original: Original data
        y_fitted: Fitted data
        method: Evaluation method ('integral', 'mse', 'r2', 'all')
        
    Returns:
        Fitting quality evaluation results
    """
    
    results = {}
    
    # Integral difference evaluation
    if method in ['integral', 'all']:
        original_integral = np.trapz(y_original, x_data)
        fitted_integral = np.trapz(y_fitted, x_data)
        integral_difference = abs(fitted_integral - original_integral) / original_integral
        results['integral_difference'] = integral_difference
        results['original_integral'] = original_integral
        results['fitted_integral'] = fitted_integral
    
    # Mean squared error evaluation
    if method in ['mse', 'all']:
        mse = np.mean((y_original - y_fitted) ** 2)
        results['mse'] = mse
    
    # R² evaluation
    if method in ['r2', 'all']:
        ss_res = np.sum((y_original - y_fitted) ** 2)
        ss_tot = np.sum((y_original - np.mean(y_original)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        results['r_squared'] = r_squared
    
    # Maximum absolute error
    if method in ['all']:
        max_abs_error = np.max(np.abs(y_original - y_fitted))
        results['max_absolute_error'] = max_abs_error
    
    return results


def symbolic_expression_from_parameters(parameters, variable_symbol='z'):
    """
    Generate symbolic expression from parameters
    
    Parameters:
        parameters: Array of Gaussian function parameters
        variable_symbol: Variable symbol
        
    Returns:
        Symbolic expression
    """
    
    import sympy as sp
    
    variable = sp.symbols(variable_symbol)
    num_gaussians = len(parameters) // 3
    
    symbolic_expr = 0
    
    for gaussian_index in range(num_gaussians):
        amplitude = parameters[gaussian_index * 3]
        mean = parameters[gaussian_index * 3 + 1]
        std_dev = parameters[gaussian_index * 3 + 2]
        
        # Add Gaussian term to the symbolic expression
        symbolic_expr += amplitude * sp.exp(-((variable - mean) ** 2) / (2 * std_dev ** 2))
    
    return symbolic_expr