import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import interpolate
from scipy.constants import e, m_e
from tqdm import tqdm
from daceypy import DA, array

# Import custom library functions
from axial_field_model import glaser_magnetic_lens_symbolic
from electromagnetic_fields_expansion import (
    compute_magnetic_field_expansion, 
    compute_electrostatic_field_expansion
)
import numerical_integrators
import electron_trajectory
import gaussian_wavelet_fitting


def find_intersection(x1, y1, x2, y2):
    """Compute the intersection point of two straight lines"""
    m = (y2 - y1) / (x2 - x1)  # slope
    x_intersection = x1 + (1 - y1) / m  # compute intersection with y=1
    return x_intersection


def has_multiple_intersections(y_values):
    """Check whether the function has multiple zero crossings"""
    sign_changes = 0
    prev_sign = np.sign(y_values[0])
    
    for y in y_values[1:]:
        current_sign = np.sign(y)
        if current_sign != prev_sign and current_sign != 0 and prev_sign != 0:
            sign_changes += 1
        prev_sign = current_sign

    return sign_changes > 1

def main():
    # Set Chinese font to SimSun
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    # Set global font size
    plt.rcParams['font.size'] = 16  # base font size
    plt.rcParams['axes.titlesize'] = 16  # title font size
    plt.rcParams['axes.labelsize'] = 16  # axis label font size
    plt.rcParams['xtick.labelsize'] = 16  # x-axis tick font size
    plt.rcParams['ytick.labelsize'] = 16  # y-axis tick font size
    plt.rcParams['legend.fontsize'] = 16  # legend font size
    # Set global scientific notation format
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['axes.formatter.limits'] = [-3, 3]  # use scientific notation when <10^-3 or >10^3

    # ============================================================================
    # 1. Basic parameter settings
    # ============================================================================

    # Electron basic parameters
    q = e                    # electron charge
    m = m_e                  # electron mass
    eta = np.sqrt(q/(2*m))   # electromagnetic field constant

    # Magnetic lens parameters
    B0 = 0.048   # peak magnetic field strength (T)
    z0 = 0.0     # magnetic field center position (m)
    a = 0.01     # magnetic field half-width (m)

    # Computational parameters
    dz = 1e-5    # axial step size
    V = 25000    # accelerating voltage (V)

    # Computational range
    object_position = -0.25   # object plane position
    image_range_end = 0.1     # end of image-side calculation range


    # ============================================================================
    # 2. Magnetic field modeling and fitting
    # ============================================================================

    print("=" * 60)
    print("Electron Lens Aberration Calculation")
    print("=" * 60)

    # 2.1 Generate analytical magnetic field distribution
    z_symbol = sp.symbols('z')
    B_expr = glaser_magnetic_lens_symbolic(B0, z0, a)  # Glaser magnetic lens model
    B_func = sp.lambdify(z_symbol, B_expr, 'numpy')

    # 2.2 Magnetic field data sampling
    z_M = np.linspace(-0.25, 0.25, 10000)  # sampling range
    B_discrete = B_func(z_M)                # discrete magnetic field data

    # 2.3 Gaussian wavelet fitting of magnetic field distribution
    print("\n1. Gaussian fitting of magnetic field distribution...")
    fitted_curve, fitted_params, num_gaussians, integral_diff = (
        gaussian_wavelet_fitting.adaptive_gaussian_fitting(
            z_M, B_discrete, max_gaussians=30, tolerance=1e-3
        )
    )

    # 2.4 Fitting quality evaluation
    fit_quality = gaussian_wavelet_fitting.evaluate_fit_quality(
        z_M, B_discrete, fitted_curve, method='all'
    )

    # 2.5 Generate fitted symbolic expression
    fitted_expr = gaussian_wavelet_fitting.symbolic_expression_from_parameters(
        fitted_params, 'z'
    )

    # 2.6 Print fitting results
    print("   Fitted symbolic expression:")
    sp.pprint(fitted_expr)
    print(f"   Number of Gaussian functions: {num_gaussians}")
    print(f"   Integral difference: {integral_diff:.6%}")
    print(f"   Coefficient of determination R²: {fit_quality.get('r_squared', 'N/A'):.10f}")

    # 2.7 Plot fitting results
    # plt.figure(figsize=(10, 6))
    # plt.plot(z_M, B_discrete, label="Analytical magnetic field distribution", linewidth=2, color='blue')
    # plt.plot(z_M, fitted_curve, 
    #         label=f"Gaussian fitted curve ({num_gaussians} Gaussian functions)", 
    #         linewidth=2, color='red', linestyle='--')
    # plt.xlabel("Axial position z (m)")
    # plt.ylabel("Magnetic field strength B (T)")
    # plt.title("Gaussian Wavelet Fitting of Magnetic Field Distribution")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()


    # ============================================================================
    # 3. Field function preparation and preprocessing
    # ============================================================================

    print("\n2. Field function preprocessing...")

    # 3.1 Numerical interpolation of magnetic field (for linear calculations)
    factor_B = 1000  # interpolation scaling factor
    B_numerical = B_discrete * factor_B
    B_numerical = interpolate.UnivariateSpline(z_M, B_numerical, s=0.1, k=3)
    B_numerical_ = B_numerical.derivative(n=1)

    # 3.2 Symbolic magnetic field function (for higher-order calculations)
    glaser = fitted_expr
    glaser_ = glaser.diff(z_symbol)
    B = sp.lambdify(z_symbol, glaser)      # magnetic field function
    B_ = sp.lambdify(z_symbol, glaser_)    # first derivative of magnetic field

    # 3.3 Electric potential function preparation
    factor_V = 1  # potential scaling factor
    Laplace_V = z_symbol * 0 + V           # uniform potential distribution
    Laplace_V_ = Laplace_V.diff(z_symbol, 1)    # first derivative
    Laplace_V__ = Laplace_V.diff(z_symbol, 2)   # second derivative

    V_func = sp.lambdify(z_symbol, Laplace_V)        # potential function
    V_deriv = sp.lambdify(z_symbol, Laplace_V_)      # first derivative of potential
    V_second_deriv = sp.lambdify(z_symbol, Laplace_V__)  # second derivative of potential


    # ============================================================================
    # 4. Linear optical property calculation
    # ============================================================================

    print("\n3. Linear optical property calculation...")

    # 4.1 Set computational grid
    axial_coordinates = np.arange(object_position, image_range_end + dz, dz)
    steps = len(axial_coordinates)

    # 4.2 Initialize trajectory arrays
    r = np.zeros(steps)   # radial position
    v = np.zeros(steps)   # radial slope
    z_array = np.zeros(steps)

    # 4.3 Compute two fundamental trajectories
    # Trajectory 1: on-axis point, initial slope 1 rad
    r[0] = 0
    v[0] = 1
    z_array[0] = object_position

    h, h_ = electron_trajectory.linearized_electromagnetic_euler_cromer(
        r.copy(), v.copy(), axial_coordinates, B_numerical, dz, steps, 
        V_func, V_second_deriv, V_deriv, 
        q, m, factor_V, factor_B
    )

    # Trajectory 2: off-axis 1 m, initial slope 0
    r[0] = 1
    v[0] = 0

    g, g_ = electron_trajectory.linearized_electromagnetic_euler_cromer(
        r.copy(), v.copy(), axial_coordinates, B_numerical, dz, steps, 
        V_func, V_second_deriv, V_deriv, 
        q, m, factor_V, factor_B
    )

    # 4.4 Compute focal position and focal length
    z_f = axial_coordinates[np.argmin(abs(g[100:])) + 100]
    zpos = find_intersection(z_f, 0, 
                            axial_coordinates[np.argmin(abs(g[100:])) + 50], 
                            g[np.argmin(abs(g[100:])) + 50])
    f = z_f - zpos

    print(f"   Focal position: {z_f:.6f} m")
    print(f"   Principal plane position: {zpos:.6f} m")
    print(f"   Focal length: {f:.6f} m")

    # 4.5 Compute Gaussian image plane and magnification
    z_Image_Plane = axial_coordinates[np.argmin(abs(h[1000:])) + 1000]
    M_ = g[np.argmin(abs(h[1000:])) + 1000]

    print(f"   Gaussian image plane position: {z_Image_Plane:.6f} m")
    print(f"   Lateral magnification: {M_:.6f}")

    # 4.6 Plot electron trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(axial_coordinates, h, label='On-axis trajectory (r₀=0, r₀\'=1 rad)', linewidth=2)
    plt.plot(axial_coordinates, g, label='Off-axis trajectory (r₀=1 m, r₀\'=0)', linewidth=2)
    plt.axvline(x=z_f, color='red', linestyle='--', alpha=0.7, 
            label=f'Focal position ({z_f:.4f} m)')
    plt.axvline(x=z_Image_Plane, color='green', linestyle='--', alpha=0.7, 
            label=f'Gaussian image plane ({z_Image_Plane:.4f} m)')
    plt.xlabel('Axial position z (m)')
    plt.ylabel('Radial position r (m)')
    plt.title('Electron Beam Trajectory Tracking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()


    # ============================================================================
    # 5. Geometric aberration coefficient calculation
    # ============================================================================

    print("\n4. Geometric aberration coefficient calculation...")

    # 5.1 Reset computation range (object plane to image plane)
    axial_coordinates = np.arange(object_position, z_Image_Plane + dz, dz)
    steps = len(axial_coordinates)

    # 5.2 Compute 3D electromagnetic field expansion
    order = 5  # expansion order

    print("   Computing 3D magnetic field expansion...")
    B_jit, B_lambda, _ = compute_magnetic_field_expansion(glaser, order)

    print("   Computing 3D electrostatic field expansion...")
    E_jit, U_jit, E_lambda, U_lambda = compute_electrostatic_field_expansion(
        Laplace_V, order
    )

    # 5.3 Initialize differential algebra variables
    DA.init(5, 5)  # initialize DA environment, 5 variables, 5th order

    # 5.4 Define initial conditions (including perturbations)
    x0 = 0
    y0 = 0
    x0_slope = 0
    y0_slope = 0
    d0 = 0

    x = array([
        x0 + DA(1),      # x = x₀ + δx
        x0_slope + DA(2), # x' = x₀' + δx'
        y0 + DA(3),      # y = y₀ + δy
        y0_slope + DA(4),  # y' = y₀' + δy'
        d0 + DA(5)
    ])

    # 5.5 Integrate equations of motion
    print("   Integrating equations of motion...")
    z_map, x_map = numerical_integrators.runge_kutta_4th_order(
        electron_trajectory.electromagnetic_motion_equation,
        object_position,
        x,
        z_Image_Plane,
        steps,
        (eta, E_lambda, U_lambda, V_func, V_deriv, B_lambda, B, B_)
    )

    # 5.6 Extract aberration coefficients
    print("   Extracting aberration coefficients...")
    B_coef = x_map[0].getCoefficient([0, 3, 0, 0, 0])
    F_coef = x_map[0].getCoefficient([1, 0, 0, 2, 0])
    C_coef = x_map[0].getCoefficient([1, 0, 1, 1, 0])/2
    D_coef = x_map[0].getCoefficient([0, 1, 2, 0, 0])
    E_coef = x_map[0].getCoefficient([3, 0, 0, 0, 0])
    f_coef = x_map[2].getCoefficient([1, 0, 0, 2, 0])/3
    c_coef = x_map[2].getCoefficient([1, 0, 1, 1, 0])/2
    e_coef = x_map[2].getCoefficient([3, 0, 0, 0, 0])
    CF_coef = x_map[0].getCoefficient([0, 1, 0, 0, 1])/(x_map[0].getCoefficient([1, 0, 0, 0, 0]))

    # 5.7 Organize and output results
    print("\n" + "=" * 60)
    print("Summary of Calculation Results")
    print("=" * 60)
    print(f"\nLinear optical properties:")
    print(f"  Focal length: {f:.6f} m")
    print(f"  Magnification: {M_:.6f}")
    print(f"  Gaussian image plane position: {z_Image_Plane:.6f} m")
    print(f"\nAberration coefficients:")
    print(f"  B: {B_coef:.8e}")
    print(f"  F: {F_coef:.8e}")
    print(f"  C: {C_coef:.8e}")
    print(f"  D: {D_coef:.8e}")
    print(f"  E: {E_coef:.8e}")
    print(f"  f: {f_coef:.8e}")
    print(f"  c: {c_coef:.8e}")
    print(f"  e: {e_coef:.8e}")
    print(f"  CF: {CF_coef:.8e}")
    print('Cs:', B_coef*M_**3)
    print('Cc:', CF_coef*M_**2)
    with open("Aberration_coefficients.txt", "w", encoding="utf-8") as f:
        f.write(str(x_map))
    plt.show()

if __name__ == "__main__":
    main()