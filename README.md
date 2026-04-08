# DA-Based Aberration Coefficient Calculator for Electron Lenses

This repository provides a Python 3.12 implementation of an aberration-coefficient calculation framework for electron lenses based on the differential algebra (DA) method.

The code supports both round lenses and multipole lenses, enabling high-order aberration analysis using analytical field reconstruction, numerical integration, and DA-based trajectory expansion.

---

## Requirements

Python version:

Python 3.12

Required packages:

numpy  
scipy  
sympy  
matplotlib  
tqdm  
numba  
daceypy  

---

## Repository Structure

.
├── axial_field_model.py
├── electrostatics.py
├── electromagnetic_fields_expansion.py
├── electron_trajectory.py
├── field_evaluator.py
├── field_fitting.py
├── field_io.py
├── gaussian_wavelet_fitting.py
├── interpolation.py
├── numerical_integrators.py
├── plane_index.py
├── tracking.py
├── run_example_round.py
├── run_example_multipole.py
└── lens_ideal_Q.txt

---

## Description

### Example Scripts

run_example_round.py  
Example for round lens calculation using analytical axial field modeling, Gaussian fitting, 3D field expansion, and DA-based aberration computation.

run_example_multipole.py  
Example for multipole lens calculation using discrete field data, local interpolation, and DA-based tracking.

---

### Core Functional Modules

axial_field_model.py  
Defines symbolic models for axial electromagnetic fields.

electromagnetic_fields_expansion.py  
Generates three-dimensional electromagnetic field expressions from axial models.

gaussian_wavelet_fitting.py  
Performs Gaussian wavelet superposition fitting for axial field data.

field_io.py  
Loads electromagnetic field data from text files.

plane_index.py  
Builds spatial indexing structures for discrete field data.

interpolation.py  
Performs Lagrange interpolation along the axial direction.

field_fitting.py  
Performs local polynomial fitting for multipole field reconstruction.

field_evaluator.py  
Constructs local analytical field expressions for use in trajectory computation.

---

### Trajectory and Differential Algebra

electron_trajectory.py  
Solves electron trajectories in electromagnetic fields.

tracking.py  
Implements DA-based particle tracking.

numerical_integrators.py  
Provides numerical integration routines.

electrostatics.py  
Defines electrostatic potential models.

---

## Usage

Run round lens example:

python run_example_round.py

Run multipole lens example:

python run_example_multipole.py

---

## Input Data Format

The field data file should contain:

x  y  z  Bx  By  Bz

x, y, z are in mm (internally converted to meters)  
Bx, By, Bz are field components  

Example file:

lens_ideal_Q.txt

---

## Citation

If you use this code, please cite:

Zhicheng Liu, Shikai Zhu, Xu Liu, Bin Qin
Calculation of Aberration Coefficients for Round Lenses and Multipole Lenses Based on the Differential Algebra Method,  
Ultramicroscopy (submitted).

---

## License

MIT License
