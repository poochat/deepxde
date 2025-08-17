# Fluid Dynamics Simulation with Physics-Informed Neural Networks (PINNs)

This repository contains a comprehensive implementation of fluid dynamics simulation using Physics-Informed Neural Networks (PINNs) with the DeepXDE library. The code demonstrates both forward and inverse problems for the Navier-Stokes equations.

## Overview

Physics-Informed Neural Networks (PINNs) are a novel approach to solving partial differential equations (PDEs) by incorporating physical laws into the neural network training process. This implementation focuses on the Navier-Stokes equations for incompressible fluid flow, which are fundamental in fluid dynamics.

The repository includes:

1. **Data Generation**: Synthetic data generation for flow around a cylinder
2. **Forward Problem**: Predict the flow field given known parameters (Reynolds number)
3. **Inverse Problem**: Identify unknown parameters (convection and diffusion coefficients) from flow field data
4. **Visualization**: Tools for visualizing the flow field and training results

## Requirements

- Python 3.7+
- DeepXDE
- NumPy
- Matplotlib
- SciPy

Install the required packages using:

```bash
pip install deepxde numpy matplotlib scipy
```

## Project Structure

- `fluid_dynamics_simulation.py`: Data generation and visualization functions
- `forward_navier_stokes.py`: Implementation of the forward problem
- `inverse_navier_stokes.py`: Implementation of the inverse problem
- `main.py`: Main script to run the entire simulation
- `results/`: Directory for saving results
  - `data/`: Generated data
  - `forward/`: Results from the forward problem
  - `inverse/`: Results from the inverse problem
- `models/`: Directory for saving trained models

## Usage

### Running the Complete Simulation

To run the complete simulation with default parameters:

```bash
python main.py
```

### Customizing the Simulation

You can customize the simulation using command-line arguments:

```bash
python main.py --nx 51 --ny 31 --nt 50 --reynolds 100 --iterations 10000
```

### Running Specific Steps

You can run specific steps of the simulation:

```bash
# Generate new data
python main.py --generate_data

# Train forward model
python main.py --forward

# Train inverse model
python main.py --inverse

# Run all steps
python main.py --all
```

## Examples

### Forward Problem

The forward problem predicts the flow field given the Reynolds number:

```python
from forward_navier_stokes import train_forward_navier_stokes, predict_flow_field
from fluid_dynamics_simulation import generate_cylinder_flow_data

# Generate data
data = generate_cylinder_flow_data(nx=51, ny=31, nt=50, Re=100)

# Train forward model
model, losshistory, train_state = train_forward_navier_stokes(
    data=data,
    Re=100,
    iterations=10000,
    save_model=True
)

# Predict flow field
predictions = predict_flow_field(
    model=model,
    data=data,
    save_path="results/forward/prediction"
)
```

### Inverse Problem

The inverse problem identifies unknown parameters from flow field data:

```python
from inverse_navier_stokes import train_inverse_navier_stokes
from fluid_dynamics_simulation import generate_cylinder_flow_data

# Generate data
data = generate_cylinder_flow_data(nx=51, ny=31, nt=50, Re=100)

# Train inverse model
model, params, losshistory, train_state = train_inverse_navier_stokes(
    data=data,
    true_Re=100,
    iterations=10000,
    save_model=True
)

# Print identified parameters
C1_value, C2_value = params
print(f"Identified Reynolds number: {1/C2_value:.6f}")
```

## Results

The simulation produces the following results:

1. **Data Generation**:
   - Synthetic flow field data around a cylinder
   - Visualization of velocity magnitude, pressure, and velocity vectors

2. **Forward Problem**:
   - Predicted flow field
   - Loss history during training
   - Visualization of predicted velocity magnitude, pressure, and velocity vectors

3. **Inverse Problem**:
   - Identified parameters (convection and diffusion coefficients)
   - Evolution of parameters during training
   - Relative error in identified parameters

## Theory

### Navier-Stokes Equations

The Navier-Stokes equations for incompressible flow in 2D are:

1. **Continuity Equation**:
   ∂u/∂x + ∂v/∂y = 0

2. **Momentum Equations**:
   - ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + (1/Re)(∂²u/∂x² + ∂²u/∂y²)
   - ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + (1/Re)(∂²v/∂x² + ∂²v/∂y²)

where:
- u, v are the velocity components
- p is the pressure
- Re is the Reynolds number

### Physics-Informed Neural Networks (PINNs)

PINNs incorporate the PDE residuals into the loss function during training:

Loss = MSE_data + MSE_physics

where:
- MSE_data is the mean squared error between the predicted and observed data
- MSE_physics is the mean squared error of the PDE residuals

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. SIAM Review, 63(1), 208-228.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
