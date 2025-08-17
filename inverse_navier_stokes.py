# =============================================
# Part 3: Inverse Problem - Navier-Stokes Equations
# =============================================

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def navier_stokes_inverse_2d(x, y, C1, C2):
    """
    Define the Navier-Stokes equations for incompressible flow in 2D with unknown parameters.
    
    Args:
        x: Input coordinates (x, y, t)
        y: Output variables (u, v, p)
        C1: Convection coefficient (to be identified)
        C2: Diffusion coefficient (to be identified)
        
    Returns:
        List of PDE residuals [continuity, x-momentum, y-momentum]
    """
    # Extract variables
    u = y[:, 0:1]  # x-velocity
    v = y[:, 1:2]  # y-velocity
    p = y[:, 2:3]  # pressure
    
    # Compute derivatives
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    u_t = dde.grad.jacobian(y, x, i=0, j=2)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    v_t = dde.grad.jacobian(y, x, i=1, j=2)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)
    
    # PDE residuals with unknown parameters
    continuity = u_x + v_y
    x_momentum = u_t + C1 * (u * u_x + v * u_y) + p_x - C2 * (u_xx + u_yy)
    y_momentum = v_t + C1 * (u * v_x + v * v_y) + p_y - C2 * (v_xx + v_yy)
    
    return [continuity, x_momentum, y_momentum]

def train_inverse_navier_stokes(data=None, true_Re=100, iterations=20000, save_model=True):
    """
    Train a PINN to solve the inverse Navier-Stokes problem.
    
    Args:
        data: Data dictionary containing the flow field data
        true_Re: True Reynolds number (for validation)
        iterations: Number of training iterations
        save_model: Whether to save the trained model
        
    Returns:
        Trained model, identified parameters, and loss history
    """
    print("Training inverse Navier-Stokes PINN...")
    
    # If no data is provided, generate it
    if data is None:
        from fluid_dynamics_simulation import generate_cylinder_flow_data
        data = generate_cylinder_flow_data(nx=51, ny=31, nt=50, Re=true_Re)
    
    # Domain dimensions from data
    x = data['x']
    y = data['y']
    t = data['t']
    U_star = data['U_star']
    V_star = data['V_star']
    
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    t_min, t_max = np.min(t), np.max(t)
    
    # Define the geometry and time domain
    space_domain = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
    time_domain = dde.geometry.TimeDomain(t_min, t_max)
    geom_time = dde.geometry.GeometryXTime(space_domain, time_domain)
    
    # Define the parameters to be identified
    C1 = dde.Variable(1.0)  # Convection coefficient (initialized with a guess)
    C2 = dde.Variable(0.01)  # Diffusion coefficient (initialized with a guess)
    
    # Define the PDE problem with Navier-Stokes equations
    def pde(x, y):
        return navier_stokes_inverse_2d(x, y, C1, C2)
    
    # Prepare observation data
    n_obs = 5000  # Number of observation points
    
    # Randomly select observation points
    idx = np.random.choice(data['X_star'].shape[0], n_obs, replace=False)
    
    # Create observation points
    X_obs = np.zeros((n_obs, 3))
    u_obs = np.zeros((n_obs, 1))
    v_obs = np.zeros((n_obs, 1))
    
    # Randomly select time points
    t_idx = np.random.choice(len(t), n_obs, replace=True)
    
    # Extract observation data
    for i in range(n_obs):
        X_obs[i, 0] = data['X_star'][idx[i], 0]  # x-coordinate
        X_obs[i, 1] = data['X_star'][idx[i], 1]  # y-coordinate
        X_obs[i, 2] = t[t_idx[i]]  # t-coordinate
        u_obs[i, 0] = U_star[idx[i], t_idx[i]]  # u-velocity
        v_obs[i, 0] = V_star[idx[i], t_idx[i]]  # v-velocity
    
    # Create observation constraints
    observe_u = dde.icbc.PointSetBC(X_obs, u_obs, component=0)
    observe_v = dde.icbc.PointSetBC(X_obs, v_obs, component=1)
    
    # Create the PDE problem
    data_pde = dde.data.TimePDE(
        geom_time, 
        pde, 
        [observe_u, observe_v],
        num_domain=5000,
        num_boundary=1000,
        num_initial=500,
        anchors=X_obs,
        num_test=1000
    )
    
    # Create the neural network
    layer_size = [3] + [50] * 6 + [3]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    # Create the model
    model = dde.Model(data_pde, net)
    
    # Create variable callback to track parameter evolution
    var_cb = dde.callbacks.VariableValue(
        [C1, C2], 
        period=100, 
        filename="results/inverse/variables.dat"
    )
    
    # Compile and train the model
    model.compile("adam", lr=1e-3, external_trainable_variables=[C1, C2])
    
    # Train the model
    start_time = time.time()
    losshistory, train_state = model.train(
        iterations=iterations//2,
        callbacks=[var_cb],
        display_every=100,
        disregard_previous_best=True
    )
    
    # Reduce learning rate and continue training
    model.compile("adam", lr=1e-4, external_trainable_variables=[C1, C2])
    losshistory, train_state = model.train(
        iterations=iterations//2,
        callbacks=[var_cb],
        display_every=100,
        disregard_previous_best=True
    )
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    # Save the model
    if save_model:
        model.save("models/inverse_model.h5")
        print("Model saved to models/inverse_model.h5")
    
    # Plot loss history
    dde.utils.plot_loss_history(losshistory)
    plt.savefig("results/inverse/loss_history.png")
    plt.close()
    
    # Get the identified parameters
    C1_value = train_state.best_y[0]
    C2_value = train_state.best_y[1]
    
    print(f"Identified parameters:")
    print(f"C1 (convection coefficient): {C1_value:.6f}")
    print(f"C2 (diffusion coefficient): {C2_value:.6f}")
    print(f"True Reynolds number: {true_Re}")
    print(f"Identified Reynolds number: {1/C2_value:.6f}")
    
    # Plot the evolution of the parameters
    plot_parameter_evolution("results/inverse/variables.dat", true_Re)
    
    return model, (C1_value, C2_value), losshistory, train_state

def plot_parameter_evolution(filename, true_Re):
    """
    Plot the evolution of the identified parameters during training.
    
    Args:
        filename: Path to the file containing the parameter values
        true_Re: True Reynolds number for comparison
    """
    # Read the parameter values
    data = np.loadtxt(filename)
    iterations = np.arange(0, len(data) * 100, 100)
    
    # Extract parameter values
    C1_values = data[:, 0]
    C2_values = data[:, 1]
    Re_values = 1 / C2_values
    
    # True values
    C1_true = 1.0
    C2_true = 1.0 / true_Re
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot C1 evolution
    ax1.plot(iterations, C1_values, 'b-', label='Identified C1')
    ax1.axhline(y=C1_true, color='r', linestyle='--', label='True C1')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('C1 (Convection Coefficient)')
    ax1.set_title('Evolution of Convection Coefficient')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Reynolds number evolution
    ax2.plot(iterations, Re_values, 'g-', label='Identified Re')
    ax2.axhline(y=true_Re, color='r', linestyle='--', label=f'True Re = {true_Re}')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Reynolds Number')
    ax2.set_title('Evolution of Reynolds Number')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/inverse/parameter_evolution.png")
    plt.close()
    
    # Plot relative error
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Relative error in C1
    rel_error_C1 = np.abs(C1_values - C1_true) / C1_true * 100
    ax1.semilogy(iterations, rel_error_C1, 'b-')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Relative Error (%)')
    ax1.set_title('Relative Error in C1')
    ax1.grid(True)
    
    # Relative error in Re
    rel_error_Re = np.abs(Re_values - true_Re) / true_Re * 100
    ax2.semilogy(iterations, rel_error_Re, 'g-')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Relative Error in Reynolds Number')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/inverse/parameter_error.png")
    plt.close()
    
    return
