# =============================================
# Part 2: Forward Problem - Navier-Stokes Equations
# =============================================

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time

def navier_stokes_2d(x, y, Re=100):
    """
    Define the Navier-Stokes equations for incompressible flow in 2D.
    
    Args:
        x: Input coordinates (x, y, t)
        y: Output variables (u, v, p)
        Re: Reynolds number
        
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
    
    # PDE residuals
    continuity = u_x + v_y
    x_momentum = u_t + u * u_x + v * u_y + p_x - (1.0/Re) * (u_xx + u_yy)
    y_momentum = v_t + u * v_x + v * v_y + p_y - (1.0/Re) * (v_xx + v_yy)
    
    return [continuity, x_momentum, y_momentum]

def train_forward_navier_stokes(data=None, Re=100, iterations=20000, save_model=True):
    """
    Train a PINN to solve the forward Navier-Stokes problem.
    
    Args:
        data: Optional data dictionary for validation
        Re: Reynolds number
        iterations: Number of training iterations
        save_model: Whether to save the trained model
        
    Returns:
        Trained model and loss history
    """
    print("Training forward Navier-Stokes PINN...")
    
    # If no data is provided, generate it
    if data is None:
        from fluid_dynamics_simulation import generate_cylinder_flow_data
        data = generate_cylinder_flow_data(nx=51, ny=31, nt=50, Re=Re)
    
    # Domain dimensions from data
    x = data['x']
    y = data['y']
    t = data['t']
    
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    t_min, t_max = np.min(t), np.max(t)
    
    # Define the geometry and time domain
    space_domain = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
    time_domain = dde.geometry.TimeDomain(t_min, t_max)
    geom_time = dde.geometry.GeometryXTime(space_domain, time_domain)
    
    # Define the PDE problem with Navier-Stokes equations
    def pde(x, y):
        return navier_stokes_2d(x, y, Re=Re)
    
    # Boundary and initial conditions
    
    # Cylinder boundary (no-slip condition)
    def cylinder_boundary(x, on_boundary):
        x_c, y_c, r_c = 1.0, 0.0, 0.5
        return on_boundary and np.sqrt((x[0] - x_c)**2 + (x[1] - y_c)**2) <= r_c + 1e-3
    
    # Inlet boundary
    def inlet_boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_min)
    
    # Top and bottom boundaries
    def wall_boundary(x, on_boundary):
        return on_boundary and (np.isclose(x[1], y_min) or np.isclose(x[1], y_max))
    
    # Outlet boundary
    def outlet_boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_max)
    
    # Inlet velocity profile
    def inlet_velocity_u(x):
        return 1.5 * np.ones_like(x[:, 0:1])
    
    def inlet_velocity_v(x):
        return np.zeros_like(x[:, 0:1])
    
    # No-slip boundary conditions
    bc_cylinder_u = dde.icbc.DirichletBC(geom_time, lambda x: 0, cylinder_boundary, component=0)
    bc_cylinder_v = dde.icbc.DirichletBC(geom_time, lambda x: 0, cylinder_boundary, component=1)
    
    # Inlet boundary conditions
    bc_inlet_u = dde.icbc.DirichletBC(geom_time, inlet_velocity_u, inlet_boundary, component=0)
    bc_inlet_v = dde.icbc.DirichletBC(geom_time, inlet_velocity_v, inlet_boundary, component=1)
    
    # Wall boundary conditions
    bc_wall_u = dde.icbc.DirichletBC(geom_time, lambda x: 0, wall_boundary, component=0)
    bc_wall_v = dde.icbc.DirichletBC(geom_time, lambda x: 0, wall_boundary, component=1)
    
    # Initial conditions from data
    def initial_u(x):
        return np.zeros_like(x[:, 0:1])
    
    def initial_v(x):
        return np.zeros_like(x[:, 0:1])
    
    def initial_p(x):
        return np.zeros_like(x[:, 0:1])
    
    ic_u = dde.icbc.IC(geom_time, initial_u, lambda _, on_initial: on_initial, component=0)
    ic_v = dde.icbc.IC(geom_time, initial_v, lambda _, on_initial: on_initial, component=1)
    ic_p = dde.icbc.IC(geom_time, initial_p, lambda _, on_initial: on_initial, component=2)
    
    # Create the PDE problem
    data_pde = dde.data.TimePDE(
        geom_time, 
        pde, 
        [bc_cylinder_u, bc_cylinder_v, bc_inlet_u, bc_inlet_v, bc_wall_u, bc_wall_v, ic_u, ic_v, ic_p],
        num_domain=10000,
        num_boundary=2000,
        num_initial=1000,
        num_test=1000
    )
    
    # Create the neural network
    layer_size = [3] + [50] * 6 + [3]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    # Create the model
    model = dde.Model(data_pde, net)
    
    # Compile and train the model
    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    
    # Use a learning rate scheduler
    def learning_rate_schedule(epoch):
        if epoch < iterations * 0.4:
            return 1e-3
        elif epoch < iterations * 0.8:
            return 1e-4
        else:
            return 1e-5
    
    # Create callbacks
    checkpointer = dde.callbacks.ModelCheckpoint(
        "models/forward_model.ckpt", 
        verbose=1, 
        save_better_only=True, 
        period=1000
    )
    
    # Train the model
    start_time = time.time()
    losshistory, train_state = model.train(
        iterations=iterations,
        callbacks=[checkpointer],
        display_every=100
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    # Save the model
    if save_model:
        model.save("models/forward_model.h5")
        print("Model saved to models/forward_model.h5")
    
    # Plot loss history
    dde.utils.plot_loss_history(losshistory)
    plt.savefig("results/forward/loss_history.png")
    plt.close()
    
    return model, losshistory, train_state

def predict_flow_field(model, data, save_path=None):
    """
    Predict the flow field using the trained model.
    
    Args:
        model: Trained DeepXDE model
        data: Data dictionary containing the domain information
        save_path: Path to save the visualization
        
    Returns:
        Dictionary containing the predicted flow field
    """
    print("Predicting flow field...")
    
    # Extract domain information
    x = data['x']
    y = data['y']
    t = data['t']
    
    nx = len(x)
    ny = len(y)
    nt = len(t)
    
    # Create meshgrid for visualization
    X, Y = np.meshgrid(x, y)
    
    # Initialize arrays for predictions
    u_pred = np.zeros((ny, nx, nt))
    v_pred = np.zeros((ny, nx, nt))
    p_pred = np.zeros((ny, nx, nt))
    
    # Predict flow field at each time step
    for k in range(nt):
        print(f"Predicting time step {k+1}/{nt}...")
        
        # Create input points for current time step
        points = np.zeros((nx * ny, 3))
        for i in range(nx):
            for j in range(ny):
                points[i * ny + j, 0] = x[i]
                points[i * ny + j, 1] = y[j]
                points[i * ny + j, 2] = t[k]
        
        # Predict
        output = model.predict(points)
        
        # Reshape predictions
        u_pred[:, :, k] = output[:, 0].reshape(ny, nx)
        v_pred[:, :, k] = output[:, 1].reshape(ny, nx)
        p_pred[:, :, k] = output[:, 2].reshape(ny, nx)
    
    # Visualize predictions
    for k in range(0, nt, nt//5):  # Visualize a few time steps
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Velocity magnitude
        vel_mag = np.sqrt(u_pred[:, :, k]**2 + v_pred[:, :, k]**2)
        im1 = axes[0].contourf(X, Y, vel_mag, 100, cmap='viridis')
        axes[0].set_title(f'Predicted Velocity Magnitude at t = {t[k]:.2f}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        # Pressure
        im2 = axes[1].contourf(X, Y, p_pred[:, :, k], 100, cmap='coolwarm')
        axes[1].set_title(f'Predicted Pressure at t = {t[k]:.2f}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        # Velocity vectors
        skip = 5
        axes[2].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                      u_pred[::skip, ::skip, k], v_pred[::skip, ::skip, k], 
                      vel_mag[::skip, ::skip], cmap='viridis')
        axes[2].set_title(f'Predicted Velocity Vectors at t = {t[k]:.2f}')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_t{k}.png")
        
        plt.show()
    
    # Create animation of velocity magnitude
    fig, ax = plt.subplots(figsize=(10, 5))
    
    def update(frame):
        ax.clear()
        vel_mag = np.sqrt(u_pred[:, :, frame]**2 + v_pred[:, :, frame]**2)
        
        im = ax.contourf(X, Y, vel_mag, 100, cmap='viridis')
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                 u_pred[::skip, ::skip, frame], v_pred[::skip, ::skip, frame], 
                 vel_mag[::skip, ::skip], cmap='viridis')
        ax.set_title(f'Predicted Flow Field at t = {t[frame]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        return [im]
    
    ani = FuncAnimation(fig, update, frames=min(20, nt), interval=200)
    
    if save_path:
        ani_path = f"{save_path}_animation.gif"
        ani.save(ani_path, writer='pillow', fps=5)
        print(f"Animation saved to {ani_path}")
    
    plt.show()
    
    # Return predictions
    predictions = {
        'u_pred': u_pred,
        'v_pred': v_pred,
        'p_pred': p_pred
    }
    
    return predictions
