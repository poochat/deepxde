"""
Comprehensive Fluid Dynamics Simulation using DeepXDE and Physics-Informed Neural Networks (PINNs)

This script demonstrates how to use DeepXDE to solve both forward and inverse problems
in fluid dynamics using Physics-Informed Neural Networks (PINNs).

The script includes:
1. 2D Navier-Stokes equations for incompressible flow
2. Forward problem: Predict flow field given known parameters
3. Inverse problem: Identify unknown parameters from flow field data
4. Data generation, training, and visualization

Author: Codegen
Date: August 17, 2025
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time

# Create directories for saving results
os.makedirs("results", exist_ok=True)
os.makedirs("results/forward", exist_ok=True)
os.makedirs("results/inverse", exist_ok=True)
os.makedirs("results/data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(1234)

# =============================================
# Part 1: Data Generation for Fluid Simulation
# =============================================

def generate_cylinder_flow_data(nx=101, ny=51, nt=100, Re=100):
    """
    Generate synthetic data for flow around a cylinder.
    
    Args:
        nx: Number of points in x direction
        ny: Number of points in y direction
        nt: Number of time steps
        Re: Reynolds number
        
    Returns:
        Dictionary containing the generated data
    """
    print("Generating synthetic data for flow around a cylinder...")
    
    # Domain dimensions
    Lx, Ly = 10.0, 2.0
    
    # Grid points
    x = np.linspace(0, Lx, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    t = np.linspace(0, 10, nt)
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Cylinder parameters
    x_c, y_c, r_c = 1.0, 0.0, 0.5
    
    # Initialize velocity and pressure fields
    u = np.zeros((ny, nx, nt))
    v = np.zeros((ny, nx, nt))
    p = np.zeros((ny, nx, nt))
    
    # Generate synthetic data using analytical functions and physical constraints
    for k in range(nt):
        time = t[k]
        
        # Inlet velocity profile (parabolic)
        u_inlet = 1.5 * np.ones_like(y)
        
        # Initialize time step with inlet profile
        for j in range(ny):
            u[j, 0, k] = u_inlet[j]
        
        # Simple analytical approximation of flow around cylinder
        for i in range(nx):
            for j in range(ny):
                # Distance from cylinder center
                dx = X[j, i] - x_c
                dy = Y[j, i] - y_c
                r = np.sqrt(dx**2 + dy**2)
                
                # Skip points inside the cylinder
                if r < r_c:
                    u[j, i, k] = 0
                    v[j, i, k] = 0
                    p[j, i, k] = 0
                    continue
                
                # Angle from cylinder center
                theta = np.arctan2(dy, dx)
                
                # Analytical approximation based on potential flow + time-dependent wake
                if X[j, i] > x_c:
                    # Wake region
                    wake_effect = np.exp(-(r - r_c) / Re) * np.sin(2*np.pi*time/5 + X[j, i]/2)
                    u[j, i, k] = u_inlet[j] * (1 - (r_c/r)**2 * np.cos(2*theta)) + 0.1 * wake_effect
                    v[j, i, k] = -u_inlet[j] * (r_c/r)**2 * np.sin(2*theta) + 0.2 * wake_effect
                else:
                    # Upstream region
                    u[j, i, k] = u_inlet[j] * (1 - (r_c/r)**2 * np.cos(2*theta))
                    v[j, i, k] = -u_inlet[j] * (r_c/r)**2 * np.sin(2*theta)
                
                # Pressure field (simplified Bernoulli)
                p[j, i, k] = 1 - 0.5 * (u[j, i, k]**2 + v[j, i, k]**2)
                
                # Add some noise
                u[j, i, k] += 0.05 * np.random.randn()
                v[j, i, k] += 0.05 * np.random.randn()
                p[j, i, k] += 0.05 * np.random.randn()
    
    # Reshape data for DeepXDE format
    X_star = np.vstack((X.flatten(), Y.flatten())).T
    T_star = t.reshape(-1, 1)
    
    # Create space-time points
    XX, TT = np.meshgrid(np.arange(nx), np.arange(nt))
    YY, _ = np.meshgrid(np.arange(ny), np.arange(nt))
    
    X_sim = np.vstack((XX.flatten(), YY.flatten(), TT.flatten())).T
    
    # Flatten data
    U_star = np.zeros((nx*ny, nt))
    V_star = np.zeros((nx*ny, nt))
    P_star = np.zeros((nx*ny, nt))
    
    for k in range(nt):
        U_star[:, k] = u[:, :, k].flatten()
        V_star[:, k] = v[:, :, k].flatten()
        P_star[:, k] = p[:, :, k].flatten()
    
    # Save data
    data = {
        'x': x,
        'y': y,
        't': t,
        'X_star': X_star,
        'T_star': T_star,
        'U_star': U_star,
        'V_star': V_star,
        'P_star': P_star,
        'X_sim': X_sim,
        'Re': Re
    }
    
    np.save('results/data/cylinder_flow_data.npy', data)
    print("Data generation completed and saved.")
    
    return data

# Visualize the generated data
def visualize_flow_data(data, save_path=None):
    """
    Visualize the generated flow data.
    
    Args:
        data: Dictionary containing the flow data
        save_path: Path to save the visualization
    """
    print("Visualizing flow data...")
    
    x = data['x']
    y = data['y']
    t = data['t']
    U_star = data['U_star']
    V_star = data['V_star']
    P_star = data['P_star']
    
    nx = len(x)
    ny = len(y)
    nt = len(t)
    
    # Create a figure for visualization
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Select a time step to visualize
    time_idx = nt // 2
    
    # Reshape data for the selected time step
    u = U_star[:, time_idx].reshape(ny, nx)
    v = V_star[:, time_idx].reshape(ny, nx)
    p = P_star[:, time_idx].reshape(ny, nx)
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)
    
    # Plot velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    im1 = axes[0].contourf(X, Y, vel_mag, 100, cmap='viridis')
    axes[0].set_title(f'Velocity Magnitude at t = {t[time_idx]:.2f}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot pressure
    im2 = axes[1].contourf(X, Y, p, 100, cmap='coolwarm')
    axes[1].set_title(f'Pressure at t = {t[time_idx]:.2f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot velocity vectors
    skip = 5  # Skip some points for better visualization
    axes[2].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  u[::skip, ::skip], v[::skip, ::skip], 
                  vel_mag[::skip, ::skip], cmap='viridis')
    axes[2].set_title(f'Velocity Vectors at t = {t[time_idx]:.2f}')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 5))
    
    def update(frame):
        ax.clear()
        u = U_star[:, frame].reshape(ny, nx)
        v = V_star[:, frame].reshape(ny, nx)
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Plot velocity magnitude
        im = ax.contourf(X, Y, vel_mag, 100, cmap='viridis')
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                 u[::skip, ::skip], v[::skip, ::skip], 
                 vel_mag[::skip, ::skip], cmap='viridis')
        ax.set_title(f'Flow Field at t = {t[frame]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        return [im]
    
    ani = FuncAnimation(fig, update, frames=min(20, nt), interval=200)
    
    if save_path:
        ani_path = save_path.replace('.png', '.gif')
        ani.save(ani_path, writer='pillow', fps=5)
        print(f"Animation saved to {ani_path}")
    
    plt.show()
