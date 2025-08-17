"""
Main script to run the fluid dynamics simulation using DeepXDE and PINNs.

This script demonstrates the complete workflow:
1. Generate synthetic data for flow around a cylinder
2. Visualize the generated data
3. Train a forward PINN model to predict the flow field
4. Train an inverse PINN model to identify unknown parameters
5. Compare the results

Author: Codegen
Date: August 17, 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

# Import the modules
from fluid_dynamics_simulation import generate_cylinder_flow_data, visualize_flow_data
from forward_navier_stokes import train_forward_navier_stokes, predict_flow_field
from inverse_navier_stokes import train_inverse_navier_stokes

def main(args):
    """
    Main function to run the fluid dynamics simulation.
    
    Args:
        args: Command-line arguments
    """
    # Create directories for saving results
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/forward", exist_ok=True)
    os.makedirs("results/inverse", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(1234)
    
    # Step 1: Generate synthetic data
    if args.generate_data or not os.path.exists("results/data/cylinder_flow_data.npy"):
        print("Step 1: Generating synthetic data...")
        data = generate_cylinder_flow_data(
            nx=args.nx, 
            ny=args.ny, 
            nt=args.nt, 
            Re=args.reynolds
        )
        
        # Visualize the generated data
        visualize_flow_data(data, save_path="results/data/flow_visualization.png")
    else:
        print("Step 1: Loading existing data...")
        data = np.load("results/data/cylinder_flow_data.npy", allow_pickle=True).item()
    
    # Step 2: Train forward model
    if args.forward or args.all:
        print("\nStep 2: Training forward Navier-Stokes model...")
        forward_model, losshistory, train_state = train_forward_navier_stokes(
            data=data,
            Re=args.reynolds,
            iterations=args.iterations,
            save_model=True
        )
        
        # Predict flow field using the trained model
        print("\nPredicting flow field using the forward model...")
        predictions = predict_flow_field(
            model=forward_model,
            data=data,
            save_path="results/forward/prediction"
        )
    
    # Step 3: Train inverse model
    if args.inverse or args.all:
        print("\nStep 3: Training inverse Navier-Stokes model...")
        inverse_model, params, losshistory, train_state = train_inverse_navier_stokes(
            data=data,
            true_Re=args.reynolds,
            iterations=args.iterations,
            save_model=True
        )
        
        # Print the identified parameters
        C1_value, C2_value = params
        print("\nIdentified parameters:")
        print(f"C1 (convection coefficient): {C1_value:.6f}")
        print(f"C2 (diffusion coefficient): {C2_value:.6f}")
        print(f"True Reynolds number: {args.reynolds}")
        print(f"Identified Reynolds number: {1/C2_value:.6f}")
        print(f"Relative error in Reynolds number: {abs(1/C2_value - args.reynolds)/args.reynolds*100:.2f}%")
    
    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fluid Dynamics Simulation using DeepXDE and PINNs")
    
    # Data generation parameters
    parser.add_argument("--nx", type=int, default=51, help="Number of points in x direction")
    parser.add_argument("--ny", type=int, default=31, help="Number of points in y direction")
    parser.add_argument("--nt", type=int, default=50, help="Number of time steps")
    parser.add_argument("--reynolds", type=float, default=100, help="Reynolds number")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=10000, help="Number of training iterations")
    
    # Execution control
    parser.add_argument("--generate_data", action="store_true", help="Generate new data")
    parser.add_argument("--forward", action="store_true", help="Train forward model")
    parser.add_argument("--inverse", action="store_true", help="Train inverse model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    # If no specific steps are selected, run all
    if not (args.generate_data or args.forward or args.inverse or args.all):
        args.all = True
    
    # Run the main function
    main(args)
