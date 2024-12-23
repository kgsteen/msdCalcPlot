"""
Module: calcMSD.py
Author: Krista G. Steenbergen
Date: 23 Dec 2024

Purpose:
    This module calculates the Mean Squared Displacement (MSD) of particles
    in a molecular dynamics (MD) simulation. MSD is used to quantify particle 
    diffusion and mobility over time.
    NOTE:  Uses multiple time origins!  Some MSD algorithms use only t=0 as time origin...
           but here, we create a "evenly-spaced" time-origin array of length simTime/2 for each
           ... so MSD(tau) = [sum_{i=1,all time origins} msd(time-origin_i + tau) ] / n_timeOrigins

Functions:
    - calculate_msd: Computes the Mean Squared Displacement (MSD) for given 
                     particle positions over set of tau time-lags.

Dependencies:
    - numpy

Usage Example:
    from calcMSD import calculate_msd
    msd = calculate_msd(time_array, position_array)
"""
import numpy as np

###############################################
###############################################
################################
def calculate_msd(time, positions):
    """
    Calculate the Mean Squared Displacement (MSD) with a fixed number of time origins 
    for each time lag, limited to meaningful tau values.

    Parameters:
        time (ndarray): 1D array of time points corresponding to the positions.
        positions (ndarray): 2D or 3D array of atomic positions over time.
                             - Shape (steps, coordinates) for 2D input (single atom).
                             - Shape (steps, atoms, coordinates) for 3D input.

    Returns:
        ndarray: 2D array of MSD values for each time lag and coordinate.
                 - Shape: (halfNstp, n_columns)
                 - msd[:, 0]: MSD for x-axis
                 - msd[:, 1]: MSD for y-axis
                 - msd[:, 2]: MSD for z-axis
    """
    # Ensure positions have 3 dimensions (steps, atoms, coordinates)
    if positions.ndim == 2:
        positions = positions[:, np.newaxis, :]  # Add an extra dimension at axis=1
    
    n_steps, n_atoms, n_columns = positions.shape  # Extract array dimensions
    
    # Define a consistent number of time origins and limit tau
    halfNstp = n_steps // 2  # Integer division for safety
    
    # Initialize the MSD array
    msd = np.zeros((halfNstp, n_columns))
    
    # Loop over each spatial dimension (e.g., x, y, z)
    for col in range(n_columns):
        # Loop only up to half the number of steps
        for tau in range(halfNstp):
            if tau == 0:
                msd[tau, col] = 0  # No displacement at zero lag
            else:
                # Evenly sample time origins
                time_origins = np.linspace(0, n_steps - tau - 1, halfNstp).astype(int)
                
                # Calculate squared displacements at selected time origins
                squared_displacements = [
                    (positions[t + tau, :, col] - positions[t, :, col])**2
                    for t in time_origins
                ]
                
                # Compute mean MSD for the sampled time origins
                msd[tau, col] = np.mean(squared_displacements)
    
    return msd




