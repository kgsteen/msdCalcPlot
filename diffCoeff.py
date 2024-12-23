"""
Module: diffCoeff.py
Author: Krista G. Steenbergen
Date: 23 Dec 2024

Purpose:
    This module calculates diffusion coefficients (D) from Mean Squared Displacement (MSD) data
    by performing a linear regression fit in a specified time range.

Functions:
    - diffFunc: Calculates the diffusion coefficient from MSD data.

Dependencies:
    - numpy
    - matplotlib.pyplot
    - scipy.stats.linregress

Usage Example:
    from diffCoeff import diffFunc
    diffC_arr, D_cm2_per_s, time_linear = diffFunc(tm, msd3c, 'npt')
"""
import numpy as np
from scipy.stats import linregress
import sys


#########################################
#########################################
############################
def diffFunc(tm, msd3c, typ):
    """
    Calculate diffusion coefficients (D) from Mean Squared Displacement (MSD) data.

    Parameters:
        tm (ndarray): 1D array of time points (in ps).
        msd3c (ndarray): 2D array of MSD values across x, y, and z axes.
                         Shape: (n_steps, 3) → [msd_x, msd_y, msd_z].
        typ (str): A string identifier for the simulation type (e.g., 'npt', 'nvt').

    Returns:
        tuple:
            - diffC_arr (ndarray): Diffusion coefficients for x, y, z axes, and the total MSD.
                                   Shape: (4,)
            - D_cm2_per_s (float): Total diffusion coefficient in cm²/s.
            - time_linear (ndarray): Time values used in the linear fit.

    Notes:
        - The diffusion coefficient is extracted from the linear region of MSD data.
        - D is calculated using the Einstein relation: D = slope / 6.
        - Units are converted from Å²/ps to cm²/s using a conversion factor.
    """
    # Ensure time array length matches MSD array
    tm = tm[:len(msd3c[:, 0])]

    # Combine MSD data into a 2D array (x, y, z, total)
    msd = np.array([
        msd3c[:, 0],  # MSD_x
        msd3c[:, 1],  # MSD_y
        msd3c[:, 2],  # MSD_z
        msd3c[:, 0] + msd3c[:, 1] + msd3c[:, 2]  # MSD_total
    ])

    # Identify the linear region for diffusion coefficient calculation
    linear_region_start = 0
    linear_region_end_index = np.where(tm > 5.0)[0][0]  # Find first index where tm > 10.0
    linear_region_mask = (tm >= tm[linear_region_start]) & (tm <= tm[linear_region_end_index])

    # Time values within the linear region
    time_linear = tm[linear_region_mask]

    # Initialize array for diffusion coefficients ... [diff_x, diff_y, diff_z, diff_tot]
    diffC_arr = np.zeros(4)

    # Loop over x, y, z, and total MSD
    for ax in range(4):
        # MSD values within the linear region
        msd_linear = msd[ax, linear_region_mask]

        # Perform the linear regression in the selected time range
        slope, intercept, r_value, p_value, std_err = linregress(time_linear, msd_linear)

        # Calculate the diffusion coefficient D 
		# Assumes 3D:  Einstein relation: D = slope / 6
        D_A2_per_ps = slope / 6  										# Diffusion coefficient in Å²/ps

        # Convert to cm²/s (1 Å²/ps = 1e-4 cm²/s)
        conversion_factor = 1e-4
        D_cm2_per_s = D_A2_per_ps * conversion_factor
        D_cm2_per_s_4disp = D_A2_per_ps * conversion_factor * 1e6  		# For display without scientific notation

        # Store the calculated diffusion coefficient per x, y, z or total
        diffC_arr[ax] = D_cm2_per_s_4disp

    # Output the total diffusion coefficient for display
    print(f"Diffusion Coefficient in cm²/s: {D_cm2_per_s_4disp:.3f}")
    sys.stdout.flush()

    return diffC_arr, time_linear





#########################################
#########################################
############################
def diffFunc2(tm, msd3c, typ):
    """
    Different way to calculate the linear regime
    """

    # Ensure time array length matches MSD array
    tm = tm[:len(msd3c[:, 0])]

    # Combine MSD data into a 2D array (x, y, z, total)
    msd = np.array([
        msd3c[:, 0],  # MSD_x
        msd3c[:, 1],  # MSD_y
        msd3c[:, 2],  # MSD_z
        msd3c[:, 0] + msd3c[:, 1] + msd3c[:, 2]  # MSD_total
    ])

    # Define a sliding window size
    window_size = 20  # Number of points per window

    # Initialize tracking variables
    best_r2 = -np.inf  # Start with the worst R² value
    best_fit_range = (0, window_size)  # Default to the first window

    # Iterate over possible windows in the time range
    for start in range(len(tm) - window_size):
        end = start + window_size
        time_window = tm[start:end]
        msd_window = msd[3,start:end]
        
        # Perform linear regression on the current window
        slope, intercept, r_value, p_value, std_err = linregress(time_window, msd_window)
        
        # Update the best-fit window if R² improves
        if r_value**2 > best_r2:
            best_r2 = r_value**2
            best_fit_range = (start, end)

    # Extract the best-fit time and MSD ranges
    best_start, best_end = best_fit_range
    linear_region_mask = (tm >= tm[best_start]) & (tm <= tm[best_end])

    # Time values within the linear region
    time_linear = tm[linear_region_mask]

    # Initialize array for diffusion coefficients ... [diff_x, diff_y, diff_z, diff_tot]
    diffC_arr = np.zeros(4)

    # Loop over x, y, z, and total MSD
    for ax in range(4):
        # MSD_x/y/z/tot within the linear region
        msd_linear = msd[ax, linear_region_mask]

        # Perform the linear regression in the selected time range
        slope, intercept, r_value, p_value, std_err = linregress(time_linear, msd_linear)

        # Calculate the diffusion coefficient D 
		# Assumes 3D:  Einstein relation: D = slope / 6
        D_A2_per_ps = slope / 6  										# Diffusion coefficient in Å²/ps

        # Convert to cm²/s (1 Å²/ps = 1e-4 cm²/s)
        conversion_factor = 1e-4
        D_cm2_per_s = D_A2_per_ps * conversion_factor
        D_cm2_per_s_4disp = D_A2_per_ps * conversion_factor * 1e6  		# For display without scientific notation

        # Store the calculated diffusion coefficient per x, y, z or total
        diffC_arr[ax] = D_cm2_per_s_4disp

    # Output the total diffusion coefficient for display
    print(f"Diffusion Coefficient in cm²/s: {D_cm2_per_s_4disp:.3f}")
    sys.stdout.flush()

    return diffC_arr, time_linear

