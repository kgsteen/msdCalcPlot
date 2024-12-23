"""
Module: trkC.py
Author: Krista G. Steenbergen
Date: 23 Dec 2024

Purpose:
    This module provides functionality to track and extract atomic coordinates 
    from molecular dynamics (MD) simulation trajectories. It generates a structured 
    array containing time-stamped x, y, and modified z coordinates for specified atoms.

Functions:
    - trkCfnct: Tracks atomic positions across simulation frames and organizes them 
                into a structured array with time stamps.

Dependencies:
    - numpy
    - fndMxZ (imported from mkArr.py)

Usage Example:
    from trkC import trkCfnct
    result = trkCfnct(trajLiq, tmFct, atom_indices=[0, 1, 2])
"""
import numpy as np
from mkArr import fndMxZ


#########################################
#########################################
############################
def trkCfnct(trajLiq, tmFct, atom_indices=None):
    """
    Track atomic coordinates across trajectory frames and generate a time-structured array.

    Parameters:
        trajLiq (ndarray): Filtered trajectory array of shape (frames, atoms, coordinates).
                           - frames: Number of time steps in the simulation.
                           - atoms: Number of atoms per frame.
                           - coordinates: x, y, z positions of each atom.
        tmFct (float): Time scaling factor for simulation frames.
                      - Typically converts time steps into nanoseconds.
        atom_indices (list of int, optional): Indices of atoms to track.
                                              - If None, all atoms are included.

    Returns:
        ndarray: Array containing time, x, y, and modified z coordinates for tracked atoms.
                 - Shape: (frames, 1 + len(atom_indices) * 3)
                 - Format per row:
                   [time, x1, x2, ..., y1, y2, ..., z1, z2, ...]

    Notes:
        - Time is scaled by `tmFct` and converted to nanoseconds.
        - The modified Z-coordinate is calculated as: (mxZ[i] - trajLiq[i][j][2])
    """
    # Default to all atoms if atom_indices is not provided
    if atom_indices is None:
        atom_indices = range(trajLiq.shape[1])

    # Number of frames in the trajectory
    nfrm = len(trajLiq)

    # Generate time values (scaled and converted to nanoseconds)
    time_values = np.arange(1, nfrm + 1) * tmFct / 1000.0

    # Calculate the average maximum Z-coordinate per frame (avg of top 20 z-coords of liquid-only)
    mxZ = fndMxZ(trajLiq, nfrm)

    # Initialize an array to store time and coordinates
    crdTrc_array = []

    # Loop over each frame
    for i in range(nfrm):
        # Add the time value for the current frame
        time_value = (i + 1) * tmFct / 1000.0
        row = [time_value]

        # Coordinate extraction using NumPy slicing
        row.extend(trajLiq[i, atom_indices, 0])                     # Add X-coordinates
        row.extend(trajLiq[i, atom_indices, 1])                     # Add Y-coordinates
        row.extend(mxZ[i] - trajLiq[i, atom_indices, 2])            # Add modified Z-coordinates, normalised to the maxZ of that frame

        # Append the row to the result array
        crdTrc_array.append(row)

    return np.array(crdTrc_array)



