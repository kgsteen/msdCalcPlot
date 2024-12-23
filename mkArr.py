"""
Module: mkArr.py
Author: Krista G. Steenbergen
Date: 22 Dec 2024

Purpose:
    This module contains functions for processing molecular dynamics (MD) simulation data.
    It includes functions for parsing trajectory files, extracting atomic coordinates, 
    filtering trajectories, calculating z-coordinate maxima, and exporting XYZ files.

Functions:
    - arrMk_npt: Parse NPT simulation data and extract atomic trajectories.
    - arrMk: Parse NVT simulation data and extract atomic trajectories.
    - arrExcl: Filter trajectory data to exclude Ga2O3 atoms.
    - fndMxZ: Calculate the maximum Z-coordinate average for each frame.
    - mkXYZ: Export atomic trajectory data to an XYZ file.

Dependencies:
    - numpy

Usage Example:
    from mkArr import arrMk_npt, arrExcl, fndMxZ, mkXYZ
"""
import numpy as np

#########################################
#########################################
############################
def arrMk_npt(x,rpt):
    """
    Process NPT simulation trajectory data to extract atomic coordinates.

    Parameters:
        x (list of str): Raw trajectory data lines.
        rpt (list of int): Repetition pattern for the unit cell ([x, y, z] repetitions).

    Returns:
        nfrm (int): 			Number of frames.
        a (ndarray): 			Scaled cell matrix or lattice parameter array (dynamic per frame).
        supNarr (ndarray):		Array of atom counts per species after repetition.
        N (int): 				Total number of atoms after repetition.
        traj (ndarray): 		Trajectory array (frames × atoms × coordinates).
        strAr (list of str): 	Atom species/types array.

    Notes:
        - In NPT simulations, the lattice matrix (a) is updated for every frame, 
          as cell dimensions can vary during the simulation.
    """

    nfrm = sum('Direct' in s for s in x)  				# Count frames marked by 'Direct'
    cnt = 8  											# Starting line number for atomic position data

    natAr = np.genfromtxt(x[6:7], comments='\n', dtype=np.int32, ndmin=1)			# atom counts (1 interger / atom type)
    N = np.sum(natAr)																# total number of atoms / frame
    strAr = x[5].split()															# atomic types (string data), e.g., Ga  In

    # Initialize trajectory arrays
    traj = np.zeros((nfrm, N * rpt[0] * rpt[1] * rpt[2], 3))			# will hold trajectory data
    atAr = np.zeros((nfrm, N, 3))						# temp array for each frame's read as [1,1,1] cell
    pb = np.zeros((N * rpt[0] * rpt[1] * rpt[2], 3))				# array to account for possibility of "wrapped" data

    # Loop through each frame
    for i in range(nfrm):
        a = np.genfromtxt(x[cnt - 6:cnt - 3], comments='\n')  			# Update lattice matrix per frame
        atAr[i] = np.genfromtxt(x[cnt:cnt + N], comments='\n')  		# Read atomic positions in i-th frame

        if i > 0:					# if not the first frame....
            pb += np.round(atAr[i - 1] - atAr[i])  		# Correct periodic boundary crossings

        # Replicate atomic positions using the function
        nA = replicate_atoms(atAr[i], natAr, rpt)

        # Apply periodic boundary corrections to replicated atoms
        nA += pb

        # Apply lattice transformation and save in trajectory array
        traj[i] = np.dot(nA, a)

        cnt += N + 8  # Increment counter for next frame

    # Exclude oxide layers
    trajNoOx = arrExcl(traj, rpt)

    # Multiply natAr atom counts by unit cell repeats
    supNarr = np.atleast_1d(natAr * rpt[0] * rpt[1] * rpt[2])

    return nfrm, a * rpt, supNarr, N * rpt[0] * rpt[1] * rpt[2], trajNoOx, strAr



#########################################
#########################################
############################
def arrMk(x,rpt):
    """
    Process NVT simulation trajectory data to extract atomic coordinates.

    Parameters:
        x (list of str): Raw trajectory data lines.
        rpt (list of int): Repetition pattern for the unit cell ([x, y, z] repetitions).

    Returns:
        nfrm (int): 			Number of frames.
        a (ndarray): 			Scaled cell matrix or lattice parameter array (static across frames).
        supNarr (ndarray): 		Array of atom counts per species after repetition.
        N (int): 				Total number of atoms after repetition.
        traj (ndarray): 		Trajectory array (frames x atoms x coordinates).
        strAr (list of str): 	Atom species/types array.

    Notes:
        - In NVT simulations, the lattice matrix (a) is read once and remains constant 
          throughout the entire simulation, as cell dimensions are fixed.
    """

    nfrm = sum('Direct' in s for s in x)                                            # Count frames marked by 'Direct'
    cnt = 8                                                                         # Starting line number for atomic position data
    a = np.genfromtxt(x[2:5],comments='\n')                                         # read/create h-matrix

    natAr = np.genfromtxt(x[6:7],comments='\n', dtype=np.int32, ndmin=1)            # atom counts (1 interger / atom type)
    N = np.sum(natAr)																# total number of atoms / frame
    strAr = x[5].split()															# atomic types (string data), e.g., Ga  In

    traj = np.zeros((nfrm, N * rpt[0] * rpt[1] * rpt[2], 3))			# will hold trajectory data
    atAr = np.zeros((nfrm, N, 3))										# temp array for each frame's read as [1,1,1] cell
    pb = np.zeros((N * rpt[0] * rpt[1] * rpt[2], 3))					# array to account for possibility of "wrapped" data

    for i in range((int)(nfrm)):
        atAr[i] = np.genfromtxt(x[cnt:cnt+N],comments='\n')			# array of atoms
        if (i>0):
            pb += np.round(atAr[i-1]-atAr[i])
                  
        # Replicate atomic positions using the function
        nA = replicate_atoms(atAr[i], natAr, rpt)
            
        # Apply periodic boundary corrections to replicated atoms
        nA += pb
            
        # Apply lattice transformation and save in trajectory array
        traj[i] = np.dot(nA, a)

        # Move to the next frame block in the raw data
        cnt += N + 1

    # Calculate supNarr array for repeated atom counts per species
    supNarr = np.atleast_1d(natAr * rpt[0] * rpt[1] * rpt[2])

    return nfrm, a*rpt, supNarr, N*rpt[0]*rpt[1]*rpt[2], traj, strAr


#########################################
#########################################
############################
def replicate_atoms(atAr, natAr, rpt):
    """
    Replicate atoms based on repetition pattern.

    Parameters:
        atAr (ndarray): 		Array of atomic positions for a frame.
        natAr (ndarray): 		Number of atoms per species.
        rpt (list of int): 		Unit cell replication as [n1, n2, n3]

    Returns:
        nA (ndarray): Replicated atomic positions.
    """
    nA = np.zeros((np.sum(natAr) * rpt[0] * rpt[1] * rpt[2], 3))
    cntNa = 0

    for nSp in range(len(natAr)):
        curSt = sum(natAr[:nSp]) if nSp > 0 else 0
        nTyp = natAr[nSp]

        for xr in range(rpt[0]):
            for yr in range(rpt[1]):
                for zr in range(rpt[2]):
                    addM = [xr, yr, zr]
                    nA[cntNa:cntNa + nTyp] = atAr[curSt:curSt + nTyp] + addM
                    cntNa += nTyp

    return nA


#########################################
#########################################
###########################
def arrExcl(traj,rpt):
    """
    Filter atomic trajectories to include only atoms with indices below a threshold.

    Parameters:
        traj (ndarray): Trajectory array of shape (frames, atoms, coordinates),
                        where:
                        - frames: Number of simulation frames.
                        - atoms: Number of atoms per frame.
                        - coordinates: x, y, z positions per atom.
        max_index (int): Maximum valid atom index to include (default: 4376).

    Returns:
        trajLiq (ndarray): Filtered trajectory array containing only valid atoms.
                           Shape: (frames, included_atoms, coordinates)

    Notes:
        - Atom indices are assumed to start from 0 (Python indexing).
    """
    # Generate an array of all atom indices (0, 1, 2, ..., traj.shape[1]-1)
    atom_indices = np.arange(traj.shape[1])

    # Create max_index = number of liquid atoms(3608) times rptX*rptY*rptZ
    max_index = 3608 * rpt[0] * rpt[1] * rpt[2]
    
    # Create a mask for indices less than the max_index
    mask = atom_indices < max_index
    
    # Apply the mask to filter the trajectory array along the atom axis
    trajLiq = traj[:, mask, :]
    
    return trajLiq


#########################################
#########################################
###########################
def fndMxZ(traj, nfrm):
    """
    Calculate the maximum Z-coordinate average for each frame.

    Parameters:
        traj (ndarray): Trajectory array (frames x atoms x 3).  Should contain only liquid atoms (no oxide).
        nfrm (int): Number of frames.

    Returns:
        mxZ (ndarray): Average maximum Z-coordinate of the liquid atoms per frame.
    """
    mxZ = np.zeros(nfrm)
    threshold = 60

    for i in range(nfrm):
        srt_arr = np.array(sorted(traj[i], key=lambda x: x[2]))
        index = np.searchsorted(srt_arr[:, 2], threshold, side='right')         # threshhold in case some atoms fly off top / bottom-but-wrap
        mxZ[i] = np.mean(srt_arr[index - 20:index, 2])                          # average over the top 20 atoms in liquid

    return mxZ


###########################
###########################
def mkXYZ(traj, nfrm, natAr, strAr, fwstr):
    """
    Export trajectory data to an XYZ movie file.

    Parameters:
        traj (ndarray): Trajectory array (frames x atoms x coordinates).
        nfrm (int): Number of frames.
        natAr (list of int): Number of atoms per species.
        strAr (list of str): Atom species/types array.
        fwstr (str):  path + filename of movie file

    Returns:
        None
    """
    nAr = [strAr[nSp] for nSp, count in enumerate(natAr) for _ in range(count)]
    N = sum(natAr)

    with open(fwstr, 'w') as fw:
        for i in range(0, nfrm, 3):
            fw.write(f"{N}\nblah\n")
            for j in range(len(traj[i])):
                fw.write(f"{nAr[j]}   {'  '.join(map(str, traj[i][j]))}\n")




