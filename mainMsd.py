"""
Script Name: mainMsd.py
Author: Krista G. Steenbergen
Date: 22 Dec 2024

Purpose:
    This script analyzes molecular dynamics (MD) simulation trajectories to calculate 
    Mean Squared Displacement (MSD) and diffusion coefficients for selected atoms.

Inputs (Command-line arguments):
    1. dop (str): Dopant type or identifier.
    2. atSt (int): Starting atom index.
    3. stN (int): Start simulation number.
    4. endN (int): End simulation number.

Outputs:
    - MSD data files.
    - Diffusion coefficient files.
    - PDF plots showing MSD trends.

Dependencies:
    - numpy
    - matplotlib.pyplot
    - mkArr (arrMk, arrMk_npt, mkXYZ)
    - calcMsd (calculate_msd)
    - plotMSD (pltMsdMid)
    - trkC (trkCfnct)
    - diffCoeff (diffFunc)
	
*** Needs to be set:  
    - dirXd          :  directory for where to find XDATCAR ... assumes the form run1_n ... n = number of runs
	- atTyp          :  label for plot file name
    - title          :  Title used in the pdf plot (set right now for this specific analysis)
    - fwRt           :  Filename used to save pdf plots.
	- simTyp         :  'npt' or 'nvt'
	- atom_indices   :  for which set of atoms are we calculating the MSD (right now, only set to do in a range, e.g., 1-20)
	                     ... but can be changed easily
						 
*** Assumes:
    - npt = with oxide
	- nvt = without oxide

Usage Example:
    python msd_analysis.py dopant_name 0 1 10
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Custom module imports
from mkArr import arrMk, arrMk_npt, mkXYZ			# arrMk and arrMk_npt -> turns XDATCAR into traj[nfrm,N,3]; 
													# mkXYZ creates an xyz-formatted movie file; 
from calcMsd import calculate_msd				# Calculates Mean Squared Displacement
from trkC import trkCfnct					# Tracks each (x, y, z) coordinate over MD time
from diffCoeff import diffFunc,diffFunc2				# Calculates diffusion coefficient ... x, y, z + total
from plotMSD import pltMsd				# Plots MSD data + trajectory tracing


# Parse command-line arguments
dop = sys.argv[1]  					# Dopant type or identifier
atSt = int(sys.argv[2])  				# Starting atom index
stN = int(sys.argv[3])  				# Start simulation number
endN = int(sys.argv[4])  				# End simulation number

# Directory and file naming conventions
dirXd = '../'  							# Base directory for input files
dopNm = dop.split('ga')  					# Extract dopant information
simTyp = 'nvt' 							# Simulation type ... determines arrMk or arrMk_npt
rpt = [1, 1, 1]  						# Number of unit cells in the [v1,v2,v3] dimensions
evN = 1 								# Frequency for sampling frames

###########################

# Define atom indices (select a range of atoms for analysis)
#atom_indices = np.arange(atSt, atSt + 20)  					# Select 20 atoms starting from atSt, e.g., 4-24 if atSt=4
atom_indices = np.arange(atSt, atSt + 1)  						# Select atSt only for analysis
atom_str = ", ".join(str(i + 1) for i in atom_indices)  		# 1-based indexing for output readability

# Output settings
atTyp = 'all20'  						# label for plot file name
title = f'All {dopNm[0]}'  				# Title for plots
fwRt = f'outFiles/msd_all_{dop}'  		# Base output file path

# Ensure the output directory exists
os.makedirs('outFiles', exist_ok=True)


# Loop through each simulation run
fwstrT = f'outFiles/diffC_{atTyp}_{stN}_{endN}.dat'
with open(fwstrT,'w') as fwTot:
	for num in range(stN,endN+1):

		fwPlotName = f'{fwRt}_{num}.pdf'					# File name for pdf of plot	
		frstr = f'{dirXd}/run1_{num}/XDATCAR'  				# Input trajectory file path
	
		# Time factor to multiply every "traj" frame by to get equiv MD simulation time
		tmFact = 1.0 * 100 * evN  							#  1.0fs time step * output every 100 time steps * evN iter from XDATCAR

		# Read trajectory file
		with open(frstr, 'r') as fr:
			x = fr.readlines()

		############# create Cartesian atom arrays
		if (simTyp == 'npt'):
			nfrm, a, natAr, N, traj, strAr = arrMk_npt(x,rpt)
		elif (simTyp == 'nvt'):
			nfrm, a, natAr, N, traj, strAr = arrMk(x,rpt)
		else:
			print(f"Error: Invalid simulation type '{simTyp}'. Expected 'npt' or 'nvt'.")
			sys.exit(1)
		# nfrm: Number of frames
		# a: Cell matrix or lattice parameter array
		# natAr: Number of atoms per type
		# N: Total number of atoms
		# traj: Trajectory array, size:  numFrames × numAtoms × 3
		# strAr: Atom species/types array	

	#	mkXYZ(traj,nfrm,natAr,strAr,'newBlah.xyz')

		trajNew = traj[0:nfrm:evN]									# Downsample trajectory to only retain every evN frame
																	# ...every time step isn't necessary, and this speeds up MSD calc

		tm = np.arange(len(trajNew))*tmFact/1000.0					# Time array in ps
			
		# Calculate Mean Squared Displacement (MSD)
		# Note1:  Since traj is already converted from frac -> cart, 
		# ... this is a 2d array:  msd[nTau,3] ... 3 columns = msd_x, msd_y, msd_z
		# Note2: For simplicity, msd is calculated for time-lags "tau" that span the entire trajectory
		# ... e.g., tau -> [0,total simulation time] ... however, based on how 'msd' is calculated, 
		# ... any 'msd' past 1/2 of the trajectory is meaningless.
		msd = calculate_msd(tm, trajNew[:, atom_indices, :])

		# Track coordinate over MD time
		# ptTrc is an array:  nfrm rows of [x1, x2, x3, ..., y1, y2, y3, ..., z1, z2, z3, ...]
		ptTrc = trkCfnct(trajNew,tmFact,atom_indices)				
		

        # Calculate diffusion coefficients
		diffC_arr, diffLinTm = diffFunc2(tm, msd, simTyp)
		# return is:
		# diffC_arr - [diff_x, diff_y, diff_z, diff_tot]
		# diffLinTm = linear region over which diffC's are calculated

		fwTot.write(f"{'    '.join(f'{val:.4f}' for val in diffC_arr)}\n")
		fwTot.flush()															# write the diffusion coefficients to file:
																				# f'outFiles/diffC_{atTyp}_{stN}_{endN}.dat'

		# Plot the MSD and coordinate tracing data
		pltMsd(diffC_arr[3]*1e-6, diffLinTm, tm, msd, ptTrc, dopNm, title, fwPlotName, atom_indices, simTyp)

		################################

		# Write MSD data to file
		fwstr = f'outFiles/msd_{dop}_{atTyp}_{num}.dat'
		with open(fwstr, 'w') as fw:
			for i in range(len(msd[:,0])):
				fw.write(f"{tm[i]:.2f}    {'    '.join(f'{msd[i,j]:.4f}' for j in range(3))}    {sum(msd[i,:]):.4f}\n")

		# Write atom coordinte trace.  Each row:  x1 x2 x3... y1 y2 y3... z1 z2 z3 ...
		fwstr = f'outFiles/atTr_{dop}_{atTyp}_{num}.dat'
		with open(fwstr, 'w') as fw:
			for i in range(len(ptTrc)):
				outstr = f"{ptTrc[i,0]:.2f}"
				for ind in range(len(atom_indices)):
					x = ptTrc[i, ind + 1 + len(atom_indices)]
					y = ptTrc[i, ind + 1]
					z = ptTrc[i, ind + 1 + len(atom_indices) * 2]
					outstr += f"    {x:.4f}    {y:.4f}    {z:.4f}"
				fw.write(outstr + '\n')




