# msdCalcPlot
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
