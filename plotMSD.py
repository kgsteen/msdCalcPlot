"""
Module: plotMSD.py
Author: Krista G. Steenbergen
Date: 23 Dec 2024

Purpose:
    This module generates multi-panel plots for analyzing atom-traces 
    (z-dist from interface and xy-distance from origin) 
    and Mean Squared Displacement (MSD) from molecular dynamics simulations.

Functions:
    - pltMsd: Generate and save a multi-panel plot for MSD and displacement data.
    - setYaxLim: Adjust the Y-axis limits based on axis span and minimum value.
    - setXaxLim: Adjust the X-axis limits based on axis span.

Dependencies:
    - numpy
    - matplotlib.pyplot

Usage Example:
    from plotMsd import pltMsdMid
    pltMsd(diffC_tot, diffTm, tm, msd, pTrc, dopNm, title, fwstr, atom_indices)
"""
import matplotlib.pyplot as plt
import numpy as np

#########################################
#########################################
############################
def pltMsd(diffC_tot, diffTm, tm, msd, pTrc, dopNm, title, fwstr, atom_indices, simTyp):
    """
    Generate a multi-panel plot showing atomic displacements and Mean Squared Displacement (MSD).

    Parameters:
        diffC_tot (float): Diffusion coefficient (total diffusion) value for display on the plot.
        diffTm (list or ndarray): Time range used for diffusion coefficient fitting.
        tm (ndarray): Time array corresponding to MSD values.
        msd (ndarray): MSD values (shape: n_steps x 3) for x, y, and z axes.
        pTrc (ndarray): Array containing displacement data for plotting.
        dopNm (list of str): Dopant names.
        title (str): Plot title.
        fwstr (str): Output file path for saving the plot.
        atom_indices (list of int): Indices of selected atoms for tracking.

    Returns:
        None: The plot is saved as a PDF file.
    """
    nat = len(atom_indices)							# total number of atoms for atomTrace plots (typically number of dopants)
    lenTm = round(len(tm) / 2.0)					# total amount of time for the MSD plot - 1/2 total simulation time
    tm = tm[:lenTm]									# cut the "tm" array
    msd = msd[:lenTm, :]

    # Create figure with a 3-row layout
    fig = plt.figure(figsize=(7.5, 8))
    outer_grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2])
    upper_grid = outer_grid[0:2].subgridspec(2, 1, height_ratios=[1, 1], hspace=0)
    lower_grid = outer_grid[2].subgridspec(1, 1, hspace=0.2)

    # Define individual axes
    ax1 = fig.add_subplot(upper_grid[0, 0])  # Top panel (z-displacement)
    ax2 = fig.add_subplot(upper_grid[1, 0])  # Middle panel (xy-displacement)
    ax3 = fig.add_subplot(lower_grid[0, 0])  # Bottom panel (MSD)

    # Plot z and xy displacements
    for ind in range(nat):
        if (ind==0):
            ax1.plot(pTrc[:, 0], pTrc[:, ind + 1 + nat * 2], label='z-dist')
            ax2.plot(pTrc[:, 0], np.sqrt(pTrc[:, ind + 1]**2 + pTrc[:, ind + 1 + nat]**2), label='xy')
        else:
            ax1.plot(pTrc[:, 0], pTrc[:, ind + 1 + nat * 2])
            ax2.plot(pTrc[:, 0], np.sqrt(pTrc[:, ind + 1]**2 + pTrc[:, ind + 1 + nat]**2))

    # Plot MSD
    ax3.plot(tm[::4], msd[::4, 0] + msd[::4, 1], marker='x', linestyle='-', color='b', markerfacecolor='none', label='xy')
    ax3.plot(tm[::4], msd[::4, 2], marker='s', linestyle='-', color='b', markerfacecolor='none', label='z')
    ax3.plot(tm[::4], msd[::4, 0] + msd[::4, 1] + msd[::4, 2], marker='o', linestyle='-', color='b', label='total')

    ###################
    # Adjust axis limits 
    # ... some steps are redundant / have been changed multiple times for best "visual"
    minY = np.zeros(6)
    spnAx = np.zeros(6)
    xSpn = np.zeros(6)
    for i, ax in enumerate([ax1, ax2, ax3]):
        spnAx[i], minY[i] = setYaxLim(ax)
        xSpn[i] = setXaxLim(ax)

    for i, ax in enumerate([ax1, ax2]):
        ax.set_ylim([minY[i], minY[i] + max(spnAx[0:2])])
        ax.set_xlim([0, max(xSpn[0:2])])

    ax3.set_ylim([0, 30])
    ax3.set_xlim([0, xSpn[2]])
    
    ###################

    # Customize plot aesthetics
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    ax1.set_xticklabels([])
    ax1.grid(True)
    ax1.legend()
    ax2.tick_params(axis='x', labelsize=12)
    ax2.grid(True)
    ax2.legend()

    if (simTyp=='npt'):
        ax1.set_title(f"GaLiq + {dopNm[1]}, with oxide", fontsize=14)
        ax1.set_ylabel(
            r'$\perp$ distance to liquid surface' + '\n' +
            r'oxide interface ($\AA$)', 
            fontsize=12
        )
    else:
        ax1.set_title(f"GaLiq + {dopNm[1]}, no oxide", fontsize=14)
        ax1.set_ylabel(
            r'$\perp$ distance to liquid surface,' + '\n' +
            r'vacuum interface ($\AA$)', 
            fontsize=12
        )

    ax3.set_xlabel('Time Interval (ps)', fontsize=14)
    ax3.set_ylabel(r'Mean Squared Displacement ($\AA^2$)', fontsize=14)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.grid(True)
    ax3.legend()

    ax3.text(0.12, 0.65, f'Diff={diffC_tot:3.3e}', fontsize=12, ha='center', va='center', transform=ax3.transAxes)
    ax3.axvspan(min(diffTm), max(diffTm), color='grey', alpha=0.3)

    fig.suptitle(title, fontsize=16, y=0.96)

    # Save and close plot
    plt.savefig(fwstr, format='pdf')
    plt.close(fig)


#########################################
#########################################
############################
def setYaxLim(ax):
    """
    Get Y-axis limits and span for an axis.

    Parameters:
        ax (matplotlib.axes.Axes): Axis object.

    Returns:
        tuple: (span_ax, minV)
            - span_ax (float): Range of Y-axis.
            - minV (float): Minimum Y-axis value (floored).
    """
    ymin_ax, ymax_ax = ax.get_ylim()
    span_ax = ymax_ax - ymin_ax
    minV = np.floor(ymin_ax)
    
    return span_ax, minV


#########################################
#########################################
############################
def setXaxLim(ax):
    """
    Get X-axis limits for an axis.

    Parameters:
        ax (matplotlib.axes.Axes): Axis object.

    Returns:
        float: Range of X-axis.
    ** Only max is returned because all x-axis min = 0
    """
    xmin_ax, xmax_ax = ax.get_xlim()
    
    return xmax_ax




