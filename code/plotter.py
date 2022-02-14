# **************************************************
# * File Name : plotter.py
# * Creation Date : 2022-01-31
# * Created By : kstoreyf
# * Description :
# **************************************************

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import sys

sys.path.insert(1, '/home/ksf293/external')
import illustris_python as il


tng_path_hydro = '/scratch/ksf293/gnn-cosmology/data/TNG50-4'
tng_path_dark = '/scratch/ksf293/gnn-cosmology/data/TNG50-4-Dark'
base_path_hydro = '/scratch/ksf293/gnn-cosmology/data/TNG50-4/output'
base_path_dark = '/scratch/ksf293/gnn-cosmology/data/TNG50-4-Dark/output'
snap_num = 99


def plot_halos_dark_and_hydro(halo_dicts, nrows_outer, ncols_outer, titles):
    
    sub_width, sub_height = 5.5, 5
    fig = plt.figure(figsize=(sub_width*ncols_outer, sub_height*nrows_outer*2))

    outer = gridspec.GridSpec(nrows_outer, ncols_outer, wspace=0.5, hspace=0.25)

    nrows_inner = 2
    ncols_inner = 1
    
    for i_hd, halo_dict in enumerate(halo_dicts):

        inner = gridspec.GridSpecFromSubplotSpec(nrows_inner, ncols_inner,
                        subplot_spec=outer[i_hd],
                        height_ratios=[1,1],
                        hspace=0.0)

        ax0 = plt.Subplot(fig, inner[0])
        ax1 = plt.Subplot(fig, inner[1])
        axarr = [ax0, ax1]
        
        ax0.set_title(titles[i_hd], pad=10)
        alpha = 0.5
        
        # Dark sim
        idx_halo_dark = halo_dict['idx_halo_dark']
        halo_dark_dm = il.snapshot.loadHalo(base_path_dark,snap_num,idx_halo_dark,'dm')
        x_halo_dark_dm = halo_dark_dm['Coordinates']
        ax0.scatter(x_halo_dark_dm[:,0], x_halo_dark_dm[:,1], 
                   s=30, alpha=alpha, marker='.', color='darkblue', label='Dark halo DM')
        
        # Hydro sim
        idx_halo_hydro = halo_dict['idx_halo_hydro']
        halo_hydro_dm = il.snapshot.loadHalo(base_path_hydro,snap_num,idx_halo_hydro,'dm')
        x_halo_hydro_dm = halo_hydro_dm['Coordinates']
        ax1.scatter(x_halo_hydro_dm[:,0], x_halo_hydro_dm[:,1], 
                   s=30, alpha=alpha, marker='.', color='blue', label='Hydro halo DM')
        
        halo_hydro_stars = il.snapshot.loadHalo(base_path_hydro,snap_num,idx_halo_hydro,'stars')
        if halo_hydro_stars['count'] > 0:
            x_halo_hydro_stars = halo_hydro_stars['Coordinates']
            ax1.scatter(x_halo_hydro_stars[:,0], x_halo_hydro_stars[:,1], 
                       s=60, alpha=alpha, marker='.', color='orange', label='Hydro halo stars')

        idx_subhalo_hydro = halo_dict['idx_subhalo_hydro']
        subhalo_hydro_stars = il.snapshot.loadSubhalo(base_path_hydro,snap_num,idx_subhalo_hydro,'stars')
        if subhalo_hydro_stars['count'] > 0:
            x_subhalo_hydro_stars = subhalo_hydro_stars['Coordinates']
            ax1.scatter(x_subhalo_hydro_stars[:,0], x_subhalo_hydro_stars[:,1],
                       s=60, alpha=alpha, marker='.', color='yellow', label='Hydro subhalo stars')

        # Set labels 
        ax0.set_ylabel(r'$y$')
        
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        
        ax0.text(0.5, 0.9, 'dark', fontsize=16, horizontalalignment='center',
                 verticalalignment='center', transform=ax0.transAxes)
        ax1.text(0.5, 0.9, 'hydro', fontsize=16, horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes)
        

        # Set limits
        x_min = np.min([ax0.get_xlim()[0], ax1.get_xlim()[0]])
        x_max = np.max([ax0.get_xlim()[1], ax1.get_xlim()[1]])
        ax0.set_xlim([x_min, x_max])
        ax1.set_xlim([x_min, x_max])
        
        y_min = np.min([ax0.get_ylim()[0], ax1.get_ylim()[0]])
        y_max = np.max([ax0.get_ylim()[1], ax1.get_ylim()[1]])
        ax0.set_ylim([y_min, y_max])
        ax1.set_ylim([y_min, y_max])
        ax0.set_aspect('equal', adjustable='datalim')
        ax1.set_aspect('equal', adjustable='datalim')
        
        plt.setp(ax0.get_xticklabels(), visible=False)
    
        # Crosshairs at center of mass of DM particles of halos
        dark_color = 'grey'
        lw = 0.8
        com_dark = np.mean(x_halo_dark_dm, axis=0)
        ax0.axvline(com_dark[0], c=dark_color, lw=lw, label='CoM of dark halo DM particles')
        ax0.axhline(com_dark[1], c=dark_color, lw=lw)
        ax1.axvline(com_dark[0], c=dark_color, lw=lw)
        ax1.axhline(com_dark[1], c=dark_color, lw=lw)
        
        light_color = 'skyblue'
        com_hydro = np.mean(x_halo_hydro_dm, axis=0)
        ax0.axvline(com_hydro[0], c=light_color, lw=lw, label='CoM of hydro halo DM particles')
        ax0.axhline(com_hydro[1], c=light_color, lw=lw)
        ax1.axvline(com_hydro[0], c=light_color, lw=lw)
        ax1.axhline(com_hydro[1], c=light_color, lw=lw)

        # Plot R200
        radius = halo_dict['r_mean200_dark_halo']
        circle = plt.Circle((com_dark[0], com_dark[1]), radius, color='forestgreen', fill=False, label='R200')
        ax0.add_patch(circle)

        # Add subplots
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)
        fig.align_ylabels(axarr)

    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles = np.concatenate((handles0, handles1))
    labels = np.concatenate((labels0, labels1))
    plt.legend(handles, labels, fontsize=18, loc=(1.2, 4))
