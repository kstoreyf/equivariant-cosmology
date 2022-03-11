# **************************************************
# * File Name : plotter.py
# * Creation Date : 2022-01-31
# * Created By : kstoreyf
# * Description :
# **************************************************

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import socket
import sys

if 'jupyter' not in socket.gethostname():
    sys.path.insert(1, '/home/ksf293/external')
import illustris_python as il


tng_path_hydro = '/scratch/ksf293/gnn-cosmology/data/TNG50-4'
tng_path_dark = '/scratch/ksf293/gnn-cosmology/data/TNG50-4-Dark'
base_path_hydro = '/scratch/ksf293/gnn-cosmology/data/TNG50-4/output'
base_path_dark = '/scratch/ksf293/gnn-cosmology/data/TNG50-4-Dark/output'
snap_num = 99


def plot_halos_dark_and_hydro(halo_dicts, base_path_dark, base_path_hydro, snap_num,
                             nrows_outer, ncols_outer, titles):
    
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



def plot_pred_vs_true(y_true, y_pred, y_train, y_train_pred, 
                      fitter, msfe_test, chi2_train, mass_multiplier,
                      title=None, save_fn=None):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()

    # main scatter plotting
    plt.scatter(y_train, y_train_pred, s=12, alpha=0.3, c='m', label='training')
    plt.scatter(y_true, y_pred, s=12, alpha=0.6, c='k', label='testing')


    # get limits, plot true line
    m_minmin = min(min(y_true[np.where(y_true > 0)]), 
                   min(y_pred[np.where(y_pred > 0)]))
    m_maxmax = max(max(y_true[np.where(y_true > 0)]), 
                   max(y_pred[np.where(y_pred > 0)]))
    true_line = np.linspace(0.5*m_minmin, 2*m_maxmax)
    plt.plot(true_line, true_line, color='grey', zorder=0)

    # labels & adjustments
    plt.xlabel(r'$m_\mathrm{true}$')
    plt.ylabel(r'$m_\mathrm{pred}$')
    plt.xscale('log')
    plt.yscale('log')
    ax.set_aspect('equal')
    plt.xlim(0.5*m_minmin, 2*m_maxmax)
    plt.ylim(0.5*m_minmin, 2*m_maxmax)

    n_neg = len(np.where(fitter.y_scalar_pred*mass_multiplier < 0)[0])
    plt.text(0.1, 0.9, fr'$n_\mathrm{{features}}$: {fitter.n_A_features}, rank: {fitter.rank}' '\n'
                       fr'MSFE: {msfe_test:.3f}, $n_\mathrm{{test}}$: {fitter.n_test}' '\n'
                       fr'$\chi^2$: {chi2_train:.3e}, $n_\mathrm{{train}}$: {fitter.n_train}' '\n'
                       fr'# m_pred < 0: {n_neg}', 
             transform=ax.transAxes, verticalalignment='top', fontsize=12)
    plt.title(title)
    plt.legend(loc='lower right', fontsize=12)

    # save
    if save_fn is not None:
        plt.savefig(save_fn, bbox_inches='tight')


def plot_pred_vs_mass(mass, y_true, y_pred, mass_train, y_train, y_train_pred, 
                      fitter, msfe_test, chi2_train, mass_multiplier,
                      title=None, save_fn=None, overplot_function=None,
                      logx='log', logy='log'):
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    
    # main scatter plotting
    #plt.scatter(mass_train, y_train_pred, s=12, alpha=0.3, c='m', label='training')
    plt.scatter(mass, y_true, s=12, alpha=0.3, c='c', label='true (test)')
    plt.scatter(mass, y_pred, s=12, alpha=0.2, c='k', label='predicted (test)')

    # get limits, plot true line
    mass_minmin = min(min(mass[np.where(mass > 0)]), 
                   min(mass_train[np.where(mass_train > 0)]))
    mass_maxmax = max(max(mass[np.where(mass > 0)]), 
                   max(mass_train[np.where(mass_train > 0)]))
    y_minmin = min(min(y_pred[np.where(y_pred > 0)]), 
                   min(y_train_pred[np.where(y_train_pred > 0)]))
    y_maxmax = max(max(y_pred[np.where(y_pred > 0)]), 
                   max(y_train_pred[np.where(y_train_pred > 0)]))

    # overplot power law
    if overplot_function is not None:
        masses = np.logspace(np.log10(mass_minmin), np.log10(mass_maxmax), 100)
        y_powerlaw = overplot_function(masses/mass_multiplier)*mass_multiplier
        plt.plot(masses, y_powerlaw, color='forestgreen', label='input broken power law')
    
    # labels & adjustments
    plt.xlabel(r'$M_\mathrm{halo,DM}$')
    plt.ylabel(r'$m_\mathrm{stellar,pred}$')
    plt.xscale(logx)
    plt.yscale(logy)
    #plt.xlim(0.5*mass_minmin, 2*mass_maxmax)
    #plt.xlim(0.5*mass_minmin, 2*mass_maxmax)
    #plt.ylim(0.5*y_minmin, 2*y_maxmax)

    n_neg = len(np.where(fitter.y_scalar_pred*mass_multiplier < 0)[0])
    plt.text(0.1, 0.9, fr'$n_\mathrm{{features}}$: {fitter.n_A_features}, rank: {fitter.rank}' '\n'
                       fr'MSFE: {msfe_test:.3e}, $n_\mathrm{{test}}$: {fitter.n_test}' '\n'
                       fr'$\chi^2$: {chi2_train:.3e}, $n_\mathrm{{train}}$: {fitter.n_train}' '\n'
                       fr'# m_pred < 0: {n_neg}', 
             transform=ax.transAxes, verticalalignment='top', fontsize=12)
    plt.title(title)
    plt.legend(loc='lower right', fontsize=12)

    
    # save
    if save_fn is not None:
        plt.savefig(save_fn, bbox_inches='tight')