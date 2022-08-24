# **************************************************
# * File Name : plotter.py
# * Creation Date : 2022-01-31
# * Created By : kstoreyf
# * Description :
# **************************************************

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import socket
import sys

if 'jupyter' not in socket.gethostname():
    sys.path.insert(1, '/home/ksf293/external')
import illustris_python as il
import utils



def plot_halos_dark_and_hydro(halo_arr, base_path_dark, base_path_hydro, snap_num,
                             nrows_outer, ncols_outer, titles, title=None):
    
    sub_width, sub_height = 5.5, 5
    fig = plt.figure(figsize=(sub_width*ncols_outer, sub_height*nrows_outer*2))

    if title is not None:
        fig.suptitle(title, fontsize=24, y=0.94)

    outer = gridspec.GridSpec(nrows_outer, ncols_outer, wspace=0.5, hspace=0.25)

    nrows_inner = 2
    ncols_inner = 1
    
    for i_hd, halo in enumerate(halo_arr):

        inner = gridspec.GridSpecFromSubplotSpec(nrows_inner, ncols_inner,
                        subplot_spec=outer[i_hd],
                        height_ratios=[1,1],
                        hspace=0.0)

        ax0 = plt.Subplot(fig, inner[0])
        ax1 = plt.Subplot(fig, inner[1])
        axarr = [ax0, ax1]
        
        ax0.set_title(titles[i_hd], pad=10)
        alpha = 0.3
        s_dm, s_hydro = 2, 2
                
        # Dark sim
        # want absolute positions, not shifted, so get directly from illustris
        halo_dark_dm = il.snapshot.loadHalo(base_path_dark,snap_num,halo.idx_halo_dark,'dm')
        x_halo_dark_dm = halo_dark_dm['Coordinates']
        #x_halo_dark_dm = halo.shift_x(x_halo_dark_dm, center='x_minPE')
        ax0.scatter(x_halo_dark_dm[:,0], x_halo_dark_dm[:,1], 
                   s=s_dm, alpha=alpha, marker='.', color='darkblue', label='Dark halo DM')
        
        # Hydro sim
        halo_hydro_dm = il.snapshot.loadHalo(base_path_hydro,snap_num,halo.idx_halo_hydro,'dm')
        x_halo_hydro_dm = halo_hydro_dm['Coordinates']
        #x_halo_hydro_dm = halo.shift_x(x_halo_hydro_dm, center='x_minPE')
        ax1.scatter(x_halo_hydro_dm[:,0], x_halo_hydro_dm[:,1], 
                   s=s_dm, alpha=alpha, marker='.', color='rebeccapurple', label='Hydro halo DM')
        
        halo_hydro_stars = il.snapshot.loadHalo(base_path_hydro,snap_num,halo.idx_halo_hydro,'stars')
        if halo_hydro_stars['count'] > 0:
            x_halo_hydro_stars = halo_hydro_stars['Coordinates']
            ax1.scatter(x_halo_hydro_stars[:,0], x_halo_hydro_stars[:,1], 
                       s=s_hydro, alpha=alpha, marker='.', color='darkorange', label='Hydro halo stars')

        subhalo_hydro_stars = il.snapshot.loadSubhalo(base_path_hydro,snap_num,halo.idx_subhalo_hydro,'stars')
        if subhalo_hydro_stars['count'] > 0:
            x_subhalo_hydro_stars = subhalo_hydro_stars['Coordinates']
            ax1.scatter(x_subhalo_hydro_stars[:,0], x_subhalo_hydro_stars[:,1],
                       s=s_hydro, alpha=alpha, marker='.', color='gold', label='Hydro subhalo stars')

        # Set labels 
        ax0.set_ylabel(r'$y$')
        
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        
        ax0.text(0.5, 0.9, 'dark', fontsize=16, horizontalalignment='center',
                 verticalalignment='center', transform=ax0.transAxes)
        ax1.text(0.5, 0.9, 'hydro', fontsize=16, horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes)
        
        # get pos properties
        x_minPE = halo.catalog_properties['x_minPE']
        x_minPE_hydro = halo.catalog_properties['x_minPE_hydro']

        # Set limits
        # x_min = np.min([ax0.get_xlim()[0], ax1.get_xlim()[0]])
        # x_max = np.max([ax0.get_xlim()[1], ax1.get_xlim()[1]])
        # ax0.set_xlim([x_min, x_max])
        # #ax1.set_xlim([x_min, x_max])
        
        # y_min = np.min([ax0.get_ylim()[0], ax1.get_ylim()[0]])
        # y_max = np.max([ax0.get_ylim()[1], ax1.get_ylim()[1]])
        # ax0.set_ylim([y_min, y_max])
        # #ax1.set_ylim([y_min, y_max])
        # ax0.set_aspect('equal', adjustable='datalim')
        # x_min_set, x_max_set = ax0.get_xlim()
        # y_min_set, y_max_set = ax0.get_ylim()
        # ax1.set_xlim([x_min_set, x_max_set])
        # ax1.set_ylim([y_min_set, y_max_set])
        #ax1.set_aspect('equal'), adjustable='datalim')

            

        
        # compute dark CoM
        particle0_pos = x_halo_dark_dm[0]
        x_arr_shifted_byparticle = utils.shift_points_torus(x_halo_dark_dm, particle0_pos, halo.box_size)
        com_dark = np.mean(x_arr_shifted_byparticle, axis=0) + particle0_pos

        # compute hydro CoM
        particle0_pos = x_halo_hydro_dm[0]
        x_arr_shifted_byparticle = utils.shift_points_torus(x_halo_hydro_dm, particle0_pos, halo.box_size)
        com_hydro = np.mean(x_arr_shifted_byparticle, axis=0) + particle0_pos


        # Plot central position points
        lw = 1
        dark_color = 'lightskyblue'
        ax0.axvline(x_minPE[0], c=dark_color, lw=lw, label='Most bound dark subhalo particle')
        ax0.axhline(x_minPE[1], c=dark_color, lw=lw)
        ax1.axvline(x_minPE[0], c=dark_color, lw=lw)
        ax1.axhline(x_minPE[1], c=dark_color, lw=lw)
        
        light_color = 'mediumslateblue'
        ax0.axvline(x_minPE_hydro[0], c=light_color, lw=lw, label='Most bound hydro subhalo particle')
        ax0.axhline(x_minPE_hydro[1], c=light_color, lw=lw)
        ax1.axvline(x_minPE_hydro[0], c=light_color, lw=lw)
        ax1.axhline(x_minPE_hydro[1], c=light_color, lw=lw)

        # Plot R200
        r200 = halo.catalog_properties['r200m']
        circle_r200 = plt.Circle((x_minPE[0], x_minPE[1]), r200, color='silver', fill=False, label='R200', lw=1.5)
        ax0.add_patch(circle_r200)

        circle_innerouter = plt.Circle((x_minPE[0], x_minPE[1]), 3/8*r200, color='silver', fill=False, label='Inner/outer radius', lw=1.5, ls=':')
        ax0.add_patch(circle_innerouter)

        # Plot CoM
        ax0.scatter(com_dark[0], com_dark[1], marker='+', color=dark_color, s=200, lw=3, label='CoM of dark halo DM particles')
        ax1.scatter(com_hydro[0], com_hydro[1], marker='+', color=light_color, s=200, lw=3, label='CoM of hydro halo DM particles')

        # Add subplots
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)
        fig.align_ylabels(axarr)

        # set limits
        center = 0.5*(x_minPE + x_minPE_hydro)
        radius_box = 1.5*r200
        ax0.set_xlim(center[0]-radius_box, center[0]+radius_box)
        ax1.set_xlim(center[0]-radius_box, center[0]+radius_box)
        ax0.set_ylim(center[1]-radius_box, center[1]+radius_box)
        ax1.set_ylim(center[1]-radius_box, center[1]+radius_box)
        plt.setp(ax0.get_xticklabels(), visible=False)


    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles = np.concatenate((handles0, handles1))
    labels = np.concatenate((labels0, labels1))
    plt.legend(handles, labels, fontsize=18, loc=(1.2, 3))


def plot_pred_vs_true(y_label_name, y_true, y_pred, y_train, y_train_pred, 
                      text_results='', title=None, save_fn=None,
                      x_lim=(7,12), y_lim=(7,12), colors_test=None,
                      colorbar_label=''):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()

    y_label = utils.label_dict[y_label_name]
    # main scatter plotting
    plt.scatter(y_train, y_train_pred, s=12, alpha=0.3, c='m', label='training')
    if colors_test is None:
        plt.scatter(y_true, y_pred, s=12, alpha=0.6, c='k', label='testing')
    else:
        plt.scatter(y_true, y_pred, s=12, alpha=0.6, c=colors_test, label='testing')
        plt.colorbar(label=colorbar_label)

    true_line = np.linspace(*x_lim)
    plt.plot(true_line, true_line, color='grey', zorder=0)

    # labels & adjustments
    plt.xlabel(y_label + ', true')
    plt.ylabel(y_label + ', predicted')

    ax.set_aspect('equal')
    
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.text(0.5, 0.3, text_results, 
             transform=ax.transAxes, verticalalignment='top', fontsize=12)
    plt.title(title)
    plt.legend(loc='upper left', fontsize=12)

    # save
    if save_fn is not None:
        plt.savefig(save_fn, bbox_inches='tight')


def plot_pred_vs_true_hist(y_label_name, y_true, y_pred, y_train, y_train_pred, 
                      text_results='', title=None, save_fn=None,
                      x_lim=(7,12), y_lim=(7,12), colors_test=None,
                      colorbar_label=''):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()

    y_label = utils.label_dict[y_label_name]
    # main scatter plotting
    #plt.scatter(y_train, y_train_pred, s=12, alpha=0.3, c='m', label='training')
    if colors_test is None:
        plt.hist2d(y_true, y_pred, c='k', label='testing')
    else:
        plt.hist2d(y_true, y_pred, c=colors_test, label='testing')
        plt.colorbar(label=colorbar_label)

    true_line = np.linspace(*x_lim)
    plt.plot(true_line, true_line, color='grey', zorder=0)

    # labels & adjustments
    plt.xlabel(y_label + ', true')
    plt.ylabel(y_label + ', predicted')

    ax.set_aspect('equal')
    
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    #plt.text(0.5, 0.3, text_results, 
    #         transform=ax.transAxes, verticalalignment='top', fontsize=12)
    #plt.title(title)
    plt.legend(loc='upper left', fontsize=12)

    # save
    if save_fn is not None:
        plt.savefig(save_fn, bbox_inches='tight')


def plot_pred_vs_property(x_label_name, y_label_name, x_property, y_true, y_pred,
                      text_results='', title=None, save_fn=None, overplot_function=None,
                      x_scale='linear', y_scale='linear',
                      x_lim=(10.5, 14), y_lim=(7, 12), colors_test=None,
                      colorbar_label='', mass_multiplier=1e10):
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    

    # main scatter plotting
    plt.scatter(x_property, y_true, s=12, alpha=0.3, c='r', label='true (test)')
    if colors_test is None:
        plt.scatter(x_property, y_pred, s=12, alpha=0.2, c='k', label='predicted (test)')
    else:
        plt.scatter(x_property, y_pred, s=12, alpha=0.2, c=colors_test, label='predicted (test)')
        plt.colorbar(label=colorbar_label)

    # overplot power law
    if overplot_function is not None:
        masses = np.linspace(*x_lim)
        y_powerlaw = overplot_function(masses/mass_multiplier)*mass_multiplier
        plt.plot(masses, y_powerlaw, color='forestgreen', label='input broken power law')
    
    # labels & adjustments
    x_label = utils.label_dict[x_label_name]
    y_label = utils.label_dict[y_label_name]

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(x_scale)
    plt.yscale(y_scale)

    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.text(0.5, 0.3, text_results, 
             transform=ax.transAxes, verticalalignment='top', fontsize=12)
    plt.title(title)
    plt.legend(loc='upper left', fontsize=12)
    
    # save
    if save_fn is not None:
        plt.savefig(save_fn, bbox_inches='tight')


def plot_fits(x_label_name, y_label_name, fitter, log_m_halo, test_error_type='percentile', 
              regularization_lambda=0.0, colors_test=None, colorbar_label='',
              log_mass_shift=10, x_lim=(10.5, 14), y_lim=(7,12),
              show_pred_vs_mass=False):

    # Extract arrays and plot
    y_true = fitter.y_scalar_test
    y_pred = fitter.y_scalar_pred
    
    y_train_pred = fitter.predict(fitter.x_scalar_train, fitter.y_val_current_train,
                                  x_extra=fitter.x_features_extra_train)
    y_train_true = fitter.y_scalar_train
    
    log_m_halo_test = log_m_halo[fitter.idx_test]
    log_m_halo_train = log_m_halo[fitter.idx_train]

    # Compute error
    n_bins = 6
    err_bins_mstellar = np.linspace(min(y_true), max(y_true), n_bins+1)
    err_bins_mhalo = np.linspace(min(log_m_halo_test), max(log_m_halo_test), n_bins+1) # do only for test set
    idx_bins_mstellar = np.digitize(y_true, err_bins_mstellar)
    idx_bins_mhalo = np.digitize(log_m_halo_test, err_bins_mhalo)
    groups_mstellar = []
    groups_mhalo = []
    for i_err in range(n_bins):
        groups_mstellar.append( y_pred[idx_bins_mstellar==i_err-1] ) # -1 bc of how digitize returns results
        groups_mhalo.append( y_pred[idx_bins_mhalo==i_err-1] ) # -1 bc of how digitize returns results
        
    if test_error_type=='msfe':
        frac_err = (y_pred - y_true)/y_true
        msfe_test = np.mean(frac_err**2)
        error_str = f'MSFE: {msfe_test:.3f}'
        n_outliers = len(frac_err[frac_err > 5*msfe_test])
        # TODO: finish implementing binned errors
    elif test_error_type=='percentile':
        delta_y = y_pred - y_true
        percentile_16 = np.percentile(delta_y, 16, axis=0)
        percentile_84 = np.percentile(delta_y, 84, axis=0)
        error_inner68_test = (percentile_84-percentile_16)/2

        error_str = fr"$\sigma_{{68}}$: {error_inner68_test:.3f}"
        n_outliers = len(delta_y[delta_y > 5*error_inner68_test])
        
    
    train_text = ''
    if hasattr(fitter, 'chi2'):
        train_text = fr'$\chi^2$: {fitter.chi2:.3e}; $\kappa$: {fitter.condition_number:.1e}' '\n'

    #n_neg = len(np.where(fitter.y_scalar_pred < 0)[0])
    text_results = fr'$n_\mathrm{{features}}$: {fitter.n_A_features}' '\n' \
                       fr'{error_str} ($n_\mathrm{{test}}$: {fitter.n_test})' '\n' \
                       f'{train_text}' \
                       f'\t \t' fr'($n_\mathrm{{train}}$: {fitter.n_train})' '\n' \
                       fr'$N > 5\sigma$: {n_outliers}'

    if 'm_' in y_label_name:
        y_true += log_mass_shift
        y_pred += log_mass_shift
        y_train_true += log_mass_shift
        y_train_pred += log_mass_shift

    if 'm_' in x_label_name:
        log_m_halo_test += log_mass_shift
        log_m_halo_train += log_mass_shift

    plot_pred_vs_true(y_label_name, y_true, y_pred, y_train_true, y_train_pred, 
                              text_results=text_results, 
                              colors_test=colors_test, colorbar_label=colorbar_label,
                              x_lim=y_lim, y_lim=y_lim)


    plot_pred_vs_property(x_label_name, y_label_name, log_m_halo_test, y_true, y_pred,
                              text_results=text_results, 
                              colors_test=colors_test, colorbar_label=colorbar_label, 
                              x_lim=x_lim, y_lim=y_lim)


def plot_pred_vs_true_hist(y_label_name, y_true, y_pred,
                      text_results='',
                      x_lim=(7,12), y_lim=(7,12)):
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()

    y_label = utils.label_dict[y_label_name]

    #ticks = np.arange(5, 25, 5)
    bins = np.linspace(y_lim[0], y_lim[1], 100)

    inferno_r = matplotlib.cm.inferno_r
    inferno_shifted = utils.shiftedColorMap(inferno_r, start=0.1, stop=1.0)
    plt.hist2d(y_true, y_pred, bins=bins, cmap=inferno_shifted, cmin=1)
    cbar = plt.colorbar(label='number of test objects')
    #cbar.ax.set_yticklabels(ticks)
    
    true_line = np.linspace(*x_lim)
    plt.plot(true_line, true_line, color='grey', zorder=0)

    # labels & adjustments
    plt.xlabel(y_label + ', true')
    plt.ylabel(y_label + ', predicted')

    ax.set_aspect('equal')
    
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.text(0.1, 0.9, text_results, 
             transform=ax.transAxes, verticalalignment='top', fontsize=22)



