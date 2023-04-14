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

    y_label = utils.label_dict[y_label_name] if y_label_name in utils.label_dict else y_label_name
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
    print(y_label)
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


def plot_pred_vs_true_hist2(y_label_name, y_true, y_pred, y_train, y_train_pred, 
                      text_results='', title=None, save_fn=None,
                      x_lim=None, y_lim=None, colors_test=None,
                      colorbar_label=''):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()

    y_label = utils.label_dict[y_label_name] if y_label_name in utils.label_dict else y_label_name
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
    y_label = utils.label_dict[y_label_name] if y_label_name in utils.label_dict else y_label_name

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


def plot_pred_vs_property_hist(ax, x_label_name, y_label_name, x_property, y_pred,
                               cmap, text_results='', x_lim=None, y_lim=None,
                               weight=1, weight_by_dex=False,
                               label_append=', predicted'):

    y_label = utils.label_dict[y_label_name]
    
    if x_lim is None:
        x_lim = utils.lim_dict[x_label_name]
    if y_lim is None:
        y_lim = utils.lim_dict[y_label_name]
    
    bin_width = (y_lim[1]-y_lim[0])/100
    bins_x = np.arange(x_lim[0], x_lim[1]+bin_width, bin_width)
    bins_y = np.arange(y_lim[0], y_lim[1]+bin_width, bin_width)

    if weight_by_dex:
        weight /= bin_width 
    weights = np.full(y_pred.shape, weight)
    ax.hist2d(x_property, y_pred, bins=[bins_x, bins_y], cmap=cmap, cmin=weight, weights=weights)

    # labels & adjustments
    x_label = utils.label_dict[x_label_name]
    y_label = utils.label_dict[y_label_name]
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label + label_append)
    
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.text(0.1, 0.9, text_results, 
             transform=ax.transAxes, verticalalignment='top', fontsize=22)


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


def plot_pred_vs_true_hist(ax, y_label_name, y_true, y_pred, cmap,
                      text_results='', title=None, save_fn=None,
                      colorbar_fig=None, weight=1, weight_by_dex=False,
                      colorbar_label='', x_lim=None, y_lim=None):
    
    if y_lim is None:
        y_lim = utils.lim_dict[y_label_name]
    bin_width = (y_lim[1]-y_lim[0])/100
    bins = np.arange(y_lim[0], y_lim[1]+bin_width, bin_width)

    if weight_by_dex:
        weight /= bin_width 
    weights = np.full(y_true.shape, weight)
    h = ax.hist2d(y_true, y_pred, bins=bins, cmap=cmap, cmin=weight, weights=weights)
        
    # true line
    true_line = np.linspace(*y_lim)
    ax.plot(true_line, true_line, color='grey', zorder=0)

    # labels & adjustments
    ax.set_title(title)
    y_label = utils.label_dict[y_label_name] if y_label_name in utils.label_dict else y_label_name
    ax.set_xlabel(y_label + ', true')
    ax.set_ylabel(y_label + ', predicted')
    ax.set_aspect('equal')
    
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
        
    ax.text(0.1, 0.9, text_results, 
             transform=ax.transAxes, verticalalignment='top', fontsize=22)
    
    if colorbar_fig is not None:
        cbar = colorbar_fig.colorbar(h[3], ax=ax, label=colorbar_label)#, ticks=ticks)

    return h


def plot_halo_dark_geometric(base_path_dark, snap_num, m_dmpart, halo, bin_edges_frac):
    
    r200 = halo.catalog_properties['r200m']
    bin_edges = r200*np.array(bin_edges_frac)

    n_bins = len(bin_edges)-1
    #colors = ['mediumblue', 'lightsteelblue', 'cornflowerblue']
    
    # Dark sim
    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()
    
    s_dm = 0.12
    alpha = 0.1
    # want absolute positions, not shifted, so get directly from illustris
    halo_dark_dm = il.snapshot.loadHalo(base_path_dark,snap_num,halo.idx_halo_dark,'dm')
    x_halo_dark_dm = halo_dark_dm['Coordinates']
    x_minPE = halo.catalog_properties['x_minPE']
    dists = np.sqrt((x_halo_dark_dm[:,0]-x_minPE[0])**2 +
                    (x_halo_dark_dm[:,1]-x_minPE[1])**2 +
                    (x_halo_dark_dm[:,2]-x_minPE[2])**2)
    
    #cmap = matplotlib.cm.get_cmap(cmap_shifted)
    cmap = matplotlib.cm.cool
    cmap = utils.shiftedColorMap(cmap, start=0.3, midpoint=0.65, stop=1.0)
    #m_200 = halo.catalog_properties['m200m']
    m_max = 1
    
    
    masses = []
    for i in range(n_bins):
        x_inbin = x_halo_dark_dm[(dists >= bin_edges[i]) & (dists < bin_edges[i+1])]
        m_inbin = len(x_inbin)*m_dmpart
        masses.append(m_inbin)
        
    m_200 = np.sum(masses[:2])    
    for i in range(n_bins):   
        x_inbin = x_halo_dark_dm[(dists >= bin_edges[i]) & (dists < bin_edges[i+1])]
        print(masses[i]/m_200)
        color = cmap(masses[i]/m_200)
        scat = ax.scatter(x_inbin[:,0], x_inbin[:,1], 
               s=s_dm, alpha=alpha, marker='.', color=color, label='Dark halo DM', zorder=0)  
        
        com_bin = np.mean(x_inbin, axis=0)
        ax.arrow(x_minPE[0], x_minPE[1], com_bin[0]-x_minPE[0], com_bin[1]-x_minPE[1], head_width=15,
                  facecolor=color, edgecolor='k', zorder=2)
        #ax.scatter(com_bin[0], com_bin[1], marker='o', color=color, edgecolor='k', s=150)
        
        if i < n_bins-1:
            circle_r200 = plt.Circle((x_minPE[0], x_minPE[1]), bin_edges[i+1], color='dimgrey', fill=False, lw=1)
            ax.add_patch(circle_r200)
    
    circle_r200 = plt.Circle((x_minPE[0], x_minPE[1]), 1.2*r200, color='dimgrey', fill=False, lw=1)
    ax.scatter(x_minPE[0], x_minPE[1], marker='+', color='k', s=200, lw=3, zorder=1)

    #scat.set_clim(0, 0.8)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap)
    #sm.set_clim(0.1, 0.6)
    cbar = fig.colorbar(sm, shrink=0.7, pad=0.04)
    cbar.set_label(label=r'$M_\mathrm{bin}/M_{200}$', size=28)#, extend='max')
    cbar.ax.tick_params(labelsize=24) 
    
    ax.set_aspect('equal')
    plt.axis('off')


def plot_residual_vs_property_hist(ax, x_label_name, y_label_name, x_property, y_true, y_pred,
                               cmap, text_results='', x_lim=None, y_lim=None):

    x_lim = utils.lim_dict[x_label_name]    
    bins_x = np.linspace(x_lim[0], x_lim[1], 100)
    bins_y = np.linspace(-1, 1, 100)

    # plot data
    ax.hist2d(x_property, y_pred-y_true, bins=[bins_x, bins_y], 
              cmap=cmap, cmin=1)
    
    # labels & adjustments
    x_label = utils.label_dict[x_label_name]
    y_label = utils.label_dict[y_label_name]
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'$\Delta$ ' + y_label)
    
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
        
    ax.axhline(0, color='grey')
    ax.text(0.1, 0.9, text_results, 
             transform=ax.transAxes, verticalalignment='top', fontsize=22)


def plot_multi_panel_pred(x_label_name, y_label_name, x_property, y_true, y_pred,
                      text_results='', title=None, save_fn=None,
                      colors_test=None,
                      colorbar_label=''):
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12,12),
                              gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [10, 9]},
                              )
    plt.subplots_adjust(hspace=0.2, wspace=0.33)
    
    inferno_r = matplotlib.cm.inferno_r
    cmap = utils.shiftedColorMap(inferno_r, start=0.1, stop=1.0)
    
    plot_pred_vs_property_hist(axarr[0,0], x_label_name, y_label_name, x_property, y_pred, cmap)
    
    h = plot_pred_vs_true_hist(axarr[0,1], y_label_name, y_true, y_pred, cmap, text_results=text_results)
    
    plot_residual_vs_property_hist(axarr[1,0], x_label_name, y_label_name, x_property, y_true, y_pred, cmap)

    plot_residual_vs_property_hist(axarr[1,1], y_label_name, y_label_name, y_true, y_true, y_pred, cmap)
    
    ticks = np.arange(5, 25, 5)
    cax = fig.add_axes([0.93, 0.33, 0.02, 0.33])
    cbar = plt.colorbar(h[3], cax=cax, label='number of test objects per bin', ticks=ticks)
    cbar.ax.set_yticklabels(ticks)


def plot_multi_panel_gal_props(x_label_name, y_label_name_arr, x_property, y_true_arr, y_pred_arr,
                      text_results_arr=[], title=None, save_fn=None,
                      colorbar_label=''):
    
    nprops = len(y_label_name_arr)
    fig, axarr = plt.subplots(nrows=nprops, ncols=2, figsize=(12, nprops*5),
                              gridspec_kw={'width_ratios': [1, 1.1]})
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    
    inferno_r = matplotlib.cm.inferno_r
    cmap = utils.shiftedColorMap(inferno_r, start=0.1, stop=1.0)
    
    for i in range(nprops):
        plot_pred_vs_property_hist(axarr[i,0], x_label_name, y_label_name_arr[i], x_property, y_pred_arr[i], 
                                   cmap)

        h = plot_pred_vs_true_hist(axarr[i,1], y_label_name_arr[i], y_true_arr[i], y_pred_arr[i], cmap, 
                                   text_results=text_results_arr[i], colorbar_fig=fig)


def plot_a_mfrac_accuracy(a_pred, a_true, mfracs, title='', n_show=8):
    
    locs_norm = matplotlib.colors.Normalize(vmin=0, vmax=n_show)
    cmap = matplotlib.cm.get_cmap('turbo')
    colors = [cmap(locs_norm(i)) for i in range(n_show)]

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]},
                                  figsize=(6,8))
    plt.subplots_adjust(hspace=0.03)
    fig.suptitle(title, fontsize=16)

    #errs = (a_pred - a_true)/a_true
    errs = a_pred - a_true

    np.random.seed(14)
    rand_idxs_show = np.random.randint(len(a_true), size=n_show)
    for i, i_rand in enumerate(rand_idxs_show):
        #halo = sim_reader.dark_halo_arr[i_rand]
        #a_mah, m_mah = halo.catalog_properties['MAH']
        #plt.plot(a_mah, m_mah/m_mah[0], marker='o', markersize=3, ls='None')

        label_true, label_pred = None, None
        if i==0:
            label_true = 'true value'
            label_pred = 'predicted value'

        ax0.plot(mfracs, a_true[i_rand], marker='o', markersize=4, ls='None', color=colors[i], label=label_true)
        ax0.plot(mfracs, a_pred[i_rand], color=colors[i], label=label_pred)

        ax1.plot(mfracs, errs[i_rand], color=colors[i])

    p16 = np.percentile(errs, 16, axis=0)
    p84 = np.percentile(errs, 84, axis=0)
    sig68_avg = 0.5*(p84-p16)
    ax2.plot(mfracs, p16, color='k', lw=2, label='prediction error')
    ax2.plot(mfracs, p84, color='k', lw=2)
    ax2.axhline(0.0, color='grey', lw=1)

    # print errors
    mfracs_to_print_err = [0.25, 0.5, 0.75]
    errs_to_print = []
    for mfrac in mfracs_to_print_err:
        _, idx_mfrac = utils.find_nearest(mfracs, mfrac)
        errs_to_print.append(rf"$\sigma_{{68}}(M/M_{{a=1}}={mfrac:.2f}) = {sig68_avg[idx_mfrac]:.3f}$")
    
    ax0.text(0.4, 0.1, '\n'.join(errs_to_print), fontsize=14)

    # TODO: this should be based on training set, not test, i think!
    a_true_mean = np.mean(a_true, axis=0)
    #sample_var = (y_test - y_test_mean)/y_test_mean
    sample_var = (a_true - a_true_mean)
    sample_p16 = np.percentile(sample_var, 16, axis=0)
    sample_p84 = np.percentile(sample_var, 84, axis=0)
    ax2.fill_between(mfracs, sample_p16, sample_p84, color='blue', lw=2, alpha=0.3, label='sample variance')

    ax0.set_ylabel(r'$a$, scale factor')
    #ax1.set_ylabel(r'$(a_\mathrm{pred}-a_\mathrm{true})/a_\mathrm{true}$')
    ax1.set_ylabel(r'$a_\mathrm{pred}-a_\mathrm{true}$')
    ax2.set_ylabel(r'$\sigma_{68}$')

    ax2.set_xlabel(r'$M_\mathrm{vir}(a)$/$M_\mathrm{vir}(a=1)$ of most massive progenitor halo')

    ax0.set_ylim(0,1)
    ax1.axhline(0.0, color='grey', lw=1)
    ax2.set_ylim(-0.2, 0.2)
    
    ax0.legend(fontsize=12, loc='upper left')
    ax2.legend(fontsize=12, loc='lower left')


def plot_errors_vs_mfracs(ax, a_pred_arr, a_true, mfracs, a_pred_labels, colors, 
                          lws=None, title='', legend_loc='best'):

    if lws is None:
        lws = [2]*len(y_pred_arr)

    for i_y in range(len(a_pred_arr)):
        a_pred = a_pred_arr[i_y]
        errs = a_pred - a_true

        p16 = np.percentile(errs, 16, axis=0)
        p84 = np.percentile(errs, 84, axis=0)
        sig68_avg = 0.5*(p84-p16)
        ax.plot(mfracs, p16, color=colors[i_y], lw=lws[i_y], label=a_pred_labels[i_y])
        ax.plot(mfracs, p84, color=colors[i_y], lw=lws[i_y])
        
    ax.axhline(0.0, color='grey', lw=1)

    
    # TODO: this should be based on training set, not test, i think!
    a_true_mean = np.mean(a_true, axis=0)
    #sample_var = (y_test - y_test_mean)/y_test_mean
    sample_var = (a_true - a_true_mean)
    sample_p16 = np.percentile(sample_var, 16, axis=0)
    sample_p84 = np.percentile(sample_var, 84, axis=0)
    ax.fill_between(mfracs, sample_p16, sample_p84, color='blue', lw=0, alpha=0.15, label='sample variance')

    ax.set_ylabel(r'$\sigma_{68}$')
    ax.set_xlabel(r'$M_\mathrm{vir}(a)$/$M_\mathrm{vir}(a=1)$ of most massive progenitor halo')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.17, 0.17)
    
    ax.legend(fontsize=12, loc=legend_loc)


def plot_errors_vs_a(ax, a_pred_arr, a_true, a_pred_labels, colors, 
                          lws=None, title='', legend_loc='best',
                          convert_to_Gyr=False):

    print("FOR TESTING")
    a_true = a_true[::5]
    a_pred_arr = a_pred_arr[:,::5]

    a_min, a_max = 0, 1

    if convert_to_Gyr: 
        from astropy.cosmology import Planck18, z_at_value
        cosmo = Planck18
        def a_to_Gyr(a):
            assert not np.any(a==0), "Can't handle zeros rn"
            z = 1/a - 1
            return cosmo.age(z).value # returns in Gyr
        a_max = a_to_Gyr(a_max)
        print("conv true")
        a_true = a_to_Gyr(a_true)
        print("conv pred")
        a_pred_arr = a_to_Gyr(a_pred_arr)
        print("done!")
        # print(np.sum(np.isfinite(a_true.flatten()))/len(a_true.flatten()))
        # print(np.sum(np.isfinite(a_pred_arr.flatten()))/len(a_pred_arr.flatten()))

    if lws is None:
        lws = [2]*len(y_pred_arr)

    a_bins = np.linspace(a_min, a_max, 12)
    a_bins_avg = 0.5*(a_bins[:-1] + a_bins[1:])
    # if convert_to_Gyr:
    #    a_bins_avg = a_to_Gyr(a_bins_avg)

    a_true_flat = a_true.flatten()
    for i_y in range(len(a_pred_arr)):

        a_pred_flat = a_pred_arr[i_y].flatten()

        p16_arr, p84_arr = [], []
        sample_p16_arr, sample_p84_arr = [], []
        for i in range(len(a_bins)-1):
            i_inbin = (a_true_flat >= a_bins[i]) & (a_true_flat < a_bins[i+1])
            a_true_inbin = a_true_flat[i_inbin]
            a_pred_inbin = a_pred_flat[i_inbin]
            errs = a_pred_inbin - a_true_inbin
            p16_arr.append( np.percentile(errs, 16) )
            p84_arr.append( np.percentile(errs, 84) )
            
            # TODO: can't figure out right way to do sample variance here
            # sample variance
            # should be based on training set, not test ?
            # a_true_mean = np.mean(a_true_inbin, axis=0)
            # sample_var = (a_true_inbin - a_true_mean)
            # sample_p16_arr.append( np.percentile(sample_var, 16) )
            # sample_p84_arr.append( np.percentile(sample_var, 84) )
        # if convert_to_Gyr:
        #     signs_p16 = np.sign(p16_arr)
        #     signs_p84 = np.sign(p84_arr)
            
        #     p16_arr = signs_p16*a_to_Gyr(np.abs(p16_arr))
        #     p84_arr = signs_p84*a_to_Gyr(np.abs(p84_arr))

        ax.plot(a_bins_avg, p16_arr, color=colors[i_y], lw=lws[i_y], label=a_pred_labels[i_y])
        ax.plot(a_bins_avg, p84_arr, color=colors[i_y], lw=lws[i_y])
        
    # ax.fill_between(a_bins_avg, sample_p16_arr, sample_p84_arr, color='blue', lw=0, alpha=0.15, 
    #                 label='sample variance')
    
    ax.axhline(0.0, color='grey', lw=1)

    if convert_to_Gyr:
        ax.set_ylim(-5, 5)        
        ax.set_xlabel(r'age of universe [Gyr]')
        ax.set_ylabel(r'$\sigma_{68}$, age of universe [Gyr]')
        #a_max = a_to_Gyr(a_max)
    else:
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel(r'$a$, scale factor')
        ax.set_ylabel(r'$\sigma_{68}, a$')
    ax.set_xlim(a_min, a_max)

    ax.legend(fontsize=12, loc=legend_loc)


def plot_errors_vs_property(ax, x_label_name, y_label_name, x_property, y_true, y_pred_arr,
                               y_pred_labels, colors, lws=None, x_lim=None, show_legend=True,
                               test_error_type='percentile',
                               x_bins=None, y_lowerlim=None):

    if lws is None:
        lws = [2]*len(y_pred_arr)
    
    if x_lim is None:
        x_lim = utils.lim_dict[x_label_name]    

    if x_bins is None:
        x_bins = np.linspace(x_lim[0], x_lim[1], 12)
        print("Assuming x bins are log !")
    x_bins_avg = np.log10(0.5*(10**x_bins[:-1] + 10**x_bins[1:]))

    for i_y in range(len(y_pred_arr)):
        y_pred = y_pred_arr[i_y]
        errors = []
        for bb in range(len(x_bins)-1):
            i_inbin = (x_property >= x_bins[bb]) & (x_property < x_bins[bb+1])
            error_inbin, _ = utils.compute_error(y_true[i_inbin], y_pred[i_inbin], test_error_type=test_error_type)
            errors.append(error_inbin)
        ax.plot(x_bins_avg, errors, label=y_pred_labels[i_y], color=colors[i_y], lw=lws[i_y])
    
    if y_lowerlim is not None:
        y_bottom = np.zeros(len(x_bins_avg))
        ax.fill_between(x_bins_avg, y_bottom, y_lowerlim, alpha=0.2, color='k', label='Lower limit from chaos')

    # labels & adjustments
    x_label = utils.get_label(x_label_name)
    y_label = utils.get_label(y_label_name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(fr'$\sigma_{{68}}$, {y_label}')
    if show_legend:
        ax.legend(loc='best', fontsize=12)
    
    # if x_lim is not None:
    #     ax.set_xlim(x_lim)

    ax.set_xlim(x_bins_avg[0], x_bins_avg[-1])
    ax.axhline(0, color='grey')


def plot_multi_panel_gal_props(x_label_name, y_label_name_arr, x_property, y_true_arr, y_pred_arr,
                      text_results_arr=[], title=None, save_fn=None,
                      colorbar_label=''):
    
    nprops = len(y_label_name_arr)
    fig, axarr = plt.subplots(nrows=nprops, ncols=2, figsize=(12, nprops*5),
                              gridspec_kw={'width_ratios': [1, 1]})
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    inferno_r = matplotlib.cm.inferno_r
    cmap = utils.shiftedColorMap(inferno_r, start=0.1, stop=1.0)
    
    for i in range(nprops):
        #plot_pred_vs_property_hist(axarr[i,0], x_label_name, y_label_name_arr[i], x_property, y_pred_arr[i], 
        #                           cmap)
        h = plot_pred_vs_true_hist(axarr[i,0], y_label_name_arr[i], y_true_arr[i], y_pred_arr[i], cmap, 
                                   text_results=text_results_arr[i], colorbar_fig=fig)

        plot_errors_vs_property(axarr[i,1], x_label_name, y_label_name_arr[i], 
                                x_property, 
                                y_true_arr[i], [y_pred_arr[i]],
                                ['scalars'], ['black'])


def plot_multi_panel_gal_props_errors(x_label_name, y_label_name_arr, x_property, 
                      y_true_arr, y_pred_arr, feature_labels, feature_colors,
                      j_fiducial=0,
                      weight=1, weight_by_dex=False,
                      text_results_arr=[],
                      x_bins=None, y_lowerlim_arr=None,
                      title=None, save_fn=None,
                      colorbar_label=''):
    
    y_true_arr = np.array(y_true_arr)
    y_pred_arr = np.array(y_pred_arr)

    n_labels = len(y_label_name_arr)
    n_feature_sets = len(feature_labels)
    assert n_labels==y_pred_arr.shape[0], "Wrong y shape!"
    assert n_feature_sets==y_pred_arr.shape[1], "Wrong y shape!"

    # line widths for error plot
    lws = [1]*n_feature_sets
    lws[j_fiducial] = 2

    fig, axarr = plt.subplots(nrows=n_labels, ncols=4, figsize=(24, n_labels*5),
                              gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    plt.subplots_adjust(hspace=0.3, wspace=0.5)
    
    inferno_r = matplotlib.cm.inferno_r
    cmap = utils.shiftedColorMap(inferno_r, start=0.1, stop=1.0)
    
    for i in range(n_labels):
        plot_pred_vs_property_hist(axarr[i,0], x_label_name, y_label_name_arr[i], x_property, y_true_arr[i], 
                                   cmap, weight=weight, weight_by_dex=weight_by_dex, label_append=', true')

        plot_pred_vs_property_hist(axarr[i,1], x_label_name, y_label_name_arr[i], x_property, y_pred_arr[i,j_fiducial], 
                                   cmap, weight=weight, weight_by_dex=weight_by_dex)

        h = plot_pred_vs_true_hist(axarr[i,2], y_label_name_arr[i], 
                                    y_true_arr[i,:], y_pred_arr[i,j_fiducial,:], cmap, 
                                   text_results=text_results_arr[i], colorbar_fig=fig,
                                   weight=weight, weight_by_dex=weight_by_dex, colorbar_label=colorbar_label)

        show_legend = False
        if i==n_labels-1:
            show_legend = True
        plot_errors_vs_property(axarr[i,3], x_label_name, 
                            y_label_name_arr[i], 
                            x_property, 
                            y_true_arr[i], y_pred_arr[i],
                            feature_labels, feature_colors,
                            show_legend=show_legend, lws=lws,
                            x_bins=x_bins, y_lowerlim=y_lowerlim_arr[i])