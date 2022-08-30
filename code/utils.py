import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from geometric_features import GeometricFeature, geo_name


label_dict = {'m_200m': r'log($M_\mathrm{halo} \: [h^{-1} \, M_\odot]$)',
              'm_stellar': r'log($m_\mathrm{stellar} \: [h^{-1} \, M_\odot]$)',
              'r_200m': r'log($R_\mathrm{halo} \: [h^{-1} \, \mathrm{kpc}]$)',
              'r_stellar': r'log($r_\mathrm{stellar} \: [h^{-1} \, \mathrm{kpc}]$)',
              'ssfr': r'log(sSFR $\: [\mathrm{yr}^{-1}]$)',
              'sfr': r'log(SFR $\: [M_\odot \, \mathrm{yr}^{-1}]$)',
              'ssfr1': r'log(sSFR$_\mathrm{1\,Gyr}$ $\: [\mathrm{yr}^{-1}]$)',
              'sfr1': r'log(SFR$_\mathrm{1\,Gyr}$ $\: [M_\odot \, \mathrm{yr}^{-1}]$)',
              'a_form': r'$a_\mathrm{form}$',
              'c_200c': r'$log(c_\mathrm{200c})$',
              'M_acc': r'$M_\mathrm{acc,dyn}$',
              'm_vir': r'$M_\mathrm{vir}$'
              }

sfr_zero = 1e-3

def get_alt_sim_name(sim_name):
    sim_name_dict = {'TNG100-1': 'L75n1820TNG',
                     'TNG100-1-Dark': 'L75n1820TNG_DM',
                     'TNG50-4': 'L35n270TNG',
                     'TNG50-4-Dark': 'L35n270TNG_DM'
                    }
    return sim_name_dict[sim_name]


def get_uncertainties_genel2019(gal_property, log_m_stellar, sim_name):
    # From Genel+2019, Figure 8
    # tng50-4: closest to epsilon=4 (blue)
    # tng100-1: closest to epsilon=1 (yellow)
    gal_properties = ['m_stellar', 'r_stellar', 'sfr', 'sfr1']
    assert gal_property in gal_properties, f'Property {gal_property} not in gal_properties={gal_properties}'

    logmstellar_bins = np.linspace(8.5, 11, 6)
    logmstellar_bins = np.array([5] + list(logmstellar_bins) + [13])
    err_msg = 'Passed log_m_stellar has values outside of bin edges!'
    assert np.min(log_m_stellar) > np.min(logmstellar_bins) and np.max(log_m_stellar) < np.max(logmstellar_bins), err_msg

    # added estimates on either end (low: double, high: extend)
    stdev_dict_mstellar = {'TNG50-4': np.array([0.56, 0.28, 0.23, 0.12, 0.05, 0.04, 0.04]), 
                           'TNG100-1': np.array([0.16, 0.08, 0.06, 0.04, 0.03, 0.04, 0.04]), 
                          }
    stdev_dict_rstellar = {'TNG50-4': np.array([0.42, 0.21, 0.14, 0.09, 0.07, 0.06, 0.06]), 
                           'TNG100-1': np.array([0.14, 0.07, 0.08, 0.08, 0.09, 0.08, 0.08]),
                           }

    stdev_dict_sfr = {'TNG50-4': np.array([0.72, 0.36, 0.34, 0.28, 0.30, 0.45, 0.45]), 
                      'TNG100-1': np.array([0.44, 0.22, 0.21, 0.22, 0.34, 0.62, 0.62]),
                    }

    stdev_dict_sfr1 = {'TNG50-4': np.array([0.82, 0.41, 0.28, 0.25, 0.25, 0.4, 0.4]), 
                      'TNG100-1': np.array([0.34, 0.17, 0.15, 0.16, 0.25, 0.5, 0.5]),
                    }

    gal_property_to_stdev_dict = {'m_stellar': stdev_dict_mstellar,
                                  'r_stellar': stdev_dict_rstellar,
                                  'sfr': stdev_dict_sfr, 
                                  'sfr1': stdev_dict_sfr, 
                                  }

    stdev_dict = gal_property_to_stdev_dict[gal_property]
    idxs_mbins = np.digitize(log_m_stellar, logmstellar_bins)
    uncertainties_genel2019 = stdev_dict[sim_name][idxs_mbins-1]

    return uncertainties_genel2019


def get_uncertainties_poisson_sfr(sfr, sfr_zero):
    # compute shot noise uncertainty
    N_over_zero = sfr/sfr_zero
    uncertainties_poisson = 1/np.sqrt(N_over_zero)
    # this is in natural log space, so convert to log10 space
    # sig_ln = sig_x/x
    # sig_log10 = 1/ln(10) sig_x/x
    # divide: sig_ln/sig_log10 = ln(10)
    # so sig_log10 = sig_ln / ln(10)
    uncertainties_poisson_log10 = uncertainties_poisson / np.log(10)
    return uncertainties_poisson_log10


# maybe a better way to do this, but just logging for now to be consistent
def broken_power_law(log_m200, N=1, log_m1=12, beta=1, gamma=1):
    # note that the 12 is without a log_mass_shift; defaults with shifted sample (shift of 10) should be 12-10=2
    return log_m200 + np.log10( 2*N/((log_m200/log_m1)**(-beta) + (log_m200/log_m1)**gamma) )


def fit_broken_power_law(log_m200, log_m_stellar, uncertainties, initial_guess=None,
                         log_mass_shift=10, return_initial_guess=False):
    if initial_guess is None:
        m1 = 12-log_mass_shift
        params_initial_guess = [0.01, m1, 1.5, 0.4]
    bounds = [[0]*len(params_initial_guess), [np.inf]*len(params_initial_guess)]
    params_best_fit, _ = curve_fit(broken_power_law, log_m200, log_m_stellar, sigma=uncertainties, 
                        bounds=bounds, p0=params_initial_guess)
    y_val_current_powerlaw_fit = broken_power_law(log_m200, *params_best_fit)
    if return_initial_guess:
        return y_val_current_powerlaw_fit, params_best_fit, params_initial_guess
    return y_val_current_powerlaw_fit, params_best_fit


def power_law(log_r_200m, A, c):
    return log_r_200m * A + c

def power_law_fixedslope(log_r_200m, c):
    A = 1.4
    return log_r_200m * A + c

def fit_function(function, x, y, uncertainties, params_initial_guess):

    params_best_fit, _ = curve_fit(function, x, y, sigma=uncertainties, 
                                  p0=params_initial_guess)
    y_val_current_powerlaw_fit = function(x, *params_best_fit)

    return y_val_current_powerlaw_fit, params_best_fit


def shift_points_torus(points, shift, box_size):
    return (points - shift + 0.5*box_size) % box_size - 0.5*box_size


def compute_error(y_true, y_pred, test_error_type='percentile'):

    if test_error_type=='msfe':
        frac_err = (y_pred - y_true)/y_true
        msfe_test = np.mean(frac_err**2)
        error_str = f'MSFE: {msfe_test:.3f}'
        n_outliers = len(frac_err[frac_err > 5*msfe_test])
        # TODO: finish implementing binned errors
        return msfe_test, n_outliers

    elif test_error_type=='percentile':
        delta_y = y_pred - y_true
        percentile_16 = np.percentile(delta_y, 16, axis=0)
        percentile_84 = np.percentile(delta_y, 84, axis=0)
        error_inner68_test = (percentile_84-percentile_16)/2

        error_str = fr"$\sigma_{{68}}$: {error_inner68_test:.3f}"
        n_outliers = len(delta_y[delta_y > 5*error_inner68_test])
        return error_inner68_test, n_outliers

    else:
        print(f"ERROR: {test_error_type} not recognized")
        return
        

# n_groups should be lists of the "n" to include in each group
def rebin_geometric_features(geo_feature_arr, n_groups):
    print("Rebinning geometric features")
    # TODO: implement check that bins listed in n_groups matches bins in the geo_feature_arr
    n_vals = [g.n for g in geo_feature_arr[0]] # 0 because just check first halo, features should be same
    n_groups_flat = [n for group in n_groups for n in group]
    assert set(n_groups_flat).issubset(set(n_vals)), 'Groups passed in contain bins not in geometric features!'

    geo_feature_arr_rebinned = []
    number_of_groups = len(n_groups)
    count = 0
    for geo_features_halo in geo_feature_arr:
        count += 1
        # group geometric features into n groups
        geo_feats_grouped = [[] for _ in range(number_of_groups)]
        for geo_feat in geo_features_halo:
            for i_n, n_group in enumerate(n_groups):
                if geo_feat.n in n_group:
                    geo_feats_grouped[i_n].append(geo_feat)

        # sum over same features (matching orders) in each group
        geo_features_halo_rebinned = []
        # m order same for all geo features so don't need to worry bout it
        x_order_highest = np.max([g.x_order for g in geo_features_halo])
        v_order_highest = np.max([g.v_order for g in geo_features_halo])
        for i_newn, geo_feat_group in enumerate(geo_feats_grouped):
            # plus 1 because want to include that highest order!
            for x_order in range(x_order_highest+1):
                for v_order in range(v_order_highest+1):
                    geo_feats_this_order = [g for g in geo_feat_group if g.x_order==x_order and g.v_order==v_order]
                    # continue if there are no values at this order (e.g. none at x=2, v=1)
                    if not geo_feats_this_order:
                        continue
                    geo_rebinned_value = np.sum([g.value for g in geo_feats_this_order], axis=0)
                    hermitian = geo_feats_this_order[0].hermitian # if one is hermitian, all are at this order
                    geo_rebinned = GeometricFeature(geo_rebinned_value, m_order=1, x_order=x_order, v_order=v_order, 
                                                    n=i_newn, hermitian=hermitian)
                    geo_features_halo_rebinned.append(geo_rebinned)
        geo_feature_arr_rebinned.append(geo_features_halo_rebinned)

    return geo_feature_arr_rebinned


def get_mrv_for_rescaling(sim_reader, mrv_names):
    mrv_for_rescaling = []
    for mrv_name in mrv_names:
        if mrv_name is None:
            mrv_for_rescaling.append(np.ones(len(sim_reader.dark_halo_arr)))
        else:
            sim_reader.add_catalog_property_to_halos(mrv_name)
            mrv_for_rescaling.append( [halo.catalog_properties[mrv_name] for halo in sim_reader.dark_halo_arr] )
    return np.array(mrv_for_rescaling)


def rescale_geometric_features(geo_feature_arr, Ms, Rs, Vs):
    print("Rescaling geometric features")
    N_geo_arrs = len(geo_feature_arr)
    assert len(Ms)==N_geo_arrs, "Length of Ms doesn't match geo feature arr!"
    assert len(Rs)==N_geo_arrs, "Length of Rs doesn't match geo feature arr!"
    assert len(Vs)==N_geo_arrs, "Length of Vs doesn't match geo feature arr!"
    for i_g, geo_features_halo in enumerate(geo_feature_arr):
        for geo_feat in geo_features_halo:
            geo_feat.value /= Ms[i_g] # all geometric features have single m term
            for _ in range(geo_feat.x_order):
                geo_feat.value /= Rs[i_g]
            for _ in range(geo_feat.v_order):
                geo_feat.value /= Vs[i_g]
    return geo_feature_arr


def transform_pseudotensors(geo_feature_arr):
    print("Transforming pseudotensors")
    geo_feature_arr = list(geo_feature_arr)
    for i_halo, geo_features_halo in enumerate(geo_feature_arr):
        gs_to_insert = []
        idxs_to_insert = []
        for i_feat, g in enumerate(geo_features_halo):

            if not g.hermitian and g.modification is None:
                g_value_symm = 0.5*(g.value + g.value.T)
                g_value_antisymm =  0.5*(g.value - g.value.T)
                g_symm = GeometricFeature(g_value_symm, m_order=g.m_order, x_order=g.x_order, v_order=g.v_order, 
                                            n=g.n, hermitian=True, modification='symmetrized')
                g_antisymm = GeometricFeature(g_value_antisymm, m_order=g.m_order, x_order=g.x_order, v_order=g.v_order, 
                                                n=g.n, hermitian=False, modification='antisymmetrized')
                # replace original with symmetric                
                geo_feature_arr[i_halo][i_feat] = g_symm
                # keep antsymmetric to insert right after symmetric, later
                gs_to_insert.append(g_antisymm)
                idxs_to_insert.append(i_feat+1)
        
        # inserting all at end to not mess up looping
        # for now should only have one pseudotensor per halo (C^{xv}), but may not always be true
        for idxs_to_insert, g_to_insert in zip(idxs_to_insert, gs_to_insert):
            geo_feature_arr[i_halo] = np.insert(geo_feature_arr[i_halo], idxs_to_insert, g_to_insert)

    return np.array(geo_feature_arr)


def split_train_val_test(random_ints, frac_train=0.70, frac_val=0.15, frac_test=0.15):

    tol = 1e-6
    assert abs((frac_train+frac_val+frac_test) - 1.0) < tol, "Fractions must add to 1!" 
    N_halos = len(random_ints)
    int_train = int(frac_train*N_halos)
    int_test = int((1-frac_test)*N_halos)

    idx_train = np.where(random_ints < int_train)[0]
    idx_test = np.where(random_ints >= int_test)[0]
    idx_val = np.where((random_ints >= int_train) & (random_ints < int_test))[0]

    return idx_train, idx_val, idx_test


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# little h via https://www.tng-project.org/data/downloads/TNG100-1/
def log_sfr_to_log_ssfr(log_sfr_arr, m_stellar_arr, mass_multiplier=1e10):
    h = 0.6774  
    m_stellar_Msun_arr = (m_stellar_arr*mass_multiplier)/h
    return log_sfr_arr - np.log10(m_stellar_Msun_arr)


def log_ssfr_to_log_sfr(log_ssfr_arr, m_stellar_arr, mass_multiplier=1e10):
    h = 0.6774  
    m_stellar_Msun_arr = (m_stellar_arr*mass_multiplier)/h
    return log_ssfr_arr + np.log10(m_stellar_Msun_arr)


# because we're working in log errors, the multiplicative factor of M/h is a constant that 
# just drops out with the derivative; so the uncertainties are the same! i think ?!
def uncertainty_log_sfr_to_uncertainty_log_ssfr(uncertainty_log_sfr_arr):
    return uncertainty_log_sfr_arr


label_to_target_name = {'m_stellar': 'mass_hydro_subhalo_star',
                        'ssfr1': 'sfr_hydro_subhalo_1Gyr'}

def get_y_vals(y_label_name, sim_reader, mass_multiplier=1e10):
    y_target_name = label_to_target_name[y_label_name]
    sim_reader.add_catalog_property_to_halos(y_target_name)
    y_vals = np.array([halo.catalog_properties[y_target_name] for halo in sim_reader.dark_halo_arr])

    if y_label_name=='m_stellar':
        return np.log10(y_vals)

    if y_label_name=='ssfr1':
        sim_reader.add_catalog_property_to_halos('mass_hydro_subhalo_star')
        m_stellar = np.array([halo.catalog_properties['mass_hydro_subhalo_star'] for halo in sim_reader.dark_halo_arr])
        sfr = y_vals
        idx_zerosfr = np.where(sfr==0)[0]
        sfr[idx_zerosfr] = sfr_zero
        log_sfr = np.log10(sfr)
        log_ssfr = log_sfr_to_log_ssfr(log_sfr, m_stellar, mass_multiplier=mass_multiplier)
        return log_ssfr


def get_y_uncertainties(y_label_name, sim_reader=None, y_vals=None, log_mass_shift=10):
    
    ssfr_name_to_sfr_name = {'ssfr': 'sfr', 'ssfr1': 'sfr1'}
    if y_label_name=='ssfr' or y_label_name=='ssfr1':
        assert sim_reader is not None and y_vals is not None, "Must pass sim_reader and y_vals!"
        sfr_label_name_genel2019 = ssfr_name_to_sfr_name[y_label_name]
        sim_reader.add_catalog_property_to_halos('mass_hydro_subhalo_star')
        m_stellar = np.array([halo.catalog_properties['mass_hydro_subhalo_star'] for halo in sim_reader.dark_halo_arr])
        uncertainties_genel2019_sfr = get_uncertainties_genel2019(sfr_label_name_genel2019, np.log10(m_stellar)+log_mass_shift,
                                                              sim_name=sim_reader.sim_name)
        # y_vals is ssfr in this case
        sfr = 10**log_ssfr_to_log_sfr(y_vals, m_stellar)
        uncertainties_poisson_sfr = get_uncertainties_poisson_sfr(sfr, sfr_zero)
        uncertainties_genel2019_poisson_sfr = np.sqrt(uncertainties_genel2019_sfr**2 + uncertainties_poisson_sfr**2)
        uncertainties_genel2019_poisson_ssfr = uncertainty_log_sfr_to_uncertainty_log_ssfr(uncertainties_genel2019_poisson_sfr) 
        return uncertainties_genel2019_poisson_ssfr
            
    elif y_label_name=='m_stellar':
        assert sim_reader is not None, "Must pass sim_reader!"
        sim_reader.add_catalog_property_to_halos('mass_hydro_subhalo_star')
        m_stellar = np.array([halo.catalog_properties['mass_hydro_subhalo_star'] for halo in sim_reader.dark_halo_arr])
        y_uncertainties = get_uncertainties_genel2019(y_label_name, np.log10(m_stellar)+log_mass_shift,
                                                              sim_name=sim_reader.sim_name)
        return y_uncertainties                                             