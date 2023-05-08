import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from astropy.table import Table

import sys
sys.path.insert(1, '/home/ksf293/external')
import illustris_python as il

sys.path.insert(1, '../code')



label_dict = {'log_m200m_fof': r'log($M_\mathrm{200m,FoF}) \: [h^{-1} \, M_\odot]$)',
              'log_mstellar': r'log($m_\mathrm{*} \: [h^{-1} \, M_\odot]$)',
              'log_r200m': r'log($R_\mathrm{200m} \: [h^{-1} \, \mathrm{kpc}]$)',
              'v200m_fof': r'$V_\mathrm{200m,FoF}) \: [km s^{-1}]$',
              'log_rstellar': r'log($r_\mathrm{*} \: [h^{-1} \, \mathrm{kpc}]$)',
              'log_ssfr': r'log(sSFR $\: [\mathrm{yr}^{-1}]$)',
              'sfr': r'log(SFR $\: [M_\odot \, \mathrm{yr}^{-1}]$)',
              'log_ssfr1': r'log(sSFR$_\mathrm{1\,Gyr}$ $\: [\mathrm{yr}^{-1}]$)',
              'sfr1': r'log(SFR$_\mathrm{1\,Gyr}$ $\: [M_\odot \, \mathrm{yr}^{-1}]$)',
              'a_form': r'$a_\mathrm{form}$',
              'c200c': r'$log(c_\mathrm{200c})$',
              'M_acc': r'$M_\mathrm{acc,dyn}$',
              'm_vir': r'$M_\mathrm{vir} [h^{-1} \, M_\odot]$',
              'gband': r'$g$-band magnitude',
              'gband_minus_iband': r'$g-i$ color',
              'log_jstellar': r'log($j_*$ [km/s kpc])',
              'log_mbh': r'log($M_\cdot [h^{-1} \, M_\odot]$)',
              'log_mbh_per_mstellar': r'log($M_{\cdot}/m_*$)',
              'num_mergers': 'log(number of mergers)',
              }

lim_dict = {'log_m200m_fof': (10, 14),
            'log_mstellar': (7, 12),
            'log_ssfr1': (-15,-8),
            'log_rstellar': (-1,2),
            'gband': (-24, -13),
            'gband_minus_iband': (0.0, 1.5),
            'log_jstellar': (0.5, 4.5),
            'log_mbh': (4.5, 10.5),
            'log_mbh_per_mstellar': (-5.5, -1),
            'num_mergers': (0.5, 4.5),
            }


zero_dict = {'sfr': 1e-3, #msun/yr
             'sfr1': 1e-3, #msun/yr
             'mbh': 4e5 #msun/h, half of seed mass
             }


def get_gal_prop_names(tag='galprops'):
    if tag=='galprops':
        names = ['log_mstellar', 'log_ssfr1', 
                'log_rstellar', 'log_jstellar', 
                'gband_minus_iband', 'log_mbh_per_mstellar']
    return names

def get_label(label_name):
    if label_name in label_dict:
        return label_dict[label_name]
    return label_name


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
    gal_properties = ['m_stellar', 'r_stellar', 'sfr', 'sfr1', 'bhmass']
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

    stdev_dict_bhmass = {'TNG50-4': np.array([0.26, 0.13, 0.12, 0.11, 0.08, 0.07, 0.07]), 
                      'TNG100-1': np.array([0.22, 0.11, 0.13, 0.18, 0.12, 0.06, 0.06]),
                    }

    gal_property_to_stdev_dict = {'m_stellar': stdev_dict_mstellar,
                                  'r_stellar': stdev_dict_rstellar,
                                  'sfr': stdev_dict_sfr, 
                                  'sfr1': stdev_dict_sfr, 
                                  'bhmass': stdev_dict_bhmass
                                  }

    stdev_dict = gal_property_to_stdev_dict[gal_property]
    idxs_mbins = np.digitize(log_m_stellar, logmstellar_bins)
    uncertainties_genel2019 = stdev_dict[sim_name][idxs_mbins-1]

    return uncertainties_genel2019


def get_uncertainties_poisson_sfr(sfr, sfr_label):
    # compute shot noise uncertainty
    N_over_zero = sfr/zero_dict[sfr_label]
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
        n_outliers = len(frac_err[frac_err > 5*msfe_test])
        # TODO: finish implementing binned errors
        return msfe_test, n_outliers

    elif test_error_type=='percentile':
        delta_y = y_pred - y_true
        percentile_16 = np.percentile(delta_y, 16, axis=0)
        percentile_84 = np.percentile(delta_y, 84, axis=0)
        error_inner68_test = (percentile_84-percentile_16)/2

        n_outliers = len(delta_y[delta_y > 5*error_inner68_test])
        return error_inner68_test, n_outliers

    elif test_error_type=='percentile_frac':
        frac_y = (y_pred - y_true)/y_true
        percentile_16 = np.percentile(frac_y, 16, axis=0)
        percentile_84 = np.percentile(frac_y, 84, axis=0)
        error_inner68_test = (percentile_84-percentile_16)/2

        n_outliers = len(frac_y[frac_y > 5*error_inner68_test])
        return error_inner68_test, n_outliers

    elif test_error_type=='stdev':
        delta_y = y_pred - y_true
        stdev = np.std(delta_y, axis=0)

        n_outliers = len(delta_y[delta_y > 5*stdev])
        return stdev, n_outliers

    else:
        print(f"ERROR: {test_error_type} not recognized")
        return
        

# def get_mrv_for_rescaling(sim_reader, mrv_names):
#     mrv_for_rescaling = []
#     for mrv_name in mrv_names:
#         if mrv_name is None:
#             mrv_for_rescaling.append(np.ones(len(sim_reader.dark_halo_arr)))
#         else:
#             sim_reader.add_catalog_property_to_halos(mrv_name)
#             mrv_for_rescaling.append( [halo.catalog_properties[mrv_name] for halo in sim_reader.dark_halo_arr] )
#     return np.array(mrv_for_rescaling)



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


label_to_target_name = {'r_stellar': 'radius_hydro_subhalo_star',
                        'm_stellar': 'mass_hydro_subhalo_star',
                        'ssfr1': 'sfr_hydro_subhalo_1Gyr'}

def get_y_vals(y_label_name, sim_reader, mass_multiplier=1e10, halo_tag=''):
    if y_label_name.startswith('a_mfrac'):
        assert halo_tag is not None, "Must pass halo_tag to get a_mfrac!"
    
    y_target_name = label_to_target_name[y_label_name] if y_label_name in label_to_target_name else y_label_name
    sim_reader.add_catalog_property_to_halos(y_target_name, halo_tag=halo_tag)
    y_vals = np.array([halo.catalog_properties[y_target_name] for halo in sim_reader.dark_halo_arr])

    if y_label_name=='m_stellar' or y_label_name=='r_stellar' \
        or y_label_name=='num_mergers' or y_label_name=='j_stellar':
        return np.log10(y_vals)

    elif y_label_name=='ssfr1':
        sim_reader.add_catalog_property_to_halos('mass_hydro_subhalo_star')
        m_stellar = np.array([halo.catalog_properties['mass_hydro_subhalo_star'] for halo in sim_reader.dark_halo_arr])
        sfr = y_vals
        idx_zerosfr = np.where(sfr==0)[0]
        sfr[idx_zerosfr] = sfr_zero
        log_sfr = np.log10(sfr)
        log_ssfr = log_sfr_to_log_ssfr(log_sfr, m_stellar, mass_multiplier=mass_multiplier)
        return log_ssfr
    
    elif y_label_name=='bhmass':
        bhmass = y_vals
        tol = 1e-10
        i_zerobhms = abs(bhmass) < tol
        bh_zero = 8e-6 #?? min in training set is 8e-5
        bhmass[i_zerobhms] = bh_zero
        return np.log10(bhmass)

    elif y_label_name=='bhmass_per_mstellar':
        bhmass_per_mstellar = y_vals
        bhmass_per_mstellar = bhmass_per_mstellar.astype(float)
        tol = 1e-10
        i_zerobhms = abs(bhmass_per_mstellar) < tol
        val_zerobhms = np.min(bhmass_per_mstellar[~i_zerobhms])/10.0  # set zero value to 1/10 of nonzero-min 
        bhmass_per_mstellar[i_zerobhms] = val_zerobhms
        return np.log10(bhmass_per_mstellar)

    else:
        return y_vals


def get_y_uncertainties(y_label_name, sim_reader=None, y_vals=None, log_mass_shift=10,
                        idx_train=None):
    
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
            
    elif y_label_name=='m_stellar' or y_label_name=='r_stellar' or y_label_name=='bhmass':
        assert sim_reader is not None, "Must pass sim_reader!"
        sim_reader.add_catalog_property_to_halos('mass_hydro_subhalo_star')
        m_stellar = np.array([halo.catalog_properties['mass_hydro_subhalo_star'] for halo in sim_reader.dark_halo_arr])
        y_uncertainties = get_uncertainties_genel2019(y_label_name, np.log10(m_stellar)+log_mass_shift,
                                                              sim_name=sim_reader.sim_name)
        # the fact that some ys are in units of 10^10 doesnt matter bc its just a constant
        return y_uncertainties            

    elif y_label_name=='bhmass_per_mstellar':
        assert sim_reader is not None, "Must pass sim_reader!"
        sim_reader.add_catalog_property_to_halos('mass_hydro_subhalo_star')
        m_stellar = np.array([halo.catalog_properties['mass_hydro_subhalo_star'] for halo in sim_reader.dark_halo_arr])
        y_uncertainties_bhmass = get_uncertainties_genel2019('bhmass', np.log10(m_stellar)+log_mass_shift,
                                                              sim_name=sim_reader.sim_name)
        y_uncertainties_mstellar = get_uncertainties_genel2019('m_stellar', np.log10(m_stellar)+log_mass_shift,
                                                              sim_name=sim_reader.sim_name)     

        # propogation of error - use subtraction formula bc working in logspace
        # i think this is ok because both bhmass and mstellar in 10^10 units...
        y_uncertainties = np.sqrt((y_uncertainties_bhmass)**2 + (y_uncertainties_mstellar)**2)

                                                                                                   
        return y_uncertainties


    elif y_label_name.startswith('a_mfrac') or y_label_name=='Mofa':
        assert idx_train is not None, "Must pass idx_train to get uncertainty for Mofa or a_mfrac!"
        #y_uncertainties = [0.02]*len(y_vals)

        y_train = y_vals[idx_train]
        y_train_mean = np.mean(y_train, axis=0)
        sample_var = y_train - y_train_mean

        sample_p16 = np.percentile(sample_var, 16, axis=0)
        sample_p84 = np.percentile(sample_var, 84, axis=0)
        y_uncertainties_persample = 0.5*(sample_p84 - sample_p16)

        print(y_uncertainties_persample)
        # same for each sample, just based on aval
        y_uncertainties = np.tile(y_uncertainties_persample, (len(y_vals), 1))
        print(y_uncertainties.shape)
        return y_uncertainties

    else:
        # this will just not do anything, bc using them as delta_y/sigma_2, or sample_weight
        #return np.ones(len(y_vals)) #TODO what should this be??                                 
        return y_vals*0.05 #TODO what should this be??                                 


def write_uncertainties_table(fn_halos, fn_unc):

    print("Loading halo table")
    tab_halos = load_table(fn_halos)
    print("Loaded")

    tab_unc = Table()

    x_label_name = 'log_m200m_fof'
    x_property = tab_halos[x_label_name]
    x_bins = np.arange(10.25, 13.7501, 0.5)
    n_bins = len(x_bins)-1
    i_bins = np.digitize(x_property, x_bins)
    i_bins -= 1
    # if outside of bounds, just take closest
    # subtracting 1 will auto do this for largest bin i think
    i_bins[i_bins==-1] = 0
    i_bins[i_bins==n_bins] = n_bins-1

    label_tag = 'galprops'
    y_label_names = get_gal_prop_names(label_tag)
    for y_label_name in y_label_names:
        _, stdevs_binned = get_butterfly_error(x_bins, y_label_name)
        tab_unc[y_label_name] = stdevs_binned[i_bins]
    
    # add poisson noise for ssfr
    # if 'log_ssfr1' in y_label_names:
    #     unc_poisson_ssfr = get_uncertainties_poisson_sfr(10**tab_halos['log_sfr1'], 'ssfr1')
    #     unc_sfr = np.sqrt(unc_poisson_sfr**2 + tab_unc['sfr1']**2)
    #     unc_ssfr = uncertainty_log_sfr_to_uncertainty_log_ssfr(unc_sfr) 

    tab_unc['idx_halo_dark'] = tab_halos['idx_halo_dark']
    print(tab_unc.columns)

    print(f"Wrote uncertainty table to {fn_unc}")
    tab_unc.write(fn_unc, overwrite=True)





def save_mah(halos, fn_mah):
    mah_dict = {}
    for halo in halos:
        mah_dict[halo.idx_halo_dark] = halo.catalog_properties['MAH']
    np.save(fn_mah, mah_dict)
    

def load_mah(halos, fn_mah):
    mah_dict = np.load(fn_mah, allow_pickle=True).item()
    for halo in halos:
        halo.set_catalog_property('MAH', mah_dict[halo.idx_halo_dark])

# not very robust, should do better
def save_merger_info(halos, fn_merger, properties=['num_mergers', 'num_major_mergers', 'ratio_last_major_merger']):
    merger_dict = {}
    for prop in properties:
        vals = np.array([halo.catalog_properties[prop] for halo in halos])
        merger_dict[prop] = vals
    np.save(fn_merger, merger_dict)
    

def load_merger_info(halos, fn_merger, properties=['num_mergers', 'num_major_mergers', 'ratio_last_major_merger']):
    merger_dict = np.load(fn_merger, allow_pickle=True).item()
    for prop in properties:
        vals = merger_dict[prop]
        for i, halo in enumerate(halos):
            halo.set_catalog_property(prop, vals[i])



# assumes y in reverse sorted order!! 
def y_interpolated(x, y, x_val):
    assert y[0]>y[1], "y def not in reverse sorted order!"
    x = np.array(x)
    y = np.array(y)
    idx_below = np.where(x<=x_val)[0]
    if len(idx_below)==0:
        return np.nan
    i_of_below = (np.abs(x[idx_below] - x_val)).argmin()
    i0 = idx_below[i_of_below]
    i1 = i0 - 1 #because reverse sorted order
    x0, x1 = x[i0], x[i1]
    y0, y1 = y[i0], y[i1]
    y_val_interp = (y0*(x1-x_val) + y1*(x_val-x0))/(x1-x0)   
    return y_val_interp


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_mfrac_vals(n):
    vals = np.linspace(0, 1, int(n+2))
    # don't include 0 or 1! maybe messing up nn
    return vals[1:-1]
    # should include 1 or no? now does
    #return np.linspace(0, 1, int(n)) # should include 1 or no? now does
    #return np.logspace(-2, 0, int(n)) # should include 1 or no? now does


def geo_feature_arr_to_values(geo_feature_arr):
    geo_features = []
    for geo_arr in geo_feature_arr:
        geo_features_halo = []
        for g in geo_arr:
            if type(g.value)==np.ndarray or type(g.value)==list:
                geo_features_halo.extend(list(g.value.flatten()))
            else:
                geo_features_halo.append(g.value)
        geo_features.append(geo_features_halo)
    geo_features = np.array(geo_features)
    return geo_features


# just get the a values of all the snapshots,
# by grabbing the first halo that has all
def get_avals(dark_halo_arr):
    n_snapshots = 100
    for halo in dark_halo_arr:
        a_vals = halo.catalog_properties['MAH'][0]
        if len(a_vals)==n_snapshots:
            return a_vals


def last_merger_ratio(tree, minMassRatio=1e-10, massPartType='dm', index=0):
    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    numMergers   = 0
    invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]

    while fpID != -1:
        fpIndex = index + (fpID - rootID)
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)

        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]
        while npID != -1:
            npIndex = index + (npID - rootID)
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)
            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass
                if ratio >= minMassRatio and ratio <= invMassRatio:
                    # for consistency
                    if ratio <= 1:
                        return ratio
                    else:
                        return 1/ratio

            npID = tree['NextProgenitorID'][npIndex]

        fpID = tree['FirstProgenitorID'][fpIndex]

    # if get here, means no major mergers above mass ratio found
    return 0


def last_merger_ratio_mpb(tree, minMassRatio=1e-10, massPartType='dm', index=0):
    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    numMergers   = 0
    invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]
    count = 0
    while fpID != -1:
        fpIndex = index + (fpID - rootID)
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)
        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        if npID != -1:
            npIndex = index + (npID - rootID)
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)

            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass
                if ratio >= minMassRatio and ratio <= invMassRatio:
                    #print(ratio)
                    if ratio <= 1:
                        return ratio
                    else:
                        return 1/ratio

        fpID = tree['FirstProgenitorID'][fpIndex]
        count += 1
    # if get here, means no major mergers above mass ratio found
    return -1

def num_mergers_mpb(tree, minMassRatio=1e-10, massPartType='dm', index=0):
    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    numMergers   = 0
    invMassRatio = 1.0 / minMassRatio
    count = 0
    
    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]
    print('main leaf:', tree['MainLeafProgenitorID'][index])
    
    while fpID != -1:
        fpIndex = index + (fpID - rootID)
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)
        print("count", count, "fpMass:", fpMass, fpID)

        count += 1
        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        if npID != -1:
            npIndex = index + (npID - rootID)
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)

            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass
                if ratio >= minMassRatio and ratio <= invMassRatio:
                    numMergers += 1

        fpID = tree['FirstProgenitorID'][fpIndex]

    return numMergers


def get_mrv_for_rescaling(tab_halos, fn_geo_clean_config,
                          use_logs=True):
    # Get MRV rescaling
    # Need halo table and MRV rescaling names for x_extra
    with open(fn_geo_clean_config, 'r') as file:
        geo_clean_params = yaml.safe_load(file)
    gcp = geo_clean_params['geo_clean']

    mrv = []
    for name in gcp['mrv_names_for_rescaling']:
        if use_logs and 'log_'+name in tab_halos.columns:
            name = 'log_'+name
        print(name)
        mrv.append(tab_halos[name])   
    return np.array(mrv).T


def load_features(feature_mode, tab_halos,
                  tab_select,
                  fn_geo_clean_config=None, fn_scalar_config=None,
                  ):

    assert feature_mode in ['scalars', 'geos', 'catalogz0', 'mrv'], f"Feature mode {feature_mode} not recognized!"

    # with open(fn_select_config, 'r') as file:
    #     select_params = yaml.safe_load(file)

    # fn_halo_config = select_params['halo']['fn_halo_config']
    # with open(fn_halo_config, 'r') as file:
    #     halo_params = yaml.safe_load(file)

    # tab_halos = load_table(halo_params['halo']['fn_halos'])
    # tab_select = load_table(select_params['select']['fn_select'])
    idxs_table = np.array(tab_select['idx_table'])

    if feature_mode=='scalars':

        assert fn_scalar_config is not None, "Must pass fn_scalar_config!"

        with open(fn_scalar_config, 'r') as file:
            scalar_params = yaml.safe_load(file)
        scp = scalar_params['scalar']

        fn_scalar_features = scp['fn_scalar_features']
        tab_scalars = load_table(fn_scalar_features)
        # remove index column, not a feature!
        tab_scalars.remove_column('idx_halo_dark')

        #print(np.array(tab_scalars))
        #print(tab_scalars)
        # as_array converts to structured array
        # view converts to regular numpy array
        # https://stackoverflow.com/a/10171321
        x = tab_scalars.as_array()
        x = x.view((float, len(x.dtype.names)))

        fn_halo_config = scalar_params['halo']['fn_halo_config']
        fn_geo_clean_config = scalar_params['geo_clean']['fn_geo_clean_config']
        x_extra = get_mrv_for_rescaling(tab_halos, fn_geo_clean_config,
                                        use_logs=True)
        
    if feature_mode=='geos':


        assert fn_geo_clean_config is not None, "Must pass fn_geo_clean_config!"

        with open(fn_geo_clean_config, 'r') as file:
            geo_clean_params = yaml.safe_load(file)
        gcp = geo_clean_params['geo_clean']

        fn_geo_clean_features = gcp['fn_geo_clean_features']
        tab_geos = load_table(fn_geo_clean_features)
        # remove index column, not a feature!
        tab_geos.remove_column('idx_halo_dark')

        # need to flatten geo features into list of components,
        # bc many are vectors or tensors
        # via https://stackoverflow.com/a/2158522
        import collections
        def _flatten(vals):
            if isinstance(vals, collections.abc.Iterable):
                return [a for i in vals for a in _flatten(i)]
            else:
                return [vals]

        x = np.array([np.array(_flatten(tab_geos[i])) for i in range(len(tab_geos))])
        fn_halo_config = geo_clean_params['halo']['fn_halo_config']
        x_extra = get_mrv_for_rescaling(tab_halos, fn_geo_clean_config,
                                        use_logs=True)
        
    elif feature_mode=='mrv':
        feature_names = ['log_m200m_fof', 'log_r200m', 'v200m_fof']
        x = np.array([tab_halos[name] for name in feature_names]).T
        x_extra = None

    elif feature_mode=='catalogz0':
        feature_names = ['c200c', 'veldisp_subhalo', 'spin_subhalo']
        x = np.array([tab_halos[name] for name in feature_names]).T

        extra_feature_names = ['log_m200m_fof', 'log_r200m', 'v200m_fof']
        x_extra = np.array([tab_halos[name] for name in feature_names]).T
        print("TODO: check shape - should be transposing extra?")

    x = np.array(x)
    if x_extra is None:
        return x[idxs_table], None
    x_extra = np.array(x_extra)

    return x[idxs_table], x_extra[idxs_table]


def load_labels(label_names, tab_halos,
                tab_select, fn_amfrac=None):
    idxs_table = tab_select['idx_table']
    if type(label_names)==str:
        label_names = [label_names]
    labels = []
    for ln in label_names:
        if ln=='amfracs':
            assert fn_amfrac is not None, "Must pass fn_amfrac!"
            labels.extend( load_amfracs(fn_amfrac) )
        else:
            labels.append( tab_halos[ln] )
        #labels.append(label)

    #labels = [tab_halos[ln] for ln in label_names]
    labels = np.array(labels).T
    print(labels.shape)
    return labels[idxs_table]


def load_amfracs(fn_amfrac):
    # the selecting is done in load_labels
    tab_amfrac = load_table(fn_amfrac)
    tab_amfrac.remove_column('idx_halo_dark')

    amfracs = tab_amfrac.as_array()
    amfracs = amfracs.view((float, len(amfracs.dtype.names)))
    print(amfracs.shape)
    return amfracs.T

def load_uncertainties(label_names, fn_unc, 
                       tab_select):
    idxs_table = tab_select['idx_table']
    tab_unc = load_table(fn_unc)
    uncs = []
    for label_name in label_names:
        uncs.append(tab_unc[label_name])
    uncs = np.array(uncs).T
    return uncs[idxs_table]







def get_butterfly_error(x_bins, y_label_name, halo_logmass_min=None, x_label_name='log_m200m'):
    arr_shadow1 = np.loadtxt('../data/butterfly_L25n512TNGs35_shadow1.csv', skiprows=1, 
                                    delimiter=',')
    arr_shadow2 = np.loadtxt('../data/butterfly_L25n512TNGs35_shadow2.csv', skiprows=1, 
                                    delimiter=',')                            

    assert arr_shadow1.shape==arr_shadow2.shape, "Shadow files should be same shape!"

    # TODO fix zero vals here!!! by eye rn
    val_zero = None
    property_divide_by = None
    if y_label_name=='log_ssfr':
        y_label_name = 'log_SFR'
        property_divide_by = 'log_mstellar'
        val_zero = np.log10(zero_dict['sfr']) # for SFR, NOT sSFR!
    elif y_label_name=='log_ssfr1':
        y_label_name = 'log_SFR1'
        property_divide_by = 'log_mstellar'
        val_zero = np.log10(zero_dict['sfr1'])
    elif y_label_name=='log_mbh_per_mstellar':
        y_label_name = 'log_mbh'
        property_divide_by = 'log_mstellar'
        val_zero = np.log10(zero_dict['mbh']) # this is for log_mbh not per mstellar! and in real not code units

    # columns: index,log10(Group_M_Mean200/Msun),log10(GroupMass/Msun),log10(SubhaloMass/Msun),log10(SubhaloMassType(5)/Msun),log10(SubhaloHalfmassRadType(5)/kpc),log10(SFR1Gyr/(Msun/yr)),g-i[mag],log10(SubhaloBHMass/Msun),log10(SubhaloSFR/(Msun/yr)),log10(SubhaloMassType(1)/Msun)),log10(j_stellar/(kpc*km/s)))
    col_names = ['index','log_m200m', 'log_mtot', 'log_msubhalo', 
                 'log_mstellar', 'log_rstellar', 'log_SFR1', 'gband_minus_iband', 'log_mbh', 'log_SFR', 'log_mgas', 'log_jstellar']
    # these are given with no h but we want in h^-1 units!
    # all other units should be same as mine
    names_convert_to_perh = ['log_m200m', 'log_mtot', 'log_msubhalo', 
                 'log_mstellar', 'log_rstellar', 'log_mbh', 'log_mgas']

    def _log_to_perh(log_val):
        h = 0.704 # from Genel+2019 paper
        return log_val + np.log10(h)

    # convert to perh! 
    for i in range(arr_shadow1.shape[1]):
        if col_names[i] in names_convert_to_perh:
            arr_shadow1[:,i] = _log_to_perh(arr_shadow1[:,i])
            arr_shadow2[:,i] = _log_to_perh(arr_shadow2[:,i])

    # Limit to shadow sources in our mass range
    if halo_logmass_min is not None:
        m200_1 = arr_shadow1[:,col_names.index('log_m200m')]
        m200_2 = arr_shadow2[:,col_names.index('log_m200m')]
        # let's say mean for now
        m200_mean = np.log10(0.5*(10**m200_1 + 10**m200_2))
        i_masscut = (m200_mean >= halo_logmass_min)
        arr_shadow1 = arr_shadow1[i_masscut]
        arr_shadow2 = arr_shadow2[i_masscut]

    # compute pairwise diffs
    i_x = col_names.index(x_label_name)
    x1 = arr_shadow1[:,i_x]
    x2 = arr_shadow2[:,i_x]
    x_mean = np.log10(0.5*(10**x1 + 10**x2))
    # x_bins = np.linspace(np.min(x_mean), np.max(x_mean) + 0.01, n_bins)

    if y_label_name not in col_names:
        print(f"Label {y_label_name} not in shadow data!")
        return x_bins, np.full(n_bins-1, np.nan)

    i_y = col_names.index(y_label_name)
    y1 = arr_shadow1[:,i_y]
    y2 = arr_shadow2[:,i_y]
    # some values are -inf (in log, so zero.) 
    # TODO figure out better vals to choose (same as do for general y!)
    if val_zero is not None:
        y1[np.isinf(y1)] = val_zero
        y2[np.isinf(y2)] = val_zero

    if property_divide_by is not None:
        # subtracting bc these are all log
        y1 -= arr_shadow1[:,col_names.index(property_divide_by)]
        y2 -= arr_shadow2[:,col_names.index(property_divide_by)]
    
    if np.sum(np.isnan(y2))>0:
        mstar = arr_shadow2[:,col_names.index('log_mstellar')]

    # bin the results
    stdevs_binned = []
    for i in range(len(x_bins)-1):
        i_inbin = (x_mean >= x_bins[i]) & (x_mean < x_bins[i+1])
        y1_inbin, y2_inbin = y1[i_inbin], y2[i_inbin]
        # we just want the raw diff (no abs val or anything) bc will be centered on zero,
        # if gaussian this would be width of gaussian
        stdev_binned = np.std(y1_inbin - y2_inbin) / np.sqrt(2)
        stdevs_binned.append(stdev_binned)

    stdevs_binned = np.array(stdevs_binned)
    return x_bins, stdevs_binned


def load_table(fn_table):
    return Table.read(fn_table)