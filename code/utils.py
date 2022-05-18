import numpy as np
from scipy.optimize import curve_fit

from geometric_features import GeometricFeature



def get_alt_sim_name(sim_name):
    sim_name_dict = {'TNG100-1': 'L75n1820TNG',
                     'TNG100-1-Dark': 'L75n1820TNG_DM',
                     'TNG50-4': 'L35n270TNG',
                     'TNG50-4-Dark': 'L35n270TNG_DM'
                    }
    return sim_name_dict[sim_name]


def get_uncertainties_genel2019(log_m_stellar, sim_name):
    logmstellar_bins = np.linspace(8.5, 11, 6)
    logmstellar_bins = np.array([5] + list(logmstellar_bins) + [13])
    err_msg = 'Passed log_m_stellar has values outside of bin edges!'
    assert np.min(log_m_stellar) > np.min(logmstellar_bins) and np.max(log_m_stellar) < np.max(logmstellar_bins), err_msg
    # added estimates on either end (low: double, high: extend)
    stdev_dict = {'TNG50-4': np.array([0.56, 0.28, 0.23, 0.12, 0.05, 0.04, 0.04]), # epsilon=4, similar to tng50-4
                'TNG100-1': np.array([0.16, 0.08, 0.06, 0.04, 0.03, 0.04, 0.04]), # epsilon=1, similar to tng100-1
                }

    idxs_mbins = np.digitize(log_m_stellar, logmstellar_bins)
    uncertainties_genel2019 = stdev_dict[sim_name][idxs_mbins-1]

    return uncertainties_genel2019

# maybe a better way to do this, but just logging for now to be consistent
def broken_power_law(log_m200, N=1, log_m1=12, beta=1, gamma=1):
    # note that the 12 is without a log_mass_shift; defaults with shifted sample (shift of 10) should be 12-10=2
    return log_m200 + np.log10( 2*N/((log_m200/log_m1)**(-beta) + (log_m200/log_m1)**gamma) )


def fit_broken_power_law(log_m200, log_m_stellar, uncertainties, initial_guess=None,
                         log_mass_shift=10, return_initial_guess=True):
    if initial_guess is None:
        m1 = 12-log_mass_shift
        initial_guess = [0.01, m1, 1.5, 0.4]
    bounds = [[0]*len(initial_guess), [np.inf]*len(initial_guess)]
    popt, _ = curve_fit(broken_power_law, log_m200, log_m_stellar, sigma=uncertainties, 
                        bounds=bounds, p0=initial_guess)
    y_val_current_powerlaw_fit = broken_power_law(log_m200, *popt)
    if return_initial_guess:
        return y_val_current_powerlaw_fit, initial_guess
    return y_val_current_powerlaw_fit


def shift_points_torus(points, shift, box_size):
    return (points - shift + 0.5*box_size) % box_size - 0.5*box_size


def compute_error(fitter, test_error_type='percentile'):
    y_true = fitter.y_scalar_test
    y_pred = fitter.y_scalar_pred

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