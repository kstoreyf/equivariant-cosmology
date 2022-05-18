import numpy as np

from fit import LinearFitter
import utils


def main():

    # scalar parameters
    m_order_max = 2
    x_order_max = 4
    v_order_max = 4
    n_groups = [[0,1,2], [3,4,5,6,7], [8,9,10]]
    scalar_tag = f'_3bins_rescaled_mord{m_order_max}_xord{x_order_max}_vord{v_order_max}'
    geo_tag = '_xminPE_rall'

    # sim / halo info
    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    snap_num_str = '099' # z = 0
    # sim_name = 'TNG100-1'
    # sim_name_dark = 'TNG100-1-Dark'
    sim_name = 'TNG50-4'
    sim_name_dark = 'TNG50-4-Dark'
    halo_dir = f'../data/halos/halos_{sim_name}'
    halo_tag = '_nstarpartmin1'
    fn_dark_halo_arr = f'{halo_dir}/halos_{sim_name}{halo_tag}.npy'

    # Load in sim reader 
    sim_reader = SimulationReader(base_dir, sim_name, sim_name_dark, snap_num_str)
    sim_reader.load_dark_halo_arr(fn_dark_halo_arr)
    sim_reader.read_simulations()
    sim_reader.add_catalog_property_to_halos('mass_hydro_subhalo_star')
    sim_reader.add_catalog_property_to_halos('m200m')

    # get halo properties
    m_stellar = np.array([dark_halo.catalog_properties['mass_hydro_subhalo_star'] for dark_halo in sim_reader.dark_halo_arr])
    m_200m = np.array([dark_halo.catalog_properties['m200m'] for dark_halo in sim_reader.dark_halo_arr])
    log_m_stellar = np.log10(m_stellar)
    log_m_200m = np.log10(m_200m)

    # geo info
    geo_dir = f'../data/geometric_features/geometric_features_{sim_name}'
    fn_geo_features = f'{geo_dir}/geometric_features{halo_tag}{geo_tag}.npy'

    # Setting up!
    geo_featurizer = GeometricFeaturizer()
    geo_featurizer.load_features(fn_geo_features)
    geo_feature_arr_rebinned = utils.rebin_geometric_features(geo_featurizer.geo_feature_arr, n_groups)

    # scalar info
    scalar_dir = f'../data/scalar_features/scalar_features_{sim_name}'
    fn_scalar_features = f'{scalar_dir}/scalar_features{halo_tag}{geo_tag}{scalar_tag}.npy'

    scalar_featurizer = ScalarFeaturizer(geo_feature_arr_rebinned)
    scalar_featurizer.compute_MXV_from_features()
    x_features_extra = np.vstack((scalar_featurizer.M_tot, 
                                scalar_featurizer.X_rms,
                                scalar_featurizer.V_rms)).T
    x_features_extra = np.log10(x_features_extra)   
    scalar_featurizer.load_features(fn_scalar_features)
    features_all = scalar_featurizer.scalar_features

    # Uncertainties, powerlaw
    uncertainties_genel2019 = utils.get_uncertainties_genel2019(log_m_stellar+log_mass_shift, sim_name=sim_name)
    y_val_current_powerlaw_fit, initial_guess = utils.fit_broken_power_law(log_m_200m, log_m_stellar, 
                                                       uncertainties=uncertainties_genel2019)

    # run and save!
    results = get_importance_order_addonein(features_all, log_m_stellar, y_val_current,
                                uncertainties=uncertainties, x_features_extra=x_features_extra)
    # save info
    feature_imp_dir = f'../data/feature_importance/feature_importance_{sim_name}'
    Path(feature_imp).mkdir(parents=True, exist_ok=True)
    feature_tag = ''
    fn_feature_imp = f'{feature_imp}/idxs_ordered_best{halo_tag}{geo_tag}{scalar_tag}{feature_tag}.npy'
    save_feature_importance(fn_feature_imp, results)


def get_importance_order_addonein(features_all, log_m_stellar, y_val_current, uncertainties=None,
                                  x_features_extra=None, top_n=None):

    N_feat = features_all.shape[1]
    idxs_feats_remaining = np.array(list(range(N_feat)))

    groups = []
    idxs_ordered_best = []
    errors_best = []
    chi2s_best = []

    if top_n is None:
        top_n = N_feat

    for n in range(top_n):
        print(f'#{n} most important')
        errors = np.full(N_feat, np.inf)
        chi2s = np.full(N_feat, np.inf)
        
        for idx in idxs_feats_remaining:

            idxs_to_use = np.append(np.array(idxs_ordered_best, dtype=int), idx)
            features = features_all[:,idxs_to_use]
            
            fitter = LinearFitter(features, log_m_stellar, 
                                y_val_current, uncertainties=uncertainties,
                                x_features_extra=x_features_extra)
            fitter.split_train_test()
            fitter.scale_and_fit(rms_x=True, log_x=False, log_y=False)
            fitter.predict_test()

            error, n_outliers = utils.compute_error(fitter, test_error_type='percentile')
            errors[idx] = error
            chi2s[idx] = fitter.chi2
        
        group = np.where((np.abs(errors - errors.min()) < 1e-8))[0]
        
        if len(group)>1:
            groups.append(group)
        idx_to_add = group[0]
        idxs_ordered_best.append( idx_to_add ) 
        errors_best.append(errors[idx_to_add])
        chi2s_best.append(chi2s[idx_to_add])

        idxs_feats_remaining = np.delete(idxs_feats_remaining, np.where(idxs_feats_remaining==idx_to_add))

    return idxs_ordered_best, errors_best, chi2s_best, groups



def save_feature_importance(self, fn_feature_imp, results):
    np.save(fn_feature_imp, results)



if __name__=='__main__':
    main()