import numpy as np
import time
from pathlib import Path

import utils
from fit import LinearFitter
from geometric_features import GeometricFeaturizer
from read_halos import SimulationReader
from scalar_features import ScalarFeaturizer


def main():

    # feature parameters
    top_n = None
    label_name = 'mstellar'
    #feature_tag = f'_top{top_n}_{label_name}'
    feature_tag = f'_{label_name}'

    # scalar parameters
    m_order_max = 2
    x_order_max = 4
    v_order_max = 4
    n_groups = [[0,1,2], [3,4,5,6,7], [8,9,10]]
    scalar_tag = f'_3bins_pseudo_rescaled_mord{m_order_max}_xord{x_order_max}_vord{v_order_max}'

    # geo parameters
    geo_tag = '_xminPEsub_rall'

    # sim / halo info
    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    snap_num_str = '099' # z = 0
    sim_name = 'TNG100-1'
    sim_name_dark = 'TNG100-1-Dark'
    # sim_name = 'TNG50-4'
    # sim_name_dark = 'TNG50-4-Dark'
    halo_dir = f'../data/halos/halos_{sim_name}'
    halo_tag = '_nstarpartmin1_twin'
    fn_dark_halo_arr = f'{halo_dir}/halos_{sim_name}{halo_tag}.npy'
    log_mass_shift = 10

    start = time.time()

    # Load in sim reader 
    sim_reader = SimulationReader(base_dir, sim_name, sim_name_dark, snap_num_str)
    sim_reader.load_dark_halo_arr(fn_dark_halo_arr)
    sim_reader.read_simulations()
    sim_reader.add_catalog_property_to_halos('mass_hydro_subhalo_star')
    sim_reader.add_catalog_property_to_halos('m200m')
    sim_reader.add_catalog_property_to_halos('r200m')
    sim_reader.add_catalog_property_to_halos('v200m')

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
    geo_feature_arr_pseudo = utils.transform_pseudotensors(geo_feature_arr_rebinned)

    # scalar info
    scalar_dir = f'../data/scalar_features/scalar_features_{sim_name}'
    fn_scalar_features = f'{scalar_dir}/scalar_features{halo_tag}{geo_tag}{scalar_tag}.npy'

    scalar_featurizer = ScalarFeaturizer(geo_feature_arr_pseudo)
    r_200m = np.array([dark_halo.catalog_properties['r200m'] for dark_halo in sim_reader.dark_halo_arr])
    v_200m = np.array([dark_halo.catalog_properties['v200m'] for dark_halo in sim_reader.dark_halo_arr])
    scalar_featurizer.rescale_geometric_features(m_200m, r_200m, v_200m)

    x_features_extra = np.vstack((m_200m, r_200m, v_200m)).T
    x_features_extra = np.log10(x_features_extra)   
    scalar_featurizer.load_features(fn_scalar_features)
    features_all = scalar_featurizer.scalar_features

    # Train test split
    frac_train = 0.70
    frac_test = 0.15
    #frac_train = 0.8
    #frac_test = 0.2
    random_ints = np.array([dark_halo.random_int for dark_halo in sim_reader.dark_halo_arr])
    idx_train, idx_val, idx_test = utils.split_train_val_test(random_ints, frac_train=frac_train, frac_test=frac_test)

    # Uncertainties, powerlaw
    uncertainties_genel2019 = utils.get_uncertainties_genel2019(log_m_stellar+log_mass_shift, sim_name=sim_name)
    y_val_current_powerlaw_fit_train, params_best_fit = utils.fit_broken_power_law(
                                                       log_m_200m[idx_train], log_m_stellar[idx_train], 
                                                       uncertainties=uncertainties_genel2019[idx_train])
    y_val_current_powerlaw_fit = utils.broken_power_law(log_m_200m, *params_best_fit)                                                

    # run and save!
    # pass validation set as test set for this
    results = get_importance_order_addonein(features_all, log_m_stellar, y_val_current_powerlaw_fit,
                                idx_train, idx_val, top_n=top_n,
                                uncertainties=uncertainties_genel2019, x_features_extra=x_features_extra)

    # save info
    feature_imp_dir = f'../data/feature_importance/feature_importance_{sim_name}'
    Path(feature_imp_dir).mkdir(parents=True, exist_ok=True)
    fn_feature_imp = f'{feature_imp_dir}/feature_importance{halo_tag}{geo_tag}{scalar_tag}{feature_tag}.npy'
    save_feature_importance(fn_feature_imp, results)
    print("Done and saved!")

    end = time.time()
    print("Time:", end-start, 'sec')

def get_importance_order_addonein(features_all, log_m_stellar, y_val_current, idx_train, idx_val, uncertainties=None,
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
            fitter.split_train_test(idx_train, idx_val)
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



def save_feature_importance(fn_feature_imp, results):
    np.save(fn_feature_imp, results)


def load_feature_importance(fn_feature_imp):
    return np.load(fn_feature_imp, allow_pickle=True)


if __name__=='__main__':
    main()
