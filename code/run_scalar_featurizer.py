import numpy as np
import time
from pathlib import Path

import utils
from read_halos import SimulationReader
from geometric_features import GeometricFeaturizer
from scalar_features import ScalarFeaturizer


def run():

    # scalar parameters
    m_order_max = 2
    x_order_max = 4
    v_order_max = 4
    n_groups_rebin = [[0,1,2], [3,4,5,6,7], [8,9,10]]
    eigenvalues_not_trace = True

    # sim / halo info
    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    snap_num_str = '099' # z = 0
    sim_name = 'TNG100-1'
    sim_name_dark = 'TNG100-1-Dark'
    #sim_name = 'TNG50-4'
    #sim_name_dark = 'TNG50-4-Dark'
    halo_dir = f'../data/halos/halos_{sim_name}'
    halo_tag = '_nstarpartmin1_twin'
    fn_dark_halo_arr = f'{halo_dir}/halos_{sim_name}{halo_tag}.npy'

    # geo info
    geo_dir = f'../data/geometric_features/geometric_features_{sim_name}'
    geo_tag = '_xminPE_rall'
    fn_geo_features = f'{geo_dir}/geometric_features{halo_tag}{geo_tag}.npy'

    # save info
    scalar_dir = f'../data/scalar_features/scalar_features_{sim_name}'
    Path(scalar_dir).mkdir(parents=True, exist_ok=True)
    trace_str = '' if eigenvalues_not_trace else '_trace'
    scalar_tag = f'_3bins_rescaled_mord{m_order_max}_xord{x_order_max}_vord{v_order_max}{trace_str}'
    fn_scalar_features = f'{scalar_dir}/scalar_features{halo_tag}{geo_tag}{scalar_tag}.npy'

    # Go!
    print("Running scalar featurizer")
    geo_featurizer = GeometricFeaturizer()
    geo_featurizer.load_features(fn_geo_features)

    geo_feature_arr_rebinned = utils.rebin_geometric_features(
                                     geo_featurizer.geo_feature_arr, n_groups_rebin)

    scalar_featurizer = ScalarFeaturizer(geo_feature_arr_rebinned)
    scalar_featurizer.compute_MXV_from_features()
    scalar_featurizer.rescale_geometric_features(scalar_featurizer.M_tot, 
                                                 scalar_featurizer.X_rms, 
                                                 scalar_featurizer.V_rms)
    start = time.time()
    scalar_featurizer.featurize(m_order_max, x_order_max=x_order_max, 
                                v_order_max=v_order_max,
                                eigenvalues_not_trace=eigenvalues_not_trace)
    print(scalar_featurizer.scalar_features.shape)
    end = time.time()
    print("Time:", end-start, 'sec')
    scalar_featurizer.save_features(fn_scalar_features)                  
    print(f'Saved scalar features to {fn_scalar_features}')



if __name__=='__main__':
    run()
