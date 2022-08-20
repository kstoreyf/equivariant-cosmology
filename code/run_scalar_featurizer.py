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
    # sim_name = 'TNG50-4'
    # sim_name_dark = 'TNG50-4-Dark'
    halo_dir = f'../data/halos/halos_{sim_name}'
    halo_tag = '_nstarpartmin50_twin'
    fn_dark_halo_arr = f'{halo_dir}/halos_{sim_name}{halo_tag}.npy'

    # geo info
    geo_dir = f'../data/geometric_features/geometric_features_{sim_name}'
    geo_tag = '_xminPEsub_rall'
    fn_geo_features = f'{geo_dir}/geometric_features{halo_tag}{geo_tag}.npy'

    # save info
    scalar_dir = f'../data/scalar_features/scalar_features_{sim_name}'
    Path(scalar_dir).mkdir(parents=True, exist_ok=True)
    trace_str = '' if eigenvalues_not_trace else '_trace'
    scalar_tag = f'_3bins_pseudo_rescaled_mord{m_order_max}_xord{x_order_max}_vord{v_order_max}{trace_str}'
    fn_scalar_features = f'{scalar_dir}/scalar_features{halo_tag}{geo_tag}{scalar_tag}.npy'

    # Go!
    print("Running scalar featurizer")

    sim_reader = SimulationReader(base_dir, sim_name, sim_name_dark, snap_num_str)
    sim_reader.load_dark_halo_arr(fn_dark_halo_arr)
    sim_reader.read_simulations()
    sim_reader.add_catalog_property_to_halos('m200m')
    sim_reader.add_catalog_property_to_halos('r200m')
    sim_reader.add_catalog_property_to_halos('v200m')

    geo_featurizer = GeometricFeaturizer()
    geo_featurizer.load_features(fn_geo_features)

    geo_feature_arr_rebinned = utils.rebin_geometric_features(
                                     geo_featurizer.geo_feature_arr, n_groups_rebin)
    geo_feature_arr_rebinned_pseudo = utils.transform_pseudotensors(geo_feature_arr_rebinned)

    scalar_featurizer = ScalarFeaturizer(geo_feature_arr_rebinned_pseudo)
    #scalar_featurizer.compute_MXV_from_features()
    m_200m = np.array([dark_halo.catalog_properties['m200m'] for dark_halo in sim_reader.dark_halo_arr])
    r_200m = np.array([dark_halo.catalog_properties['r200m'] for dark_halo in sim_reader.dark_halo_arr])
    v_200m = np.array([dark_halo.catalog_properties['v200m'] for dark_halo in sim_reader.dark_halo_arr])

    scalar_featurizer.rescale_geometric_features(m_200m, r_200m, v_200m)
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
