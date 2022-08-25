import numpy as np
import os
import time
import yaml
from pathlib import Path

import utils
from read_halos import SimulationReader
from geometric_features import GeometricFeaturizer
from scalar_features import ScalarFeaturizer


def run():

    #sim_name = 'TNG100-1'
    sim_name = 'TNG50-4'
    halo_tag = ''
    geo_tag = ''
    scalar_tag = ''
    fn_scalar_config = f'../configs/scalar_{sim_name}{halo_tag}{geo_tag}{scalar_tag}.yaml'

    with open(fn_scalar_config, 'r') as file:
        scalar_params = yaml.safe_load(file)
    scp = scalar_params['scalar']

    fn_geo_config = scalar_params['geo']['fn_geo_config']
    with open(fn_geo_config, 'r') as file:
        geo_params = yaml.safe_load(file)
    gp = geo_params['geo']

    fn_halo_config = geo_params['halo']['fn_halo_config']
    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    sp = halo_params['sim']

    fn_scalar_features = scp['fn_scalar_features']
    Path(os.path.dirname(fn_scalar_features)).mkdir(parents=True, exist_ok=True)

    # Go!
    print("Running scalar featurizer")

    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], 
                                  sp['sim_name_dark'], sp['snap_num_str'])
    sim_reader.read_simulations()
    sim_reader.load_dark_halo_arr(halo_params['halo']['fn_dark_halo_arr'])
    sim_reader.add_catalog_property_to_halos('m200m')
    sim_reader.add_catalog_property_to_halos('r200m')
    sim_reader.add_catalog_property_to_halos('v200m')

    geo_featurizer = GeometricFeaturizer()
    geo_featurizer.load_features(gp['fn_geo_features'])

    geo_feature_arr_rebinned = utils.rebin_geometric_features(
                                     geo_featurizer.geo_feature_arr, scp['n_groups_rebin'])
    geo_feature_arr_rebinned_pseudo = utils.transform_pseudotensors(geo_feature_arr_rebinned)

    scalar_featurizer = ScalarFeaturizer(geo_feature_arr_rebinned_pseudo)
    #scalar_featurizer.compute_MXV_from_features()
    if scp['rescale_geometric_features']:
        m_200m = np.array([dark_halo.catalog_properties['m200m'] for dark_halo in sim_reader.dark_halo_arr])
        r_200m = np.array([dark_halo.catalog_properties['r200m'] for dark_halo in sim_reader.dark_halo_arr])
        v_200m = np.array([dark_halo.catalog_properties['v200m'] for dark_halo in sim_reader.dark_halo_arr])
        scalar_featurizer.rescale_geometric_features(m_200m, r_200m, v_200m)

    start = time.time()
    scalar_featurizer.featurize(scp['m_order_max'], x_order_max=scp['x_order_max'], 
                                v_order_max=scp['v_order_max'],
                                eigenvalues_not_trace=scp['eigenvalues_not_trace'])
    print('Shape of scalar features:', scalar_featurizer.scalar_features.shape)

    scalar_featurizer.save_features(fn_scalar_features)                  
    print(f'Saved scalar features to {fn_scalar_features}')

    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    run()
