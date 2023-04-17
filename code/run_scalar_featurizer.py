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

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    halo_tag = '_Mmin10_nstar1'
    geo_tag = '_bins10'
    #geo_tag = '_gx1_gv1'
    #scalar_tag = '_n01'
    #scalar_tag = '_x4_v4_n5'
    #scalar_tag = '_elementary_n3'
    scalar_tag = '_n3'
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

    start = time.time()
    mrv_for_rescaling = utils.get_mrv_for_rescaling(sim_reader, scp['mrv_names_for_rescaling'])
    scalar_featurizer = ScalarFeaturizer(geo_featurizer.geo_feature_arr,
                            n_groups_rebin=scp['n_groups_rebin'], 
                            transform_pseudotensors=scp['transform_pseudotensors'], 
                            mrv_for_rescaling=mrv_for_rescaling)

    scalar_featurizer.featurize(scp['m_order_max'], 
                                x_order_max=scp['x_order_max'], 
                                v_order_max=scp['v_order_max'],
                                eigenvalues_not_trace=scp['eigenvalues_not_trace'],
                                elementary_scalars_only=scp['elementary_scalars_only'])
    print('Shape of scalar features:', scalar_featurizer.scalar_features.shape)

    scalar_featurizer.save_features(fn_scalar_features)                  
    print(f'Saved scalar features to {fn_scalar_features}')

    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    run()
