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
    halo_tag = '_mini10'
    geo_tag = ''
    geo_clean_tag = '_n3'
    scalar_tag = ''
    fn_scalar_config = f'../configs/scalar_{sim_name}{halo_tag}{geo_tag}{geo_clean_tag}{scalar_tag}.yaml'

    with open(fn_scalar_config, 'r') as file:
        scalar_params = yaml.safe_load(file)
    scp = scalar_params['scalar']

    # fn_geo_config = scalar_params['geo']['fn_geo_config']
    # with open(fn_geo_config, 'r') as file:
    #     geo_params = yaml.safe_load(file)
    # gp = geo_params['geo']

    fn_geo_clean_config = scalar_params['geo_clean']['fn_geo_clean_config']
    with open(fn_geo_clean_config, 'r') as file:
        geo_clean_params = yaml.safe_load(file)
    gcp = geo_clean_params['geo_clean']

    fn_scalar_features = scp['fn_scalar_features']
    fn_scalar_info = scp['fn_scalar_info']

    # Go!
    print("Running scalar featurizer")
    start = time.time()

    tab_geos = utils.load_table(gcp['fn_geo_clean_features'])
    tab_geo_info = utils.load_table(gcp['fn_geo_clean_info'])

    scalar_featurizer = ScalarFeaturizer(tab_geos, tab_geo_info)

    scalar_featurizer.featurize(scp['m_order_max'], 
                                x_order_max=scp['x_order_max'], 
                                v_order_max=scp['v_order_max'],
                                eigenvalues_not_trace=scp['eigenvalues_not_trace'],
                                elementary_scalars_only=scp['elementary_scalars_only'])
    print('Shape of scalar features:', scalar_featurizer.scalar_features.shape)

    scalar_featurizer.save_features(fn_scalar_features)                  
    print(f'Saved scalar features to {fn_scalar_features}')
    scalar_featurizer.save_scalar_info(fn_scalar_info)                  
    print(f'Saved scalar info to {fn_scalar_info}')
       

    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    run()
