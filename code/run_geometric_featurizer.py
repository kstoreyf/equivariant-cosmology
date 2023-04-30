import numpy as np
import os
import time
import yaml
from pathlib import Path

from read_halos import SimulationReader
from geometric_features import GeometricFeaturizer


def run():

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    #halo_tag = '_mini10'
    halo_tag = ''
    #geo_tag = '_gx1_gv1'
    #geo_tag = '_gx1_gv1'
    geo_tag = ''
    fn_geo_config = f'../configs/geo_{sim_name}{halo_tag}{geo_tag}.yaml'

    with open(fn_geo_config, 'r') as file:
        geo_params = yaml.safe_load(file)
    gp = geo_params['geo']

    fn_halo_config = geo_params['halo']['fn_halo_config']
    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    sp = halo_params['sim']
    hp = halo_params['halo']

    fn_halos = hp['fn_halos']
    fn_geo_features = gp['fn_geo_features']
    fn_geo_info = gp['fn_geo_info']

    # Go!
    start = time.time()

    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], 
                                  sp['sim_name_dark'], sp['snap_num_str'])

    geo_featurizer = GeometricFeaturizer()
    geo_featurizer.featurize(sim_reader, fn_halos,
                             gp['r_edges'],
                             gp['x_order_max'], gp['v_order_max'],
                            r_units=gp['r_units'])
    geo_featurizer.save_features(fn_geo_features)
    geo_featurizer.save_geo_info(fn_geo_info)
    print(f'Saved geometric features to {fn_geo_features} and {fn_geo_info}')

    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    run()