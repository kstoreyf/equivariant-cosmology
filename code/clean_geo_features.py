import numpy as np
import os
import time
import yaml
from pathlib import Path

import utils
from read_halos import SimulationReader
import geometric_features as gf
from geometric_features import GeometricFeaturizer
from scalar_features import ScalarFeaturizer


def run():

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    #halo_tag = '_mini10'
    halo_tag = ''
    geo_tag = ''
    geo_clean_tag = '_n3'

    overwrite = True

    fn_geo_clean_config = f'../configs/geo_clean_{sim_name}{halo_tag}{geo_tag}{geo_clean_tag}.yaml'

    with open(fn_geo_clean_config, 'r') as file:
        geo_clean_params = yaml.safe_load(file)
    gcp = geo_clean_params['geo_clean']

    fn_geo_config = geo_clean_params['geo']['fn_geo_config']
    with open(fn_geo_config, 'r') as file:
        geo_params = yaml.safe_load(file)
    gp = geo_params['geo']

    fn_halo_config = geo_clean_params['halo']['fn_halo_config']
    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    hp = halo_params['halo']
    sp = halo_params['sim']

    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], 
                                  sp['sim_name_dark'], sp['snap_num_str'])

    fn_geo_clean_features = gcp['fn_geo_clean_features']
    fn_geo_clean_info = gcp['fn_geo_clean_info']

    # Go!
    print("Running scalar featurizer")

    tab_halos = utils.load_table(hp['fn_halos'])

    start = time.time()

    print("Transforming table to objects")
    tab_geos = utils.load_table(gp['fn_geo_features'])
    tab_geo_info = utils.load_table(gp['fn_geo_info'])
    geo_feature_arr = gf.geo_table_to_objects(tab_geos, tab_geo_info)
    print("Done!")

    if gcp['n_groups_rebin'] is not None:
        geo_feature_arr = gf.rebin_geometric_features(
                                geo_feature_arr, gcp['n_groups_rebin'])
    if gcp['transform_pseudotensors']:
        geo_feature_arr = gf.transform_pseudotensors(geo_feature_arr)
        
    if gcp['mrv_names_for_rescaling'] is not None:
        mrv = [tab_halos[name] for name in gcp['mrv_names_for_rescaling']]
        geo_feature_arr = gf.rescale_geometric_features(geo_feature_arr, *mrv)

    tab_geos_clean = gf.geo_objects_to_table(geo_feature_arr, tab_geos['idx_halo_dark'])
    print(tab_geos_clean.columns)
    for c in tab_geos_clean.columns:
        print(np.min(tab_geos_clean[c]), np.max(tab_geos_clean[c]))
    tab_geos_clean.write(fn_geo_clean_features, overwrite=overwrite, format='fits')
    print(f"Wrote table to {fn_geo_clean_features}")

    # get geos for first halo; same for all halo
    tab_geo_info = gf.geos_to_info_table(geo_feature_arr[0])
    tab_geo_info.write(fn_geo_clean_info, overwrite=overwrite, format='fits')
    print(f"Wrote table to {fn_geo_clean_info}")

    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    run()
