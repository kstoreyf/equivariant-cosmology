import numpy as np
import os
import time
import yaml
from pathlib import Path

import utils
from read_halos import SimulationReader



def main():

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    halo_tag = ''
    geo_tag = ''
    fn_geo_config = f'../configs/geo_{sim_name}{halo_tag}{geo_tag}.yaml'

    with open(fn_geo_config, 'r') as file:
        geo_params = yaml.safe_load(file)
    gp = geo_params['geo']

    fn_halo_config = geo_params['halo']['fn_halo_config']
    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    sp = halo_params['sim']

    fn_geo_features = gp['fn_geo_features']
    Path(os.path.dirname(fn_geo_features)).mkdir(parents=True, exist_ok=True)

    # Go!
    start = time.time()

    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], 
                                  sp['sim_name_dark'], sp['snap_num_str'])
    sim_reader.read_simulations()
    sim_reader.load_dark_halo_arr(halo_params['halo']['fn_dark_halo_arr'])

    print("Adding merger tree info")
    start = time.time()
    sim_reader.add_merger_info_to_halos_sublink(halo_tag)
    end = time.time()
    print("Time:", end-start, 'sec')

    print("Done!")


if __name__=='__main__':
    main()
