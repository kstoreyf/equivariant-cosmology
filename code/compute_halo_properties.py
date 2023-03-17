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
    halo_tag = '_Mmin10'

    fn_halo_config = f'../configs/halos_{sim_name}{halo_tag}.yaml'
    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    sp = halo_params['sim']

    # Go!
    start = time.time()

    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], 
                                  sp['sim_name_dark'], sp['snap_num_str'])
    sim_reader.read_simulations()
    sim_reader.load_dark_halo_arr(halo_params['halo']['fn_dark_halo_arr'])

    print("Adding merger tree info")
    start = time.time()
    #sim_reader.add_merger_info_to_halos_sublink(halo_tag)

    # need to add x_minPE first to compute the mrv200mean values
    sim_reader.add_catalog_property_to_halos('x_minPE')
    sim_reader.add_catalog_property_to_halos('m200mean') # this will also add r200mean, v200mean

    sim_reader.save_dark_halo_arr(halo_params['halo']['fn_dark_halo_arr'])
    end = time.time()

    print(sim_reader.dark_halo_arr[0].catalog_properties['m200mean'])
    print("Time:", end-start, 'sec')
    print("Done!")




if __name__=='__main__':
    main()
