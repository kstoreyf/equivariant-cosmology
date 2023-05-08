import os
import time
import yaml
from pathlib import Path

import utils
from read_halos import SimulationReader



def run():

    sim_name = 'TNG100-1'
    halo_tag = '_mini10'
    #halo_tag = ''
    fn_halo_config = f'../configs/halos_{sim_name}{halo_tag}.yaml'

    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    sp = halo_params['sim']
    hp = halo_params['halo']

    overwrite_mahs = False
    fn_halos = hp['fn_halos']

    fn_unc = f'../data/halo_tables/uncertainty_table_{sim_name}{halo_tag}.fits'

    # Go!
    start = time.time()

    utils.write_uncertainties_table(fn_halos, fn_unc)

    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60} min)")


if __name__=='__main__':
    run()