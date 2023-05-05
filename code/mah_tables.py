import os
import time
import yaml
from pathlib import Path

import utils
from read_halos import SimulationReader



def run():

    sim_name = 'TNG100-1'
    #halo_tag = '_mini10'
    halo_tag = ''
    fn_halo_config = f'../configs/halos_{sim_name}{halo_tag}.yaml'

    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    sp = halo_params['sim']
    hp = halo_params['halo']

    overwrite_mahs = False
    fn_halos = hp['fn_halos']

    fn_mahs = f'../data/mahs/mah_table_{sim_name}{halo_tag}.fits'
    fn_amfrac = f'../data/mahs/amfracs_{sim_name}{halo_tag}.fits'

    # Go!
    start = time.time()

    print("Reading sims")
    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], 
                                  sp['sim_name_dark'], sp['snap_num_str'])

    if not os.path.exists(fn_mahs) or overwrite_mahs:
        print("Initializing halo table")
        sim_reader.write_MAH_table(fn_halos, fn_mahs, overwrite=overwrite_mahs)

    sim_reader.write_amfrac_table(fn_mahs, fn_amfrac)

    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60} min)")


if __name__=='__main__':
    run()