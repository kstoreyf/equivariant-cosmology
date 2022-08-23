import os
import time
import yaml
from pathlib import Path

from read_halos import SimulationReader


def run():

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    halo_tag = ''
    fn_halo_config = f'../configs/halos_{sim_name}{halo_tag}.yaml'

    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)

    fn_dark_halo_arr = halo_params['halo']['fn_dark_halo_arr']
    Path(os.path.dirname(fn_dark_halo_arr)).mkdir(parents=True, exist_ok=True)

    # Go!
    start = time.time()

    sim_reader = SimulationReader(halo_params['sim']['base_dir'], halo_params['sim']['sim_name'], 
                                  halo_params['sim']['sim_name_dark'], halo_params['sim']['snap_num_str'])
    sim_reader.read_simulations()
    sim_reader.match_twins()
    sim_reader.select_halos(num_star_particles_min=halo_params['halo']['num_star_particles_min'], 
                               halo_logmass_min=halo_params['halo']['halo_logmass_min'], 
                               halo_logmass_max=halo_params['halo']['halo_logmass_max'],
                               halo_mass_difference_factor=halo_params['halo']['halo_mass_difference_factor'],
                               subsample_frac=halo_params['halo']['subsample_frac'],
                               subhalo_mode=halo_params['halo']['subhalo_mode'])
    sim_reader.save_dark_halo_arr(fn_dark_halo_arr)
    print(f'Saved halos to {fn_dark_halo_arr}')

    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    run()