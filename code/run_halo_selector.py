import os
import time
import yaml
from pathlib import Path

from read_halos import SimulationReader


def run():

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    halo_tag = '_Mmin10_nstar1'
    fn_halo_config = f'../configs/halos_{sim_name}{halo_tag}.yaml'

    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    hp = halo_params['halo']
    sp = halo_params['sim']

    fn_dark_halo_arr = halo_params['halo']['fn_dark_halo_arr']
    Path(os.path.dirname(fn_dark_halo_arr)).mkdir(parents=True, exist_ok=True)

    # Go!
    start = time.time()

    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], 
                                  sp['sim_name_dark'], sp['snap_num_str'])
    sim_reader.read_simulations()
    sim_reader.match_twins()
    sim_reader.select_halos(num_star_particles_min=hp['num_star_particles_min'], 
                            num_gas_particles_min=hp['num_gas_particles_min'], 
                            halo_logmass_min=hp['halo_logmass_min'], 
                            halo_logmass_max=hp['halo_logmass_max'],
                            halo_mass_difference_factor=hp['halo_mass_difference_factor'],
                            subsample_frac=hp['subsample_frac'],
                            subhalo_mode=hp['subhalo_mode'],
                            must_have_SAM_match=hp['must_have_SAM_match'],
                            must_have_halo_structure_info=hp['must_have_halo_structure_info'], 
                            seed=hp['seed'])
    # add halo mrv's 
    sim_reader.save_dark_halo_arr(fn_dark_halo_arr)
    print(f'Saved halos to {fn_dark_halo_arr}')

    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    run()