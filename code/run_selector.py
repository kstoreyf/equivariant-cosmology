import os
import time
import yaml
from pathlib import Path

from read_halos import SimulationReader


def run():

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    #halo_tag = '_mini10'
    halo_tag = ''
    select_tag = ''
    fn_select_config = f'../configs/halo_selection_{sim_name}{halo_tag}.yaml'

    with open(fn_select_config, 'r') as file:
        select_params = yaml.safe_load(file)
    selp = select_params['select']

    fn_halo_config = select_params['halo']['fn_halo_config']
    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    hp = halo_params['halo']
    sp = halo_params['sim']

    fn_halos = hp['fn_halos']
    fn_select = selp['fn_select']

    # Go!
    start = time.time()

    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], 
                                  sp['sim_name_dark'], sp['snap_num_str'])

    sim_reader.select_halos(fn_halos, halo_tag, fn_select,
                            num_star_particles_min=selp['num_star_particles_min'], 
                            num_gas_particles_min=selp['num_gas_particles_min'], 
                            halo_logmass_min=selp['halo_logmass_min'], 
                            halo_logmass_max=selp['halo_logmass_max'],
                            halo_mass_difference_factor=selp['halo_mass_difference_factor'],
                            must_have_mah_info=selp['must_have_mah_info'],
                            must_have_halo_structure_info=selp['must_have_halo_structure_info'], 
                            seed=selp['seed'])

    print(f'Saved selection to {fn_select}')

    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    run()