import os
import time
import yaml
from pathlib import Path

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

    overwrite_table = False
    fn_halos = hp['fn_halos']

    # Go!
    start = time.time()

    print("Reading sims")
    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], 
                                  sp['sim_name_dark'], sp['snap_num_str'])
    #sim_reader.read_simulations()
    #sim_reader.match_twins()

    if not os.path.exists(fn_halos) or overwrite_table:
        print("Initializing halo table")
        sim_reader.construct_halo_table(fn_halos, overwrite=overwrite_table,
                                        N=hp['N'],
                                        )

    print("Adding properties")
    #sim_reader.add_properties_dark(fn_halos)
    #sim_reader.add_properties_hydro(fn_halos)
    #sim_reader.add_mv200m_fof_dark(fn_halos)
    #sim_reader.add_properties_structure(fn_halos)
    sim_reader.transform_properties(fn_halos)

    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60} min)")


if __name__=='__main__':
    run()