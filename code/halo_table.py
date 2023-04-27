import os
import time
import yaml
from pathlib import Path

from read_halos import SimulationReader


def run():

    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    sim_name = 'TNG100-1'
    sim_name_dark = 'TNG100-1-Dark'
    snap_num_str = '099'
    
    overwrite = False
    fn_halos = f'../data/halo_tables/halos_{sim_name}.fits'

    # Go!
    start = time.time()

    sim_reader = SimulationReader(base_dir, sim_name, 
                                  sim_name_dark, snap_num_str)
    sim_reader.read_simulations()
    sim_reader.match_twins()

    if not os.path.exists(fn_halos) or overwrite:
        sim_reader.construct_halo_table(fn_halos, overwrite=overwrite)

    #sim_reader.add_properties_dark(fn_halos)
    #sim_reader.add_properties_hydro(fn_halos)
    #sim_reader.add_MRV_dark(fn_halos, overwrite=False)
    sim_reader.add_MRV_dark(fn_halos)

    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60} min)")


if __name__=='__main__':
    run()