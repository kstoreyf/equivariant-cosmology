import time
from pathlib import Path

from read_halos import SimulationReader


def run():

    # sim info
    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    snap_num_str = '099' # z = 0
    sim_name = 'TNG100-1'
    sim_name_dark = 'TNG100-1-Dark'
    # sim_name = 'TNG50-4'
    # sim_name_dark = 'TNG50-4-Dark'

    # halo params 
    num_star_particles_min = 50
    halo_mass_min = 10**10.8
    halo_mass_min_str = '1e10.8'
    halo_mass_max = None
    halo_mass_max_str = None
    halo_mass_difference_factor = 3.0
    subsample_frac = None
    subhalo_mode = 'twin_subhalo'

    # save info
    halo_dir = f'../data/halos/halos_{sim_name}'
    Path(halo_dir).mkdir(parents=True, exist_ok=True)
    halo_tag = f'_nstarpartmin{num_star_particles_min}_twin'
    fn_dark_halo_arr = f'{halo_dir}/halos_{sim_name}{halo_tag}.npy'

    # Go!
    start = time.time()

    sim_reader = SimulationReader(base_dir, sim_name, sim_name_dark, snap_num_str)
    sim_reader.read_simulations()
    sim_reader.match_twins()
    sim_reader.select_halos(num_star_particles_min=num_star_particles_min, 
                               halo_mass_min=halo_mass_min, halo_mass_max=halo_mass_max,
                               halo_mass_difference_factor=halo_mass_difference_factor,
                               subsample_frac=subsample_frac,
                               subhalo_mode=subhalo_mode)
    sim_reader.save_dark_halo_arr(fn_dark_halo_arr)
    print(f'Saved halos to {fn_dark_halo_arr}')

    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    run()