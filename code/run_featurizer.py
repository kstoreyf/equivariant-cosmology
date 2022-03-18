import cProfile
import h5py
import numpy as np
import sys
from pathlib import Path

from featurize_and_fit import Featurizer, Fitter
import scalars



def run():

    tag = ''

    # feature params
    n_rbins_arr = np.array([1,2,4])
    m_order_max_arr = [1,2,3]
    x_order_max_arr = [0] 
    v_order_max_arr = [0, 2]
    include_eigenvalues = False
    include_eigenvectors = False

    r_units = 'r200'
    rms_x = True
    log_x = False
    log_y = False

    # simulation params
    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    snap_num_str = '099' # z = 0
    # sim_name = 'TNG100-1'
    # sim_name_dark = 'TNG100-1-Dark'
    sim_name = 'TNG50-4'
    sim_name_dark = 'TNG50-4-Dark'

    # halo params 
    num_star_particles_min = 1
    halo_mass_min = 10**10.8
    halo_mass_min_str = '1e10.8'
    halo_mass_max = None
    halo_mass_max_str = None
    halo_mass_difference_factor = 3.0
    subsample_frac = None

    # final things
    l_arr, p_arr = scalars.get_needed_vec_orders_scalars(max(x_order_max_arr), max(v_order_max_arr))
    if include_eigenvalues:
        tag += '_eigvals'

    # Go!
    print("Setting up featurizer")
    featurizer = Featurizer(base_dir, sim_name, sim_name_dark, snap_num_str)
    featurizer.load_halo_dicts(num_star_particles_min=num_star_particles_min, 
                               halo_mass_min=halo_mass_min, halo_mass_min_str=halo_mass_min_str, 
                               halo_mass_max=halo_mass_max, halo_mass_max_str=halo_mass_max_str, 
                               halo_mass_difference_factor=halo_mass_difference_factor,
                               subsample_frac=subsample_frac,
                               force_reload=False)
    save_dir = f'../data/features/features_{sim_name}/halos{featurizer.halo_tag}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for n_rbins in n_rbins_arr:
        r_edges = np.linspace(0, 1, n_rbins+1)
        print("Computing geometric features for n_rbins =", n_rbins)
        featurizer.compute_geometric_features(r_edges, l_arr, p_arr, r_units=r_units)
        print("Done!")


        for m_order_max in m_order_max_arr:
            for x_order_max in x_order_max_arr:
                for v_order_max in v_order_max_arr:

                    # Featurize to order in grid
                    featurizer.compute_scalar_features(m_order_max, x_order_max, v_order_max,
                                                    include_eigenvalues=include_eigenvalues, 
                                                    include_eigenvectors=include_eigenvectors)

                    scalar_tag = f'_mordmax{m_order_max}_xordmax{x_order_max}_vordmax{v_order_max}_rbins{n_rbins}{tag}'
                    scalar_fn = f'{save_dir}/scalar_features{scalar_tag}.npy'
                    np.save(scalar_fn, featurizer.x_scalar_features)
                    

if __name__=='__main__':
    run()
    #cProfile.run('run()', 'profiling/profile_1e11sliver.out')