import numpy as np
from pathlib import Path

from read_halos import SimulationReader
from geometric_features import GeometricFeaturizer


def run():

    # geo feature params
    n_rbins = 8
    r_edges = np.linspace(0, 1, n_rbins+1) # in units of r200
    #r_edges = np.linspace(0, 1, n_rbins+1) # eventually include info outside r200?
    r_edges_outsider200 = np.array([2, 3, 10])
    r_edges = np.concatenate((r_edges, r_edges_outsider200))
    print(r_edges)
    x_order_max = 2
    v_order_max = 2

    # sim / halo info
    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    snap_num_str = '099' # z = 0
    sim_name = 'TNG100-1'
    sim_name_dark = 'TNG100-1-Dark'
    # sim_name = 'TNG50-4'
    # sim_name_dark = 'TNG50-4-Dark'
    halo_dir = f'../data/halos/halos_{sim_name}'
    halo_tag = '_nstarpartmin10_twin'
    fn_dark_halo_arr = f'{halo_dir}/halos_{sim_name}{halo_tag}.npy'

    # save info
    geo_dir = f'../data/geometric_features/geometric_features_{sim_name}'
    Path(geo_dir).mkdir(parents=True, exist_ok=True)
    geo_tag = '_xminPEsub_rall'
    fn_geo_features = f'{geo_dir}/geometric_features{halo_tag}{geo_tag}.npy'

    # Go!
    sim_reader = SimulationReader(base_dir, sim_name, sim_name_dark, snap_num_str)
    sim_reader.read_simulations()
    #sim_reader.load_sim_dark_halos()
    sim_reader.load_dark_halo_arr(fn_dark_halo_arr)
    sim_reader.add_catalog_property_to_halos('x_minPE')

    geo_featurizer = GeometricFeaturizer()
    geo_featurizer.featurize(sim_reader, r_edges, x_order_max, v_order_max, 
                             center_halo='x_minPE')
    geo_featurizer.save_features(fn_geo_features)
    print(f'Saved geometric features to {fn_geo_features}')



if __name__=='__main__':
    run()