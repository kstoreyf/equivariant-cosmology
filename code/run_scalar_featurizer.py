import numpy as np
from pathlib import Path

from read_halos import SimulationReader
from geometric_features import GeometricFeaturizer
from scalar_features import ScalarFeaturizer


def run():

    # scalar parameters
    m_order_max = 2
    n_groups_rebin = np.atleast_2d(np.arange(8))
    eigenvalues_not_trace = False

    # sim / halo info
    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    snap_num_str = '099' # z = 0
    sim_name = 'TNG100-1'
    sim_name_dark = 'TNG100-1-Dark'
    # sim_name = 'TNG50-4'
    # sim_name_dark = 'TNG50-4-Dark'
    halo_dir = f'../data/halos/halos_{sim_name}'
    halo_tag = '_nstarpartmin1'
    fn_dark_halo_arr = f'{halo_dir}/halos_{sim_name}{halo_tag}.npy'

    # geo info
    geo_dir = f'../data/geometric_features/geometric_features_{sim_name}'
    geo_tag = ''
    fn_geo_features = f'{geo_dir}/geometric_features{halo_tag}{geo_tag}.npy'

    # save info
    scalar_dir = f'../data/scalar_features/scalar_features_{sim_name}'
    Path(scalar_dir).mkdir(parents=True, exist_ok=True)
    scalar_tag = '_1bin'
    fn_scalar_features = f'{scalar_dir}/scalar_features{halo_tag}{geo_tag}.npy'

    # Go!
    geo_featurizer = GeometricFeaturizer()
    geo_featurizer.load_features(fn_geo_features)

    scalar_featurizer = ScalarFeaturizer(geo_featurizer.geo_feature_arr)
    scalar_featurizer.featurize(m_order_max, n_groups_rebin=n_groups_rebin,
                            eigenvalues_not_trace=eigenvalues_not_trace)
    scalar_featurizer.save_features(fn_scalar_features)                  
    print(f'Saved scalar features to {fn_scalar_features}')



if __name__=='__main__':
    run()