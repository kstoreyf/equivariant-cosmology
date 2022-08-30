import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader

from geometric_features import GeometricFeaturizer
from scalar_features import ScalarFeaturizer
from neural_net import NNFitter, NeuralNet, seed_worker
from read_halos import SimulationReader
import utils

import random


def seed_torch(seed=1029):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():

    y_target_name = 'mass_hydro_subhalo_star'
    y_label_name = 'm_stellar'

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    halo_tag = ''
    geo_tag = ''
    scalar_tag = ''

    fit_tag = '_nntest'
    fn_model = f'../models/models_{sim_name}/model_{sim_name}{halo_tag}{geo_tag}{scalar_tag}{fit_tag}.pt'

    log_mass_shift = 10

    fn_scalar_config = f'../configs/scalar_{sim_name}{halo_tag}{geo_tag}{scalar_tag}.yaml'
    with open(fn_scalar_config, 'r') as file:
        scalar_params = yaml.safe_load(file)
    scp = scalar_params['scalar']

    fn_geo_config = scalar_params['geo']['fn_geo_config']
    with open(fn_geo_config, 'r') as file:
        geo_params = yaml.safe_load(file)
    gp = geo_params['geo']

    fn_halo_config = scalar_params['halo']['fn_halo_config']
    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    sp = halo_params['sim']

    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], sp['sim_name_dark'], 
                                  sp['snap_num_str'])
    sim_reader.load_dark_halo_arr(halo_params['halo']['fn_dark_halo_arr'])
    sim_reader.read_simulations()

    geo_featurizer = GeometricFeaturizer()
    geo_featurizer.load_features(gp['fn_geo_features'])

    mrv_for_rescaling = utils.get_mrv_for_rescaling(sim_reader, scp['mrv_names_for_rescaling'])
    scalar_featurizer = ScalarFeaturizer(geo_featurizer.geo_feature_arr,
                            n_groups_rebin=scp['n_groups_rebin'], 
                            transform_pseudotensors=scp['transform_pseudotensors'], 
                            mrv_for_rescaling=mrv_for_rescaling)

    x_features_extra = np.log10(mrv_for_rescaling).T

    print('loading')
    scalar_featurizer.load_features(scp['fn_scalar_features'])
    print('loaded')

    sim_reader.add_catalog_property_to_halos(y_target_name)
    y_label_vals = np.array([halo.catalog_properties[y_target_name] for halo in sim_reader.dark_halo_arr])
    y_val_current = np.ones(len(y_label_vals))

    # often need mstellar for the uncertainties
    # TODO: should only be allowed to get these for training set!
    sim_reader.add_catalog_property_to_halos(y_target_name)
    y_label_vals = np.array([halo.catalog_properties[y_target_name] for halo in sim_reader.dark_halo_arr])
    y_val_current = np.ones(len(y_label_vals))
    if y_label_name=='m_stellar':
        y_label_vals = np.log10(y_label_vals)

    sim_reader.add_catalog_property_to_halos('mass_hydro_subhalo_star')
    m_stellar = np.array([halo.catalog_properties[y_target_name] for halo in sim_reader.dark_halo_arr])
    log_m_stellar = np.log10(m_stellar)
    uncertainties = utils.get_uncertainties_genel2019(y_label_name, log_m_stellar+log_mass_shift, sim_name=sim_name)

    nnfitter = NNFitter(scalar_featurizer.scalar_features, y_label_vals,
                        y_val_current, x_features_extra=x_features_extra,
                        uncertainties=uncertainties)

    random_ints = np.array([dark_halo.random_int for dark_halo in sim_reader.dark_halo_arr])
    frac_train, frac_val, frac_test = 0.7, 0.15, 0.15
    idx_train, idx_val, idx_test = utils.split_train_val_test(random_ints, 
                        frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    nnfitter.split_train_test(idx_train, idx_val)

    nnfitter.set_up_data()

    #lrs = [0.0001, 0.0001, 0.0001]
    lrs = [0.00005]
    for lr in lrs:
        seed_torch(42)
        g = torch.Generator()
        g.manual_seed(0)
        nnfitter.data_loader_train = DataLoader(nnfitter.dataset_train, 
                                          batch_size=32, shuffle=True,
                                          worker_init_fn=seed_worker,
                                          generator=g, num_workers=0)
        train(nnfitter, hidden_size=128, max_epochs=70, learning_rate=lr,
             fn_model=fn_model)

        #nnfitter.save_model(fn_model)


def train(nnfitter, hidden_size=128, max_epochs=250, learning_rate=0.00005,
          fn_model=None):
    print("training:")
    print(hidden_size, max_epochs, learning_rate)

    input_size = nnfitter.n_A_features
    hidden_size = hidden_size
    nnfitter.model = NeuralNet(input_size, hidden_size=hidden_size)
    nnfitter.train(max_epochs=max_epochs, learning_rate=learning_rate,
                   fn_model=fn_model)

    nnfitter.predict_test()

    #error_nn, _ = utils.compute_error(nnfitter, test_error_type='percentile')
    #print(error_nn)


if __name__=='__main__':
    main()