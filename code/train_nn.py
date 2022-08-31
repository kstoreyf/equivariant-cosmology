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

    #y_label_name = 'm_stellar'
    #y_label_name = 'ssfr1'
    #y_label_name = 'r_stellar'
    #y_label_name = 'a_mfrac_0.75'
    y_label_name = 'a_mfrac_n32'

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    halo_tag = ''
    geo_tag = ''
    scalar_tag = ''

    # fit
    max_epochs = 150
    fit_tag = f'_{y_label_name}_nn_test'
    fn_model = f'../models/models_{sim_name}/model_{sim_name}{halo_tag}{geo_tag}{scalar_tag}{fit_tag}.pt'

    # load configs
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

    # Load in objects
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

    x_extra = np.log10(mrv_for_rescaling).T

    print('loading scalar features')
    scalar_featurizer.load_features(scp['fn_scalar_features'])
    print('loaded')

    # get y vals
    y = utils.get_y_vals(y_label_name, sim_reader, halo_tag=halo_tag)

    # print("FIX ME")
    # print(y.shape)
    # y = np.vstack((y, 2*y)).T
    # print(y.shape)
    
    y_uncertainties = utils.get_y_uncertainties(y_label_name, sim_reader=sim_reader, y_vals=y)
    print(y_uncertainties.shape)

    # Split data into train and test, only work with training data after this
    frac_train, frac_val, frac_test = 0.7, 0.15, 0.15
    random_ints = np.array([halo.random_int for halo in sim_reader.dark_halo_arr])
    idx_train, idx_val, idx_test = utils.split_train_val_test(random_ints, 
                        frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)

    y_train = y[idx_train]
    x_train = scalar_featurizer.scalar_features[idx_train]
    y_uncertainties_train = y_uncertainties[idx_train]
    y_current_train = None
    x_extra_train = x_extra[idx_train]

    nnfitter = NNFitter()
    nnfitter.load_training_data(x_train, y_train,
                        y_current_train=y_current_train, x_extra_train=x_extra_train,
                        y_uncertainties_train=y_uncertainties_train)
    nnfitter.set_up_training_data()
    
    #lrs = [0.0001, 0.0001, 0.0001]
    lrs = [0.0001]
    for lr in lrs:
        seed_torch(42)
        # g = torch.Generator()
        # g.manual_seed(0)
        # nnfitter.data_loader_train = DataLoader(nnfitter.dataset_train, 
        #                                   batch_size=32, shuffle=True,
        #                                   worker_init_fn=seed_worker,
        #                                   generator=g, num_workers=0)
        train(nnfitter, hidden_size=128, max_epochs=max_epochs, learning_rate=lr,
              fn_model=fn_model)
        #nnfitter.save_model(fn_model)


def train(nnfitter, hidden_size=128, max_epochs=250, learning_rate=0.00005,
          fn_model=None):
    print("training:")
    print(hidden_size, max_epochs, learning_rate)

    input_size = nnfitter.A_train.shape[1]
    output_size = nnfitter.y_train.shape[-1]
    print(output_size)
    hidden_size = hidden_size
    nnfitter.model = NeuralNet(input_size, hidden_size=hidden_size, output_size=output_size)
    nnfitter.train(max_epochs=max_epochs, learning_rate=learning_rate,
                   fn_model=fn_model)

    #nnfitter.predict_test()
    #error_nn, _ = utils.compute_error(nnfitter, test_error_type='percentile')
    #print(error_nn)


if __name__=='__main__':
    main()