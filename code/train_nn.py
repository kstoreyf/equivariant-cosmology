import numpy as np
import random
import time
import yaml

import torch
from torch.utils.data import DataLoader

from geometric_features import GeometricFeaturizer
from scalar_features import ScalarFeaturizer
from neural_net import NNFitter, NeuralNet, seed_worker
from read_halos import SimulationReader
import utils



def seed_torch(seed=1029):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    #y_label_names = ['m_stellar']
    #y_label_names = ['gband']
    y_label_names = ['bhmass']
    #y_label_names = ['m_stellar', 'ssfr1', 'r_stellar']
    #y_label_names = ['a_mfrac_n19']
    #y_label_names = ['a_mfrac_0.75']
    #y_label_names = ['a_mfrac_n39']
    #y_label_names = ['Mofa']
    run(y_label_names)
    # ns_top_features = [1, 5, 10, 50, 100, 567]
    # for nn in range(len(ns_top_features)):
    #     run(y_label_names, n_top_features=ns_top_features[nn])


def run(y_label_names, n_top_features=None):

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    halo_tag = ''
    geo_tag = ''
    scalar_tag = ''
    #scalar_tag = '_gx1_gv1_n5'
    frac_subset = 1.0
    #n_top_features = 1
    #info_metric = 'spearman'
    #info_metric = 'pca_top50'
    info_metric = None

    # fit
    max_epochs = 500
    lr = 5e-5
    hidden_size = 128

    feature_mode = 'scalars'
    #feature_mode = 'geos'
    #feature_mode = 'catalog'
    #feature_mode = 'mrv'
    #feature_mode = 'mrvc'
    assert feature_mode in ['scalars', 'geos', 'catalog', 'mrv', 'mrvc'], "Feature mode not recognized!"

    y_str = '_'.join(y_label_names)
    frac_tag, info_tag = '', ''
    if frac_subset != 1.0:
        frac_tag = f'_f{frac_subset}'
    if info_metric is not None:
        info_tag = f'_{info_metric}_n{n_top_features}'

    fit_tag = f'_{y_str}_nn_{feature_mode}_epochs{max_epochs}_lr{lr}_hs{hidden_size}{frac_tag}{info_tag}'
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

    if feature_mode=='scalars' or feature_mode=='geos':
        geo_featurizer = GeometricFeaturizer()
        geo_featurizer.load_features(gp['fn_geo_features'])

        mrv_for_rescaling = utils.get_mrv_for_rescaling(sim_reader, scp['mrv_names_for_rescaling'])
        scalar_featurizer = ScalarFeaturizer(geo_featurizer.geo_feature_arr,
                                n_groups_rebin=scp['n_groups_rebin'], 
                                transform_pseudotensors=scp['transform_pseudotensors'], 
                                mrv_for_rescaling=mrv_for_rescaling)
        x_extra = np.log10(mrv_for_rescaling).T
        
        if feature_mode=='geos':
            # need to grab from scalar featurizer bc its doing the rebinning, rescaling 
            # and transforming for us (TODO: check if should be doing transforming here)
            x = utils.geo_feature_arr_to_values(scalar_featurizer.geo_feature_arr)

        elif feature_mode=='scalars':
            print('loading scalar features')
            scalar_featurizer.load_features(scp['fn_scalar_features'])
            print('loaded')
            x = scalar_featurizer.scalar_features

    elif feature_mode=='mrv':
        mrv_for_rescaling = utils.get_mrv_for_rescaling(sim_reader, scp['mrv_names_for_rescaling'])
        x = np.log10(mrv_for_rescaling).T
        x_extra = None

    elif feature_mode=='catalog':
        catalog_feature_names = ['M200c', 'c200c', 'a_form']
        sim_reader.get_structure_catalog_features(catalog_feature_names)
        x = sim_reader.x_catalog_features
        x_extra = None

    elif feature_mode=='mrvc':
        mrv_for_rescaling = utils.get_mrv_for_rescaling(sim_reader, scp['mrv_names_for_rescaling'])
        mrv = np.log10(mrv_for_rescaling).T
        x_extra = mrv

        catalog_feature_names = ['c200c']
        sim_reader.get_structure_catalog_features(catalog_feature_names)
        x = sim_reader.x_catalog_features


    if info_metric is not None:
        assert len(y_label_names)==1, "Info currently only computed for single labels"
        y_label_name = y_label_names[0]
        feature_info_dir = f'../data/feature_info/feature_info_{sim_name}'
        if info_metric.startswith('pca'):
            label_tag = ''
        else:
            label_tag = f'_{y_label_name}'
        fn_feature_info = f'{feature_info_dir}/feature_info_{sim_name}{halo_tag}{geo_tag}{scalar_tag}_{info_metric}{label_tag}.npy'
        values = np.load(fn_feature_info, allow_pickle=True)
        i_info = np.argsort(values)[::-1][:n_top_features]
        print(len(i_info))
        print(values[i_info])
        print("Original feature (x) shape:", x.shape)
        print("Taking top", n_top_features)
        x = x[:,i_info]
        
    print("Feature (x) shape:", x.shape)

    # Split data into train, validation, and test
    frac_train, frac_val, frac_test = 0.7, 0.15, 0.15
    random_ints = np.array([halo.random_int for halo in sim_reader.dark_halo_arr])
    idx_train, idx_valid, idx_test = utils.split_train_val_test(random_ints, 
                        frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    if frac_subset != 1.0: 
        rng = np.random.default_rng(seed=42)
        idx_train = rng.choice(idx_train, size=int(frac_subset*len(idx_train)))
        idx_valid = rng.choice(idx_valid, size=int(frac_subset*len(idx_valid)))
    print("N_train:", len(idx_train), "N_valid:", len(idx_valid))

    y = [] 
    y_uncertainties = []
    for i in range(len(y_label_names)):
        y_single = utils.get_y_vals(y_label_names[i], sim_reader, halo_tag=halo_tag)
        y_single = np.array(y_single)
        y_uncertainties_single = utils.get_y_uncertainties(y_label_names[i], sim_reader=sim_reader, y_vals=y_single,
                                                           idx_train=idx_train)
        if y_single.ndim>1:
            y.extend(y_single.T)
            y_uncertainties_single = np.array(y_uncertainties_single)
            y_uncertainties.extend(y_uncertainties_single.T)
        else:
            y.append(y_single)
            y_uncertainties.append(y_uncertainties_single)

    y = np.array(y).T
    y_uncertainties = np.array(y_uncertainties).T
    print('Label (y) shape:', y.shape)
    print('Uncertainty (y_uncertainties) shape:', y_uncertainties.shape)

    # print("SETTING FEATURES = LABELS AS TEST, CAREFUL")
    # x = y
    # print("Feature (x) shape:", x.shape)


    # training
    x_train = x[idx_train]
    y_train = y[idx_train]
    y_uncertainties_train = y_uncertainties[idx_train]
    y_current_train = None
    if feature_mode=='catalog' or feature_mode=='mrv':
        x_extra_train = None
    else:
        x_extra_train = x_extra[idx_train]

    # validation
    x_valid = x[idx_valid]
    y_valid = y[idx_valid]
    y_uncertainties_valid = y_uncertainties[idx_valid]
    y_current_valid = None
    if feature_mode=='catalog' or feature_mode=='mrv':
        x_extra_valid = None
    else:
        x_extra_valid = x_extra[idx_valid]
    


    nnfitter = NNFitter()
    nnfitter.load_training_data(x_train, y_train,
                        y_current_train=y_current_train, x_extra_train=x_extra_train,
                        y_uncertainties_train=y_uncertainties_train)
    nnfitter.set_up_training_data()
    nnfitter.load_validation_data(x_valid, y_valid,
                        y_current_valid=y_current_valid, x_extra_valid=x_extra_valid,
                        y_uncertainties_valid=y_uncertainties_valid)
    nnfitter.set_up_validation_data()
    
    start = time.time()
    print(f"Starting training with learning rate {lr:.3f}")
    seed_torch(42)
    train(nnfitter, hidden_size=hidden_size, max_epochs=max_epochs, learning_rate=lr,
            fn_model=fn_model)
    end = time.time()
    print(f"Time: {end-start} s = {(end-start)/60.0} min")
    print("Saved to", fn_model)



def train(nnfitter, hidden_size=128, max_epochs=250, learning_rate=0.00005,
          fn_model=None):
    print("training:")
    print(hidden_size, max_epochs, learning_rate)

    input_size = nnfitter.A_train.shape[1]
    if nnfitter.y_train.ndim==1:
        output_size = 1
    else:
        output_size = nnfitter.y_train.shape[-1]
    print("Output size:", output_size)
    hidden_size = hidden_size
    nnfitter.model = NeuralNet(input_size, hidden_size=hidden_size, output_size=output_size)
    nnfitter.train(max_epochs=max_epochs, learning_rate=learning_rate,
                   fn_model=fn_model)

    #nnfitter.predict_test()
    #error_nn, _ = utils.compute_error(nnfitter, test_error_type='percentile')
    #print(error_nn)


if __name__=='__main__':
    main()