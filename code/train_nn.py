import numpy as np
import random
import time
import yaml

import torch
from torch.utils.data import DataLoader

import utils
from neural_net import NNFitter, NeuralNet, NeuralNetList, seed_worker
from regressor import BoosterFitter, RFFitter, TabNetFitter
from read_halos import SimulationReader



def seed_torch(seed=1029):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    y_label_names = ['m_stellar']
    #y_label_names = ['j_stellar']
    #y_label_names = ['gband']
    #y_label_names = ['num_mergers']
    #y_label_names = ['m_stellar', 'ssfr1', 'r_stellar', 'gband_minus_iband', 'bhmass_per_mstellar', 'j_stellar']
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
    halo_tag = '_Mmin10_nstar1'
    # geo_tag = '_bins10'
    # scalar_tag = '_n3'
    geo_tag = None
    scalar_tag = None
    #scalar_tag = '_gx1_gv1_n5'
    frac_subset = 1.0
    #n_top_features = 1
    #info_metric = 'spearman'
    #info_metric = 'pca_top50'
    info_metric = None

    # fit parameters
    #max_epochs = 1000
    #lr = 5e-5
    max_epochs = 300
    lr = 0.1
    hidden_size = 128

    #feature_mode = 'scalars'
    #feature_mode = 'geos'
    #feature_mode = 'catalog'
    feature_mode = 'catalog_z0'
    #feature_mode = 'catalog_mergers_noaform'
    #feature_mode = 'mrv'
    #feature_mode = 'mrvc'

    y_str = '_'.join(y_label_names)
    frac_tag, info_tag = '', ''
    # if frac_subset != 1.0:
    #     frac_tag = f'_f{frac_subset}'
    # if info_metric is not None:
    #     info_tag = f'_{info_metric}_n{n_top_features}'
    #fit_tag = '_list_nl9_bn'
    #model_tag = 'nn'
    model_tag = 'hgboost'
    #model_tag = 'gboost'
    #model_tag = 'rf'
    #model_tag = 'tabnet'
    #fit_tag = '_nest300'
    fit_tag = ''
    #fit_tag = '_list_nl6'
    if feature_mode=='scalars':
        fit_tag += geo_tag
        fit_tag += scalar_tag
    if feature_mode=='geos':
         fit_tag += geo_tag       

    fit_tag += f'_{y_str}_{model_tag}_{feature_mode}_epochs{max_epochs}_lr{lr}_hs{hidden_size}'
    if frac_subset != 1.0:
        fit_tag += f'_f{frac_subset}'
    if info_metric is not None:
        fit_tag += f'_{info_metric}_n{n_top_features}'    
    
    fn_model = f'../models/models_{sim_name}/model_{sim_name}{halo_tag}{fit_tag}.pt'
    fn_pred = f'../predictions/predictions_{sim_name}/predictions_{sim_name}{halo_tag}{fit_tag}.npy'

    # Load config
    fn_halo_config = f'../configs/halos_{sim_name}{halo_tag}.yaml'
    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    sp = halo_params['sim']

    # Load in objects
    print("Setting up SimulationReader")
    sim_reader = SimulationReader(sp['base_dir'], sp['sim_name'], sp['sim_name_dark'], 
                                  sp['snap_num_str'])
    sim_reader.load_dark_halo_arr(halo_params['halo']['fn_dark_halo_arr'])
    sim_reader.read_simulations()

    print("Loading features")
    fn_scalar_config, fn_geo_config = None, None
    if feature_mode=='scalars':
        fn_scalar_config = f'../configs/scalar_{sim_name}{halo_tag}{geo_tag}{scalar_tag}.yaml'
    elif feature_mode=='geos':
        fn_geo_config = f'../configs/geo_{sim_name}{halo_tag}{geo_tag}.yaml'
        # geos needs scalar config too bc that's what tells it the bins and transformations!
        # bad design but not a clear/straightforward better way to do this
        fn_scalar_config = f'../configs/scalar_{sim_name}{halo_tag}{geo_tag}{scalar_tag}.yaml'
    x, x_extra = utils.load_features(feature_mode, sim_reader,
                                     fn_geo_config=fn_geo_config,
                                     fn_scalar_config=fn_scalar_config)


    if info_metric is not None:
        print("Loading feature info")
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
        idx_test = rng.choice(idx_valid, size=int(frac_subset*len(idx_valid)))
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

    # validation
    x_valid = x[idx_valid]
    y_valid = y[idx_valid]
    y_uncertainties_valid = y_uncertainties[idx_valid]

    #test 
    x_test = x[idx_test]
    y_test = y[idx_test]
    y_uncertainties_test = y_uncertainties[idx_test]

    # x_extra
    if x_extra is None:
        x_extra_train, x_extra_valid, x_extra_test = None, None, None
    else:
        x_extra_train = x_extra[idx_train]
        x_extra_valid = x_extra[idx_valid]
        x_extra_test = x_extra[idx_test]

    if model_tag=='nn':
        nnfitter = NNFitter()
    elif model_tag=='hgboost' or model_tag=='gboost':
        nnfitter = BoosterFitter()
    elif model_tag=='rf':
        nnfitter = RFFitter()
    elif model_tag=='tabnet':
        nnfitter = TabNetFitter()
    else:
        raise ValueError(f"Model {model} not recognized!")
    nnfitter.load_training_data(x_train, y_train,
                        x_extra_train=x_extra_train,
                        y_uncertainties_train=y_uncertainties_train)
    nnfitter.set_up_training_data()
    nnfitter.load_validation_data(x_valid, y_valid,
                        x_extra_valid=x_extra_valid,
                        y_uncertainties_valid=y_uncertainties_valid)
    nnfitter.set_up_validation_data()
    
    start = time.time()
    print(f"Starting training with learning rate {lr:.3f}")
    seed_torch(42)
    train(nnfitter, hidden_size=hidden_size, max_epochs=max_epochs, learning_rate=lr,
            fn_model=fn_model)
    end = time.time()

    print(f"Applying to test set")
    y_pred = nnfitter.predict(x_test, x_extra=x_extra_test)
    np.save(fn_pred, y_pred)
    print(f"Saved test to {fn_pred}")

    print(f"Time: {end-start} s = {(end-start)/60.0} min")
    print("Saved to", fn_model)


def train_booster():
    #from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor 
    booster = HistGradientBoostingRegressor()
    booster.fit(X, y)


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
    nnfitter.model = NeuralNetList(input_size, hidden_size=hidden_size, output_size=output_size)
    nnfitter.train(max_epochs=max_epochs, learning_rate=learning_rate,
                   fn_model=fn_model)

    #nnfitter.predict_test()
    #error_nn, _ = utils.compute_error(nnfitter, test_error_type='percentile')
    #print(error_nn)


if __name__=='__main__':
    main()