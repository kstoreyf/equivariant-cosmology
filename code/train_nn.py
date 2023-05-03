import numpy as np
import random
import time
import yaml

import torch
from torch.utils.data import DataLoader

import utils
from neural_net import NNFitter, NeuralNet, NeuralNetList, seed_worker
from regressor import BoosterFitter, XGBoosterFitter, RFFitter, TabNetFitter
from read_halos import SimulationReader



def seed_torch(seed=1029):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    y_label_names = ['log_mstellar']
    #y_label_names = ['j_stellar']
    #y_label_names = ['gband']
    #y_label_names = ['num_mergers']
    #y_label_names = ['m_stellar', 'ssfr1', 'r_stellar', 'gband_minus_iband', 'bhmass_per_mstellar', 'j_stellar']
    #y_label_names = ['m_stellar', 'ssfr1', 'r_stellar', 'bhmass_per_mstellar']
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
    #halo_tag = '_mini10'
    geo_tag = ''
    geo_clean_tag = '_n3'
    scalar_tag = ''
    select_tag = ''

    frac_subset = 1.0
    info_metric = None

    # fit parameters
    # max_epochs = 1000
    # lr = 5e-5
    # hidden_size = 128
    max_epochs = 300
    lr = 0.1
    hidden_size = None
    # max_epochs = 1000
    # lr = 0.02
    # hidden_size = None
    # max_epochs = None
    # lr = 0.02
    # hidden_size = None

    #feature_mode = 'scalars'
    #feature_mode = 'geos'
    #feature_mode = 'catalogz0'
    feature_mode = 'mrv'

    y_str = '_'.join(y_label_names)
    frac_tag, info_tag = '', ''
    # if frac_subset != 1.0:
    #     frac_tag = f'_f{frac_subset}'
    # if info_metric is not None:
    #     info_tag = f'_{info_metric}_n{n_top_features}'
    #fit_tag = '_list_nl9_bn'
    #model_name = 'nn'
    model_name = 'hgboost'
    #model_name = 'gboost'
    #model_name = 'rf'
    #model_name = 'xgboost'
    #model_name = 'tabnet'
    model_tag = f'_{model_name}_yerrnan'
    #fit_tag = '_nest300'
    #fit_tag = '_scaleqt100normal'
    #fit_tag = '_yerrnan'
    #fit_tag = '_unweighted'
    #fit_tag = '_list_nl6'
    fit_tag = f'_{feature_mode}'
    if feature_mode=='scalars':
        fit_tag += f'{geo_tag}{geo_clean_tag}{scalar_tag}'
        fit_tag += scalar_tag
    if feature_mode=='geos':
        fit_tag += f'{geo_tag}{geo_clean_tag}'       

    if model_name=='nn' or model_name=='hgboost':
        model_tag += f'_epochs{max_epochs}_lr{lr}'
    if model_name=='nn':
        model_tag += f'_hs{hidden_size}'
    if model_name=='xgboost':
        model_tag += f'_lr{lr}'

    fit_tag += f'_{y_str}{model_tag}'

    if frac_subset != 1.0:
        fit_tag += f'_f{frac_subset}'
    if info_metric is not None:
        fit_tag += f'_{info_metric}_n{n_top_features}'    
    
    fn_model = f'../models/model_{sim_name}{halo_tag}{fit_tag}.pt'
    fn_pred = f'../predictions/predictions_{sim_name}{halo_tag}{fit_tag}.npy'

    print("Loading configs and tables")
    # Load config
    fn_select_config = f'../configs/halo_selection_{sim_name}{halo_tag}.yaml'
    with open(fn_select_config, 'r') as file:
        select_params = yaml.safe_load(file)
    tab_select = utils.load_table(select_params['select']['fn_select'])

    fn_halo_config = select_params['halo']['fn_halo_config']
    with open(fn_halo_config, 'r') as file:
        halo_params = yaml.safe_load(file)
    tab_halos = utils.load_table(halo_params['halo']['fn_halos'])

    print("Loading features")
    fn_scalar_config, fn_geo_clean_config = None, None
    if feature_mode=='scalars':
        fn_scalar_config = f'../configs/scalar_{sim_name}{halo_tag}{geo_tag}{geo_clean_tag}{scalar_tag}.yaml'
        print('fn_scalar_config:', fn_scalar_config)
    elif feature_mode=='geos':
        fn_geo_clean_config = f'../configs/geo_clean_{sim_name}{halo_tag}{geo_tag}{geo_clean_tag}.yaml'
        print('fn_geo_clean_config:', fn_geo_clean_config)

    x, x_extra = utils.load_features(feature_mode,
                                     tab_halos, tab_select,
                                     fn_geo_clean_config=fn_geo_clean_config,
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
    random_ints = tab_select['rand_int']
    idx_train, idx_valid, idx_test = utils.split_train_val_test(random_ints, 
                        frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    if frac_subset != 1.0: 
        rng = np.random.default_rng(seed=42)
        idx_train = rng.choice(idx_train, size=int(frac_subset*len(idx_train)))
        idx_valid = rng.choice(idx_valid, size=int(frac_subset*len(idx_valid)))
        idx_test = rng.choice(idx_valid, size=int(frac_subset*len(idx_valid)))
    print("N_train:", len(idx_train), "N_valid:", len(idx_valid))

    y = utils.load_labels(y_label_names,
                            tab_halos, tab_select)

    # For now!
    y_uncertainties = np.full(y.shape, 1)
    #y = [] 
    # y_uncertainties = []
    # for i, y_label_name in enumerate(y_label_names):
    #     #y_single = np.array(tab_halos[y_label_name])
    #     # TODO add uncertainties to table!
    #     y_uncertainties_single = np.full(y.shape, np.nan)
    #     if y_single.ndim>1:
    #         #y.extend(y_single.T)
    #         y_uncertainties.extend(y_uncertainties_single.T)
    #     else:
    #         #y.append(y_single)
    #         y_uncertainties.append(y_uncertainties_single)
   # y_uncertainties = np.array(y_uncertainties).T


    #y = np.array(y).T
    print('Label (y) shape:', y.shape)
    print('Uncertainty (y_uncertainties) shape:', y_uncertainties.shape)


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

    if model_name=='nn':
        nnfitter = NNFitter()
    elif model_name=='hgboost' or model_name=='gboost':
        nnfitter = BoosterFitter()
    elif model_name=='xgboost':
        nnfitter = XGBoosterFitter()
    elif model_name=='rf':
        nnfitter = RFFitter()
    elif model_name=='tabnet':
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

    nnfitter.input_size = nnfitter.A_train.shape[1]
    if nnfitter.y_train.ndim==1:
        nnfitter.output_size = 1
    else:
        nnfitter.output_size = nnfitter.y_train.shape[-1]
    print("Output size:", nnfitter.output_size)
    nnfitter.hidden_size = hidden_size
    nnfitter.train(max_epochs=max_epochs, learning_rate=learning_rate,
                   fn_model=fn_model)

    #nnfitter.predict_test()
    #error_nn, _ = utils.compute_error(nnfitter, test_error_type='percentile')
    #print(error_nn)


if __name__=='__main__':
    main()
