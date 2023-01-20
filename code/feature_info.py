import numpy as np
import random
import scipy.stats
import time
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gmm_mi.mi import EstimateMI
from gmm_mi.param_holders import GMMFitParamHolder, SelectComponentsParamHolder, MIDistParamHolder
from gmm_mi.utils.analytic_MI import calculate_MI_D1_analytical

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
    #y_label_names = [] # none for pca version, indep of label
    y_label_names = ['m_stellar']
    #y_label_names = ['m_stellar', 'ssfr1', 'r_stellar']
    #y_label_names = ['a_mfrac_n19']
    #y_label_names = ['a_mfrac_0.75']
    #y_label_names = ['a_mfrac_n39']
    #y_label_names = ['Mofa']
    run(y_label_names)


def run(y_label_names):

    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'
    halo_tag = ''
    geo_tag = ''
    scalar_tag = ''

    #info_metric = 'spearman'
    info_metric = 'MI'
    #n_comp_top = 50
    #info_metric = f'pca_top{n_comp_top}'

    feature_mode = 'scalars'
    #feature_mode = 'geos'
    #feature_mode = 'catalog'
    #feature_mode = 'mrv'
    #feature_mode = 'mrvc'
    assert feature_mode in ['scalars', 'geos', 'catalog'], "Feature mode not recognized!"

    start = time.time()

    # load configs
    print("Loading configs")
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

    elif feature_mode=='catalog':
        catalog_feature_names = ['M200c', 'c200c', 'a_form']
        sim_reader.get_structure_catalog_features(catalog_feature_names)
        x = sim_reader.x_catalog_features
        x_extra = None

    # Split data into train, validation, and test
    frac_train, frac_val, frac_test = 0.7, 0.15, 0.15
    random_ints = np.array([halo.random_int for halo in sim_reader.dark_halo_arr])
    idx_train, idx_valid, idx_test = utils.split_train_val_test(random_ints, 
                        frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)

    print(x.shape)
    x_train = x[idx_train]
    print(x_train.shape)

    feature_info_dir = f'../data/feature_info/feature_info_{sim_name}'
    Path(feature_info_dir).mkdir(parents=True, exist_ok=True)

    if info_metric.startswith('pca'):
        
        from sklearn.decomposition import PCA
        pca = PCA()
        projection = pca.fit_transform(x_train)

        pc_top = pca.components_[:n_comp_top]
        summed_contributions_top = np.sum(np.abs(pc_top), axis=0)
        values = summed_contributions_top
        #i_importance_summed_top = np.argsort(summed_contributions)[::-1]
        
        fn_feature_info = f'{feature_info_dir}/feature_info_{sim_name}{halo_tag}{geo_tag}{scalar_tag}_{info_metric}.npy'
        np.save(fn_feature_info, values)
        print("Done and saved!")

    else:
        for y_label_name in y_label_names:
            y = utils.get_y_vals(y_label_name, sim_reader)

            y_train = y[idx_train]
            print(y_train.shape)

            values = []

            # for each feature, compute correlation value
            for i in range(x_train.shape[1]):
                x_train_i = x_train[:,i]
                
                if info_metric=='spearman':
                    res = scipy.stats.spearmanr(x_train_i, y_train)
                    values.append(np.abs(res[0]))

                elif info_metric=='MI':
                    X = np.vstack((x_train_i, y_train)).T
                    mi_estimator = EstimateMI()
                    MI_mean, _ = mi_estimator.fit(X)
                    values.append(MI_mean)

            fn_feature_info = f'{feature_info_dir}/feature_info_{sim_name}{halo_tag}{geo_tag}{scalar_tag}_{info_metric}_{y_label_name}.npy'
            np.save(fn_feature_info, values)
            print("Done and saved!")


    end = time.time()
    print("Time:", end-start, 'sec')


if __name__=='__main__':
    main()