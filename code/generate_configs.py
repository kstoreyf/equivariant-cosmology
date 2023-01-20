import numpy as np
import yaml


config_dir = '../configs'

def main():
    
    sim_name = 'TNG100-1'
    #sim_name = 'TNG50-4'

    halo_config(sim_name)
    #geo_config(sim_name)
    #scalar_config(sim_name)
    #fit_config(sim_name)


def halo_config(sim_name):
    
    # sim info
    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    snap_num_str = '099' # z = 0
    sim_name_dark = f'{sim_name}-Dark'

    # halo params 
    num_star_particles_min = 1
    num_gas_particles_min = 1
    halo_logmass_min = 10
    halo_logmass_max = None
    halo_mass_difference_factor = None
    subsample_frac = None
    subhalo_mode = 'twin_subhalo'
    must_have_SAM_match = False
    must_have_halo_structure_info = False
    if sim_name=='TNG50-4': # this sim doesnt have this data!
        must_have_SAM_match = False
        must_have_halo_structure_info = False     

    seed = 42

    # save info
    halo_dir = f'../data/halos/halos_{sim_name}'
    halo_tag = '_mssm'
    fn_dark_halo_arr = f'{halo_dir}/halos_{sim_name}{halo_tag}.npy'

    fn_halo_config = f'{config_dir}/halos_{sim_name}{halo_tag}.yaml'

    sim_config_dict = {'base_dir': base_dir,
                'snap_num_str': snap_num_str,
                'sim_name': sim_name,
                'sim_name_dark': sim_name_dark,
                }
    halo_config_dict = {'halo_tag': halo_tag,
                        'fn_dark_halo_arr': fn_dark_halo_arr,
                        'num_star_particles_min': num_star_particles_min,
                        'num_gas_particles_min': num_gas_particles_min,
                        'halo_logmass_min': halo_logmass_min, 
                        'halo_logmass_max': halo_logmass_max, 
                        'halo_mass_difference_factor': halo_mass_difference_factor,
                        'subsample_frac': subsample_frac, 
                        'subhalo_mode': subhalo_mode,
                        'must_have_SAM_match': must_have_SAM_match, 
                        'must_have_halo_structure_info': must_have_halo_structure_info,
                        'seed': seed
                        }
    dicts = {'sim': sim_config_dict, 'halo': halo_config_dict}

    with open(fn_halo_config, 'w') as file:
        documents = yaml.safe_dump(dicts, file, sort_keys=False)
    print(f"Generated halo config file {fn_halo_config}")


def geo_config(sim_name):

    # halo info
    halo_tag = ''
    fn_halo_config = f'{config_dir}/halos_{sim_name}{halo_tag}.yaml'

    # geo feature params
    # bins
    n_rbins = 8
    r_edges = np.linspace(0, 1, n_rbins+1) # in units of r200
    r_edges_outsider200 = np.array([2, 3, 10])
    r_edges = np.concatenate((r_edges, r_edges_outsider200))
    print(list(r_edges))
    r_units = 'r200m'
    # other
    x_order_max = 1
    v_order_max = 1
    center_halo = 'x_minPE'

    # save info
    geo_dir = f'../data/geometric_features/geometric_features_{sim_name}'
    geo_tag = '_gx1_gv1'
    fn_geo_features = f'{geo_dir}/geometric_features_{sim_name}{halo_tag}{geo_tag}.npy'
    fn_geo_config = f'{config_dir}/geo_{sim_name}{halo_tag}{geo_tag}.yaml'

    geo_config_dict = {'geo_tag': geo_tag,
                'fn_geo_features': fn_geo_features,
                'r_edges': r_edges.tolist(),
                'r_units': r_units,
                'x_order_max': x_order_max,
                'v_order_max': v_order_max,
                'center_halo': center_halo,
                }

    halo_config_dict = {'halo_tag': halo_tag,
                        'fn_halo_config': fn_halo_config,
                        }

    dicts = {'halo': halo_config_dict, 'geo': geo_config_dict, }

    with open(fn_geo_config, 'w') as file:
        documents = yaml.dump(dicts, file, sort_keys=False, default_flow_style=False)
    print(f"Generated geo config file {fn_geo_config}")
   


def scalar_config(sim_name):

    # halo info
    halo_tag = ''
    fn_halo_config = f'{config_dir}/halos_{sim_name}{halo_tag}.yaml'

    # geo info
    geo_tag = '_gx1_gv1'
    fn_geo_config = f'{config_dir}/geo_{sim_name}{halo_tag}{geo_tag}.yaml'

    # scalar parameters
    m_order_max = 2
    x_order_max = 2
    v_order_max = 2
    #n_groups_rebin = [[0,1,2], [3,4,5,6,7], [8,9,10]]
    #n_groups_rebin = [[8,9,10]]
    n_groups_rebin = [[0,1], [2,3], [4,5], [6,7], [8,9,10]]
    eigenvalues_not_trace = True
    mrv_names_for_rescaling = ['m200m', 'r200m', 'v200m']
    transform_pseudotensors = True

    # save info
    scalar_dir = f'../data/scalar_features/scalar_features_{sim_name}'
    #scalar_tag = f'_x{x_order_max}_v{v_order_max}_n5'
    scalar_tag = '_n5'
    fn_scalar_features = f'{scalar_dir}/scalar_features{sim_name}{halo_tag}{geo_tag}{scalar_tag}.npy'
    fn_scalar_config = f'{config_dir}/scalar_{sim_name}{halo_tag}{geo_tag}{scalar_tag}.yaml'


    geo_config_dict = {'geo_tag': geo_tag,
                       'fn_geo_config': fn_geo_config,
                      }

    halo_config_dict = {'halo_tag': halo_tag,
                        'fn_halo_config': fn_halo_config,
                        }

    scalar_config_dict = {'scalar_tag': scalar_tag,
                          'fn_scalar_features': fn_scalar_features,
                          'm_order_max': m_order_max,
                          'x_order_max': x_order_max,
                          'v_order_max': v_order_max,
                          'n_groups_rebin': n_groups_rebin,
                          'eigenvalues_not_trace': eigenvalues_not_trace,
                          'mrv_names_for_rescaling': mrv_names_for_rescaling,
                          'transform_pseudotensors': transform_pseudotensors
                          }

    dicts = {'halo': halo_config_dict, 'geo': geo_config_dict, 'scalar': scalar_config_dict}

    with open(fn_scalar_config, 'w') as file:
        documents = yaml.dump(dicts, file, sort_keys=False, default_flow_style=False)
    print(f"Generated scalar config file {fn_scalar_config}")
   

def fit_config(sim_name):

    # # halo info
    # halo_tag = ''
    # fn_halo_config = f'{config_dir}/halos_{sim_name}{halo_tag}.yaml'

    # # geo info
    # geo_tag = ''
    # fn_geo_config = f'{config_dir}/geo_{sim_name}{halo_tag}{geo_tag}.yaml'

    # scalar info
    scalar_tag = ''
    fn_scalar_config = f'{config_dir}/scalar_{sim_name}{halo_tag}{geo_tag}{scalar_tag}.yaml'

    # fit parameters
    fitter_name = 'NNFitter'
    #input_size = 

    scalar_config_dict = {'scalar_tag': scalar_tag,
                          'fn_scalar_config': fn_scalar_config,
                          }

    dicts = {'scalar': scalar_config_dict, 'fit': fit_config_dict}

    with open(fn_scalar_config, 'w') as file:
        documents = yaml.dump(dicts, file, sort_keys=False, default_flow_style=False)
    print(f"Generated scalar config file {fn_scalar_config}")
   


if __name__=='__main__':
    main()