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
    num_star_particles_min = 50
    num_gas_particles_min = 0
    halo_logmass_min = 10.25
    halo_logmass_max = None
    halo_mass_difference_factor = 3
    must_have_SAM_match = True
    must_have_halo_structure_info = True
    if sim_name=='TNG50-4': # this sim doesnt have this data!
        must_have_SAM_match = False
        must_have_halo_structure_info = False     

    seed = 42

    # save info
    halo_tag = ''
    fn_halos = f'../data/halo_tables/halos_{sim_name}.fits'
    fn_select = f'../data/halo_selections/halo_selection_{sim_name}{halo_tag}.fits'

    fn_halo_config = f'{config_dir}/halos_{sim_name}{halo_tag}.yaml'

    sim_config_dict = {'base_dir': base_dir,
                'snap_num_str': snap_num_str,
                'sim_name': sim_name,
                'sim_name_dark': sim_name_dark,
                }
    halo_config_dict = {'halo_tag': halo_tag,
                        'fn_halos': fn_halos,
                        'fn_select': fn_select,
                        'num_star_particles_min': num_star_particles_min,
                        'num_gas_particles_min': num_gas_particles_min,
                        'halo_logmass_min': halo_logmass_min, 
                        'halo_logmass_max': halo_logmass_max, 
                        'halo_mass_difference_factor': halo_mass_difference_factor,
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
    halo_tag = '_Mmin10.25'
    fn_halo_config = f'{config_dir}/halos_{sim_name}{halo_tag}.yaml'

    # geo feature params
    # bins
    n_rbins = 10
    r_min, r_max = 0.0, 1.0
    bin_width = (r_max - r_min)/n_rbins
    print(bin_width)
    r_edges = np.arange(r_min, r_max + bin_width, bin_width)
    # round to nearest 0.01; careful, make sure want this!
    r_edges = np.array([round(r,2) for r in r_edges])
    #r_edges = np.linspace(0.0, 1.0, n_rbins+1) # in units of r200
    #r_edges_outsider200 = np.array([2, 3, 10])
    #r_edges = np.concatenate((r_edges, r_edges_outsider200))
    print(r_edges)
    r_units = 'r200m'
    # other
    x_order_max = 2
    v_order_max = 2
    center_halo = 'x_minPE'

    # save info
    geo_dir = f'../data/geometric_features/geometric_features_{sim_name}'
    #geo_tag = '_gx1_gv1'
    geo_tag = ''
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
    halo_tag = '_Mmin10.25'
    fn_halo_config = f'{config_dir}/halos_{sim_name}{halo_tag}.yaml'

    # geo info
    geo_tag = ''
    fn_geo_config = f'{config_dir}/geo_{sim_name}{halo_tag}{geo_tag}.yaml'

    # scalar parameters
    m_order_max = 2
    x_order_max = 2
    v_order_max = 2
    #n_groups_rebin = [[0,1,2], [3,4,5,6,7], [8,9,10]]
    #n_groups_rebin = [[0], [1,2], [3,4,5,6,7]]
    #n_groups_rebin = [[0,1], [2,3], [4,5], [6,7], [8,9]] # '_n5'
    #n_groups_rebin = [[0], [1,2,3], [4,5,6,7,8,9]] # '_n3'
    #n_groups_rebin = [[8,9,10]]
    #n_groups_rebin = [[0,1], [2,3], [4,5], [6,7], [8,9,10]]
    n_groups_rebin = [[i] for i in range(10)] # '_n10'
    print(n_groups_rebin)
    eigenvalues_not_trace = True
    elementary_scalars_only = True
    mrv_names_for_rescaling = ['m200m', 'r200m', 'v200m']
    transform_pseudotensors = True

    # save info
    scalar_dir = f'../data/scalar_features/scalar_features_{sim_name}'
    #scalar_tag = f'_x{x_order_max}_v{v_order_max}_n5'
    #scalar_tag = '_elementary'
    #scalar_tag = '_n3'
    scalar_tag = '_n10'
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
                          'elementary_scalars_only': elementary_scalars_only,
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