import yaml

from pathlib import Path


config_dir = '../configs'

def main():
    halo_config()

def halo_config():
    
    # sim info
    base_dir = '/scratch/ksf293/equivariant-cosmology/data'
    snap_num_str = '099' # z = 0
    # sim_name = 'TNG100-1'
    # sim_name_dark = 'TNG100-1-Dark'
    sim_name = 'TNG50-4'
    sim_name_dark = 'TNG50-4-Dark'

    # halo params 
    num_star_particles_min = 50
    halo_logmass_min = 10.8
    halo_logmass_max = None
    halo_mass_difference_factor = 3.0
    subsample_frac = None
    subhalo_mode = 'twin_subhalo'

    # save info
    halo_dir = f'../data/halos/halos_{sim_name}'
    halo_tag = ''
    fn_dark_halo_arr = f'{halo_dir}/halos_{sim_name}{halo_tag}.npy'

    fn_halo_config = f'{config_dir}/halos_{sim_name}{halo_tag}.yaml'

    sim_dict = {'base_dir': base_dir,
                'snap_num_str': snap_num_str,
                'sim_name': sim_name,
                'sim_name_dark': sim_name_dark,
                }
    halo_config_dict = {'num_star_particles_min': num_star_particles_min,
                        'halo_logmass_min': halo_logmass_min, 
                        'halo_logmass_max': halo_logmass_max, 
                        'halo_mass_difference_factor': halo_mass_difference_factor,
                        'subsample_frac': subsample_frac, 
                        'subhalo_mode': subhalo_mode,
                        'fn_dark_halo_arr': fn_dark_halo_arr,
                        }
    dicts = {'sim': sim_dict, 'halo': halo_config_dict}

    with open(fn_halo_config, 'w') as file:
        documents = yaml.dump(dicts, file, sort_keys=False)



if __name__=='__main__':
    main()