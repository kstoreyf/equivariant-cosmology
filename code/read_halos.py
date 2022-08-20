import h5py
import numpy as np
import os
import sys

sys.path.insert(1, '/home/ksf293/external')
import illustris_python as il


class DarkHalo:

    def __init__(self, idx_halo_dark, base_path, snap_num, box_size):
        self.idx_halo_dark = idx_halo_dark
        self.base_path = base_path
        self.snap_num = snap_num
        self.box_size = box_size
        self.catalog_properties = {}

    def set_associated_halos(self, idx_subhalo_dark, idx_halo_hydro, idx_subhalo_hydro):
        self.idx_subhalo_dark = idx_subhalo_dark
        self.idx_halo_hydro = idx_halo_hydro
        self.idx_subhalo_hydro = idx_subhalo_hydro

    def set_random_int(self, random_int):
        self.random_int = random_int

    def load_positions_and_velocities(self, shift=True, center='x_com'):
        halo_dark_dm = il.snapshot.loadHalo(self.base_path,self.snap_num,self.idx_halo_dark,'dm')
        x_data_halo = halo_dark_dm['Coordinates']
        v_data_halo = halo_dark_dm['Velocities']

        if shift:
            center_options = ['x_com','x_minPE']
            assert center in center_options, f"Center choice must be one of {center_options}!"
                 
            if center=='x_minPE':
                assert 'x_minPE' in self.catalog_properties, "Must first add 'x_minPE' to catalog property dict!"
            x_data_halo = self.shift_x(x_data_halo, center)
            v_data_halo = self.shift_v(v_data_halo)

        return x_data_halo, v_data_halo

    # for now, masses of all particles are assumed to be same
    def shift_x(self, x_arr, center):
        if center=='x_com':
            # Add particle position to make sure CoM falls within halo
            particle0_pos = x_arr[0]
            x_arr_shifted_byparticle = self.shift_points_torus(x_arr, particle0_pos)
            x_shift = np.mean(x_arr_shifted_byparticle, axis=0) + particle0_pos
        elif center=='x_minPE':
            x_shift = self.catalog_properties['x_minPE']
        # Subtract off shift for each halo, wrapping around torus
        x_arr_shifted = self.shift_points_torus(x_arr, x_shift)
        return x_arr_shifted

    def shift_points_torus(self, points, shift):
        return (points - shift + 0.5*self.box_size) % self.box_size - 0.5*self.box_size

    # for now, masses of all particles is considered to be same
    def shift_v(self, v_arr):
        v_arr_com = np.mean(v_arr, axis=0)
        return v_arr - v_arr_com

    def set_catalog_property(self, property_name, value):
        self.catalog_properties[property_name] = value


class SimulationReader:

    # TODO: replace snap_num_str with proper zfill (i think? check works)
    def __init__(self, base_dir, sim_name, sim_name_dark, snap_num_str):
        self.sim_name = sim_name
        self.sim_name_dark = sim_name_dark
        self.tng_path_hydro = f'{base_dir}/{self.sim_name}'
        self.tng_path_dark = f'{base_dir}/{self.sim_name_dark}'
        self.base_path_hydro = f'{base_dir}/{self.sim_name}/output'
        self.base_path_dark = f'{base_dir}/{self.sim_name_dark}/output'
        self.snap_num_str = snap_num_str
        self.snap_num = int(self.snap_num_str)
        self.halo_arr = []

        with h5py.File(f'{self.base_path_hydro}/snapdir_{self.snap_num_str}/snap_{self.snap_num_str}.0.hdf5','r') as f:
            header = dict( f['Header'].attrs.items() )
            self.m_dmpart = header['MassTable'][1] # this times 10^10 msun/h
            self.box_size = header['BoxSize'] # c kpc/h


    def load_sim_dark_halos(self):
        self.halos_dark = il.groupcat.loadHalos(self.base_path_dark,self.snap_num)

    def read_simulations(self, subhalo_fields_to_load=None):

        if subhalo_fields_to_load is None:
            subhalo_fields_to_load = ['SubhaloLenType', 'SubhaloGrNr', 'SubhaloMassType', 'SubhaloMass',
                                      'SubhaloHalfmassRadType', 'SubhaloSFR', 'SubhaloPos', 'SubhaloFlag',
                                      'SubhaloVelDisp']
        #subhalo_fields_to_load_dark = ['SubhaloFlag', 'SubhaloPos']

        self.subhalos_hydro = il.groupcat.loadSubhalos(self.base_path_hydro,self.snap_num,
                                                       fields=subhalo_fields_to_load)
        self.halos_hydro = il.groupcat.loadHalos(self.base_path_hydro,self.snap_num)

        self.subhalos_dark = il.groupcat.loadSubhalos(self.base_path_dark,self.snap_num)
                                                      #fields=subhalo_fields_to_load_dark)
        self.load_sim_dark_halos() # separate out bc will need these on own

        self.idxs_halos_hydro_all = np.arange(self.halos_hydro['count'])
        self.idxs_halos_dark_all = np.arange(self.halos_dark['count'])

        self.ipart_dm = il.snapshot.partTypeNum('dm') # 0
        self.ipart_star = il.snapshot.partTypeNum('stars') # 4


    # TODO: clean up
    def match_twins(self):
        # Load twin-matching file
        fn_match_dark_to_full = f'../data/subhalo_matching_dicts/subhalo_dark_to_full_dict_{self.sim_name}.npy'
        fn_match_full_to_dark = f'../data/subhalo_matching_dicts/subhalo_full_to_dark_dict_{self.sim_name}.npy'
        if os.path.exists(fn_match_dark_to_full) and os.path.exists(fn_match_full_to_dark):
            self.subhalo_dark_to_full_dict = np.load(fn_match_dark_to_full, allow_pickle=True).item()
            self.subhalo_full_to_dark_dict = np.load(fn_match_full_to_dark, allow_pickle=True).item()
        else:
            with h5py.File(f'{self.tng_path_hydro}/postprocessing/subhalo_matching_to_dark.hdf5','r') as f:
                # Note: there are two different matching algorithms: 'SubhaloIndexDark_LHaloTree' & 
                # 'SubhaloIndexDark_SubLink'. choosing first for now
                subhalo_full_to_dark_inds = f[f'Snapshot_{self.snap_num}']['SubhaloIndexDark_LHaloTree']

                # Build dicts to match subhalos both ways. If a full subhalo has no dark subhalo twin, exclude it.
                self.subhalo_full_to_dark_dict = {}
                self.subhalo_dark_to_full_dict = {}
                for i in range(len(subhalo_full_to_dark_inds)):
                    idx_full = i
                    idx_dark = subhalo_full_to_dark_inds[idx_full]
                    if idx_dark == -1:
                        continue
                    self.subhalo_dark_to_full_dict[idx_dark] = idx_full
                    self.subhalo_full_to_dark_dict[idx_full] = idx_dark
            np.save(fn_match_dark_to_full, self.subhalo_dark_to_full_dict)
            np.save(fn_match_full_to_dark, self.subhalo_full_to_dark_dict)


    # TODO: clean up
    def select_halos(self, num_star_particles_min, halo_mass_min, 
                     halo_mass_max, halo_mass_difference_factor, subsample_frac,
                     subhalo_mode='most_massive', seed=42):

        subhalo_mode_options = ['most_massive_subhalo', 'twin_subhalo']
        assert subhalo_mode in subhalo_mode_options, f"Input subhalo_mode {subhalo_mode} not an \
                                                      option; choose one of {subhalo_mode_options}"
        # GroupFirstSub: Index into the Subhalo table of the first/primary/most massive 
        # Subfind group within this FoF group. Note: This value is signed (or should be interpreted as signed)! 
        # In this case, a value of -1 indicates that this FoF group has no subhalos.
        self.halos_dark['GroupFirstSub'] = self.halos_dark['GroupFirstSub'].astype('int32')
        mask_has_subhalos = np.where(self.halos_dark['GroupFirstSub'] >= 0) # filter out halos with no subhalos

        idxs_halos_dark_withsubhalos = self.idxs_halos_dark_all[mask_has_subhalos]
        idxs_largestsubs_dark_all = self.halos_dark['GroupFirstSub'][mask_has_subhalos]

        # TODO: should be passing in mass multiplier? or getting from elsewhere?
        if halo_mass_min is not None:
            halo_mass_min /= 1e10 # because masses in catalog have units of 10^10 M_sun/h
        if halo_mass_max is not None:
            halo_mass_max /= 1e10 # because masses in catalog have units of 10^10 M_sun/h

        # TODO: could I reformulate this starting from the subhalo_dark_to_full_dict,
        # and then get each subhalo's parent and its twin's parent??
        # FOLLOWUP: i could but then i would have to also check that it is the most massive dark subhalo in the dark halo!
        dark_halo_arr = []
        for i, idx_halo_dark in enumerate(idxs_halos_dark_withsubhalos):
            
            idx_largestsub_dark = idxs_largestsubs_dark_all[i]
            if idx_largestsub_dark in self.subhalo_dark_to_full_dict:
                
                # This is the index of the hydro subhalo that is the twin of the largest subhalo in the dark halo
                idx_subtwin_hydro = self.subhalo_dark_to_full_dict[idx_largestsub_dark]
                # This is that hydro subhalo's parent halo in the hydro sim
                idx_halo_hydro = self.subhalos_hydro['SubhaloGrNr'][idx_subtwin_hydro]

                if subhalo_mode=='most_massive_subhalo':
                    # This is the largest hydro subhalo of that hydro halo
                    idx_subhalomassive_hydro = self.halos_hydro['GroupFirstSub'][idx_halo_hydro]
                    idx_subhalo_hydro = idx_subhalomassive_hydro
                elif subhalo_mode=='twin_subhalo':
                    idx_subhalo_hydro = idx_subtwin_hydro
                else:
                    raise ValueError("Mode not recognized! (should not get here, there's an assert above)")
                    
                # if number of stars below a minimum, exclude
                if num_star_particles_min is not None and self.subhalos_hydro['SubhaloLenType'][:,self.ipart_star][idx_subhalo_hydro] < num_star_particles_min: 
                    continue

                # if halo is below a minimum mass, exclude
                if halo_mass_min is not None and self.halos_dark['GroupMass'][idx_halo_dark] < halo_mass_min: 
                    continue
                
                # if halo is above a maximum mass, exclude
                if halo_mass_max is not None and self.halos_dark['GroupMass'][idx_halo_dark] > halo_mass_max: 
                    continue

                # if dark halo and hydro halo masses differ significantly, likely a mismatch; exclude
                if halo_mass_difference_factor is not None and self.halos_hydro['GroupMass'][idx_halo_hydro] > halo_mass_difference_factor*self.halos_dark['GroupMass'][idx_halo_dark]: 
                    continue
                
                halo = DarkHalo(idx_halo_dark, self.base_path_dark, self.snap_num, self.box_size)
                halo.set_associated_halos(idx_largestsub_dark, idx_halo_hydro, idx_subhalo_hydro)

                dark_halo_arr.append(halo)

        if subsample_frac is not None:
            np.random.seed(42)
            dark_halo_arr = np.random.choice(dark_halo_arr, size=int(subsample_frac*len(dark_halo_arr)), replace=False)        
            
        self.dark_halo_arr = np.array(dark_halo_arr)
        self.N_halos = len(self.dark_halo_arr)

        # give each a random number
        rng = np.random.default_rng(seed=seed)
        random_ints = np.arange(self.N_halos)
        rng.shuffle(random_ints) #in-place
        for i in range(self.N_halos):
            dark_halo_arr[i].set_random_int(random_ints[i])


    def add_catalog_property_to_halos(self, property_name):
        for halo in self.dark_halo_arr:
            if property_name=='r200m':
                halo.set_catalog_property(property_name, self.halos_dark['Group_R_Mean200'][halo.idx_halo_dark])
            elif property_name=='mass_hydro_subhalo_star':
                halo.set_catalog_property(property_name, self.subhalos_hydro['SubhaloMassType'][:,self.ipart_star][halo.idx_subhalo_hydro])
            elif property_name=='m200m':
                halo.set_catalog_property(property_name, self.halos_dark['Group_M_Mean200'][halo.idx_halo_dark])
            elif property_name=='v200m':
                import astropy
                import astropy.constants as const
                import astropy.units as u
                mass_multiplier = 1e10
                G = const.G.to('(kpc * km**2)/(Msun * s**2)')
                # m200m really in Msun/h and r200m in ckpc/h; the h's cancel out, and the c is comoving meaning
                # we need a factor of the scale factor, but here at z=0 just 1. if go to diff z need to 
                # make sure to include!
                v_200m = np.sqrt(G * (mass_multiplier*halo.catalog_properties['m200m']*u.Msun) / (halo.catalog_properties['r200m']*u.kpc))
                halo.set_catalog_property(property_name, v_200m.value)
            elif property_name=='x_minPE':
                halo.set_catalog_property(property_name, self.subhalos_dark['SubhaloPos'][halo.idx_subhalo_dark])
            elif property_name=='x_minPE_hydro':
                halo.set_catalog_property(property_name, self.subhalos_hydro['SubhaloPos'][halo.idx_subhalo_hydro])
            elif property_name=='x_com':
                halo.set_catalog_property(property_name, self.halos_dark['GroupCM'][halo.idx_halo_dark])
            elif property_name=='sfr_hydro_subhalo_star':
                halo.set_catalog_property(property_name, self.subhalos_hydro['SubhaloSFR'][halo.idx_subhalo_hydro])
            elif property_name=='radius_hydro_subhalo_star':
                halo.set_catalog_property(property_name, self.subhalos_hydro['SubhaloHalfmassRadType'][:,self.ipart_star][halo.idx_subhalo_hydro])
            elif property_name=='subhalo_hydro_flag':
                halo.set_catalog_property(property_name, self.subhalos_hydro['SubhaloFlag'][halo.idx_subhalo_hydro])
            elif property_name=='mass_hydro_subhalo_gas':
                self.ipart_gas = il.snapshot.partTypeNum('gas') # 0
                halo.set_catalog_property(property_name, self.subhalos_hydro['SubhaloMassType'][:,self.ipart_gas][halo.idx_subhalo_hydro])
            elif property_name=='velocity_dispersion':
                halo.set_catalog_property(property_name, self.subhalos_hydro['SubhaloVelDisp'][halo.idx_subhalo_hydro])
            else:
                raise ValueError(f"Property name {property_name} not recognized!")


    def save_dark_halo_arr(self, fn_dark_halo_arr):
        np.save(fn_dark_halo_arr, self.dark_halo_arr)


    def load_dark_halo_arr(self, fn_dark_halo_arr):
        self.dark_halo_arr = np.load(fn_dark_halo_arr, allow_pickle=True)


    def get_structure_catalog_features(self, catalog_feature_names):

        self.x_catalog_features = []
        with h5py.File(f'{self.tng_path_dark}/postprocessing/halo_structure_{self.snap_num_str}.hdf5','r') as f:

            x_catalog_features_all = []
            for i, c_feat in enumerate(catalog_feature_names):
                x_catalog_features_all.append(f[c_feat])
            x_catalog_features_all = np.array(x_catalog_features_all).T
            idxs_halos_dark = np.array([dark_halo.idx_halo_dark for dark_halo in self.dark_halo_arr])
            self.x_catalog_features = x_catalog_features_all[idxs_halos_dark]

            # Delete halos with NaNs as any feature 
            idxs_nan_structure_catalog = np.argwhere(np.isnan(self.x_catalog_features).any(axis=1)).flatten()
            print(f"{len(idxs_nan_structure_catalog)} halos with NaN values of structure properties detected!")
            #self.x_catalog_features = np.delete(self.x_catalog_features, self.idxs_nan, axis=0)
            #self.halo_dicts = np.delete(self.halo_dicts, self.idxs_nan, axis=0)
            # TODO: for now, leaving power to delete with the notebook, not here;
            # if want a completely direct comparison, will have to build this in to halo selection
        
        return idxs_nan_structure_catalog
