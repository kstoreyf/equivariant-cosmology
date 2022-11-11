import h5py
import numpy as np
import os
import sys

sys.path.insert(1, '/home/ksf293/external')
import illustris_python as il
import illustris_sam as ilsam 

import utils


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
        x_data_halo = halo_dark_dm['Coordinates'] #c kpc/h
        v_data_halo = halo_dark_dm['Velocities']

        if shift:
            center_options = ['x_com','x_minPE','x_grouppos']
            assert center in center_options, f"Center choice must be one of {center_options}!"
                 
            assert center in self.catalog_properties, f"Must first add center mode {center} to catalog property dict!"
            x_data_halo = self.shift_x(x_data_halo, center)
            v_data_halo = self.shift_v(v_data_halo)

        return x_data_halo, v_data_halo

    # for now, masses of all particles are assumed to be same
    def shift_x(self, x_arr, center):
        x_shift = self.catalog_properties[center]
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

    # don't actually use! was for checks
    def compute_mrv_200m(self, density_mean, m_dmpart_dark, mass_multiplier, center='x_com'):
        number_density_mean = density_mean / m_dmpart_dark
        factor = 200

        x_data_halo, v_data_halo = self.load_positions_and_velocities(shift=True, center=center)

        dists_from_center = np.linalg.norm(x_data_halo, axis=1)
        x_rms = np.sqrt(np.mean(dists_from_center**2))

        x_bin_edges = np.linspace(0.5*x_rms, 10*x_rms, 500)
        for i in range(len(x_bin_edges)):
            n_part_inside_edge = np.sum(dists_from_center < x_bin_edges[i])
            vol = 4/3*np.pi*(x_bin_edges[i]**3)
            number_density = n_part_inside_edge / vol 
            if number_density < 200*number_density_mean:
                self.catalog_properties['r200m'] = x_bin_edges[i]
                self.catalog_properties['m200m'] = n_part_inside_edge * m_dmpart_dark
                break

        import astropy
        import astropy.constants as const
        import astropy.units as u
        G = const.G.to('(kpc * km**2)/(Msun * s**2)')
        # m200m really in Msun/h and r200m in ckpc/h; the h's cancel out, and the c is comoving meaning
        # we need a factor of the scale factor, but here at z=0 just 1. if go to diff z need to 
        # make sure to include!
        self.catalog_properties['v200m'] = np.sqrt(G * (mass_multiplier*self.catalog_properties['m200m']*u.Msun) / (self.catalog_properties['r200m']*u.kpc)).value


    def compute_MXV_rms(self, center, m_dmpart_dark):
        x_data_halo, v_data_halo = self.load_positions_and_velocities(shift=True, center=center)
        dists_from_center = np.linalg.norm(x_data_halo, axis=1)
        self.X_rms = np.sqrt(np.mean(dists_from_center**2))
        n_part_in_X_rms = np.sum(dists_from_center < self.X_rms)
        self.M_rms = n_part_in_X_rms*m_dmpart_dark
        v_norms = np.linalg.norm(v_data_halo, axis=1)
        self.V_rms = np.sqrt(np.mean(v_norms**2))


    def get_a_mfrac(self, mfrac):
        if 'MAH' not in self.catalog_properties or len(self.catalog_properties['MAH'][0])==0:
            return 1.0 #?? 
        if 1.0 not in self.catalog_properties['MAH'][0]:
            return 1.0 # something wrong but it seems to happen
        a_vals = self.catalog_properties['MAH'][0]
        m_vals = self.catalog_properties['MAH'][1]
        i_a1 = np.where(a_vals==1.0)[0][0] #there should be exactly 1, but if not, eh take first
        m_a1 = m_vals[i_a1]
        a_mfrac_interp = utils.y_interpolated(m_vals/m_a1, a_vals, mfrac)
        if a_mfrac_interp < 0 or a_mfrac_interp > 1 or np.isnan(a_mfrac_interp): #something went horribly wrong! 
            #a_mfrac_interp = np.nan
            # pick the a closest to mfrac, even if its far away
            _, i_nearest = utils.find_nearest(m_vals/m_a1, mfrac)
            return a_vals[i_nearest]
        #if np.isnan(a_mfrac_interp):
        #    print("nan!")
        return a_mfrac_interp


    def get_Mofa(self, a2idx_dict):
        Ms = np.zeros(len(a2idx_dict))
        if 'MAH' not in self.catalog_properties or len(self.catalog_properties['MAH'][0])==0:
            return Ms
        if 1.0 not in self.catalog_properties['MAH'][0]:
            return Ms # something wrong but it seems to happen; need because getting M(a)/M(a=1)
        a_vals = self.catalog_properties['MAH'][0]
        m_vals = self.catalog_properties['MAH'][1]

        a2m_dict = dict(zip(a_vals, m_vals))
        M_a1 = a2m_dict[1.0]
        for i in range(len(a_vals)):
            idx = a2idx_dict[a_vals[i]]
            Ms[idx] = m_vals[i]/M_a1
        return Ms


class SimulationReader:

    # TODO: replace snap_num_str with proper zfill (i think? check works)
    def __init__(self, base_dir, sim_name, sim_name_dark, snap_num_str,
                 mass_multiplier=1e10):
        self.base_dir = base_dir
        self.sim_name = sim_name
        self.sim_name_dark = sim_name_dark
        self.tng_path_hydro = f'{base_dir}/{self.sim_name}'
        self.tng_path_dark = f'{base_dir}/{self.sim_name_dark}'
        self.base_path_hydro = f'{base_dir}/{self.sim_name}/output'
        self.base_path_dark = f'{base_dir}/{self.sim_name_dark}/output'
        self.snap_num_str = snap_num_str
        self.snap_num = int(self.snap_num_str)
        self.halo_arr = []
        self.mass_multiplier = mass_multiplier

        with h5py.File(f'{self.base_path_dark}/snapdir_{self.snap_num_str}/snap_{self.snap_num_str}.0.hdf5','r') as f:
            header = dict( f['Header'].attrs.items() )
            self.m_dmpart_dark = header['MassTable'][1] # this times 10^10 msun/h
            self.box_size = header['BoxSize'] # c kpc/h

        with h5py.File(f'{self.base_path_hydro}/snapdir_{self.snap_num_str}/snap_{self.snap_num_str}.0.hdf5','r') as f:
            header = dict( f['Header'].attrs.items() )
            self.m_dmpart_hydro = header['MassTable'][1] # this times 10^10 msun/h
            # box size same in dark and hydro

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


    def get_halos_with_SAM_match(self, idxs_halo_dark):
        self.base_path_sam = f'{self.base_dir}/{self.sim_name_dark}_SCSAM'
        subvolume_list = self.gen_subvolumes_SAM()

        fields = ['HalopropIndex']
        matches = True
        halos_sam = ilsam.groupcat.load_snapshot_halos(self.base_path_sam, self.snap_num, subvolume_list, fields, matches)
        idxs_halo_dark_SAM = halos_sam['HalopropFoFIndex_DM']
        i_with_SAM_match = np.in1d(idxs_halo_dark, idxs_halo_dark_SAM)
        print(f'Keeping {np.sum(i_with_SAM_match)}/{len(i_with_SAM_match)} halos with SAM matches')
        return i_with_SAM_match
        

    def select_halos(self, num_star_particles_min, halo_logmass_min, 
                     halo_logmass_max, halo_mass_difference_factor, subsample_frac,
                     subhalo_mode='most_massive', must_have_SAM_match=True,
                     must_have_halo_structure_info=True, seed=42):

        subhalo_mode_options = ['most_massive_subhalo', 'twin_subhalo']
        assert subhalo_mode in subhalo_mode_options, f"Input subhalo_mode {subhalo_mode} not an \
                                                      option; choose one of {subhalo_mode_options}"
        # These are the indices of the most massive subhalo in the FOF halo; -1 means no subhalos, filter these out
        self.halos_dark['GroupFirstSub'] = self.halos_dark['GroupFirstSub'].astype('int32')
        mask_has_subhalos = np.where(self.halos_dark['GroupFirstSub'] >= 0) # filter out halos with no subhalos
        idxs_halos_dark_withsubhalos = self.idxs_halos_dark_all[mask_has_subhalos]
        idxs_largestsubs_dark_all = self.halos_dark['GroupFirstSub'][mask_has_subhalos]
        
        halo_mass_min, halo_mass_max = None, None
        if halo_logmass_min is not None:
            halo_mass_min = 10**halo_logmass_min
            halo_mass_min /= self.mass_multiplier # because masses in catalog have units of 10^10 M_sun/h
        if halo_logmass_max is not None:
            halo_mass_max = 10**halo_logmass_max
            halo_mass_max /= self.mass_multiplier # because masses in catalog have units of 10^10 M_sun/h

        # For each dark halo that has a subhalo, get its most massive subhalo, 
        # and then check if that dark subhalo has a twin in the hydro sim.
        # (Note: I could have gone through the twin dict, but that includes non-most-massive subhalos.) 
        dark_halo_arr = []
        for i, idx_halo_dark in enumerate(idxs_halos_dark_withsubhalos):
            
            idx_largestsub_dark = idxs_largestsubs_dark_all[i]
            if idx_largestsub_dark not in self.subhalo_dark_to_full_dict:
                continue
                
            # This is the index of the hydro subhalo that is the twin of the largest subhalo in the dark halo
            idx_subtwin_hydro = self.subhalo_dark_to_full_dict[idx_largestsub_dark]
            # This is that hydro subhalo's parent halo in the hydro sim
            idx_halo_hydro = self.subhalos_hydro['SubhaloGrNr'][idx_subtwin_hydro]

            if subhalo_mode=='most_massive_subhalo':
                # This is the largest hydro subhalo of that hydro halo
                idx_subhalomassive_hydro = self.halos_hydro['GroupFirstSub'][idx_halo_hydro]
                idx_subhalo_hydro = idx_subhalomassive_hydro
            elif subhalo_mode=='twin_subhalo':
                # This is just the twin
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
            
            # Construct halo, keep track of all the indices
            halo = DarkHalo(idx_halo_dark, self.base_path_dark, self.snap_num, self.box_size)
            halo.set_associated_halos(idx_largestsub_dark, idx_halo_hydro, idx_subhalo_hydro)
            dark_halo_arr.append(halo)

        rng = np.random.default_rng(seed=seed)
        dark_halo_arr = np.array(dark_halo_arr)

        if must_have_SAM_match:
            idxs_halo_dark = [halo.idx_halo_dark for halo in dark_halo_arr]
            i_with_SAM_match = self.get_halos_with_SAM_match(idxs_halo_dark)
            dark_halo_arr = dark_halo_arr[i_with_SAM_match]

        if must_have_halo_structure_info:
            idxs_halo_dark = [halo.idx_halo_dark for halo in dark_halo_arr]
            i_with_halo_structure_info = self.has_halo_structure_info(idxs_halo_dark)
            dark_halo_arr = dark_halo_arr[i_with_halo_structure_info]

        # Subsample the dark halos if we want (for testing purposes)
        if subsample_frac is not None:
            dark_halo_arr = rng.choice(dark_halo_arr, size=int(subsample_frac*len(dark_halo_arr)), replace=False)        
            
        self.dark_halo_arr = dark_halo_arr
        self.N_halos = len(self.dark_halo_arr)

        # Give each halo a random number; will be useful later, e.g. for splitting train/test consistently
        random_ints = np.arange(self.N_halos)
        rng.shuffle(random_ints) #in-place
        for i in range(self.N_halos):
            dark_halo_arr[i].set_random_int(random_ints[i])

        print(f'Selected {self.N_halos}')


    def add_catalog_property_to_halos(self, property_name, halo_tag=None):
        if property_name=='sfr_hydro_subhalo_1Gyr':
            with h5py.File(f'{self.base_dir}/{self.sim_name}/postprocessing/star_formation_rates.hdf5', 'r') as f: 
                idxs_subhalo_hydro = f[f'Snapshot_{self.snap_num}']['SubfindID']
                sfrs1 = f[f'Snapshot_{self.snap_num}']['SFR_MsunPerYrs_in_all_1000Myrs']
                idx_subhalo_to_sfr1 = dict(zip(idxs_subhalo_hydro, sfrs1))
                for halo in self.dark_halo_arr:
                    halo.set_catalog_property(property_name, idx_subhalo_to_sfr1[halo.idx_subhalo_hydro])
            return

        if property_name.startswith('a_mfrac') or property_name=='Mofa':
            self.add_MAH_to_halos_SAM(halo_tag)

        if property_name=='Mofa':
            avals = utils.get_avals(self.dark_halo_arr)
            n_snapshots = len(avals)
            a2idx_dict = dict(zip(avals, range(n_snapshots)))


        for halo in self.dark_halo_arr:
            # if property_name=='m200m' or         
            #     halo.compute_mrv_200m(mean_density_header, sim_reader.m_dmpart_dark, sim_reader.mass_multiplier, center=center_halo)
            if property_name=='r200m':
                property_value = self.halos_dark['Group_R_Mean200'][halo.idx_halo_dark]
            elif property_name=='mass_hydro_subhalo_star':
                property_value = self.subhalos_hydro['SubhaloMassType'][:,self.ipart_star][halo.idx_subhalo_hydro]
            elif property_name=='m200m':
                property_value = self.halos_dark['Group_M_Mean200'][halo.idx_halo_dark]
            elif property_name=='v200m':
                import astropy
                import astropy.constants as const
                import astropy.units as u
                G = const.G.to('(kpc * km**2)/(Msun * s**2)')
                # m200m really in Msun/h and r200m in ckpc/h; the h's cancel out, and the c is comoving meaning
                # we need a factor of the scale factor, but here at z=0 just 1. if go to diff z need to 
                # make sure to include!
                v_200m = np.sqrt(G * (self.mass_multiplier*halo.catalog_properties['m200m']*u.Msun) / (halo.catalog_properties['r200m']*u.kpc))
                property_value = v_200m.value
            elif property_name=='x_minPE':
                # this should generally be the same as x_grouppos, bc most bound particle definition (double check)
                property_value = self.subhalos_dark['SubhaloPos'][halo.idx_subhalo_dark]
            elif property_name=='x_minPE_hydro':
                property_value = self.subhalos_hydro['SubhaloPos'][halo.idx_subhalo_hydro]
            elif property_name=='x_com':
                property_value = self.halos_dark['GroupCM'][halo.idx_halo_dark]
            elif property_name=='x_grouppos':
                property_value = self.halos_dark['GroupPos'][halo.idx_halo_dark]
            elif property_name=='sfr_hydro_subhalo_star':
                property_value = self.subhalos_hydro['SubhaloSFR'][halo.idx_subhalo_hydro]
            elif property_name=='radius_hydro_subhalo_star':
                property_value = self.subhalos_hydro['SubhaloHalfmassRadType'][:,self.ipart_star][halo.idx_subhalo_hydro]
            elif property_name=='subhalo_hydro_flag':
                property_value = self.subhalos_hydro['SubhaloFlag'][halo.idx_subhalo_hydro]
            elif property_name=='mass_hydro_subhalo_gas':
                self.ipart_gas = il.snapshot.partTypeNum('gas') # 0
                property_value = self.subhalos_hydro['SubhaloMassType'][:,self.ipart_gas][halo.idx_subhalo_hydro]
            elif property_name=='velocity_dispersion':
                property_value = self.subhalos_hydro['SubhaloVelDisp'][halo.idx_subhalo_hydro]
            elif property_name.startswith('a_mfrac_n'):
                n = int(property_name.split('_n')[-1])
                mfrac_vals = utils.get_mfrac_vals(n)
                property_value = []
                for mfrac in mfrac_vals:
                    property_value.append(halo.get_a_mfrac(float(mfrac)))
            elif property_name.startswith('a_mfrac'):
                mfrac = property_name.split('_')[-1]
                property_value = halo.get_a_mfrac(float(mfrac))
            elif property_name=='Mofa':
                property_value = halo.get_Mofa(a2idx_dict)
            else:
                raise ValueError(f"Property name {property_name} not recognized!")

            halo.set_catalog_property(property_name, property_value)
        return


    def save_dark_halo_arr(self, fn_dark_halo_arr):
        np.save(fn_dark_halo_arr, self.dark_halo_arr)


    def load_dark_halo_arr(self, fn_dark_halo_arr):
        self.dark_halo_arr = np.load(fn_dark_halo_arr, allow_pickle=True)


    def has_halo_structure_info(self, idxs_halo_dark):
        catalog_feature_names_all = ['M200c', 'c200c', 'a_form']
        with h5py.File(f'{self.tng_path_dark}/postprocessing/halo_structure_{self.snap_num_str}.hdf5','r') as f:
            x_catalog_features_all = []
            for i, c_feat in enumerate(catalog_feature_names_all):
                x_catalog_features_all.append(f[c_feat][:])
        x_catalog_features_all = np.array(x_catalog_features_all).T
        x_catalog_features = x_catalog_features_all[idxs_halo_dark]

        i_has_halo_structure_info = ~np.isnan(x_catalog_features).any(axis=1)
        print(f'Keeping {np.sum(i_has_halo_structure_info)}/{len(i_has_halo_structure_info)} halos with SAM matches')
        return i_has_halo_structure_info


    def get_structure_catalog_features(self, catalog_feature_names):
        self.x_catalog_features = []
        with h5py.File(f'{self.tng_path_dark}/postprocessing/halo_structure_{self.snap_num_str}.hdf5','r') as f:
            x_catalog_features_all = []
            for i, c_feat in enumerate(catalog_feature_names):
                x_catalog_features_all.append(f[c_feat][:])
        x_catalog_features_all = np.array(x_catalog_features_all).T
        idxs_halos_dark = np.array([dark_halo.idx_halo_dark for dark_halo in self.dark_halo_arr])
        self.x_catalog_features = x_catalog_features_all[idxs_halos_dark]
        idxs_nan_structure_catalog = np.argwhere(np.isnan(self.x_catalog_features).any(axis=1)).flatten()
        assert len(idxs_nan_structure_catalog)==0, "Halos with NaN values of structure properties detected!"


    def gen_subvolumes_SAM(self, n=5):
        subvolume_list = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    subvolume_list.append([i, j, k])
        return subvolume_list
        

    def get_Mvir_SAM(self):
        self.base_path_sam = f'{self.base_dir}/{self.sim_name_dark}_SCSAM'
        subvolume_list = self.gen_subvolumes_SAM()

        fields = ['HalopropIndex', 'HalopropMvir']
        matches = True #??
        halos_sam = ilsam.groupcat.load_snapshot_halos(self.base_path_sam, self.snap_num, subvolume_list, fields, matches)
        halo_idx_to_Mvir_dict = dict(zip(halos_sam['HalopropFoFIndex_DM'], halos_sam['HalopropMvir']))
        mvirs = [halo_idx_to_Mvir_dict[halo.idx_halo_dark] if halo.idx_halo_dark in halo_idx_to_Mvir_dict else -1 for halo in self.dark_halo_arr ]
        return mvirs


    def add_MAH_to_halos_SAM(self, halo_tag):
 
        fn_mah = f'../data/mahs/mahs_SAM_{self.sim_name}{halo_tag}.npy'
        if os.path.exists(fn_mah):
            utils.load_mah(self.dark_halo_arr, fn_mah)
            return
    
        self.base_path_sam = f'{self.base_dir}/{self.sim_name_dark}_SCSAM'
        subvolume_list = self.gen_subvolumes_SAM()

        # should be HalopropFoFIndex_DM
        # 'HalopropIndex' ? 'HalopropFoFIndex_DM' ?? (included in former)
        fields = ['HalopropIndex', 'HalopropRootHaloID']
        #fields = ['HalopropIndex_Snapshot', 'HalopropRootHaloID']
        matches = True #??
        halos_sam = ilsam.groupcat.load_snapshot_halos(self.base_path_sam, self.snap_num, subvolume_list, fields, matches)
        halo_idx_to_root_idx_dict = dict(zip(halos_sam['HalopropFoFIndex_DM'], halos_sam['HalopropRootHaloID']))
        
        # halo_idxs = np.array([halo.idx_halo_dark for halo in self.dark_halo_arr])
        # in1d = np.in1d(halo_idxs, halos_sam['HalopropFoFIndex_DM'])
        # print(len(halo_idxs))
        # print(len(halo_idxs[in1d]))

        for halo in self.dark_halo_arr:
            # should be haloprop or galprop??
            if halo.idx_halo_dark not in halo_idx_to_root_idx_dict: 
                halo.set_catalog_property('MAH', np.nan)
                continue

            root_idx = halo_idx_to_root_idx_dict[halo.idx_halo_dark]
            #print(halo.idx_halo_dark, root_idx)

            mtree = ilsam.merger.load_tree_haloprop(self.base_path_sam, root_idx, 
                                fields=['HalopropRedshift', 'HalopropMvir'], most_massive=True,
                                matches=True)
            scale_factors = 1/(1+mtree[root_idx]['HalopropRedshift'])
            halo.set_catalog_property('MAH', [scale_factors, mtree[root_idx]['HalopropMvir']])

        utils.save_mah(self.dark_halo_arr, fn_mah)        


    def add_MAH_to_halos_sublink(self):

        # via https://www.tng-project.org/data/forum/topic/369/snapshots-and-redshifts/
        fn_snaps_redshifts = f'../tables/snapnums_redshifts.dat'
        if os.path.exists(fn_snaps_redshifts):
            snap_num_list, redshift_list = np.loadtxt(fn_snaps_redshifts, usecols=(0,1), unpack=True)
            snap_to_redshift_dict = dict(zip(snap_num_list, redshift_list))
        else:
            raise ValueError(f"Snap num redshift file {fn_snaps_redshifts} doesn't exist!")

        fields = ['SubhaloMass','SubfindID','SnapNum']
        count = 0
        for halo in self.dark_halo_arr:
            mtree = il.sublink.loadTree(self.base_path_dark, self.snap_num, halo.idx_subhalo_dark,
                                        fields=fields, onlyMPB=True)
            redshifts = np.array([snap_to_redshift_dict[i_snap] for i_snap in mtree['SnapNum']])
            scale_factors = 1/(1+redshifts)
            
            property_name = 'MAH'                            
            # this is just the subhalo mass - do i want to sum over the subhalos in the
            # fof group to get the halo mass accretion history?
            if count % 1000 == 0:
                print(count)
                print(halo.idx_halo_dark, halo.idx_subhalo_dark)
                #print(scale_factors)
                #print(mtree['SubhaloMass'])
            halo.set_catalog_property(property_name, [scale_factors, mtree['SubhaloMass']])
            count += 1


    def get_mean_density_from_mr200m(self):
        center = 'x_grouppos'
        factor = 200
        mean_densities = []
        for halo in self.dark_halo_arr:
            m_200m = halo.catalog_properties['m200m']
            vol = 4/3*np.pi*(halo.catalog_properties['r200m']**3)
            density_in_r200m = m_200m/vol
            mean_dens = density_in_r200m / factor
            mean_densities.append(mean_dens) # mean of the calculated mean densities for all halos
        return np.mean(mean_densities)


    def get_mean_density_from_header(self):
        print(f'{self.base_path_dark}/snapdir_{self.snap_num_str}/snap_{self.snap_num_str}.0.hdf5')
        # This seems to give wrong value for TNG100-1?? Says Npart = 1200**3, should be 1820**3! 
        # TNG50-4 seems right tho
        # with h5py.File(f'{self.base_path_dark}/snapdir_{self.snap_num_str}/snap_{self.snap_num_str}.0.hdf5','r') as f:
        #     header = dict( f['Header'].attrs.items() )
        #     n_part_dm_dark = header['NumPart_Total'][1] 
        #     print(header['NumPart_Total'])
        if self.sim_name=='TNG100-1':
            n_part_dm_dark = 1820**3
        elif self.sim_name=='TNG50-4':
            n_part_dm_dark = 270**3
        else:
            raise ValueError('Sim not recognized!')
        number_density_mean = n_part_dm_dark/(self.box_size**3)
        print(n_part_dm_dark, n_part_dm_dark**(1/3), self.box_size, number_density_mean)
        mean_density = number_density_mean*self.m_dmpart_dark
        return mean_density