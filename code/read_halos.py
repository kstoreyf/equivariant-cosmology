import h5py
import numpy as np
import os
import sys
from astropy.table import Table
import astropy.constants as const
import astropy.units as u

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

    def load_positions_and_velocities(self, shift=True, pos_center=None):
        halo_dark_dm = il.snapshot.loadHalo(self.base_path,self.snap_num,self.idx_halo_dark,'dm')
        x_data_halo = halo_dark_dm['Coordinates'] #c kpc/h
        v_data_halo = halo_dark_dm['Velocities']

        if shift:
            assert pos_center is not None, "Must pass pos_center to shift!"
            x_data_halo = self.shift_x(x_data_halo, pos_center)
            v_data_halo = self.shift_v(v_data_halo)

        return x_data_halo, v_data_halo

    # for now, masses of all particles are assumed to be same
    def shift_x(self, x_arr, pos_center):
        # Subtract off shift for each halo, wrapping around torus
        x_arr_shifted = self.shift_points_torus(x_arr, pos_center)
        return x_arr_shifted

    def shift_points_torus(self, points, shift):
        return (points - shift + 0.5*self.box_size) % self.box_size - 0.5*self.box_size

    # for now, masses of all particles is considered to be same
    def shift_v(self, v_arr):
        v_arr_com = np.mean(v_arr, axis=0)
        return v_arr - v_arr_com

    def set_catalog_property(self, property_name, value):
        self.catalog_properties[property_name] = value


    def compute_m200m_fof(self, r200m, pos_center, m_dmpart_dark):
        x_data_halo, v_data_halo = self.load_positions_and_velocities(shift=True, pos_center=pos_center)
        dists_from_center = np.linalg.norm(x_data_halo, axis=1)
        n_part_200m = np.sum(dists_from_center < r200m)
        m200m_fof = n_part_200m * m_dmpart_dark
        return m200m_fof


    def compute_mrv_200m_fof(self, density_mean, m_dmpart_dark, log_mass_shift, pos_center, r_max=None):
        number_density_mean = density_mean / m_dmpart_dark
        factor = 200

        x_data_halo, v_data_halo = self.load_positions_and_velocities(shift=True, pos_center=pos_center)

        dists_from_center = np.linalg.norm(x_data_halo, axis=1)
        x_rms = np.sqrt(np.mean(dists_from_center**2))
        if r_max is None:
            r_max = 2*x_rms
        r_min = 0.1*x_rms

        #print(r_max)
        #print(np.min(dists_from_center), np.max(dists_from_center))
        #print(x_rms)
        # Loop backwards because we expect it to be close to r200
        #x_bin_edges = np.arange(r_max, r_min, -0.01)
        x_bin_edges = np.linspace(r_max, r_min, 1000)
        for i in range(len(x_bin_edges)):
            n_part_inside_edge = np.sum(dists_from_center < x_bin_edges[i])
            vol = 4/3*np.pi*(x_bin_edges[i]**3)
            number_density = n_part_inside_edge / vol 
            #print(number_density)
            if number_density > factor*number_density_mean:
                #self.catalog_properties['r200mean'] = x_bin_edges[i]
                r200m_fof = x_bin_edges[i]
                #print(r200m_fof)
                #self.catalog_properties['m200mean'] = n_part_inside_edge * m_dmpart_dark
                m200m_fof = np.log10(n_part_inside_edge * m_dmpart_dark) + log_mass_shift
                break
        #print(i)
        if i==0 or i==len(x_bin_edges)-1:
            raise ValueError(f"Did not find R200 properly! (idx_dark_halo={self.idx_halo_dark}, i={i}), r_max={r_max}, r_min={r_min}, number_density={number_density}, ref num dens={factor*number_density_mean}")

        import astropy
        import astropy.constants as const
        import astropy.units as u
        G = const.G.to('(kpc * km**2)/(Msun * s**2)')
        # m200m really in Msun/h and r200m in ckpc/h; the h's cancel out, and the c is comoving meaning
        # we need a factor of the scale factor, but here at z=0 just 1. if go to diff z need to 
        # make sure to include!
        v200m_fof = np.sqrt(G * (10**m200m_fof*u.Msun) / (r200m_fof*u.kpc)).value
        return m200m_fof, r200m_fof, v200m_fof


    def compute_MXV_rms(self, center_mode, m_dmpart_dark):
        x_data_halo, v_data_halo = self.load_positions_and_velocities(shift=True, center_mode=center_mode)
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
    def __init__(self, base_dir, sim_name, sim_name_dark, snap_num_str):
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
        self.mass_multiplier = 1e10
        self.log_mass_shift = 10

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
                                      'SubhaloVelDisp', 'SubhaloBHMass', 'SubhaloSpin']
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
        self.ipart_gas = il.snapshot.partTypeNum('gas')
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
        print(f'{np.sum(i_with_SAM_match)}/{len(i_with_SAM_match)} halos have with SAM matches')
        return i_with_SAM_match



    def construct_halo_table(self, fn_halos, overwrite=False, N=None):

        # Construct main halo table, above a conservative mass cut for M200_mean
        # Only require that dark halos have >=1 subhalo, and a hydro halo match
        halo_logmass_min = 10

        print(f"Starting with all {len(self.halos_dark)} halos in {self.sim_name}")

        # Make mass cut
        i_select = self.log_m(self.halos_dark['Group_M_Mean200']) >= halo_logmass_min 
        print(f"After mass cut: N = {np.sum(i_select)}")

        # Cut out halos with no subhalos
        # (These are the indices of the most massive subhalo in the FOF halo; -1 means no subhalos)
        self.halos_dark['GroupFirstSub'] = self.halos_dark['GroupFirstSub'].astype('int32')
        i_has_subhalos = self.halos_dark['GroupFirstSub'] >= 0
        i_select = i_select & i_has_subhalos
        print(f"After no-subhalos cut: N = {np.sum(i_select)}")

        # Cut out halos with no match in hydro 
        i_has_hydro_match = np.isin(self.halos_dark['GroupFirstSub'], list(self.subhalo_dark_to_full_dict.keys()))
        i_select = i_select & i_has_hydro_match
        print(f"After no-hydro-match cut: N = {np.sum(i_select)}")

        # Get indices for halos and subhalos of selected objects
        # Note that idxs are the indices into the relevant info arrays! 
        idxs_halos_dark = self.idxs_halos_dark_all[i_select]
        idxs_subhalos_dark = self.halos_dark['GroupFirstSub'][i_select]

        # This is the index of the hydro subhalo that is the twin of the largest subhalo in the dark halo
        idxs_subtwins_hydro = [self.subhalo_dark_to_full_dict[idx] for idx in idxs_subhalos_dark]
        # This is that hydro subhalo's parent halo in the hydro sim
        idxs_halos_hydro = [self.subhalos_hydro['SubhaloGrNr'][idx] for idx in idxs_subtwins_hydro]

        tab_halos = Table([idxs_halos_dark, idxs_subhalos_dark, idxs_subtwins_hydro, idxs_halos_hydro], 
                    names=('idx_halo_dark', 'idx_subhalo_dark', 'idx_subhalo_hydro', 'idx_halo_hydro'))

        # for mini halo table, only for testing
        if N is not None:
            rng = np.random.default_rng(42)
            i_keep = rng.choice(np.arange(len(tab_halos)), size=N)
            tab_halos = tab_halos[i_keep]

        tab_halos.write(fn_halos, overwrite=overwrite)
        print(f"Wrote table to {fn_halos} with N={len(tab_halos)}")
        return tab_halos


    def log_m(self, m_tng_units):
        return np.log10(m_tng_units) + self.log_mass_shift


    def compute_velocity(self, mass_msunperh, radius_ckpcperh):
        # m200m in Msun/h and r200m in ckpc/h; the h's cancel out, and the c is comoving meaning
        # we need a factor of the scale factor, but here at z=0 just 1. if go to diff z need to 
        # make sure to include!
        G = const.G.to('(kpc * km**2)/(Msun * s**2)')
        velocity_kpcpers = np.sqrt(G * (mass_msunperh*u.Msun) / (radius_ckpcperh*u.kpc))
        return velocity_kpcpers.value

    def add_properties_dark(self, fn_halos, overwrite=True):

        print(f"Loading halo table {fn_halos}")
        tab_halos = Table.read(fn_halos)

        print("Adding dark halo & subhalo properties")
        idxs_halos_dark = tab_halos['idx_halo_dark']
        idxs_subhalos_dark = tab_halos['idx_subhalo_dark']

        ### M, R, V 200 mean; M200crit
        tab_halos['m200m'] = self.halos_dark['Group_M_Mean200'][idxs_halos_dark]
        tab_halos['r200m'] = self.halos_dark['Group_R_Mean200'][idxs_halos_dark]
        tab_halos['v200m'] = self.compute_velocity(self.mass_multiplier * self.halos_dark['Group_M_Mean200'][idxs_halos_dark], self.halos_dark['Group_R_Mean200'][idxs_halos_dark])
        tab_halos['m200c'] = self.halos_dark['Group_M_Crit200'][idxs_halos_dark]

        ### positions
        tab_halos['x_com'] = self.halos_dark['GroupCM'][idxs_halos_dark]
        tab_halos['x_grouppos'] = self.halos_dark['GroupCM'][idxs_halos_dark]
        tab_halos['x_minPE'] = self.subhalos_dark['SubhaloPos'][idxs_subhalos_dark]

        ### subhalo spin 
        spin_x, spin_y, spin_z = self.subhalos_dark['SubhaloSpin'][idxs_subhalos_dark].T
        tab_halos['spin_subhalo'] = np.sqrt(spin_x**2 + spin_y**2 + spin_z**2)

        ### velocity dispersion
        tab_halos['veldisp_subhalo'] = self.subhalos_dark['SubhaloVelDisp'][idxs_subhalos_dark]

        tab_halos.write(fn_halos, overwrite=overwrite)


    def add_properties_hydro(self, fn_halos, overwrite=True):

        print(f"Loading halo table {fn_halos}")
        tab_halos = Table.read(fn_halos)

        print("Adding hydro halo & subhalo properties")
        idxs_halos_hydro = tab_halos['idx_halo_hydro']
        idxs_subhalos_hydro = tab_halos['idx_subhalo_hydro']

        ### Masses, radii, number
        # TODO: what to do about zeros?? letting them fail for now, get -infs.
        tab_halos['m200m_hydro'] = self.halos_hydro['Group_M_Mean200'][idxs_halos_hydro]
        tab_halos['mstellar'] = self.subhalos_hydro['SubhaloMassType'][:,self.ipart_star][idxs_subhalos_hydro]
        tab_halos['rstellar'] = self.subhalos_hydro['SubhaloHalfmassRadType'][:,self.ipart_star][idxs_subhalos_hydro]
        tab_halos['mgas'] = self.subhalos_hydro['SubhaloMassType'][:,self.ipart_gas][idxs_subhalos_hydro]
        tab_halos['mbh'] = self.subhalos_hydro['SubhaloBHMass'][idxs_subhalos_hydro]
        # don't need to deal with 10^10 units bc they divide out
        tab_halos['mbh_per_mstellar'] = self.subhalos_hydro['SubhaloBHMass'][idxs_subhalos_hydro]/self.subhalos_hydro['SubhaloMassType'][:,self.ipart_star][idxs_subhalos_hydro]
        tab_halos['npartstellar'] = self.subhalos_hydro['SubhaloLenType'][:,self.ipart_star][idxs_subhalos_hydro]
        tab_halos['npartgas'] = self.subhalos_hydro['SubhaloLenType'][:,self.ipart_gas][idxs_subhalos_hydro]

        ### Star formation
        tab_halos['sfr'] = self.subhalos_hydro['SubhaloSFR'][idxs_subhalos_hydro]
        with h5py.File(f'{self.base_dir}/{self.sim_name}/postprocessing/star_formation_rates.hdf5', 'r') as f: 
            idxs_subhalos_hydro_sfrfile = f[f'Snapshot_{self.snap_num}']['SubfindID']
            sfrs1 = f[f'Snapshot_{self.snap_num}']['SFR_MsunPerYrs_in_all_1000Myrs']
            idx_subhalo_to_sfr1 = dict(zip(idxs_subhalos_hydro_sfrfile, sfrs1))
            tab_halos['sfr1'] = [idx_subhalo_to_sfr1[idx] if idx in idx_subhalo_to_sfr1 else np.nan for idx in idxs_subhalos_hydro]

        ### Photometry and colors
        # 2nd dim columns: sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, wfc_acs_f606w, des_y, jwst_f150w
        # 3rd dimension is viewing angles, just take first for now (0)
        phot_file = f'{self.tng_path_hydro}/postprocessing/stellar_photometry/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_{self.snap_num_str}.hdf5'
        f_phot = h5py.File(phot_file)
        phot = np.array(f_phot['Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc'])
        tab_halos['gband'] = phot[idxs_subhalos_hydro,1,0]
        tab_halos['gband_minus_iband'] = tab_halos['gband'] - phot[idxs_subhalos_hydro,3,0]

        ### Angular momentum, j_stellar
        fn_stellar = f'{self.tng_path_hydro}/postprocessing/circularities_aligned_allstars_L75n1820TNG099.hdf5'
        f_stellar = h5py.File(fn_stellar)
        j_stellar_all = np.array(f_stellar['SpecificAngMom']).flatten()
        tab_halos['jstellar'] = j_stellar_all[idxs_subhalos_hydro]

        tab_halos.write(fn_halos, overwrite=overwrite)



    def add_mv200m_fof_dark(self, fn_halos, overwrite=True):

        print(f"Loading halo table {fn_halos}")
        tab_halos = utils.load_table(fn_halos)

        print("Adding dark halo M & V 200_mean_fof properties")
        idxs_halos_dark = tab_halos['idx_halo_dark']

        m200m_fof = np.empty(len(idxs_halos_dark))
        v200m_fof = np.empty(len(idxs_halos_dark))

        # so can see how long will take better
        rng = np.random.default_rng(42)
        i_shuffle = np.arange(len(tab_halos))
        rng.shuffle(i_shuffle)
        #for i, idx_halo_dark in enumerate(idxs_halos_dark[i_shuffle]):
        count = 0
        for i in i_shuffle:
        #for i in range(len(idxs_halos_dark)-1, 0, -1):
            idx_halo_dark = idxs_halos_dark[i]
            halo = DarkHalo(idx_halo_dark, self.base_path_dark, self.snap_num, self.box_size)
            # units of 10^10 msun bc that's what m_dmpart_dark is in
            m200m_fof[i] = halo.compute_m200m_fof(tab_halos['r200m'][i], tab_halos['x_minPE'][i], self.m_dmpart_dark)
            if count % 1000 == 0:
                print("count", count, flush=True)
                print(tab_halos['m200m'][i], m200m_fof[i], tab_halos['r200m'][i], flush=True)
            count += 1

        tab_halos['m200m_fof'] = m200m_fof
        tab_halos['v200m_fof'] = self.compute_velocity(self.mass_multiplier * m200m_fof, tab_halos['r200m'])

        tab_halos.write(fn_halos, overwrite=overwrite)
        print(f"Wrote m200m_fof and v200m_fof to {fn_halos}")


    def add_properties_structure(self, fn_halos, overwrite=True):

        print(f"Loading halo table {fn_halos}")
        tab_halos = utils.load_table(fn_halos)

        catalog_feature_names = ['c200c', 'a_form', 'M200c']
        self.x_catalog_features = []
        with h5py.File(f'{self.tng_path_dark}/postprocessing/halo_structure_{self.snap_num_str}.hdf5','r') as f:
            x_catalog_features_all = []
            for i, name_feat in enumerate(catalog_feature_names):
                vals_feature = f[name_feat][:]
                # index the catalog features with the dark halo index (checked this by comparing M200c)
                name_table = name_feat
                if name_feat=='M200c':
                    name_table = 'log_M200c_Msun_structure'
                tab_halos[name_table] = vals_feature[tab_halos['idx_halo_dark']]

        tab_halos.write(fn_halos, overwrite=overwrite)
        print(f"Wrote structure properties to {fn_halos}")



    def transform_properties(self, fn_halos, overwrite=True):

        print(f"Loading halo table {fn_halos}")
        tab_halos = utils.load_table(fn_halos)
        
        names_mass = ['m200m', 'm200m_fof', 'm200m_hydro',
                           'mstellar', 'mgas']
        for name_mass in names_mass:
            tab_halos['log_'+name_mass] = self.log_m(tab_halos[name_mass])

        names_to_log = ['rstellar', 'r200m', 'jstellar']
        for name_to_log in names_to_log:
            tab_halos['log_'+name_to_log] = np.log10(tab_halos[name_to_log])

        # Note that the master table still includes some 
        # mstellar == 0, so will have infinities etc in here!

        # SFR -> ssfr, log and handle zeros
        names_sfr = ['sfr', 'sfr1']
        # sfr is in msun/yr
        # to estimate "zero": avg mass gas cell: 10^6 Msun, divided by 1 Gyr (longest sfr timescale) = 10^9 yr
        # 10^6 / 10^9 yr = 10^-3 Msun/yr
        sfr_zero = 1e-3 #msun/yr
        tol = 1e-10
        for name_sfr in names_sfr:
            i_zerosfr = np.abs(tab_halos[name_sfr])<tol
            sfr = tab_halos[name_sfr].copy()
            sfr[i_zerosfr] = sfr_zero
            tab_halos['log_s'+name_sfr] = self.log_sfr_to_log_ssfr(np.log10(sfr), tab_halos['mstellar']*self.mass_multiplier)

        # Black hole masses and ratios
        # make zero min/2, bc that's where might just hit resolution issues (aka rounding)
        i_zerombh = tab_halos['mbh']==0
        mbh_zero = np.min(tab_halos['mbh'])/2.0  
        mbh = tab_halos['mbh'].copy()
        mbh[i_zerombh] = mbh_zero
        tab_halos['log_mbh'] = self.log_m(mbh)
        # use the fixed-zero bh masses for mbh_per_mstellar
        tab_halos['log_mbh_per_mstellar'] = tab_halos['log_mbh'] - tab_halos['log_mstellar']


        tab_halos.write(fn_halos, overwrite=overwrite)


    # little h via https://www.tng-project.org/data/downloads/TNG100-1/
    def log_sfr_to_log_ssfr(self, log_sfr_arr, m_stellar_Msunperh_arr):
        h = 0.6774  
        m_stellar_Msun_arr = (m_stellar_Msunperh_arr)/h
        return log_sfr_arr - np.log10(m_stellar_Msun_arr)


    def select_halos(self, fn_halos, fn_select, 
                     num_star_particles_min=0, num_gas_particles_min=0, halo_logmass_min=None, 
                     halo_logmass_max=None, halo_mass_difference_factor=None,
                     must_have_SAM_match=True,
                     must_have_halo_structure_info=True, seed=42):

        if halo_logmass_min is None:
            halo_logmass_min = -np.inf
        if halo_logmass_max is None:
            halo_logmass_max = np.inf

        print(f"Loading halo table {fn_halos}")
        tab_halos = Table.read(fn_halos)

        i_select = np.full(len(tab_halos), True)

        i_Nstellar_abovemin = tab_halos['npartstellar'] >= num_star_particles_min
        i_select = i_select & i_Nstellar_abovemin

        i_mhalo_inbounds = (tab_halos['log_m200m_fof'] >= halo_logmass_min) & \
                           (tab_halos['log_m200m_fof'] < halo_logmass_max)
        i_select = i_select & i_mhalo_inbounds

        if halo_mass_difference_factor is not None:
            # cleaner way to do this?
            # here we don't use fof, because we don't have for hydro,
            # and it's just a way to get an idea of if there's a bad mismatch
            mhalo_dark = tab_halos['m200m']*self.mass_multiplier
            mhalo_hydro = tab_halos['m200m_hydro']*self.mass_multiplier
            i_mdiff_abovemin = (mhalo_dark/mhalo_hydro <  halo_mass_difference_factor) & \
                               (mhalo_dark/mhalo_hydro >  1/halo_mass_difference_factor) & \
                               (mhalo_hydro/mhalo_dark <  halo_mass_difference_factor) & \
                               (mhalo_hydro/mhalo_dark > 1/halo_mass_difference_factor)
            i_select = i_select & i_mdiff_abovemin

        if must_have_SAM_match:
            i_with_SAM_match = self.get_halos_with_SAM_match(tab_halos['idx_halo_dark'])
            i_select = i_select & i_with_SAM_match

        if must_have_halo_structure_info:
            i_with_halo_structure_info = self.get_halos_with_structure_info(tab_halos['idx_halo_dark'])
            i_select = i_select & i_with_halo_structure_info      

        N_halos = np.sum(i_select)
        print(f"Selecting N={N_halos}")

        # Give each halo a random number; will be useful later, e.g. for splitting train/test consistently
        rng = np.random.default_rng(seed)
        random_ints = np.arange(N_halos)
        rng.shuffle(random_ints) #in-place

        idxs_table = np.arange(len(tab_halos))

        tab_select = Table([tab_halos['idx_halo_dark'][i_select], 
                            idxs_table[i_select], 
                            random_ints], 
                    names=('idx_halo_dark', 'idx_table', 'rand_int'))
        tab_select.write(fn_select, overwrite=True)
        print(f"Wrote table to {fn_select} with N={len(tab_select)} halos")
        return tab_select


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

        if 'merger' in property_name:
            self.add_merger_info_to_halos_sublink(halo_tag)
            return

        if property_name=='Mofa':
            avals = utils.get_avals(self.dark_halo_arr)
            n_snapshots = len(avals)
            a2idx_dict = dict(zip(avals, range(n_snapshots)))

        if 'band' in property_name:
            phot_file = f'{self.tng_path_hydro}/postprocessing/stellar_photometry/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_{self.snap_num_str}.hdf5'
            f_phot = h5py.File(phot_file)
            phot = f_phot['Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc']

        if property_name=='j_stellar':
            fn_stellar = f'{self.tng_path_hydro}/postprocessing/circularities_aligned_allstars_L75n1820TNG099.hdf5'
            f_stellar = h5py.File(fn_stellar)
            j_stellar_all = np.array(f_stellar['SpecificAngMom']).flatten()

        if property_name=='m200mean' or property_name=='r200mean' or property_name=='v200mean':      
            mean_density_header = self.get_mean_density_from_header()

        # just do this here bc sets the values internally
        if property_name=='m200mean' or property_name=='r200mean' or property_name=='v200mean':     
            for halo in self.dark_halo_arr: 
                halo.compute_mrv_200m(mean_density_header, self.m_dmpart_dark, self.mass_multiplier, center_mode='x_minPE')
            return

        catalog_feature_names = ['M200c', 'c200c', 'a_form']
        if property_name in catalog_feature_names:
            self.get_structure_catalog_features([property_name])
            prop_vals = self.x_catalog_features[:,0] #only 1 feature so 2nd dim should be 1 
            for i, halo in enumerate(self.dark_halo_arr):
                halo.set_catalog_property(property_name, prop_vals[i])
            return


        for halo in self.dark_halo_arr:
            if 'merger' in property_name:
                print("here")
                # already done above loop
                continue
                #total_merger_count, merger_mass_ratio, major_merger_count = get_major_merger_count(f, index)
            
            elif property_name=='r200m':
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
            elif property_name=='bhmass':
                property_value = self.subhalos_hydro['SubhaloBHMass'][halo.idx_subhalo_hydro]
            elif property_name=='bhmass_per_mstellar':
                m_stellar = self.subhalos_hydro['SubhaloMassType'][:,self.ipart_star][halo.idx_subhalo_hydro]
                bhmass = self.subhalos_hydro['SubhaloBHMass'][halo.idx_subhalo_hydro]
                # doing this because these are both in 10^10 Msun units, but if we work in logspace 
                # and are doing their difference (==ratio) this factor doesn't matter!
                property_value = 10**(np.log10(bhmass) - np.log10(m_stellar)) if bhmass!=0 else 0
            elif property_name=='gband':
                # 2nd dim columns: sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, wfc_acs_f606w, des_y, jwst_f150w
                # 3rd dimension is viewing angles, just take first for now (0)
                # sdss_g is index 1
                property_value = phot[halo.idx_subhalo_hydro,1,0]
            elif property_name=='gband_minus_iband':
                # 2nd dim columns: sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, wfc_acs_f606w, des_y, jwst_f150w
                # 3rd dimension is viewing angles, just take first for now (0)
                # sdss_g is index 1, sdss_i is index 3
                property_value = phot[halo.idx_subhalo_hydro,1,0] - phot[halo.idx_subhalo_hydro,3,0]
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
                Mofa_arr = halo.get_Mofa(a2idx_dict)
                # remove a=1 because always Mfrac=1
                idx_a1 = a2idx_dict[1]
                Mofa_arr = np.delete(Mofa_arr, idx_a1)
                #property_value = Mofa_arr

                #avals_subset = [0.25, 0.5, 0.75]
                avals_subset = [0.25]
                idxs_subset = []
                for aval in avals_subset:
                    aval_closest, _ = utils.find_nearest(avals, aval)
                    idxs_subset.append( a2idx_dict[aval_closest] )

                property_value = Mofa_arr[idxs_subset]
            elif property_name=='j_stellar':
                property_value = j_stellar_all[halo.idx_subhalo_hydro]
            elif property_name=='veldisp_dm':
                property_value = self.subhalos_dark['SubhaloVelDisp'][halo.idx_subhalo_dark]
            elif property_name=='spin_dm':
                spin_x, spin_y, spin_z = self.subhalos_dark['SubhaloSpin'][halo.idx_subhalo_dark]
                property_value = np.sqrt(spin_x**2 + spin_y**2 + spin_z**2)
            else:
                raise ValueError(f"Property name {property_name} not recognized!")

            halo.set_catalog_property(property_name, property_value)
        return


    def save_dark_halo_arr(self, fn_dark_halo_arr):
        np.save(fn_dark_halo_arr, self.dark_halo_arr)


    def load_dark_halo_arr(self, fn_dark_halo_arr):
        self.dark_halo_arr = np.load(fn_dark_halo_arr, allow_pickle=True)


    def get_halos_with_structure_info(self, idxs_halo_dark):
        catalog_feature_names_all = ['c200c', 'a_form']
        with h5py.File(f'{self.tng_path_dark}/postprocessing/halo_structure_{self.snap_num_str}.hdf5','r') as f:
            x_catalog_features_all = []
            for i, c_feat in enumerate(catalog_feature_names_all):
                x_catalog_features_all.append(f[c_feat][:])
        x_catalog_features_all = np.array(x_catalog_features_all).T
        x_catalog_features = x_catalog_features_all[idxs_halo_dark]

        i_has_halo_structure_info = ~np.isnan(x_catalog_features).any(axis=1)
        print(f'{np.sum(i_has_halo_structure_info)}/{len(i_has_halo_structure_info)} halos have halo structure info')
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


    def add_MAH_to_halos_SAM(self, halo_tag, most_massive=True):
 
        mah_tag = '' if most_massive else '_allprogenitors' 
        fn_mah = f'../data/mahs/mahs_SAM_{self.sim_name}{halo_tag}{mah_tag}.npy'
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
                                fields=['HalopropRedshift', 'HalopropMvir'], most_massive=most_massive,
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


    def add_merger_info_to_halos_sublink(self, halo_tag):

        properties = ['num_mergers', 'num_major_mergers', 'ratio_last_major_merger']

        fn_merger = f'../data/merger_info/merger_info_{self.sim_name}{halo_tag}_mpb.npy'
        if os.path.exists(fn_merger):
            utils.load_merger_info(self.dark_halo_arr, fn_merger, properties=properties)
            return

        fields = ['SubhaloID','NextProgenitorID','MainLeafProgenitorID','FirstProgenitorID','SubhaloMassType']
        count = 0
        ratio = 1./3.
        for halo in self.dark_halo_arr:
            tree = il.sublink.loadTree(self.base_path_dark, self.snap_num, halo.idx_subhalo_dark,
                               fields=fields)
            numMergers = il.sublink.numMergers(tree,massPartType='dm')
            #numMajorMergers = il.sublink.numMergers(tree,minMassRatio=ratio,massPartType='dm')
            numMajorMergers = utils.num_mergers_mpb(tree,minMassRatio=ratio,massPartType='dm')
            ratioLastMajorMerger = utils.last_merger_ratio_mpb(tree,minMassRatio=ratio,massPartType='dm')
            if count % 1000 == 0:
                print(count)
                print(halo.idx_halo_dark, halo.idx_subhalo_dark)

            halo.set_catalog_property('num_mergers', numMergers)
            halo.set_catalog_property('num_major_mergers', numMajorMergers)
            halo.set_catalog_property('ratio_last_major_merger', ratioLastMajorMerger)
            count += 1

        utils.save_merger_info(self.dark_halo_arr, fn_merger, properties=properties)        


    def get_mean_density_from_mr200m(self):
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
