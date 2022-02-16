import numpy as np
import sys
import h5py

# shouldn't need this if have illustris_python properly in python path! todo: check if fixed upon reload
sys.path.insert(1, '/home/ksf293/external')
import illustris_python as il

import utils
import scalars



def run():
    # choose parameters
    save_tag = ''
    #r_edges = np.logspace(np.log10(1), np.log10(1000), 4) 
    r_edges = np.array([0, 100])
    print(r_edges)

    m_order_max = 1
    x_order_max = 0
    l_arr = scalars.get_needed_ls_scalars(m_order_max, x_order_max)

    featurizer = Featurizer(r_edges)
    featurizer.read_simulations()
    featurizer.match_twins()
    featurizer.select_halos()
    featurizer.add_info_to_halo_dicts()
    featurizer.compute_geometric_features(l_arr)
    featurizer.compute_scalar_features(m_order_max=m_order_max, x_order_max=x_order_max)
    featurizer.set_y_labels()

    fitter = Fitter(featurizer.x_scalar_features, featurizer.y_scalar, 
                    featurizer.x_scalar_dicts)
    fitter.split_train_test()
    fitter.scale_and_fit()
    fitter.predict()


# set up paths
class Featurizer:
        
    def __init__(self):

        self.tng_path_hydro = '/scratch/ksf293/gnn-cosmology/data/TNG50-4'
        self.tng_path_dark = '/scratch/ksf293/gnn-cosmology/data/TNG50-4-Dark'
        self.base_path_hydro = '/scratch/ksf293/gnn-cosmology/data/TNG50-4/output'
        self.base_path_dark = '/scratch/ksf293/gnn-cosmology/data/TNG50-4-Dark/output'
        self.snap_num_str = '099'
        self.snap_num = int(self.snap_num_str)


    def read_simulations(self):
        
        with h5py.File(f'{self.base_path_hydro}/snapdir_{self.snap_num_str}/snap_{self.snap_num_str}.0.hdf5','r') as f:
            header = dict( f['Header'].attrs.items() )
            self.m_dmpart = header['MassTable'][1] # this times 10^10 msun/h
            self.box_size = header['BoxSize'] # c kpc/h


        fields = ['SubhaloMass','SubhaloPos','SubhaloMassType', 'SubhaloLenType', 'SubhaloHalfmassRad', 'SubhaloGrNr']

        self.subhalos_hydro = il.groupcat.loadSubhalos(self.base_path_hydro,self.snap_num,fields=fields)
        self.halos_hydro = il.groupcat.loadHalos(self.base_path_hydro,self.snap_num)

        self.subhalos_dark = il.groupcat.loadSubhalos(self.base_path_dark,self.snap_num,fields=fields)
        self.halos_dark = il.groupcat.loadHalos(self.base_path_dark,self.snap_num)

        self.idxs_halos_hydro_all = np.array(list(range(self.halos_hydro['count'])))
        self.idxs_halos_dark_all = np.array(list(range(self.halos_dark['count'])))

        self.ipart_dm = il.snapshot.partTypeNum('dm') # 0
        self.ipart_star = il.snapshot.partTypeNum('stars') # 4


    def match_twins(self):
        # Load twin-matching file
        f = h5py.File(f'{self.tng_path_hydro}/postprocessing/subhalo_matching_to_dark.hdf5','r')
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

    
    def select_halos(self, num_star_particles_min=1, halo_mass_min=1e10, 
                     halo_mass_difference_factor=3.0):
        # GroupFirstSub: Index into the Subhalo table of the first/primary/most massive 
        # Subfind group within this FoF group. Note: This value is signed (or should be interpreted as signed)! 
        # In this case, a value of -1 indicates that this FoF group has no subhalos.
        self.halos_dark['GroupFirstSub'] = self.halos_dark['GroupFirstSub'].astype('int32')
        mask_has_subhalos = np.where(self.halos_dark['GroupFirstSub'] >= 0) # filter out halos with no subhalos

        idxs_halos_dark_withsubhalos = self.idxs_halos_dark_all[mask_has_subhalos]
        idxs_largestsubs_dark_all = self.halos_dark['GroupFirstSub'][mask_has_subhalos]

        halo_dicts = []
        for i, idx_halo_dark in enumerate(idxs_halos_dark_withsubhalos):
            
            idx_largestsub_dark = idxs_largestsubs_dark_all[i]
            if idx_largestsub_dark in self.subhalo_dark_to_full_dict:
                
                halo_dict = {}
                
                # This is the index of the hydro subhalo that is the twin of the largest subhalo in the dark halo
                idx_subtwin_hydro = self.subhalo_dark_to_full_dict[idx_largestsub_dark]
                # This is that hydro subhalo's parent halo in the hydro sim
                idx_halo_hydro = self.subhalos_hydro['SubhaloGrNr'][idx_subtwin_hydro]
                # This is the largest hydro subhalo of that hydro halo
                idx_subhalo_hydro = self.halos_hydro['GroupFirstSub'][idx_halo_hydro]

                # if no stars in this subhalo, exclude
                if self.subhalos_hydro['SubhaloLenType'][:,self.ipart_star][idx_subhalo_hydro] < num_star_particles_min: 
                    continue

                # if halo is below a minimum mass, exclude
                halo_mass_min /= 1e10 # because Mass is in units of 10^10 M_sun/h
                if self.halos_dark['GroupMass'][idx_halo_dark] < halo_mass_min: 
                    continue
                    
                # if dark halo and hydro halo masses differ significantly, likely a mismatch; exclude
                if self.halos_hydro['GroupMass'][idx_halo_hydro] > halo_mass_difference_factor*self.halos_dark['GroupMass'][idx_halo_dark]: 
                    continue
                
                halo_dict['idx_halo_dark'] = idx_halo_dark
                halo_dict['idx_subhalo_hydro'] = idx_subhalo_hydro
                halo_dict['idx_subhalo_dark'] = idx_largestsub_dark
                halo_dict['idx_halo_hydro'] = idx_halo_hydro
                
                halo_dicts.append(halo_dict)
                
        self.halo_dicts = np.array(halo_dicts)
        self.N_halos = len(halo_dicts)
        self.idx_halos_in_halodict = np.arange(self.N_halos)

    
    def add_info_to_halo_dicts(self):
        for i_hd, halo_dict in enumerate(self.halo_dicts):
            
            idx_halo_dark = halo_dict['idx_halo_dark']
            halo_dict['r_crit200_dark_halo'] = self.halos_dark['Group_R_Crit200'][idx_halo_dark]
            halo_dict['r_mean200_dark_halo'] = self.halos_dark['Group_R_Mean200'][idx_halo_dark]
            halo_dict['mass_crit200_dark_halo_dm'] = self.halos_dark['Group_M_Crit200'][idx_halo_dark]
            halo_dict['mass_mean200_dark_halo_dm'] = self.halos_dark['Group_M_Mean200'][idx_halo_dark]
            halo_dict['mass_dark_halo_dm'] = self.halos_dark['GroupMassType'][:,self.ipart_dm][idx_halo_dark]

            idx_halo_hydro = halo_dict['idx_halo_hydro']
            halo_dict['mass_hydro_halo_dm'] = self.halos_hydro['GroupMassType'][:,self.ipart_dm][idx_halo_hydro]
            halo_dict['mass_hydro_halo_star'] = self.halos_hydro['GroupMassType'][:,self.ipart_star][idx_halo_hydro]
            
            idx_subhalo_hydro = halo_dict['idx_subhalo_hydro']
            halo_dict['mass_hydro_subhalo_star'] = self.subhalos_hydro['SubhaloMassType'][:,self.ipart_star][idx_subhalo_hydro]
            
    
    def set_y_labels(self, y_scalar_feature_name='mass_hydro_halo_star'):
        self.y_scalar = np.empty(self.N_halos) # 1 mass dimension
        for i_hd, halo_dict in enumerate(self.halo_dicts):
            self.y_scalar[i_hd] = halo_dict[y_scalar_feature_name]

    
    def shift_points_torus(self, points, shift):
        return (points - shift + 0.5*self.box_size) % self.box_size - 0.5*self.box_size


    def compute_geometric_features(self, r_edges, l_arr, r_units='r200'):

        self.r_edges = np.array(r_edges).astype(float)
        n_rbins = len(r_edges) - 1
        self.n_arr = np.arange(n_rbins)
        self.r_units = r_units
        self.l_arr = np.array(l_arr)

        self.g_arrs_halos = []
        self.g_normed_arrs_halos = []

        for i_hd, halo_dict in enumerate(self.halo_dicts):
            idx_halo_dark = halo_dict['idx_halo_dark']
            halo_dark_dm = il.snapshot.loadHalo(self.base_path_dark,self.snap_num,idx_halo_dark,'dm')
            x_halo_dark_dm = halo_dark_dm['Coordinates']
            # particle0_pos is the first particle, choosing random one as proxy for pos of halo
            particle0_pos = x_halo_dark_dm[0]
            x_data_halo_shifted = self.shift_points_torus(x_halo_dark_dm, particle0_pos)

            x_halo_dark_dm_com = np.mean(x_data_halo_shifted, axis=0) + particle0_pos
            #print("com", x_halo_dark_dm_com)
            # Subtract off center of mass for each halo
            x_data_halo = self.shift_points_torus(x_halo_dark_dm, x_halo_dark_dm_com)

            if self.r_units=='r200':
                r_edges = self.r_edges * halo_dict['r_mean200_dark_halo']
            else:
                r_edges = self.r_edges
    
            g_arrs, g_normed_arrs = scalars.get_geometric_features(x_data_halo, r_edges, self.l_arr, self.n_arr, self.m_dmpart)
            self.g_arrs_halos.append(g_arrs)
            self.g_normed_arrs_halos.append(g_normed_arrs)


    def compute_scalar_features(self, m_order_max, x_order_max, feature_names_to_include_also=[]):
        
        self.x_scalar_dicts = np.empty(self.N_halos, dtype=object)
        self.x_scalar_features = []
        for i_hd in range(self.N_halos):
            scalar_dict_i = scalars.featurize_scalars(self.g_arrs_halos[i_hd], self.n_arr, m_order_max, x_order_max, l_arr=self.l_arr)
            scalars_i = []
            for key_name, scalar_ns in scalar_dict_i.items():
                for key_ns, scalar in scalar_ns.items():
                    if ((scalar['x_order'] <= x_order_max and scalar['m_order'] <= m_order_max) 
                            or key_name in feature_names_to_include_also):
                        scalars_i.append(scalar['value'])
            self.x_scalar_dicts[i_hd] = scalar_dict_i
            self.x_scalar_features.append(scalars_i)
        self.x_scalar_features = np.array(self.x_scalar_features)
        self.n_features = self.x_scalar_features.shape[1]



class Fitter:

    def __init__(self, x_scalar_features, y_scalar, x_scalar_dicts, 
                 y_val_current, uncertainties=None):
        assert x_scalar_features.shape[0]==y_scalar.shape[0], "Must have same number of x features and y labels!"
        assert x_scalar_features.shape[0]==len(x_scalar_dicts), "Must have same number of x features and feature dicts!"
        self.x_scalar_features = x_scalar_features
        self.y_scalar = y_scalar
        self.x_scalar_dicts = x_scalar_dicts
        self.N_halos = x_scalar_features.shape[0]
        #TODO: document
        self.y_val_current = y_val_current
        if uncertainties is None:
            uncertainties = np.ones(self.N_halos)
        self.uncertainties = uncertainties


    def scale_x_features(self, x_input):
        x = np.copy(x_input)
        if self.log_x:
           x = np.log10(x)
        return x


    def scale_y(self, y_input):
        y = np.copy(y_input)
        if self.log_y:
            y = np.log10(y)
        return y

    def unscale_y(self, y_input):
        y = np.copy(y_input)
        if self.log_y:
            y = 10**y
        return y



    def split_train_test(self, frac_test=0.2, seed=42):
        # split indices and then obtain training and test x and y, so can go back and get the full info later
        np.random.seed(seed)
        idx_traintest = np.arange(self.N_halos)
        self.idx_test = np.random.choice(idx_traintest, size=int(frac_test*self.N_halos), replace=False)
        self.idx_train = np.setdiff1d(idx_traintest, self.idx_test, assume_unique=True)

        # Split train and test arrays
        self.x_scalar_train = self.x_scalar_features[self.idx_train]
        self.x_scalar_test = self.x_scalar_features[self.idx_test]
        self.y_scalar_train = self.y_scalar[self.idx_train]
        self.y_scalar_test = self.y_scalar[self.idx_test]

        # Split lists of feature dicts
        self.x_scalar_dicts_train = self.x_scalar_dicts[self.idx_train]
        self.x_scalar_dicts_test = self.x_scalar_dicts[self.idx_test]
        self.uncertainties_train = self.uncertainties[self.idx_train]
        self.uncertainties_test = self.uncertainties[self.idx_test]

        self.y_val_current_train = self.y_val_current[self.idx_train]
        self.y_val_current_test = self.y_val_current[self.idx_test]

        self.n_train = len(self.x_scalar_train)
        self.n_test = len(self.x_scalar_test)
        self.n_x_features = self.x_scalar_features.shape[1]

        #print(f'n_train: {self.n_train}, n_test: {self.n_test}')
        #print(f'n_features: {self.n_features}')
        if self.n_x_features > self.n_train/2:
            print('WARNING!!! Number of features ({self.n_features}) is close to the number of training samples ({self.n_train})')


    def scale_y_values(self):
        self.y_scalar_train_scaled = self.scale_y(self.y_scalar_train)
        self.y_scalar_test_scaled = self.scale_y(self.y_scalar_test)
        self.uncertainties_train_scaled = self.scale_y(self.uncertainties_train)
        self.y_val_current_train_scaled = self.scale_y(self.y_val_current_train)
        self.y_val_current_test_scaled = self.scale_y(self.y_val_current_test)


    def construct_feature_matrix(self, x_features, y_current, training_mode=False):
        ones_feature = np.ones((x_features.shape[0], 1))
        y_current = np.atleast_2d(y_current).T
        A = np.concatenate((ones_feature, y_current, x_features), axis=1)
        if training_mode:
            self.x_scales = np.concatenate(([1.0, 1.0], self.x_scales))
            self.n_A_features = self.n_x_features + 2
        return A

    def scale_and_fit(self, rms_x=False, log_x=False, log_y=False, check_cond=False):
        self.log_x, self.log_y = log_x, log_y
        self.scale_y_values()
        self.x_scalar_train_scaled = self.scale_x_features(self.x_scalar_train)
        if rms_x:
            self.x_scales = np.sqrt(np.mean(self.x_scalar_train_scaled**2, axis=0))
        else:
            self.x_scales = np.ones(x.shape[0])

        self.A_train = self.construct_feature_matrix(self.x_scalar_train_scaled, self.y_val_current_train_scaled,
                                                     training_mode=True)

        # in this code, A=x_vals, diag(C_inv)=inverse_variances, Y=y_vals
        if check_cond:
            u, s, v = np.linalg.svd(self.A, full_matrices=False)
            print('x_vals condition number:',  np.max(s)/np.min(s))
        inverse_variances = 1/self.uncertainties_train_scaled**2
        AtCinvA = self.A_train.T @ (inverse_variances[:,None] * self.A_train)
        AtCinvY = self.A_train.T @ (inverse_variances * self.y_scalar_train_scaled)
        self.res_scalar = np.linalg.lstsq(AtCinvA, AtCinvY, rcond=None)

        self.theta_scalar = self.res_scalar[0]/self.x_scales
        # This is in units of the data as given, so does not include the mass_multiplier
        self.chi2 = np.sum((self.y_scalar_train_scaled - self.A_train*self.x_scales @ self.theta_scalar)**2 * inverse_variances)

        assert self.res_scalar[0].shape[0] == self.A_train.shape[1], 'Number of coefficients from theta vector should equal number of features!'

    
    def predict(self):
        #self.y_scalar_pred = self.x_scalar_test_scaled @ self.theta_scalar
        self.x_scalar_test_scaled = self.scale_x_features(self.x_scalar_test)
        self.A_test = self.construct_feature_matrix(self.x_scalar_test_scaled, self.y_val_current_test_scaled)
        # x_scales is already in theta_scalar, so we also need to multiply by x_scales here
        self.y_scalar_pred_scaled = self.A_test*self.x_scales @ self.theta_scalar
        self.y_scalar_pred = self.unscale_y(self.y_scalar_pred_scaled)

            
if __name__=='__main__':
    run()
