import copy
import itertools
import numpy as np
from collections import defaultdict

import utils
from geometric_features import GeometricFeature



class ScalarFeaturizer:

    def __init__(self, geo_feature_arr=None):
        self.geo_feature_arr = geo_feature_arr
        #self.geo_feature_arr_orig = geo_feature_arr
        # must use deepcopy because our array has opjects! np.copy doesn't work
        #self.geo_feature_arr = copy.deepcopy(geo_feature_arr)
        self.N_halos = len(self.geo_feature_arr)


    def featurize(self, m_order_max, 
                  x_order_max=np.inf, v_order_max=np.inf,
                  eigenvalues_not_trace=False):

        self.scalar_feature_arr = []
        self.scalar_features = []
        for geo_features_halo in self.geo_feature_arr:
            scalar_arr_i = self.get_scalar_features(geo_features_halo, m_order_max,
                                                    x_order_max, v_order_max,
                                                    eigenvalues_not_trace=eigenvalues_not_trace)
            scalar_vals = np.array([s.value for s in scalar_arr_i])
            self.scalar_feature_arr.append(scalar_arr_i)
            self.scalar_features.append(scalar_vals)

        self.scalar_features = np.array(self.scalar_features)
        self.n_features = self.scalar_features.shape[1]


    def get_scalar_features(self, geometric_features, m_order_max, x_order_max, v_order_max,
                            eigenvalues_not_trace=False):
        scalar_features = []
        num_terms = np.arange(1, m_order_max+1)
        # Get all combinations of geometric features with m_order_max terms or fewer
        for nt in num_terms:
            geo_term_combos = list(itertools.combinations_with_replacement(geometric_features, nt))
            for geo_terms in geo_term_combos:
                s_features = self.make_scalar_feature(list(geo_terms), m_order_max,
                                        x_order_max, v_order_max,
                                        eigenvalues_not_trace=eigenvalues_not_trace)
                if s_features != -1:
                    scalar_features.extend(s_features)
        return scalar_features


    def compute_MXV_from_features(self):
        # to see what n's we have, get a set of them for just one halo (should all be same)
        ns_all = list(set([g.n for g in self.geo_feature_arr[0]]))
        geo_feature_arr_onebin = utils.rebin_geometric_features(self.geo_feature_arr, [ns_all])
        self.compute_M_tot(geo_feature_arr_onebin)
        self.compute_X_rms(geo_feature_arr_onebin)
        self.compute_V_rms(geo_feature_arr_onebin)


    def compute_M_tot(self, geo_feature_arr_onebin):
        self.M_tot = np.empty(self.N_halos)
        for i_g, geo_features_halo in enumerate(self.geo_feature_arr):
            self.M_tot[i_g] = np.sum([g.value for g in geo_feature_arr_onebin[i_g] if g.x_order==0 and g.v_order==0])


    def compute_X_rms(self, geo_feature_arr_onebin):
        x_tensor_traces = np.empty(self.N_halos)
        # rms is sqrt(mean(x^2)), and tr(g_20n/M_tot) = sum(x^2) (this einsum is the trace)
        for i_g, geo_features_halo in enumerate(self.geo_feature_arr):
            x_tensors = np.array([g.value for g in geo_feature_arr_onebin[i_g] if g.x_order==2 and g.v_order==0])
            assert len(x_tensors)==1, "Should be exactly one second order tensor for each x and v!"
            x_tensor_traces[i_g] = np.einsum('jj', x_tensors[0])
        self.X_rms = np.sqrt( x_tensor_traces / self.M_tot )


    def compute_V_rms(self, geo_feature_arr_onebin):
        v_tensor_traces = np.empty(self.N_halos)
        # rms is sqrt(mean(v^2)), and tr(g_02n/M_tot) = sum(v^2) (this einsum is the trace)
        for i_g, geo_features_halo in enumerate(self.geo_feature_arr):
            v_tensors = np.array([g.value for g in geo_feature_arr_onebin[i_g] if g.x_order==0 and g.v_order==2])
            assert len(v_tensors)==1, "Should be exactly one second order tensor for each x and v!"
            v_tensor_traces[i_g] = np.einsum('jj', v_tensors[0])
        self.V_rms = np.sqrt( v_tensor_traces / self.M_tot )


    # does rescaling in-place in self.geo_feature_arr!
    def rescale_geometric_features(self, Ms, Xs, Vs):
        
        for i_g, geo_features_halo in enumerate(self.geo_feature_arr):
            for geo_feat in geo_features_halo:
                geo_feat.value /= Ms[i_g] # all have single m term
                for _ in range(geo_feat.x_order):
                    geo_feat.value /= Xs[i_g]
                for _ in range(geo_feat.v_order):
                    geo_feat.value /= Vs[i_g]


    def make_scalar_feature(self, geo_terms, m_order_max, x_order_max, v_order_max, 
                            eigenvalues_not_trace=False):

        #print("GEO TERMS:", [g.to_string() for g in geo_terms])

        # should these orders be degree?
        x_order_tot = np.sum([g.x_order for g in geo_terms])
        v_order_tot = np.sum([g.v_order for g in geo_terms])
        xv_order_tot = x_order_tot + v_order_tot
        xv_orders_allowed = [0, 2, 4] # allowing two-tensor terms, but not odd-order!
        # i think if the first case is satisfied the others will be too, but keeping to be safe
        if xv_order_tot not in xv_orders_allowed or x_order_tot > x_order_max or v_order_tot > v_order_max:
            return -1
                
        subcombo_table = []
        geo_vals_contracted = []
        geo_vals_from_tensors = []
        names_contracted = []
        names_tensor = []

        # construct symmetric and antisymmetric tensors
        geo_terms_xv_antisymm = []
        geo_vals_xv_antisymm = []
        for i_g, g in enumerate(geo_terms):
            if not g.hermitian:
                # g = x(outer)v, so g^T = v(outer)x
                xv_symm = 0.5*(g.value + g.value.T)
                xv_antisymm =  0.5*(g.value - g.value.T)
                geo_vals_xv_antisymm.append(xv_antisymm)
                geo_terms_xv_antisymm.append(g)
                # replace g value with its symmetric value
                geo_symm = GeometricFeature(xv_symm, m_order=g.m_order, x_order=g.x_order, v_order=g.v_order, n=g.n, hermitian=True)
                geo_terms[i_g] = geo_symm
        # for antisymmetric tensors, only think we can do is multiply them with each other to recover the symmetry
        if len(geo_terms_xv_antisymm)==2:
            name = f'[A({geo_terms_xv_antisymm[0].to_string()})]_jk [A({geo_terms_xv_antisymm[1].to_string()})]_jk'
            value = np.einsum('jk,jk', *geo_vals_xv_antisymm)
            subcombo_table.append([name, value, geo_terms_xv_antisymm])
                    
        # multi t terms
        # two-vector terms
        geo_terms_vec = [g for g in geo_terms if g.x_order+g.v_order==1]
        assert len(geo_terms_vec) <= 2, "not going above 3 terms so shouldnt have more than 2 vector terms for scalars!"
        # t10n t10n', t01n t01n', t10n t01n'
        if len(geo_terms_vec)==2:
            # vector & vector: take inner product
            geo_vals_vec = [g.value for g in geo_terms_vec]
            #geo_vals_contracted.append((np.einsum('j,j', *geo_vals_vec), 2))
            #names_contracted.append( f'[{geo_terms_vec[0].to_string()}]_j [{geo_terms_vec[1].to_string()}]_j' )
            name = f'[{geo_terms_vec[0].to_string()}]_j [{geo_terms_vec[1].to_string()}]_j'
            value = np.einsum('j,j', *geo_vals_vec)
            subcombo_table.append([name, value, geo_terms_vec])

        # two tensor terms
        # shouldn't be any non-hermitian left in here, but making sure!
        geo_terms_tensor = [g for g in geo_terms if g.x_order+g.v_order==2 and g.hermitian] 
        assert len(geo_terms_tensor) <= 2, "not going above 4th order so shouldnt have more than 2 tensor terms!"
        if len(geo_terms_tensor)==2:
            geo_vals_tensor = [g.value for g in geo_terms_tensor]
            # tensor and tensor: do contraction
            # not adding to geo_vals_from_tensors bc those get multiplied in everywhere 
            # (only single-tensor contraction or eigenvalues)
            # (but shouldnt matter when limiting to two terms)
            #geo_vals_contracted.append((np.einsum('jk,jk', *geo_vals_tensor), 2))
            #names_contracted.append( f'[{geo_terms_tensor[0].to_string()}]_jk [{geo_terms_tensor[1].to_string()}]_jk' )
            name = f'[{geo_terms_tensor[0].to_string()}]_jk [{geo_terms_tensor[1].to_string()}]_jk'
            value = np.einsum('jk,jk', *geo_vals_tensor)
            subcombo_table.append([name, value, geo_terms_tensor])

        # single t terms
        for g in geo_terms:
            xv_order = g.x_order + g.v_order
            if xv_order==0:
                # t00n
                # geo_vals_contracted.append((g.value, 1))
                # names_contracted.append(g.to_string())
                name = g.to_string()
                subcombo_table.append([name, g.value, [g]])
            elif xv_order==2:
                # t20n, t02n, t11n
                # if g is hermitian, include trace OR eigenvalues
                if g.hermitian:
                    if eigenvalues_not_trace:
                        # eigenvalues
                        eigenvalues = np.linalg.eigvalsh(g.value) # returns in ascending order
                        #geo_vals_from_tensors.extend(list(zip(eigenvalues, np.ones(len(eigenvalues)))))
                        #names_tensor.extend( [f'e{eig_num}({g.to_string()})' for eig_num in range(len(eigenvalues))] )
                        for i, eigenvalue in enumerate(eigenvalues):
                            eig_num = 3 - i  # this makes lambda_1 = max, lambda_3 = min
                            name = f'e{eig_num}({g.to_string()})'
                            subcombo_table.append([name, eigenvalue, [g]])
                    else:
                        # trace
                        #geo_vals_from_tensors.append((np.einsum('jj', g.value), 1))
                        #names_tensor.append( f'[{g.to_string()}]_jj' )
                        name = f'[{g.to_string()}]_jj'
                        value = np.einsum('jj', g.value)
                        subcombo_table.append([name, value, [g]])

        subcombo_table = np.array(subcombo_table, dtype=object)
        n_geo_terms = len(geo_terms)
        geo_terms_names = [g.to_string() for g in geo_terms]
        geo_term_names_set = set(geo_terms_names)

        s_features = []
        num_subcombos = np.arange(1, m_order_max+1) #+1 to include m_order_max; this is max possible subcombos
        # Get all combinations of geometric features with m_order_max terms or fewer
        for nsc in num_subcombos:
            idx_subcombos = np.arange(len(subcombo_table))
            idx_lists_combos = list(itertools.combinations(idx_subcombos, nsc)) # no replacement!

            for idx_list in idx_lists_combos:
                subcombo = subcombo_table[idx_list,:]
                subcombo = np.atleast_2d(subcombo)
                subcombo_geo_terms = subcombo[:,2]
                subcombo_geo_terms_flat = [g for g_subcombo in subcombo_geo_terms for g in g_subcombo]

                # # Only if each original geo_term appears in subcombo exactly once does it make a valid term
                if len(subcombo_geo_terms_flat)==n_geo_terms:
                    subcombo_geo_term_names = [g.to_string() for g in subcombo_geo_terms_flat]
                    if set(subcombo_geo_term_names)==geo_term_names_set: 
                        subcombo_vals = subcombo[:,1]
                        value = np.product(subcombo_vals)
                        name = ' '.join(subcombo[:,0])
                        #print("including:", name)
                        s_features.append(ScalarFeature(value, subcombo_geo_terms_flat, name=name))
            
        # geo_val_product = np.product(geo_vals_contracted)
        # name_product = ' '.join(names_contracted)
        # # if no vectors or tensors, still need to add feature with rest of values
        # if not geo_vals_from_tensors:
        #     geo_vals_from_tensors.append(1.0)
        #     names_tensor.append('')
        # for i, g_tensor in enumerate(geo_vals_from_tensors):
        #     value = g_tensor * geo_val_product
        #     name = f'{name_product} {names_tensor[i]}' 
        #     s_features.append(ScalarFeature(value, geo_terms, name=name))

        return s_features
    

    def save_features(self, fn_scalar_features):
        np.save(fn_scalar_features, self.scalar_feature_arr)


    def load_features(self, fn_scalar_features):
        self.scalar_feature_arr = np.load(fn_scalar_features, allow_pickle=True)
        scalar_features = []
        for i in range(self.scalar_feature_arr.shape[0]):
            scalar_vals = np.array([s.value for s in self.scalar_feature_arr[i]])
            scalar_features.append(scalar_vals)
        self.scalar_features = np.array(scalar_features)



class ScalarFeature:

    def __init__(self, value, geo_terms, name=''):
        self.value = value
        self.geo_terms = geo_terms
        self.m_order = np.sum([g.m_order for g in self.geo_terms])
        self.x_order = np.sum([g.x_order for g in self.geo_terms])
        self.v_order = np.sum([g.v_order for g in self.geo_terms])
        self.name = name

    def to_string(self):
        if self.name:
            return self.name
        else:
            return ' '.join(np.array([g.to_string() for g in self.geo_terms]))
            

        