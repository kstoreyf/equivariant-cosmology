import copy
import itertools
import numpy as np
from collections import defaultdict

import utils
from geometric_features import GeometricFeature, geo_name



class ScalarFeaturizer:

    def __init__(self, geo_feature_arr=None):
        self.geo_feature_arr_orig = geo_feature_arr
        # must use deepcopy because our array has bpjects! np.copy doesn't work
        self.geo_feature_arr = copy.deepcopy(geo_feature_arr)
        self.N_halos = len(self.geo_feature_arr)


    def featurize(self, m_order_max, 
                  x_order_max=np.inf, v_order_max=np.inf,
                  eigenvalues_not_trace=False):

        self.m_order_max = m_order_max
        self.x_order_max = x_order_max
        self.v_order_max = v_order_max

        self.scalar_feature_arr = []
        self.scalar_features = []
        for geo_features_halo in self.geo_feature_arr:
            scalar_arr_i = self.get_scalar_features(geo_features_halo, m_order_max,
                                                    x_order_max, v_order_max,
                                                    eigenvalues_not_trace=eigenvalues_not_trace)

            scalar_vals = np.array([s.value for s in scalar_arr_i])
            self.scalar_feature_arr.append(scalar_arr_i)
            self.scalar_features.append(scalar_vals)
            # print("BREAKING for now")
            # break

        self.scalar_feature_arr = np.array(self.scalar_feature_arr, dtype=object)
        self.scalar_features = np.array(self.scalar_features)
        self.scalar_features, self.scalar_feature_arr = self.sort_scalar_features(
                                                        self.scalar_features, self.scalar_feature_arr)
        # for s in self.scalar_feature_arr[0]:
        #     print(scalar_name(s, self.geo_feature_arr))
        self.n_features = self.scalar_features.shape[1]


    def get_scalar_features(self, geometric_features, m_order_max, x_order_max, v_order_max,
                            eigenvalues_not_trace=False):

        assert m_order_max >= 1 and m_order_max <= 2, "m_order_max must be 1 or 2 (for now!)"

        N_geo = len(geometric_features)

        scalar_features = []
        scalar_features_single = []

        for i_geo_term in range(N_geo):
            g1 = geometric_features[i_geo_term]
            if g1.x_order > x_order_max or g1.v_order > v_order_max:
                continue
            
            ### Single term features

            # Scalar: raw value
            if g1.x_order + g1.v_order == 0:
                value = g1.value
                operations = ['']
                scalar_features_single.append( ScalarFeature(value, [i_geo_term], 
                                                             g1.m_order, g1.x_order, g1.v_order, [g1.n], operations) )

            # Tensor: contraction OR eigenvalues
            # Can only do these operations if g1 hermitian, bc need real eigenalues, and trace of 
            # antisymmetric matrix is 0 (took difference)
            elif g1.x_order + g1.v_order == 2 and g1.hermitian:
                if eigenvalues_not_trace:
                    eigenvalues = np.linalg.eigvalsh(g1.value) # returns in ascending order
                    for i_e, value in enumerate(eigenvalues):
                        eig_num = 3 - i_e  # this makes lambda_1 = max, lambda_3 = min
                        operations = [f'\lambda_{eig_num}']
                        scalar_features_single.append( ScalarFeature(value, [i_geo_term], 
                                                             g1.m_order, g1.x_order, g1.v_order, [g1.n], operations) )
                else:
                    value = np.einsum('jj', g1.value)
                    operations = ['jj']
                    scalar_features_single.append( ScalarFeature(value, [i_geo_term], 
                                                             g1.m_order, g1.x_order, g1.v_order, [g1.n], operations) )
                
        if m_order_max >= 2:
            ### Two-term features
            # this includes self-pairs but not duplicate permutations
            for i_geo_term in range(N_geo):
                g1 = geometric_features[i_geo_term]
                for j_geo_term in range(i_geo_term, N_geo):
                    g2 = geometric_features[j_geo_term]
                    value = None

                    x_order = g1.x_order + g2.x_order
                    v_order = g1.v_order + g2.v_order
                    if x_order > x_order_max or v_order > v_order_max:
                        continue

                    # 2 vectors: inner product
                    if g1.x_order + g1.v_order == 1 and g2.x_order + g2.v_order == 1:
                        value = np.einsum('j,j', g1.value, g2.value)
                        operations = ['j','j']

                    # 2 tensors or 2 pseudotensors: full contraction
                    elif g1.x_order + g1.v_order == 2 and g2.x_order + g2.v_order == 2:
                        if g1.hermitian == g2.hermitian:
                            if not g1.hermitian or not g2.hermitian:
                                msg = 'Non-hermitian terms not antisymmetrized! Did you forget to transform_pseudotensors()?'
                                assert g1.modification=='antisymmetrized' and g2.modification=='antisymmetrized', msg
                            value = np.einsum('jk,jk', g1.value, g2.value)
                            operations = ['jk','jk']
                            
                    # Get scalar feature properties and add to list
                    # only add to list if we assigned it a value, aka it matched one of the criteria above
                    if value is not None:
                        m_order = g1.m_order + g2.m_order
                        ns = [g1.n, g2.n]
                        scalar_features.append( ScalarFeature(value, [i_geo_term, j_geo_term], 
                                                            m_order, x_order, v_order, ns, operations) )

        # Combine single-term features into two-term features
        for i_s in range(len(scalar_features_single)):
            s1 = scalar_features_single[i_s]
            for j_s in range(i_s, len(scalar_features_single)):
                s2 = scalar_features_single[j_s]

                value = s1.value * s2.value
                idxs_geo_terms = np.concatenate((s1.idxs_geo_terms, s2.idxs_geo_terms))
                m_order = s1.m_order + s2.m_order
                x_order = s1.x_order + s2.x_order
                v_order = s1.v_order + s2.v_order
                ns = np.concatenate((s1.ns, s2.ns))
                
                operations = np.concatenate((s1.operations, s2.operations))

                scalar_features.append( ScalarFeature(value, idxs_geo_terms, 
                                                        m_order, x_order, v_order, ns, operations) )

        # Add in single-term features on their own
        scalar_features.extend(scalar_features_single)

        # order is handled in another function, so doesn't matter here
        return scalar_features        


    def sort_scalar_features(self, scalar_features, scalar_feature_arr):
        scalar_features_single = scalar_feature_arr[0] # list of features for single halo

        dtypes = [('name', str), ('m_order', int), ('x_order', int), ('v_order', int)]
        n_col_names = [f'n_{i}' for i in range(self.m_order_max)]
        dtypes.extend([(n_col_name, int) for n_col_name in n_col_names])
        scalar_feature_table = np.empty(len(scalar_features_single), dtype=dtypes)

        scalar_feature_table['name'] = [scalar_name(s, self.geo_feature_arr) for s in scalar_features_single]
        scalar_feature_table['m_order'] = [s.m_order for s in scalar_features_single]
        scalar_feature_table['x_order'] = [s.x_order for s in scalar_features_single]
        scalar_feature_table['v_order'] = [s.v_order for s in scalar_features_single]

        for i in range(self.m_order_max):
            # -1 is #magic bc cant use nans for int structured array
            scalar_feature_table[f'n_{i}'] = [s.ns[i] if len(s.ns) > i else -1 for s in scalar_features_single]
        order = ['m_order','x_order','v_order'] + n_col_names + ['name']
        i_sort = np.argsort(scalar_feature_table, order=order)
        return scalar_features[:,i_sort], scalar_feature_arr[:,i_sort]


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
            assert len(x_tensors)==1, "Should be exactly one second order tensor for x!"
            x_tensor_traces[i_g] = np.einsum('jj', x_tensors[0])
        self.X_rms = np.sqrt( x_tensor_traces / self.M_tot )


    def compute_V_rms(self, geo_feature_arr_onebin):
        v_tensor_traces = np.empty(self.N_halos)
        # rms is sqrt(mean(v^2)), and tr(g_02n/M_tot) = sum(v^2) (this einsum is the trace)
        for i_g, geo_features_halo in enumerate(self.geo_feature_arr):
            v_tensors = np.array([g.value for g in geo_feature_arr_onebin[i_g] if g.x_order==0 and g.v_order==2])
            assert len(v_tensors)==1, "Should be exactly one second order tensor for v!"
            v_tensor_traces[i_g] = np.einsum('jj', v_tensors[0])
        self.V_rms = np.sqrt( v_tensor_traces / self.M_tot )


    # does rescaling in-place in self.geo_feature_arr!
    def rescale_geometric_features(self, Ms, Rs, Vs):
        
        for i_g, geo_features_halo in enumerate(self.geo_feature_arr):
            for geo_feat in geo_features_halo:
                geo_feat.value /= Ms[i_g] # all geometric features have single m term
                for _ in range(geo_feat.x_order):
                    geo_feat.value /= Rs[i_g]
                for _ in range(geo_feat.v_order):
                    geo_feat.value /= Vs[i_g]
    

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

    def __init__(self, value, idxs_geo_terms, m_order, x_order, v_order, ns, 
                operations, modification=None):
        self.value = value
        self.idxs_geo_terms = idxs_geo_terms
        self.m_order = m_order
        self.x_order = x_order
        self.v_order = v_order
        self.ns = ns
        self.operations = operations


def scalar_name(scalar_feature, geo_feature_arr, mode='readable'):
    name_parts = []
    for i, idx_geo_term in enumerate(scalar_feature.idxs_geo_terms):
        # the 0 just gets the first halo, should all be same features
        g = geo_feature_arr[0][idx_geo_term] 
        g_name = geo_name(g, mode=mode)
        if scalar_feature.operations[i]=='':
            name_parts.append(f'{g_name}')
        elif scalar_feature.operations[i]=='j':
            name_parts.append(f'[{g_name}]_{{j}}')
        elif scalar_feature.operations[i]=='jk':
            name_parts.append(f'[{g_name}]_{{jk}}')
        elif scalar_feature.operations[i]=='jj':
            name_parts.append(f'[{g_name}]_{{jk}}')
        elif 'lambda' in scalar_feature.operations[i]:
            name_parts.append(f'{scalar_feature.operations[i]}\\left({g_name}\\right)')
        
    name = ' \, '.join(name_parts)
    return '$'+name+'$'
    
