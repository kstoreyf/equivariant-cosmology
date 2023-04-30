import copy
import itertools
import numpy as np
from collections import defaultdict
from astropy.table import Table

import utils
import geometric_features as gf
from geometric_features import GeometricFeature, geo_name



class ScalarFeaturizer:

    def __init__(self, tab_geos, tab_geo_info):

        self.geo_feature_arr = gf.geo_table_to_objects(tab_geos, tab_geo_info)
        self.idxs_halos_dark = tab_geos['idx_halo_dark']
        self.N_halos = len(self.geo_feature_arr)



    def featurize(self, m_order_max, 
                  x_order_max=np.inf, v_order_max=np.inf,
                  eigenvalues_not_trace=False,
                  elementary_scalars_only=False):

        self.m_order_max = m_order_max
        self.x_order_max = x_order_max
        self.v_order_max = v_order_max

        self.scalar_feature_arr = []
        self.scalar_features = []
        for geo_features_halo in self.geo_feature_arr:
            scalar_arr_i = self.get_scalar_features(geo_features_halo, m_order_max,
                                                    x_order_max, v_order_max,
                                                    eigenvalues_not_trace=eigenvalues_not_trace,
                                                    elementary_scalars_only=elementary_scalars_only)

            scalar_vals = np.array([s.value for s in scalar_arr_i])
            self.scalar_feature_arr.append(scalar_arr_i)
            self.scalar_features.append(scalar_vals)

        self.scalar_feature_arr = np.array(self.scalar_feature_arr, dtype=object)
        self.scalar_features = np.array(self.scalar_features)
        self.scalar_features, self.scalar_feature_arr = self.sort_scalar_features(
                                                        self.scalar_features, self.scalar_feature_arr)

        self.n_features = self.scalar_features.shape[1]


    def get_scalar_features(self, geometric_features, m_order_max, x_order_max, v_order_max,
                            eigenvalues_not_trace=False, elementary_scalars_only=False):

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
                scalar_features_single.append( ScalarFeature(value, 
                                                            [geo_name(g1, mode='multipole')],
                                                            [geo_name(g1, mode='readable')], 
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
                        scalar_features_single.append( ScalarFeature(value, 
                                                             [geo_name(g1, mode='multipole')],
                                                             [geo_name(g1, mode='readable')] , 
                                                             g1.m_order, g1.x_order, g1.v_order, [g1.n], operations) )
                else:
                    value = np.einsum('jj', g1.value)
                    operations = ['jj']
                    scalar_features_single.append( ScalarFeature(value, 
                                                             [geo_name(g1, mode='multipole')],
                                                             [geo_name(g1, mode='readable')],
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
                        scalar_features.append( ScalarFeature(value, 
                                                            [geo_name(g, mode='multipole') for g in [g1, g2]],
                                                            [geo_name(g, mode='readable') for g in [g1, g2]], 
                                                            m_order, x_order, v_order, ns, operations) )

        # Combine single-term features into two-term features
        if not elementary_scalars_only:
            for i_s in range(len(scalar_features_single)):
                s1 = scalar_features_single[i_s]
                for j_s in range(i_s, len(scalar_features_single)):
                    s2 = scalar_features_single[j_s]

                    value = s1.value * s2.value
                    geo_keys = np.concatenate((s1.geo_key, s2.geo_key))
                    geo_names = np.concatenate((s1.geo_name, s2.geo_name))
                    m_order = s1.m_order + s2.m_order
                    x_order = s1.x_order + s2.x_order
                    v_order = s1.v_order + s2.v_order
                    ns = np.concatenate((s1.ns, s2.ns))
                    
                    operations = np.concatenate((s1.operations, s2.operations))

                    scalar_features.append( ScalarFeature(value, geo_key, geo_name, 
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

        scalar_feature_table['name'] = [scalar_name(s, mode='readable') for s in scalar_features_single]
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

    

    def save_features(self, fn_scalar_features, save_format='table', overwrite=True):
        if save_format=='table':
            tab_scalars = scalar_objects_to_table(self.scalar_feature_arr, self.idxs_halos_dark)
            tab_scalars.write(fn_scalar_features, overwrite=overwrite, format='fits')
            print(f"Wrote table to {fn_scalar_features}")
        elif save_format=='numpy':
            np.save(fn_scalar_features, self.scalar_feature_arr)
        else:
            raise ValueError(f'Save format {save_format} not recognized!')

    def save_scalar_info(self, fn_scalar_info, overwrite=True):

        # get geos for first halo; same for all halos
        scalars = self.scalar_feature_arr[0]
        tab_scalar_info = scalars_to_info_table(scalars)
        tab_scalar_info.write(fn_scalar_info, overwrite=overwrite, format='fits')
        print(f"Wrote table to {fn_scalar_info}")
        
        #np.save(fn_geo_features, self.geo_feature_arr)
        return tab_scalar_info


    def load_features(self, fn_scalar_features, save_format='numpy'):
        if save_format=='fits':
            tab_scalars = utils.load_table(fn_scalar_features)
            return tab_scalars
        if save_format=='numpy':
            self.scalar_feature_arr = np.load(fn_scalar_features, allow_pickle=True)
            scalar_features = []
            for i in range(self.scalar_feature_arr.shape[0]):
                scalar_vals = np.array([s.value for s in self.scalar_feature_arr[i]])
                scalar_features.append(scalar_vals)
            self.scalar_features = np.array(scalar_features)
        else:
            raise ValueError(f'Save format {save_format} not recognized!')


class ScalarFeature:

    def __init__(self, value, geo_keys, geo_names, m_order, x_order, v_order, ns, 
                operations, modification=None):
        self.value = value
        self.geo_keys = geo_keys
        self.geo_names = geo_names
        self.m_order = m_order
        self.x_order = x_order
        self.v_order = v_order
        self.ns = ns
        self.operations = operations


def scalar_name(scalar_feature, mode='readable'):
    name_parts = []
    if mode=='readable':
        g_names = scalar_feature.geo_names
    elif mode=='multipole':
        g_names = scalar_feature.geo_keys
    else:
        raise KeyError("Mode not recgonized!")

    for i, g_name in enumerate(g_names):
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
    return name
    

### Data structure swappings

def scalar_table_to_objects(tab_scalars, tab_scalar_info):

    scalar_feature_arr = []
    # i indexes halo
    for i in range(len(tab_geos)):
        scalar_features_halo = []
        # j indexes scalar feature
        for j in range(len(tab_geo_info)):
            scalar_key = tab_scalar_info['scalar_key'][j]
            geo_keys = tab_scalar_info['geo_keys'][j]
            geo_keys = geo_keys[geo_keys!='']
            geo_names = tab_scalar_info['geo_names'][j]
            geo_names = geo_names[geo_names!='']
            ns = tab_scalar_info['ns'][j]
            ns = ns[~np.isnan(ns)]
            operations = tab_scalar_info['operations'][j]
            operations = operations[operations!='']
            scalar = ScalarFeature(value=tab_scalars[i][scalar_key], 
                                geo_keys=geo_keys,
                                geo_names=geo_names,
                                m_order=tab_scalar_info['m_order'][j], 
                                x_order=tab_scalar_info['x_order'][j], 
                                v_order=tab_scalar_info['v_order'][j], 
                                ns=ns,
                                operations=operations,
                                )
            scalar_features_halo.append(scalar)  
        scalar_feature_arr.append(scalar_features_halo)

    return scalar_feature_arr


def scalar_objects_to_table(scalar_feature_arr, idxs_halos_dark):

    # get geo names from first halo; should be same for all halos
    # these are the columns; number N_geos
    scalar_keys = [scalar_name(s, mode='multipole') for s in scalar_feature_arr[0]]
    # vals is a 2nd array of (N_halos, N_geos)
    scalar_vals = np.array([[s.value for s in scalars] for scalars in scalar_feature_arr])
    
    tab_scalars = Table()
    tab_scalars['idx_halo_dark'] = np.array(idxs_halos_dark)

    for j in range(scalar_vals.shape[1]):
        tab_scalars[scalar_keys[j]] = np.stack(scalar_vals[:,j])

    return tab_scalars


# input: single set of geo features
def scalars_to_info_table(scalars):

    scalar_keys = [scalar_name(s, mode='multipole') for s in scalars]        
    scalar_names = [scalar_name(g, mode='readable') for g in scalars]        
    geo_keys = [s.geo_keys for s in scalars]
    geo_names = [s.geo_names for s in scalars]
    m_orders = [s.m_order for s in scalars]
    x_orders = [s.x_order for s in scalars]
    v_orders = [s.v_order for s in scalars]
    ns = [s.ns for s in scalars]
    operations = [s.operations for s in scalars]

    # handle ragged arrays; assumes number of geos is consistent
    num_ns = [len(nvals) for nvals in ns]
    num_ns_max = np.max(num_ns)
    geo_keys_nonragged = np.full((len(ns), num_ns_max), '')
    geo_names_nonragged = np.full((len(ns), num_ns_max), '')
    ns_nonragged = np.full((len(ns), num_ns_max), np.nan)
    operations_nonragged = np.full((len(ns), num_ns_max), '')
    for i in range(len(ns)):
        ns_nonragged[i,:num_ns[i]] = ns[i]
    tab_scalar_info = Table([scalar_keys, scalar_names, 
                          geo_keys_nonragged, geo_names_nonragged, 
                          m_orders, x_orders, v_orders,
                          ns_nonragged, operations_nonragged],
                          names=('scalar_key', 'scalar_name', 'geo_key', 'geo_name', 
                          'm_order', 'x_order', 'v_order', 
                          'ns', 'operations'),
                        dtype=(str, str, str, str, int, int, int, int, str))
    return tab_scalar_info