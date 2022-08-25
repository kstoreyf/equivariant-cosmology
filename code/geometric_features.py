import numpy as np
from collections import defaultdict
import itertools


class GeometricFeaturizer:

    def featurize(self, sim_reader, r_edges, x_order_max, v_order_max, 
                  center_halo='x_minPE', r_units='r200m'):
        
        r_edges = np.array(r_edges)
        self.sim_reader = sim_reader
        self.geo_feature_arr = []

        l_arr = np.arange(x_order_max+1)
        p_arr = np.arange(v_order_max+1)

        if r_units is not None:
            print(f"Adding property {r_units} for radial bins")
            self.sim_reader.load_sim_dark_halos()
            self.sim_reader.add_catalog_property_to_halos(r_units)

        print("Computing geometric features for all dark halos")
        for dark_halo in self.sim_reader.dark_halo_arr:
            x_halo, v_halo = dark_halo.load_positions_and_velocities(shift=True, center=center_halo)

            r_edges_scaled = r_edges
            if r_units is not None:
                r_edges_scaled = r_edges * dark_halo.catalog_properties[r_units]
            self.geo_feature_arr.append(self.get_geometric_features(x_halo, v_halo, 
                                r_edges_scaled, l_arr, p_arr))


    def get_geometric_features(self, delta_x_data_halo, delta_v_data_halo, r_edges, l_arr, p_arr, n_dim=3):

        # input checks
        N, n_dim_from_data = delta_x_data_halo.shape
        assert n_dim_from_data == n_dim, f"Number of dimensions in data {n_dim_from_data} does not equal global n_dim, {n_dim}!"
        assert delta_x_data_halo.shape==delta_v_data_halo.shape, "Position and velocity arrays should have same shape!"

        # set up bin indices
        n_rbins = len(r_edges) - 1
        n_arr = np.arange(n_rbins)

        # get mass info
        m_dm = self.sim_reader.m_dmpart
        m_total = N*m_dm
        
        # vectorized for all particles N
        rs = np.linalg.norm(delta_x_data_halo, axis=1)
        assert len(rs) == N, "rs should have length N!"
        window_vals = window(n_arr, rs, r_edges, n_rbins)

        # compute x and v outer product terms
        # this is pretty memory-hungry, but is faster because we need them multiple times
        x_outers_arr = []
        v_outers_arr = []
        for l in l_arr:
            x_outers = vector_outer_product_power(l, delta_x_data_halo)
            x_outers_arr.append(x_outers)
        for p in p_arr:
            v_outers = vector_outer_product_power(p, delta_v_data_halo)
            v_outers_arr.append(v_outers)

        geo_features_halo = []
        for i_l, l in enumerate(l_arr):
            for i_p, p in enumerate(p_arr):

                if l+p>2:
                    continue

                for i_n, n in enumerate(n_arr):
                    
                    if l==1 and p==1:
                        hermitian = False
                        xv_terms = vector_outer_product(x_outers_arr[i_l], v_outers_arr[i_p])
                    else:
                        hermitian = True
                        xv_terms = multiply_vector_terms(x_outers_arr[i_l], v_outers_arr[i_p])

                    g_lpn = np.sum( multiply_vector_terms(window_vals[i_n], xv_terms), axis=0) * m_dm

                    geo = GeometricFeature(g_lpn, m_order=1, x_order=l, v_order=p, n=n, hermitian=hermitian)
                    geo_features_halo.append(geo)  

        return geo_features_halo


    def save_features(self, fn_geo_features):
        np.save(fn_geo_features, self.geo_feature_arr)


    def load_features(self, fn_geo_features):
        self.geo_feature_arr = np.load(fn_geo_features, allow_pickle=True)


class GeometricFeature:

    def __init__(self, value, m_order, x_order, v_order, n, 
                 hermitian=True, modification=None):
        self.value = value
        self.m_order = m_order
        self.x_order = x_order
        self.v_order = v_order
        self.n = n
        self.hermitian = hermitian
        self.modification = modification


# should be called multiply_arbitrary_dim_terms ?
def multiply_vector_terms(v1, v2):
    v1_copy = np.copy(v1)
    v2_copy = np.copy(v2)
    if v1_copy.ndim < v2_copy.ndim:
        for _ in np.arange(1, v2_copy.ndim):
            v1_copy = v1_copy[:,None]
    elif v1_copy.ndim > v2_copy.ndim:
        for _ in np.arange(1, v1_copy.ndim):
            v2_copy = v2_copy[:,None]
    return v1_copy * v2_copy


def window(n_arr, rs, r_edges, n_rbins):
    inds = np.digitize(rs, r_edges)
    Ws = np.zeros((n_rbins, len(rs)))
    for k, n in enumerate(n_arr):
        idxs_r_n = np.where(inds==k+1) #+1 because bin 0 means outside of range
        Ws[k,idxs_r_n] = 1
    return Ws 


def vector_outer_product_power(l, vec_arr):
    if l==0:
        return np.ones(len(vec_arr))
    if l==1:
        return vec_arr
    # if l is 2 or greater, will take outer product with itself
    return vector_outer_product(vec_arr, vec_arr)


def vector_outer_product(v1, v2):
    return np.einsum('...j,...k->...jk',v1,v2)


def geo_name(geometric_feature, mode='readable'):
    assert mode in ['readable', 'multipole'], 'Name mode not recognized!'

    n_str = geometric_feature.n
    if geometric_feature.n > 9:
        n_str = f'({geometric_feature.n})'
    if mode=='multipole':
        # double curly braces escape f-string formatting, make single brace
        name = f"g_{{{geometric_feature.x_order}{geometric_feature.v_order}{n_str}}}"
        if not geometric_feature.hermitian:
            name += '^A'
    elif mode=='readable':
        geo_name_dict = {(0,0): 'm',
                         (0,1): 'v',
                         (0,2): 'C^{vv}',
                         (1,0): 'x',
                         (1,1): 'C^{xv}',
                         (2,0): 'C^{xx}'}
        if geometric_feature.modification=='symmetrized':
            name = f'\\frac{{1}}{{2}} (C^{{xv}}_{n_str} + C^{{vx}}_{n_str})'  
        elif geometric_feature.modification=='antisymmetrized':          
            name = f'\\frac{{1}}{{2}} (C^{{xv}}_{n_str} - C^{{vx}}_{n_str})'
        else:
            name = geo_name_dict[(geometric_feature.x_order, geometric_feature.v_order)] + f'_{n_str}'
    return name