import numpy as np
from collections import defaultdict
import itertools


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


# def multiply_xv_terms(x_term, v_term):
#     # ndim==2 mean these are arrays of vectors (a vector for each particle)
#     assert x_term.ndim + v_term.ndim <= 4, "Dimensions for x and v should not be this high! Can't multiply"
#     if x_term.ndim==2 and v_term.ndim==2:
#         return 0.5*(vector_outer_product(x_term, v_term) + vector_outer_product(v_term, x_term))
#     # else, one of them must be ndim=0 because limited to max combined order of 2, so just multiply
#     return multiply_vector_terms(x_term, v_term)


class GeometricFeaturizer:

    def __init__(self, halo_reader):

        self.halo_reader = halo_reader


    def featurize(self, r_edges, l_arr, p_arr, n_arr, m_dm):
        self.geo_feature_arr = []
        for halo in self.halo_reader.dark_halos:
            x_halo = halo.load_positions()
            v_halo = halo.load_velocities()
            self.geo_feature_arr.append(get_geometric_features(x_halo, v_halo, r_edges,
                                   l_arr, p_arr, n_arr, m_dm))


    def get_geometric_features(self, delta_x_data_halo, delta_v_data_halo, r_edges, l_arr, p_arr, n_arr, m_dm, n_dim=3):

        n_rbins = len(r_edges) - 1
        N, n_dim_from_data = delta_x_data_halo.shape
        assert n_dim_from_data == n_dim, f"Number of dimensions in data {n_dim_from_data} does not equal global n_dim, {n_dim}!"
        assert delta_x_data_halo.shape==delta_v_data_halo.shape, "Position and velocity arrays should have same shape!"
        m_total = N*m_dm
        
        # vectorized for all particles N
        rs = np.linalg.norm(delta_x_data_halo, axis=1)
        assert len(rs) == N, "rs should have length N!"
        window_vals = window(n_arr, rs, r_edges, n_rbins)

        x_outers_prev, v_outers_prev = None, None
        x_outers_arr = []
        v_outers_arr = []
        for l in l_arr:
            x_outers = vector_outer_product_power(l, delta_x_data_halo, x_outers_prev)
            x_outers_arr.append(x_outers)
            x_outers_prev = x_outers
        for p in p_arr:
            v_outers = vector_outer_product_power(p, delta_v_data_halo, v_outers_prev)
            v_outers_arr.append(v_outers)
            v_outers_prev = v_outers

        geo_features_halo = []
        for i_l, l in enumerate(l_arr):
            for i_p, p in enumerate(p_arr):

                if l+p>2:
                    continue

                for i_n, n in enumerate(n_arr):
                    # TODO: figure out how to robustly outer multiply x and v outers (i think) ??
                    #g_ln = np.sum( window_vals[k,:] * x_outers.T, axis=-1) * m_dm
                    hermitian = False
                    if l==1 and p==1:
                        hermitian = True

                    xv_terms = multiply_vector_terms((x_outers_arr[i_l], v_outers_arr[i_p]))
                    g_lpn = np.sum( multiply_vector_terms(window_vals[i_n], xv_terms), axis=0) * m_dm

                    geo = GeometricFeature(g_lpn, m_order=1, x_order=l, v_order=p, n=n, hermitian=hermitian)
                    geo_features_halo.append(geo)  
        return geo_features_halo
    
    def save_features(self, fn_geo_features):
        np.save(fn_geo_features, self.geo_feature_arr)


class GeometricFeature:

    def __init__(self, value, m_order, x_order, v_order, n, hermitian=True):
        self.value = value
        self.m_order = m_order
        self.x_order = x_order
        self.v_order = v_order
        self.n = n
        self.hermitian = hermitian

    def to_string(self):
        return f"g_{self.x_order}{self.v_order}{self.n}"

