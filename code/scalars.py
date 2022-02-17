import numpy as np
from collections import defaultdict
import itertools


### Functions for geometric features

def window(n_arr, rs, r_edges, n_rbins):
    inds = np.digitize(rs, r_edges)
    Ws = np.zeros((n_rbins, len(rs)))
    for k, n in enumerate(n_arr):
        idxs_r_n = np.where(inds==k+1) #+1 because bin 0 means outside of range
        Ws[k,idxs_r_n] = 1
    return Ws 

def vector_outer_product(l, vec_arr, vec_outers_prev, n_dim=3):
    # can't figure out how to vectorize better - TODO: profile 
    if l==0:
        return np.ones(len(vec_arr))
    if l==1:
        return vec_arr
    else:
        vec_outers = np.empty([len(vec_arr)] + [n_dim]*l)
    for i, vec in enumerate(vec_arr):
        vec_outers[i,:] = np.multiply.outer(vec_outers_prev[i], vec)
    return vec_outers

###

def get_needed_vec_orders_scalars(m_order_max, vec_order_max):
    if vec_order_max==1:
        vec_order_max = 0 #because no x_order=1 for scalars
    if vec_order_max==3:
        vec_order_max = 2 #because no x_order=3 for scalars
    if vec_order_max>3:
        raise ValueError(f'ERROR: vec_order_max must be <=3, \
              higher order scalars not yet computed (input was {vec_order_max})')
    needed_l_dict = {(0, 0): [0],
                     (1, 0): [0],
                     (2, 0): [0],
                     (3, 0): [0],
                     (0, 2): [0],
                     (1, 2): [0,2],
                     (2, 2): [0,1,2],
                     (3, 2): [0,1,2]}
    return needed_l_dict[(m_order_max, vec_order_max)]


class GeometricFeature:

    def __init__(self, value, m_order, x_order, n):
        self.value = value
        self.m_order = m_order
        self.x_order = x_order
        self.n = n


class ScalarFeature:

    def __init__(self, value, geo_terms):
        self.value = value
        self.geo_terms = geo_terms
        self.m_order = np.sum([g.m_order for g in self.geo_terms])
        self.x_order = np.sum([g.x_order for g in self.geo_terms])


def make_scalar_feature(geo_terms, x_order_max, v_order_max):
    x_order = np.sum([g.x_order for g in geo_terms])
    # TODO: just for now! for testing
    if v_order_max > 0:
        return -1
    if x_order > x_order_max or x_order % 2 != 0:
        return -1
    geo_vals_contracted = []
    geo_vals_x1 = [g.value for g in geo_terms if g.x_order==1]
    geo_terms_x1 = [g for g in geo_terms if g.x_order==1]
   
    assert len(geo_terms_x1) <= 2, "not going above 3 terms so shouldnt have more than 2 x=1 terms for scalars!"
    if len(geo_terms_x1)>0:
        geo_vals_contracted.append(np.einsum('j,j', *geo_vals_x1))
    for g in geo_terms:
        if g.x_order==0:
            geo_vals_contracted.append(g.value)
        elif g.x_order==2:
            geo_vals_contracted.append(np.einsum('jj', g.value))
    return ScalarFeature(np.product(geo_vals_contracted), geo_terms)
    

def featurize_scalars(g_features, m_order_max, x_order_max, v_order_max):
    scalar_features = []
    num_terms = np.arange(1, m_order_max+1)
    for nt in num_terms:
        geo_term_combos = list(itertools.combinations_with_replacement(g_features, nt))
        for geo_terms in geo_term_combos:
            s = make_scalar_feature(geo_terms, x_order_max, v_order_max)
            if s != -1:
                scalar_features.append(s)
    return scalar_features

    
def get_geometric_features(delta_x_data_halo, v_data_halo, r_edges, l_arr, p_arr, n_arr, m_dm, n_dim=3):
    n_rbins = len(r_edges) - 1
    N, n_dim_from_data = delta_x_data_halo.shape
    assert n_dim_from_data == n_dim, f"Number of dimensions in data {n_dim_from_data} does not equal global n_dim, {n_dim}!"
    assert delta_x_data_halo.shape==v_data_halo.shape, "Position and velocity arrays should have same shape!"
    m_total = N*m_dm
    g_features = []
    g_normed_features = []

    # vectorized for all particles N
    rs = np.linalg.norm(delta_x_data_halo, axis=1)
    assert len(rs) == N, "rs should have length N!"
    window_vals = window(n_arr, rs, r_edges, n_rbins)

    x_outers_prev, v_outers_prev = None, None
    for j, l in enumerate(l_arr):
        g_arr = np.empty([len(n_arr)] + [n_dim]*l)
        g_normed_arr = np.empty([len(n_arr)] + [n_dim]*l)

        x_outers = vector_outer_product(l, delta_x_data_halo, x_outers_prev, n_dim=n_dim)
        v_outers = vector_outer_product(l, v_data_halo, v_outers_prev, n_dim=n_dim)

        for k, n in enumerate(n_arr):
            # TODO: figure out how to robustly outer multiply x and v outers (i think)
            g_ln = np.sum( window_vals[k,:] * x_outers.T @ v_outers, axis=-1) * m_dm

            geo = GeometricFeature(g_ln, m_order=1, x_order=l, n=n)
            g_features.append(geo)  

            geo_norm = GeometricFeature(g_ln / m_total, m_order=1, x_order=l, n=n) 
            g_normed_features.append(geo_norm)  

        x_outers_prev = x_outers
        v_outers_prev = v_outers

    return g_features, g_normed_features


# TODO: write as dictionary, as in scalar function above
def featurize_vectors(g_arr, l_arr, n_arr, n_dim=3):#, tensor_normed_arr):
    g_0, g_1, g_2, g_3 = g_arr[:4] #only need up to l=3 for now
    #g_normed_0, g_normed_1, g_normed_2, g_normed_3 = tensor_normed_arr[:4]

    vector_features = []

    vector_features.append( np.ones(n_dim) ) # v0
    for n0 in n_arr:
        vector_features.append( g_1[n0] ) # v1
        #vector_features.append( np.nan_to_num(g_1[n0] / g_normed_1[n0]) ) # v1 normed (for l=1, normalize out the mass)
        vector_features.append( np.einsum('jjk', g_3[n0]) ) # v4
        for n1 in n_arr:
            vector_features.append( g_0[n0] * g_1[n0] ) # v2
            vector_features.append( g_0[n0] * np.einsum('jjk', g_3[n1]) ) # v5
            vector_features.append( np.einsum('jj,k', g_2[n0], g_1[n1]) ) # v6
            vector_features.append( np.einsum('jk,j', g_2[n0], g_1[n1]) ) # v7
            for n2 in n_arr:
                vector_features.append( g_0[n0] * g_0[n1] * g_1[n2] ) # v3
                vector_features.append( g_0[n0] * g_0[n1] * np.einsum('jjk', g_3[n2]) ) # v8
                vector_features.append( np.einsum('j,j,k', g_1[n0], g_1[n1], g_1[n2]) ) # v9
                vector_features.append( g_0[n0] * np.einsum('jj,k', g_2[n1], g_1[n2]) ) # v10
                vector_features.append( g_0[n0] * np.einsum('jk,j', g_2[n1], g_1[n2]) ) # v11

    return vector_features