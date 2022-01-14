import numpy as np
from collections import defaultdict


### Functions for geometric features

def window(n_arr, rs, r_edges, n_rbins):
    inds = np.digitize(rs, r_edges)
    Ws = np.zeros((n_rbins, len(rs)))
    for k, n in enumerate(n_arr):
        idxs_r_n = np.where(inds==k+1) #+1 because bin 0 means outside of range
        Ws[k,idxs_r_n] = 1
    return Ws 

def x_outer_product(l, delta_x_arr, x_outers_prev, n_dim=3):
    # can't figure out how to vectorize better
    if l==0:
        return np.ones(len(delta_x_arr))
    if l==1:
        return delta_x_arr
    else:
        x_outers = np.empty([len(delta_x_arr)] + [n_dim]*l)
    for i, delta_x in enumerate(delta_x_arr):
        x_outers[i,:] = np.multiply.outer(x_outers_prev[i], delta_x)
    return x_outers

### 

def get_geometric_features(delta_x_data_halo, r_edges, l_arr, n_arr, m_dm, n_dim=3):
    n_rbins = len(r_edges) - 1
    N, n_dim_from_data = delta_x_data_halo.shape
    m_total = N*m_dm
    assert n_dim_from_data == n_dim, f"Number of dimensions in data {n_dim_from_data} does not equal global n_dim, {n_dim}!"
    g_arrs = []
    g_normed_arrs = []

    # vectorized for all particles N
    rs = np.linalg.norm(delta_x_data_halo, axis=1)
    assert len(rs) == N, "rs should have length N!"
    window_vals = window(n_arr, rs, r_edges, n_rbins)

    x_outers_prev = None
    for j, l in enumerate(l_arr):
        g_arr = np.empty([len(n_arr)] + [n_dim]*l)
        g_normed_arr = np.empty([len(n_arr)] + [n_dim]*l)

        #x_outers = x_outer_vec(l, delta_x_data_halo)
        x_outers = x_outer_product(l, delta_x_data_halo, x_outers_prev, n_dim=n_dim)

        for k, n in enumerate(n_arr):
            g_ln = np.sum( window_vals[k,:] * x_outers.T, axis=-1)
            g_normalization_ln = np.sum(window_vals[k,:])

            g_arr[k,...] = g_ln * m_dm # can pull the mass multiplier out here bc equal mass particles 
            #g_normed_arr[k,...] = g_ln / g_normalization_ln if g_normalization_ln != 0 else 0
            g_normed_arr[k,...] = g_ln * m_dm / m_total

        g_arrs.append(g_arr)
        g_normed_arrs.append(g_normed_arr)

        x_outers_prev = x_outers

    return g_arrs, g_normed_arrs  


def featurize_scalars(g_arr, n_arr):

    assert len(g_arr) >= 2, "Need up to at least l=2 to compute all scalars here!"

    scalar_features = defaultdict(dict)
    # (-1) n tuple for consistency with others
    scalar_features['s0'][(-1)] = {'value': 1, 
                             'ns': [], 'ls': [], 'm_order': 0, 'x_order': 0}
    for n0 in n_arr:
        scalar_features['s1'][(n0)] = {'value': g_arr[0][n0], 
                                    'ns': [n0], 'ls': [0], 'm_order': 1, 'x_order': 0}
        scalar_features['s4'][(n0)] = {'value': np.einsum('jj', g_arr[2][n0]), 
                                    'ns': [n0], 'ls': [2], 'm_order': 1, 'x_order': 2}
        for n1 in n_arr:
            scalar_features['s2'][(n0,n1)] = {'value':  g_arr[0][n0] *  g_arr[0][n1], 
                                 'ns': [n0,n1], 'ls': [0], 'm_order': 2, 'x_order': 0}
            scalar_features['s5'][(n0,n1)] = {'value':  np.einsum('j,j', g_arr[1][n0], g_arr[1][n1]), 
                                 'ns': [n0,n1], 'ls': [1], 'm_order': 2, 'x_order': 2}
            scalar_features['s6'][(n0,n1)] = {'value':  g_arr[0][n0] * np.einsum('jj', g_arr[2][n1]),
                                 'ns': [n0,n1], 'ls': [0,2], 'm_order': 2, 'x_order': 2}
            for n2 in n_arr:
                scalar_features['s3'][(n0,n1,n2)] = {'value':  g_arr[0][n0] * g_arr[0][n1] * g_arr[0][n2],
                                 'ns': [n0,n1,n2], 'ls': [0], 'm_order': 3, 'x_order': 0}
                scalar_features['s7'][(n0,n1,n2)] = {'value':
                                 g_arr[0][n0] * np.einsum('j,j', g_arr[1][n1], g_arr[1][n2]),
                                 'ns': [n0,n1,n2], 'ls': [0,1], 'm_order': 3, 'x_order': 2}
                scalar_features['s8'][(n0,n1,n2)] = {'value':  
                                 g_arr[0][n0] * g_arr[0][n1] * np.einsum('jj', g_arr[2][n2]),
                                 'ns': [n0,n1,n2], 'ls': [0,2], 'm_order': 3, 'x_order': 2}

    return scalar_features

# TODO: write as dictionary, as in scalar function above
def featurize_vectors(g_arr, n_arr, n_dim=3):#, tensor_normed_arr):
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