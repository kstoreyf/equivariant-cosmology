
### Functions for geometric features

def window(n_arr, rs, r_edges):
    inds = np.digitize(rs, r_edges)
    Ws = np.zeros((n_rbins, len(rs)))
    for k, n in enumerate(n_arr):
        idxs_r_n = np.where(inds==k+1) #+1 because bin 0 means outside of range
        Ws[k,idxs_r_n] = 1
    return Ws 

def x_outer_product(l, delta_x_arr, x_outers_prev):
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

def get_geometric_features(delta_x_data_halo):
    N, n_dim_from_data = delta_x_data_halo.shape
    m_total = N*m_dm
    assert n_dim_from_data == n_dim, f"Number of dimensions in data {n_dim_from_data} does not equal global n_dim, {n_dim}!"
    g_arrs = []
    g_normed_arrs = []

    # vectorized for all particles N
    rs = np.linalg.norm(delta_x_data_halo, axis=1)
    assert len(rs) == N, "rs should have length N!"
    window_vals = W_vec(n_arr, rs)

    x_outers_prev = None
    for j, l in enumerate(l_arr):
        g_arr = np.empty([len(n_arr)] + [n_dim]*l)
        g_normed_arr = np.empty([len(n_arr)] + [n_dim]*l)

        #x_outers = x_outer_vec(l, delta_x_data_halo)
        x_outers = x_outer_vec(l, delta_x_data_halo, x_outers_prev)

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


def featurize_scalars(g_arr):
    g_0, g_1, g_2 = g_arr[:3] #only need up to l=2 for now

    scalar_features = []

    scalar_features.append( 1 ) # s0
    for n0 in n_arr:
        scalar_features.append( g_0[n0] ) # s1
        scalar_features.append( np.einsum('jj', g_2[n0]) ) # s4
        for n1 in n_arr:
            scalar_features.append( g_0[n0] * g_0[n1] ) # s2
            scalar_features.append( np.einsum('j,j', g_1[n0], g_1[n1]) ) # s5 
            scalar_features.append( g_0[n0] * np.einsum('jj', g_2[n1]) ) # s6
            for n2 in n_arr:
                scalar_features.append( g_0[n0] * g_0[n1] * g_0[n2] ) # s3
                scalar_features.append( g_0[n0] * np.einsum('j,j', g_1[n1], g_1[n2]) ) # s7
                scalar_features.append( g_0[n0] * g_0[n1] * np.einsum('jj', g_2[n2]) ) # s8

    return scalar_features


def featurize_vectors(g_arr):#, tensor_normed_arr):
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