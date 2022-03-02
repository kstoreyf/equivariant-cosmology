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

def vector_outer_product_power(l, vec_arr, vec_outers_prev):
    # can't figure out how to vectorize better - TODO: profile 
    if l==0:
        return np.ones(len(vec_arr))
    if l==1:
        return vec_arr
    # if l is 2 or greater, will take outer product of itself with previous
    return vector_outer_product(vec_arr, vec_outers_prev)

def vector_outer_product(v1, v2):
    return np.einsum('...j,...k->...jk',v1,v2)

def multiply_xv_terms(x_term, v_term):
    # ndim==2 mean these are arrays of vectors (a vector for each particle)
    assert x_term.ndim + v_term.ndim <= 4, "Dimensions for x and v should not be this high! Can't multiply"
    if x_term.ndim==2 and v_term.ndim==2:
        return 0.5*(vector_outer_product(x_term, v_term) + vector_outer_product(v_term, x_term))
    # else, one of them must be ndim=0 because limited to max combined order of 2, so just multiply
    # print(x_term.shape, v_term.shape, np.atleast_2d(v_term).shape)
    # print(x_term.ndim, v_term.ndim)
    return multiply_vector_terms(x_term, v_term)

# must be a better way to do this...
def multiply_vector_terms(v1, v2):

    # sum(product, axis=1)

    # hogg says this is good - i can clean up by single return 
    if v1.ndim < v2.ndim:
        v1_exp = v1
        for _ in np.arange(1, v2.ndim):
            v1_exp = v1_exp[:,None]
        return v2 * v1_exp
    elif v1.ndim > v2.ndim:
        v2_exp = v2
        for _ in np.arange(1, v1.ndim):
            v2_exp = v2_exp[:,None]
        return v1 * v2_exp 
    else:
        return v1 * v2

###

def get_needed_vec_orders_scalars(x_order_max, v_order_max):
    vec_order_max = x_order_max + v_order_max
    vec_orders_allowed = [0, 2]
    if vec_order_max not in vec_orders_allowed:
        raise ValueError('ERROR: x_order_max+v_order_max must be 0 or 2 - '\
              'even because otherwise we will not obtain scalars, and low because '\
              'higher order scalars not yet computed (input was ' \
              f'x_order_max={x_order_max}, v_order_max={v_order_max})')
    
    l_arr = np.arange(0, x_order_max+1)
    p_arr = np.arange(0, v_order_max+1)
    return l_arr, p_arr


class GeometricFeature:

    def __init__(self, value, m_order, x_order, v_order, n):
        self.value = value
        self.m_order = m_order
        self.x_order = x_order
        self.v_order = v_order
        self.n = n
        
    def to_string(self):
        return f"g_{self.x_order}{self.v_order}{self.n}"


class ScalarFeature:

    def __init__(self, value, geo_terms):
        self.value = value
        self.geo_terms = geo_terms
        self.m_order = np.sum([g.m_order for g in self.geo_terms])
        self.x_order = np.sum([g.x_order for g in self.geo_terms])
        self.v_order = np.sum([g.v_order for g in self.geo_terms])

    def to_string(self):
        return ' '.join(np.array([g.to_string() for g in self.geo_terms]))
            

def are_geometric_features_same(g1, g2):
    if g1.m_order==g2.m_order and g1.x_order==g2.x_order and g1.v_order==g2.v_order and g1.n==g2.n:
        return True
    else:
        return False


def make_scalar_feature(geo_terms, x_order_max, v_order_max, 
                        include_eigenvalues=False, include_eigenvectors=False):
    x_order_tot = np.sum([g.x_order for g in geo_terms])
    v_order_tot = np.sum([g.v_order for g in geo_terms])
    xv_order_tot = x_order_tot + v_order_tot
    xv_orders_allowed = [0, 2] # make sure aligns with get_needed_vec_orders_scalars()
    # i think if the first case is satisfied the others will be too, but keeping to be safe
    if xv_order_tot not in xv_orders_allowed or x_order_tot > x_order_max or v_order_tot > v_order_max:
        return -1
  
    geo_vals_contracted = []
    geo_vals_from_vectors = []
    geo_vals_from_tensors = []
    
    # multi t terms
    geo_vals_vec = [g.value for g in geo_terms if g.x_order+g.v_order==1]
    assert len(geo_vals_vec) <= 2, "not going above 3 terms so shouldnt have more than 2 vector terms for scalars!"
    # t10n t10n', t01n t01n', t10n t01n'
    # if 0, continue; if include_eigenvectors, this term will be included later
    if len(geo_vals_vec)==2 and not include_eigenvectors:
        geo_vals_from_vectors.append(np.einsum('j,j', *geo_vals_vec))

    # single t terms
    for g in geo_terms:
        xv_order = g.x_order + g.v_order
        if xv_order==0:
            # t00n
            geo_vals_contracted.append(g.value)
        elif xv_order==2:
            # t20n, t02n, t11n
            # include trace only if don't include eigenvalues
            if not include_eigenvalues:
                geo_vals_from_tensors.append(np.einsum('jj', g.value))
            # if no vectors, no point in computing eigenvectors to project them onto
            if include_eigenvalues and (not include_eigenvectors or len(geo_vals_vec)==0):
                eigenvalues = np.linalg.eigvalsh(g.value)
                geo_vals_from_tensors.extend(eigenvalues)
            if include_eigenvectors and len(geo_vals_vec)>0:
                # TODO: deal with - with current setup, never reaches!
                # because no terms with tensors and vectors
                eigenvalues, eigenvectors = np.linalg.eigh(g.value)
                if include_eigenvalues:
                    geo_vals_from_tensors.extend(eigenvalues)
                # if get here, have exactly 2 eigenvectors, and computing the possible
                # scalar projection prefactor combos
                g0_projs = [ np.dot(geo_vals_vec[0], e) / np.dot(e, e) * e for e in eigenvectors ]
                g0_projs.append(1.0)
                if are_geometric_features_same(geo_vals_vec[0], geo_vals_vec[1]):
                    projection_combos = list(itertools.combinations_with_replacement(g0_projs, 2))
                    for proj_combo in projection_combos:
                        geo_vals_from_vectors.append( np.einsum('j,j', *proj_combo) )
                else:
                    g1_projs = [ np.dot(geo_vals_vec[1], e) / np.dot(e, e) * e for e in eigenvectors ]
                    g1_projs.append(1.0) # for the case where we don't multiply by any eigenvector
                    for g0_proj in g0_projs:
                        for g1_proj in g1_projs:
                            geo_vals_from_vectors.append( np.einsum('j,j', g0_proj, g1_proj) )

    # if there are any geometric values from a tensor, multiply each of these
    # separately into the rest of the contracted scalar term, because of eigenvalues
    s_features = []
    geo_val_product = np.product(geo_vals_contracted)
    # if no vectors or tensors, still need to add feature with rest of values
    if not geo_vals_from_tensors:
        geo_vals_from_tensors.append(1.0)
    if not geo_vals_from_vectors:
        geo_vals_from_vectors.append(1.0)
    for g_vector in geo_vals_from_vectors:
        for g_tensor in geo_vals_from_tensors:
            value = g_vector * g_tensor * geo_val_product
            s_features.append(ScalarFeature(value, geo_terms))

    return s_features
    

def featurize_scalars(g_features, m_order_max, x_order_max, v_order_max,
                      include_eigenvalues=False, include_eigenvectors=False):
    scalar_features = []
    num_terms = np.arange(1, m_order_max+1)
    for nt in num_terms:
        geo_term_combos = list(itertools.combinations_with_replacement(g_features, nt))
        for geo_terms in geo_term_combos:
            s_features = make_scalar_feature(geo_terms, x_order_max, v_order_max,
                                    include_eigenvalues=include_eigenvalues, 
                                    include_eigenvectors=include_eigenvectors)
            if s_features != -1:
                scalar_features.extend(s_features)
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
    x_outers_arr = []
    v_outers_arr = []
    for l in l_arr:
        x_outers = vector_outer_product_power(l, delta_x_data_halo, x_outers_prev)
        x_outers_arr.append(x_outers)
        x_outers_prev = x_outers
    for p in p_arr:
        v_outers = vector_outer_product_power(p, v_data_halo, v_outers_prev)
        v_outers_arr.append(v_outers)
        v_outers_prev = v_outers
    for i_l, l in enumerate(l_arr):
        for i_p, p in enumerate(p_arr):
            for i_n, n in enumerate(n_arr):
                # TODO: figure out how to robustly outer multiply x and v outers (i think) ??
                #g_ln = np.sum( window_vals[k,:] * x_outers.T, axis=-1) * m_dm
                xv_terms = multiply_xv_terms(x_outers_arr[i_l], v_outers_arr[i_p])
                g_lpn = np.sum( multiply_vector_terms(window_vals[i_n], xv_terms), axis=0) * m_dm
                # print(i_l, i_p, i_n)
                # print(window_vals[i_n].shape, xv_terms.shape)
                # print(multiply_vector_terms(window_vals[i_n], xv_terms).shape)
                # print(g_ln.shape)

                geo = GeometricFeature(g_lpn, m_order=1, x_order=l, v_order=p, n=n)
                g_features.append(geo)  

                geo_norm = GeometricFeature(g_lpn / m_total, m_order=1, x_order=l, v_order=p, n=n) 
                g_normed_features.append(geo_norm)

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