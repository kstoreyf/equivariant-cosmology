import numpy as np
from collections import defaultdict
import itertools
from astropy.table import Table

import utils
from read_halos import DarkHalo


class GeometricFeaturizer:

    def featurize(self, sim_reader, fn_halos, 
                  r_edges, x_order_max, v_order_max, 
                  r_units='r200m_fof'):
        
        r_edges = np.array(r_edges)
        self.sim_reader = sim_reader
        self.geo_feature_arr = []

        l_arr = np.arange(x_order_max+1)
        p_arr = np.arange(v_order_max+1)

        print(f"Loading halo table {fn_halos}")
        tab_halos = utils.load_table(fn_halos)
        self.idxs_halos_dark = tab_halos['idx_halo_dark']

        print("Computing geometric features for all dark halos")
        for i, idx_halo_dark in enumerate(self.idxs_halos_dark):
            halo = DarkHalo(idx_halo_dark, sim_reader.base_path_dark, sim_reader.snap_num, sim_reader.box_size)

            x_halo, v_halo = halo.load_positions_and_velocities(shift=True, pos_center=tab_halos['x_minPE'][i])

            r_edges_scaled = r_edges
            if r_units is not None:
                r_edges_scaled = r_edges * tab_halos[r_units][i]
            self.geo_feature_arr.append(self.get_geometric_features(x_halo, v_halo, 
                                r_edges_scaled, l_arr, p_arr, self.sim_reader.m_dmpart_dark))


    def get_geometric_features(self, delta_x_data_halo, delta_v_data_halo, r_edges, l_arr, p_arr, m_dm, n_dim=3):

        # input checks
        N, n_dim_from_data = delta_x_data_halo.shape
        assert n_dim_from_data == n_dim, f"Number of dimensions in data {n_dim_from_data} does not equal global n_dim, {n_dim}!"
        assert delta_x_data_halo.shape==delta_v_data_halo.shape, "Position and velocity arrays should have same shape!"

        # set up bin indices
        n_rbins = len(r_edges) - 1
        n_arr = np.arange(n_rbins)

        # get mass info
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


    def save_features(self, fn_geo_features, overwrite=True):

        # get geo names from first halo; should be same for all halos
        # these are the columns; number N_geos
        tab_geos = geo_objects_to_table(self.geo_feature_arr, self.idxs_halos_dark)

        tab_geos.write(fn_geo_features, overwrite=overwrite, format='fits')
        print(f"Wrote table to {fn_geo_features}")
        
        #np.save(fn_geo_features, self.geo_feature_arr)
        return tab_geos


    def save_geo_info(self, fn_geo_info, overwrite=True):

        # get geos for first halo; same for all halos
        geos = self.geo_feature_arr[0]
        tab_geo_info = geos_to_info_table(geos)
        tab_geo_info.write(fn_geo_info, overwrite=overwrite, format='fits')
        print(f"Wrote table to {fn_geo_info}")
        
        #np.save(fn_geo_features, self.geo_feature_arr)
        return tab_geo_info


    def load_features(self, fn_geo_features):

        tab_geos = utils.load_table(fn_geo_features)
        #self.geo_feature_arr = np.load(fn_geo_features, allow_pickle=True)
        return tab_geos


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
            name += '_A'
        if geometric_feature.modification is not None:
            name += '_'+geometric_feature.modification
    elif mode=='readable':
        geo_name_dict = {(0,0): 'm',
                         (0,1): 'v',
                         (0,2): 'C^{vv}',
                         (1,0): 'x',
                         (1,1): 'C^{xv}',
                         (2,0): 'C^{xx}'}
        # currently these are the only symm/antisymm features,
        # so hardcoded; careful here if change!
        if geometric_feature.modification=='symmetrized':
            name = f'\\frac{{1}}{{2}} (C^{{xv}}_{n_str} + C^{{vx}}_{n_str})'  
        elif geometric_feature.modification=='antisymmetrized':          
            name = f'\\frac{{1}}{{2}} (C^{{xv}}_{n_str} - C^{{vx}}_{n_str})'
        else:
            name = geo_name_dict[(geometric_feature.x_order, geometric_feature.v_order)] + f'_{n_str}'
    return name

### Cleaning functions

### Rebinning is just summing over the features at that order! 
# because the geo features are all sums; haven't divided out
# anything yet
# n_groups should be lists of the "n" to include in each group
def rebin_geometric_features(geo_feature_arr, n_groups):
    from geometric_features import GeometricFeature
    print("Rebinning geometric features")
    # TODO: implement check that bins listed in n_groups matches bins in the geo_feature_arr
    n_vals = [g.n for g in geo_feature_arr[0]] # 0 because just check first halo, features should be same
    n_groups_flat = [n for group in n_groups for n in group]
    assert set(n_groups_flat).issubset(set(n_vals)), 'Groups passed in contain bins not in geometric features!'

    geo_feature_arr_rebinned = []
    number_of_groups = len(n_groups)
    count = 0
    for geo_features_halo in geo_feature_arr:
        count += 1
        # group geometric features into n groups
        geo_feats_grouped = [[] for _ in range(number_of_groups)]
        for geo_feat in geo_features_halo:
            for i_n, n_group in enumerate(n_groups):
                if geo_feat.n in n_group:
                    geo_feats_grouped[i_n].append(geo_feat)

        # sum over same features (matching orders) in each group
        geo_features_halo_rebinned = []
        # m order same for all geo features so don't need to worry bout it
        x_order_highest = np.max([g.x_order for g in geo_features_halo])
        v_order_highest = np.max([g.v_order for g in geo_features_halo])
        for i_newn, geo_feat_group in enumerate(geo_feats_grouped):
            # plus 1 because want to include that highest order!
            for x_order in range(x_order_highest+1):
                for v_order in range(v_order_highest+1):
                    geo_feats_this_order = [g for g in geo_feat_group if g.x_order==x_order and g.v_order==v_order]
                    # continue if there are no values at this order (e.g. none at x=2, v=1)
                    if not geo_feats_this_order:
                        continue
                    geo_rebinned_value = np.sum([g.value for g in geo_feats_this_order], axis=0)
                    hermitian = geo_feats_this_order[0].hermitian # if one is hermitian, all are at this order
                    geo_rebinned = GeometricFeature(geo_rebinned_value, m_order=1, x_order=x_order, v_order=v_order, 
                                                    n=i_newn, hermitian=hermitian)
                    geo_features_halo_rebinned.append(geo_rebinned)
        geo_feature_arr_rebinned.append(geo_features_halo_rebinned)

    return geo_feature_arr_rebinned


def rescale_geometric_features(geo_feature_arr, Ms, Rs, Vs):
    print("Rescaling geometric features")
    N_geo_arrs = len(geo_feature_arr)
    assert len(Ms)==N_geo_arrs, "Length of Ms doesn't match geo feature arr!"
    assert len(Rs)==N_geo_arrs, "Length of Rs doesn't match geo feature arr!"
    assert len(Vs)==N_geo_arrs, "Length of Vs doesn't match geo feature arr!"
    for i_g, geo_features_halo in enumerate(geo_feature_arr):
        for geo_feat in geo_features_halo:
            geo_feat.value /= Ms[i_g] # all geometric features have single m term
            for _ in range(geo_feat.x_order):
                geo_feat.value /= Rs[i_g]
            for _ in range(geo_feat.v_order):
                geo_feat.value /= Vs[i_g]
    return geo_feature_arr


def transform_pseudotensors(geo_feature_arr):
    print("Transforming pseudotensors")
    from geometric_features import GeometricFeature
    geo_feature_arr = list(geo_feature_arr)
    for i_halo, geo_features_halo in enumerate(geo_feature_arr):
        gs_to_insert = []
        idxs_to_insert = []
        for i_feat, g in enumerate(geo_features_halo):

            if not g.hermitian and g.modification is None:
                g_value_symm = 0.5*(g.value + g.value.T)
                g_value_antisymm =  0.5*(g.value - g.value.T)
                g_symm = GeometricFeature(g_value_symm, m_order=g.m_order, x_order=g.x_order, v_order=g.v_order, 
                                            n=g.n, hermitian=True, modification='symmetrized')
                g_antisymm = GeometricFeature(g_value_antisymm, m_order=g.m_order, x_order=g.x_order, v_order=g.v_order, 
                                                n=g.n, hermitian=False, modification='antisymmetrized')
                # replace original with symmetric                
                geo_feature_arr[i_halo][i_feat] = g_symm
                # keep antisymmetric to insert right after symmetric, later
                gs_to_insert.append(g_antisymm)
                idxs_to_insert.append(i_feat+1)
        
        # inserting all at end to not mess up looping
        # for now should only have one pseudotensor per halo (C^{xv}), but may not always be true
        for idxs_to_insert, g_to_insert in zip(idxs_to_insert, gs_to_insert):
            geo_feature_arr[i_halo] = np.insert(geo_feature_arr[i_halo], idxs_to_insert, g_to_insert)

    return np.array(geo_feature_arr)


### Data structure swappings

def geo_table_to_objects(tab_geos, tab_geo_info):
    print(tab_geos.columns)
    geo_feature_arr = []
    # i indexes halo
    for i in range(len(tab_geos)):
        geo_features_halo = []
        # j indexes geo feature
        for j in range(len(tab_geo_info)):
            geo_key = tab_geo_info['geo_key'][j]
            geo = GeometricFeature(tab_geos[i][geo_key], 
                                   m_order=tab_geo_info['m_order'][j], 
                                   x_order=tab_geo_info['x_order'][j], 
                                   v_order=tab_geo_info['v_order'][j], 
                                   n=tab_geo_info['n'][j],
                                   hermitian=tab_geo_info['hermitian'][j],
                                   modification=tab_geo_info['modification'][j])
            geo_features_halo.append(geo)  
        geo_feature_arr.append(geo_features_halo)

    return geo_feature_arr

def geo_objects_to_table(geo_feature_arr, idxs_halos_dark):

    # get geo names from first halo; should be same for all halos
    # these are the columns; number N_geos
    geo_keys = [geo_name(g, mode='multipole') for g in geo_feature_arr[0]]
    # vals is a 2nd array of (N_halos, N_geos)
    geo_vals = np.array([[g.value for g in geos] for geos in geo_feature_arr])
    
    tab_geos = Table()
    tab_geos['idx_halo_dark'] = np.array(idxs_halos_dark)

    for j in range(geo_vals.shape[1]):
        tab_geos[geo_keys[j]] = np.stack(geo_vals[:,j])

    return tab_geos

# input: single set of geo features
def geos_to_info_table(geos):

    geo_keys = [geo_name(g, mode='multipole') for g in geos]        
    geo_names = [geo_name(g, mode='readable') for g in geos]        
    m_orders = [g.m_order for g in geos]
    x_orders = [g.x_order for g in geos]
    v_orders = [g.v_order for g in geos]
    ns = [g.n for g in geos]
    hermitians = [g.hermitian for g in geos]
    modifications = [g.modification for g in geos]
    
    tab_geo_info = Table([geo_keys, geo_names, m_orders, x_orders, v_orders,
                        ns, hermitians, modifications],
                        names=('geo_key', 'geo_name', 'm_order', 'x_order', 
                                'v_order', 'n', 'hermitian', 'modification'),
                        dtype=(str, str, int, int, int, int, bool, str))
    return tab_geo_info