import copy
import itertools
import numpy as np

from geometric_features import GeometricFeature



class ScalarFeaturizer:

    def __init__(self, geo_feature_arr=None):
        self.geo_feature_arr_orig = geo_feature_arr
        # must use deepcopy because our array has opjects! np.copy doesn't work
        self.geo_feature_arr = copy.deepcopy(geo_feature_arr)
        self.N_halos = len(self.geo_feature_arr)


    def featurize(self, m_order_max, n_groups_rebin=None, 
                  x_order_max=np.inf, v_order_max=np.inf,
                  eigenvalues_not_trace=False):

        if n_groups_rebin is not None:
            print(f"Rebinning")
            self.geo_feature_arr = self.rebin_geometric_features(self.geo_feature_arr, n_groups_rebin)
            print(f"Rebinned to {len(n_groups_rebin)} bins!")

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
                s_features = self.make_scalar_feature(list(geo_terms), 
                                        x_order_max=x_order_max, v_order_max=v_order_max,
                                        eigenvalues_not_trace=eigenvalues_not_trace)
                if s_features != -1:
                    scalar_features.extend(s_features)
        return scalar_features

    # n_groups should be lists of the "n" to include in each group
    def rebin_geometric_features(self, geo_feature_arr, n_groups):
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


    def compute_MXV_from_features(self):
        # to see what n's we have, get a set of them for just one halo (should all be same)
        ns_all = list(set([g.n for g in self.geo_feature_arr[0]]))
        geo_feature_arr_onebin = self.rebin_geometric_features(self.geo_feature_arr, [ns_all])
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


    def make_scalar_feature(self, geo_terms, x_order_max, v_order_max, eigenvalues_not_trace=False):

        # should these orders be degree?
        x_order_tot = np.sum([g.x_order for g in geo_terms])
        v_order_tot = np.sum([g.v_order for g in geo_terms])
        xv_order_tot = x_order_tot + v_order_tot
        xv_orders_allowed = [0, 2, 4] # allowing two-tensor terms, but not odd-order!
        # i think if the first case is satisfied the others will be too, but keeping to be safe
        if xv_order_tot not in xv_orders_allowed or x_order_tot > x_order_max or v_order_tot > v_order_max:
            return -1
                
        geo_vals_contracted = []
        geo_vals_from_tensors = []

        # construct symmetric and antisymmetric tensors
        geo_vals_xv_antisymm = []
        for i_g, g in enumerate(geo_terms):
            if not g.hermitian:
                # g = x(outer)v, so g^T = v(outer)x
                xv_symm = 0.5*(g.value + g.value.T)
                xv_antisymm =  0.5*(g.value - g.value.T)
                geo_vals_xv_antisymm.append(xv_antisymm)
                # replace g value with its symmetric value
                geo_symm = GeometricFeature(xv_symm, m_order=g.m_order, x_order=g.x_order, v_order=g.v_order, n=g.n, hermitian=True)
                geo_terms[i_g] = geo_symm
        # for antisymmetric tensors, only think we can do is multiply them with each other to recover the symmetry
        if len(geo_vals_xv_antisymm)==2:
            geo_vals_contracted.append(np.einsum('jk,jk', *geo_vals_xv_antisymm))
        
        # multi t terms
        # two-vector terms
        geo_vals_vec = [g.value for g in geo_terms if g.x_order+g.v_order==1]
        assert len(geo_vals_vec) <= 2, "not going above 3 terms so shouldnt have more than 2 vector terms for scalars!"
        # t10n t10n', t01n t01n', t10n t01n'
        # TODO: i don't understand this rn - why not when eigenvalues!
        # if 0, continue; if include_eigenvectors, this term will be included later
        if len(geo_vals_vec)==2: #and not eigenvalues_not_trace:
            # vector & vector: take inner product
            geo_vals_contracted.append(np.einsum('j,j', *geo_vals_vec))

        # two tensor terms
        # shouldn't be any non-hermitian left in here, but making sure!
        geo_vals_tensor = [g.value for g in geo_terms if g.x_order+g.v_order==2 and g.hermitian] 
        assert len(geo_vals_tensor) <= 2, "not going above 4th order so shouldnt have more than 2 tensor terms!"
        if len(geo_vals_tensor)==2:
            # tensor and tensor: do contraction
            # not adding to geo_vals_from_tensors bc those get multiplied in everywhere, like eigenvalues
            # (but shouldnt matter when limiting to two terms)
            geo_vals_contracted.append(np.einsum('jk,jk', *geo_vals_tensor))

        # single t terms
        for g in geo_terms:
            xv_order = g.x_order + g.v_order
            if xv_order==0:
                # t00n
                geo_vals_contracted.append(g.value)
            elif xv_order==2:
                # t20n, t02n, t11n
                # if g is hermitian, include trace OR eigenvalues
                if g.hermitian:
                    if eigenvalues_not_trace:
                        # eigenvalues
                        eigenvalues = np.linalg.eigvalsh(g.value)
                        geo_vals_from_tensors.extend(eigenvalues)
                    else:
                        # trace
                        geo_vals_from_tensors.append(np.einsum('jj', g.value))

        # if there are any geometric values from a tensor, multiply each of these
        # separately into the rest of the contracted scalar term, because of eigenvalues
        s_features = []
        geo_val_product = np.product(geo_vals_contracted)
        # if no vectors or tensors, still need to add feature with rest of values
        if not geo_vals_from_tensors:
            geo_vals_from_tensors.append(1.0)
        for g_tensor in geo_vals_from_tensors:
            value = g_tensor * geo_val_product
            s_features.append(ScalarFeature(value, geo_terms))

        return s_features
    

    def save_features(self, fn_scalar_features):
        np.save(fn_scalar_features, self.scalar_feature_arr)


    def load_features(self, fn_scalar_features):
        self.scalar_feature_arr = np.load(fn_scalar_features, allow_pickle=True)


class ScalarFeature:

    def __init__(self, value, geo_terms):
        self.value = value
        self.geo_terms = geo_terms
        self.m_order = np.sum([g.m_order for g in self.geo_terms])
        self.x_order = np.sum([g.x_order for g in self.geo_terms])
        self.v_order = np.sum([g.v_order for g in self.geo_terms])

    def to_string(self):
        return ' '.join(np.array([g.to_string() for g in self.geo_terms]))
            

        