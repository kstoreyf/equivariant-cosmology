import numpy as np


def unroll_array(arr, n_dim=3):
    # shape (N, n_feat, n_dim) to (N*n_dim, n_feat)
    # ... indexes all previous dimensions with :
    arr_unrolled = np.vstack([arr[...,d] for d in range(n_dim)])
    return arr_unrolled

def unroll_array_2d(arr, n_dim=3):
    # shape (N, n_dim) to (N*n_dim)
    arr_unrolled = np.hstack([arr[:,d] for d in range(n_dim)])
    return arr_unrolled

def roll_array(arr, n_dim=3):
    # shape (N*n_dim) to (N, n_dim)
    assert arr.shape[0] % n_dim == 0, "Unrolled vector shape must be a multiple of n_dim!"
    N = int( arr.shape[0] / n_dim )
    arr_rolled = np.array([arr[d*N:(d+1)*N] for d in range(n_dim)]).T
    return arr_rolled

def get_alt_sim_name(sim_name):
    sim_name_dict = {'TNG100-1': 'L75n1820TNG',
                     'TNG100-1-Dark': 'L75n1820TNG_DM',
                     'TNG50-4': 'L35n270TNG',
                     'TNG50-4-Dark': 'L35n270TNG_DM'
                    }
    return sim_name_dict[sim_name]


def get_uncertainties_genel2019(log_m_stellar, sim_name):
    logmstellar_bins = np.linspace(8.5, 11, 6)
    logmstellar_bins = np.array([5] + list(logmstellar_bins) + [13])
    err_msg = 'Passed log_m_stellar has values outside of bin edges!'
    assert np.min(log_m_stellar) > np.min(logmstellar_bins) and np.max(log_m_stellar) < np.max(logmstellar_bins), err_msg
    # added estimates on either end (low: double, high: extend)
    stdev_dict = {'TNG50-4': np.array([0.56, 0.28, 0.23, 0.12, 0.05, 0.04, 0.04]), # epsilon=4, similar to tng50-4
                'TNG100-1': np.array([0.16, 0.08, 0.06, 0.04, 0.03, 0.04, 0.04]), # epsilon=1, similar to tng100-1
                }

    idxs_mbins = np.digitize(log_m_stellar, logmstellar_bins)
    uncertainties_genel2019 = stdev_dict[sim_name][idxs_mbins-1]

    return uncertainties_genel2019