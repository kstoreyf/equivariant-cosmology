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