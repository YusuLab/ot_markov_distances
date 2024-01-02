import torch
from torch import Tensor

from .sinkhorn import sinkhorn
from .utils import markov_measure, densify, cost_matrix_index as make_cost_matrix_index,\
        reindex_cost_matrix, re_project_C, degree_markov, dummy_densify

def wl_k(MX: Tensor, MY: Tensor, 
        l1: Tensor|None = None, 
        l2: Tensor|None = None,
        *, 
        cost_matrix: Tensor | None = None,
        k: int,
        muX: Tensor | None = None,
        muY: Tensor | None = None, 
        reg: float=.1, 
        sinkhorn_parameters: dict = dict()
        ):
    """computes the WL distance

    computes the WL distance between two markov transition matrices 
    (represented as torch tensor)

    Batched over first dimension (b)

    Args:
        MX: (b, n, n) first transition tensor
        MY: (b, m, m) second transition tensor
        l1: (b, n,) label values for the first space
        l2: (b, m,) label values for the second space
        k: number of steps (k parameter for the WL distance)
        muX: stationary distribution for MX (if omitted, will be recomuputed)
        muY: stationary distribution for MY (if omitted, will be recomuputed)
        reg: regularization parameter for sinkhorn
    """
    b, n, n_ = MX.shape
    b_, m, m_ = MY.shape
    assert (n==n_) and (m == m_) and (b == b_)
    if cost_matrix is None:
        assert (l1 is not None) and (l2 is not None)
        cost_matrix = (l1[:, :, None] - l2[:, None, :]).abs()
    
    for i in range(k):
        cost_matrix = sinkhorn(
                MX[:, :, None, :], # b, n, 1, n
                MY[:, None, :, :], # b, 1, m, m
                cost_matrix[:, None, None, :, :], # b, 1, 1, n, m
                epsilon=reg, 
                **sinkhorn_parameters
        ) # b, n, m

    if muX is None: 
        muX = markov_measure(MX)
    if muY is None:
        muY = markov_measure(MY)

    return sinkhorn(muX, muY, cost_matrix, reg)


def wl_k_sparse(MX: Tensor, MY: Tensor, 
        l1: Tensor|None = None, 
        l2: Tensor|None = None,
        *, 
        cost_matrix: Tensor | None = None,
        k: int,
        muX: Tensor | None = None,
        muY: Tensor | None = None, 
        reg: float=.1, 
        sinkhorn_parameters: dict = dict()
        ):
    """computes the WL distance, sparse version

    computes the WL distance between two markov transition matrices 
    (represented as torch tensor)

    Batched over first dimension (b)

    Args:
        MX: (b, n, n) first transition tensor
        MY: (b, m, m) second transition tensor
        l1: (b, n,) label values for the first space
        l2: (b, m,) label values for the second space
        k: number of steps (k parameter for the WL distance)
        muX: stationary distribution for MX (if omitted, will be recomuputed)
        muY: stationary distribution for MY (if omitted, will be recomuputed)
        reg: regularization parameter for sinkhorn
    """
    b, n, n_ = MX.shape
    b_, m, m_ = MY.shape
    assert (n==n_) and (m == m_) and (b == b_)
    if cost_matrix is None:
        assert (l1 is not None) and (l2 is not None)
        cost_matrix = (l1[:, :, None] - l2[:, None, :]).abs()

    mx_index, mx_mask, mx_dense = densify(MX) 
    my_index, my_mask, my_dense = densify(MY)
    mx_dense = mx_dense[:, :, None, :] # b, n, 1, dx
    my_dense = my_dense[:, None, :, :] # b, 1, m, dy
    cost_matrix_index, cost_matrix_mask = make_cost_matrix_index(cost_matrix, mx_index, my_index, mx_mask, my_mask)
    
    for i in range(k):
        cost_matrix_sparse = reindex_cost_matrix(cost_matrix, 
                                                 cost_matrix_index, 
                                                 cost_matrix_mask) 
        cost_matrix = sinkhorn(
                mx_dense, # b, n, 1, dx
                my_dense, # b, 1, m, dy
                cost_matrix_sparse[:, None, None, :, :], # b, 1, 1, dx, dy
                epsilon=reg, 
                **sinkhorn_parameters
        ) # b, n, m


    if muX is None: 
        muX = markov_measure(MX)
    if muY is None:
        muY = markov_measure(MY)

    return sinkhorn(muX, muY, cost_matrix, reg)
