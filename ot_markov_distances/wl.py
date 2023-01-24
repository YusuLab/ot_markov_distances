import torch
from torch import Tensor

from .sinkhorn import sinkhorn
from .utils import markov_measure

def wl_k(MX: Tensor, MY: Tensor, 
        l1: Tensor, l2: Tensor,
        k: int,
        muX: Tensor | None = None,
        muY: Tensor | None = None, 
        reg: float=.1, 
        sinkhorn_iter: int= 100,
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
        sinkhorn_iter: number of sinkhorn iterations for a step
    """
    b, n, n_ = MX.shape
    b_, m, m_ = MY.shape
    assert (n==n_) and (m == m_) and (b == b_)
    cost_matrix = (l1[:, :, None] - l2[:, None, :]).abs()
    
    for i in range(k):
        cost_matrix = sinkhorn(
                MX[:, :, None, :], # b, n, 1, n
                MY[:, None, :, :], # b, 1, m, m
                cost_matrix[:, None, None, :, :], # b, 1, 1, n, m
                epsilon=reg, 
                k= sinkhorn_iter
        ) # b, n, m

    if muX is None: 
        muX = markov_measure(MX)
    if muY is None:
        muY = markov_measure(MY)

    return sinkhorn(muX, muY, cost_matrix, reg)
