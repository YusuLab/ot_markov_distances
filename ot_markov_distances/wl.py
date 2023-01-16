import torch
from torch import Tensor

from .sinkhorn import sinkhorn

def markov_measure(M: Tensor) -> Tensor:
    """Takes a (batched) markov transition matrix, 
    and outputs its stationary distribution

    Args:
        M: (*b, n, n) the markov transition matrix

    Returns:
        Tensor: m (*b, n)  so that m @ b  = m
    """
    *b, n, n_ = M.shape
    assert n == n_
    target = torch.zeros((*b, n+1, 1), device=M.device)
    target[..., n, :] = 1
    
    equations = (M.transpose(-1, -2) - torch.eye(n, device=M.device))
    equations = torch.cat([equations, torch.ones((*b, 1, n), device=M.device)], dim=-2)
    
    sol, *_ = torch.linalg.lstsq(equations, target)
    return sol.abs().squeeze(-1)


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
