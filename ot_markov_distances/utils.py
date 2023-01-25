import networkx as nx
import torch
from torch import Tensor

def weighted_transition_matrix(G:nx.Graph, q: float):
    """Create a LMMC from a Graph

    See def 5 in :citet:`chenWLdistance2022`,
    
    It returns the transition matrix of the random walk on graph G, 
    where the probability of staying in the same node is q,
    and the probablity of transiting to one of the neighboring nodes
    is equiprobable

    Args:
        G: the graph
        q: the q parameter for the q-markov chain

    Returns:
        Array: weighted transition matrix associated to q-random walks on G
    """
    A: Tensor= torch.as_tensor(nx.adjacency_matrix(G, weight=None).todense())#type:ignore
    n, _n = A.shape; assert n == _n
    D = A.sum(1, keepdim=True)
    mask = D == 0
    D[mask] = 1
    D = D.reshape((A.shape[0], 1))
    A = (1 - q)*A/D
    A = A + q * torch.eye(n)
    single_node_inds = torch.nonzero(mask)[:, 0]
    A[single_node_inds, single_node_inds] = 1
    return A

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


def double_last_dimension(t: Tensor) -> Tensor:
    """
    doubles last dimension of a tensor, by making it into a diagonal matrix
    Args:
        t: (..., d)
    Returns:
        t' (..., d, d) where
            `t'[..., i, i] = t[..., i]` and
            `t'[..., i, k] = 0 if i != k`
    """
    *batch, i = t.shape
    new_tensor = torch.zeros((*batch, i, i), device=t.device)
    new_tensor = torch.diagonal_scatter(new_tensor, t, dim1=-2, dim2=-1)
    return new_tensor

def draw_markov(M: Tensor,pos, node_color=None, ax=None):
    import numpy as np
    match node_color:
        case None:
            nc = {}
        case np.ndarray():
            nc = {"node_color":node_color}
        case torch.Tensor():
            nc = {"node_color":node_color.numpy(force=True)}
        case _:
            raise TypeError
    
    G = nx.from_numpy_array(M.numpy(force=True), create_using=nx.DiGraph)
    if ax is None:
        import matplotlib.pyplot as plt #type:ignore
        ax = plt.gca()
    nx.draw_networkx_edges(G, 
                pos,
                ax=ax,
                alpha=np.array(list(nx.get_edge_attributes(G, "weight").values())),
                    )
    nx.draw_networkx_nodes(G, pos, ax=ax, **nc)
