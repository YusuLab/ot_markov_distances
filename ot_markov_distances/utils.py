"""
This module contains a lot of miscellanous functions.

It contains

- Functions on markov chains: :py:func:`weighted_transition_matrix` , :py:func:`weighted_transition_matrix`, :py:func:`degree_markov`
- Display functions :py:func:`draw_markov`
- Support for sparse markov chains: :py:func:`densify`, :py:func:`dummy_densify`, :py:func:`cost_matrix_index` etc.


It would gain to be broken up into submodules, and merged with misc.py
(as utils.markov, utils.display, utils.misc, utils.sparse)
(TODO for a next version)
"""
import torch
from torch import Tensor, LongTensor, FloatTensor, BoolTensor
import networkx as nx

from .misc import all_equal

def weighted_transition_matrix(G:nx.Graph, q: float):
    """Create a LMMC from a Graph

    This function is largely inspired from the same one here
    https://github.com/chens5/WL-distance/blob/main/utils/utils.py

    See def 5 in :cite:t:`chen2022weisfeilerlehman`,
    
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
    r"""Takes a (batched) markov transition matrix, 
    and outputs its stationary distribution

    Args:
        M: (\*b, n, n) the markov transition matrix

    Returns:
        Tensor: m (\*b, n)  so that m @ b  = m
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

def draw_markov(M: Tensor,pos=None, node_color=None, ax=None):
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
    if pos is None:
        pos  = nx.spring_layout(G)
    if ax is None:
        import matplotlib.pyplot as plt #type:ignore
        ax = plt.gca()
    nx.draw_networkx_edges(G, 
                pos,
                ax=ax,
                alpha=np.array(list(nx.get_edge_attributes(G, "weight").values())),
                    )
    nx.draw_networkx_nodes(G, pos, ax=ax, **nc)

def degree_markov(m: torch.Tensor) -> int:
    return m.count_nonzero(dim=-1).max().item() #type:ignore


def densify(m: torch.FloatTensor)-> tuple[Tensor, Tensor, Tensor]:
    r"""make a sparse markov matrix into a dense one

    Takes a (batched) markov matrix `m` where every transition is very sparse.

    Call :math:`d := \max_i(\deg(m_i)) = \max_i #\{j, m_{ij} \neq 0 \} `
    (maxmized over the batch)

    Assume d is low (so every one of the :math:`m_i` is sparse)

    call, for :math:`0 \leq i < [n]`, 
    :math:`\text{index}_{i, 0} \ldots \text{index}_{i, d-1}`
    the indices where :math:`m_i` is nonzero, in increasing order
    (eventually 0-padded if there are less than d)

    this gives us an `index` array of size `(*batch, n, d)`

    call also `index_mask` the mask of size `(*batch, n, d)` corresponding to 
    the 0-padding we did on `index` (so `index_mask[*b, i, j] = True` 
    if `index[*b, i, j]` is an actual index, False if it’s just padding)

    Finally, call `m_dense` so that

    .. math::

        m_dense[*b, i, j] = \begin{cases}
            m[*b, i, index[*b, i, j]]  \text{ if } index_mask[*b, i, j] \\
            0 \text{ otherwise }
        \end{cases}

    Args:
        m: `(*batch, n, n)` input markov matrix

    Returns:
        m_indices: `(*batch, n, d)`, LongTensor
        m_indices_mask: `(*batch, n, d)`, BoolTensor
        m_dense: `(*batch, ), Floattensor

    """

    *batch, n, n_ = m.shape
    assert all_equal(n, n_)

    # flatten stuff (working with less dimensions is easier)
    m_flat = m.view(-1, n)
    n_flat, _ = m_flat.shape #n_flat = n * batch

    
    # max degree
    m_flat_degree = (m_flat>0).sum(-1)
    m_max_degree:int = m_flat_degree.max().item() #type:ignore


    # indices and padding masks
    nonzero_indices = torch.zeros((*batch, n, m_max_degree), 
                                  dtype=torch.long, device=m.device)
    nonzero_indices_mask = torch.zeros((*batch, n, m_max_degree), 
                                       dtype=torch.bool, device=m.device)
    nzi_flat = nonzero_indices.view(-1, m_max_degree)
    nzi_flat_mask = nonzero_indices_mask.view(-1, m_max_degree)

    for i in range(n_flat):
        line, = m_flat[i].nonzero(as_tuple=True)
        line_len = len(line)
        nzi_flat[i, :line_len] = line
        nzi_flat_mask[i, :line_len] = 1

    
    #densified
    # we want mx_dense[*b, i, k] = mx[*b, i, nonzero_indices[*b, i, k]] if nonzero_indices_mask[*b, i, k] else 0
    m_dense = torch.empty((*batch, n, m_max_degree), 
                          dtype=m.dtype, device=m.device)
    m_dense_flat = m_dense.view((-1, m_max_degree))
    m_dense_flat[...] = torch.where(nzi_flat_mask, m_flat.gather(1, nzi_flat), 0)#type:ignore

    return nonzero_indices, nonzero_indices_mask, m_dense

def dummy_densify(m: torch.FloatTensor)\
        -> tuple[LongTensor, BoolTensor, FloatTensor]:
    """Same as densify, but returns dummy results:
        the densified matrix is the full matrix, 
        the index is simply (1, 2, ..., n)
        and the mask is all true
    """
    _, n, n_ = m.shape
    assert n == n_
    m_index: LongTensor = torch.arange(n, dtype=torch.long, device=m.device)\
            [None, None, :]\
            .expand_as(m)#type:ignore 
    m_mask: BoolTensor = torch.ones_like(m, dtype=torch.bool, device=m.device)#type:ignore
    return m_index, m_mask, m

def cost_matrix_index(C: Tensor, mx_index: Tensor, my_index: Tensor, 
                      mx_mask: Tensor, my_mask: Tensor):
    r"""reindex the cost matrix to go with densified distributions

    takes the indices mx_index and my_index output by densify, as well as a cost matrix,
    and reindexes the cost matrix to go with the densified distributions.

    More precisely, if

    ```
    mx_index, mx_mask, mx_dense =  densify(mx)
    my_index, my_mask, my_dense =  densify(my)
    C_index, C_mask = cost_matrix_index(mx_index, my_index, mx_mask, my_mask)
    C_dense = C.view(-1)[C_index]
    ```

    then 

    ```
    sinkhorn(mx_dense[:, :, None, :], my_dense[:, None, :, :], C_dense, epsilon) ==  sinkhorn(mx[:, :, None, :], my[:, None, :, :], C[:, None, None, :, :])
    ```

    Args:
        mx_index: `(*batch, n, dx)`, LongTensor
        my_index: `(*batch, m, dy)`, LongTensor
        

    Returns:
        C_index: 
        C_mask

    """
    b, _, _ = C.shape
    assert C.is_contiguous()
    C_index = C.stride(1) * mx_index[:, :, None, :, None] +  my_index[:, None, :, None, :] \
            + torch.arange(0, b * C.stride(0), step=C.stride(0), dtype=torch.long, device=C.device)[:, None, None, None, None]
    C_mask = (mx_mask[:, :, None, :, None] & my_mask[:, None, :, None, :])

    return C_index, C_mask

def reindex_cost_matrix(C, C_index, C_mask):
    assert C.is_contiguous()
    C_dense = C.view(-1)[C_index]
    C_dense[C_mask == False] = 1.
    return C_dense


def re_project_C(C_dense: FloatTensor, C_index: LongTensor, 
                 C_mask: BoolTensor|None=None):
    """Inverse of reindex_cost_matrix

    Takes a (b, n, m, dx, dy) reindexed cost matrix (or matching matrix)
    and outputs a (b, n, m, n, m) cost matrix C_sparse
    where C_sparse[b, i, j, mx_index[b, i, j, k], my_index[b, i, j, l]] 
        = C_dense[b, i, j, k, l]
    And the rest are filled with zeros

    Args:
        C_dense: (b, n, m, dx, dy)
        C_index: (b, n, m, dx, dy)
        C_mask: (b, n, m, dx, dy)
    """
    b, n, m, dx, dy = C_dense.shape
    if C_mask is not None:
        C_dense = C_dense.clone() #type: ignore
        C_dense[C_mask==0]=0
    C_dense = torch.einsum("bijxy->ijbxy", C_dense) #type: ignore
    C_index = torch.einsum("bijxy->ijbxy", C_index) #type: ignore
    C = torch.zeros((n, m, b, n, m), dtype=C_dense.dtype, 
                    device=C_dense.device).view(n, m, -1) 
    C.scatter_add_(-1, C_index.reshape(n, m, -1), C_dense.reshape(n, m, -1))
    return torch.einsum("ijbxy->bijxy", C.view(n, m, b, n, m))

def pad_one_markov_to(M: Tensor, mu: Tensor,  n: int):
    """Pad a markov chain to n states

    Pads a markov chain to n states by adding 
    (n - m) states with zero distribution mass, and transitions to themselves wp 1.

    Args:
        M: (m, m) transition matrix
        mu: (m) distribution
        n: the number of states to pad to
    """
    m, m_ = M.shape
    assert m == m_
    assert m <= n

    M_pad = torch.eye((n, n), dtype=M.dtype, device=M.device)
    M_pad[:m, :m] = M
    mu_pad = torch.zeros((n,), dtype=mu.dtype, device=mu.device)
    mu_pad[:m] = mu
    return M_pad, mu_pad


def pad_markovs(transition_matrices: list[Tensor], distributions: list[Tensor]):
    """Pad markov chains to the same number of states

    Pads markov chains to the same number of states by adding 
    states with zero distribution mass, and transitions to themselves wp 1.

    Args:
        transition_matrices: list of  transition matrices (of shape (m_i, m_i))
        distributions: list of distributions (of shape (m_i))
    """
    sizes = []
    for M, mu in zip(transition_matrices, distributions):
        m, m_ = M.shape
        m__, = mu.shape
        assert all_equal(m, m_, m__)
        sizes.append(m)
    n = max(sizes)
    transition_matrices_pad = [ pad_one_markov_to(M, mu, n) for M, mu in zip(transition_matrices, distributions)]
    return torch.stack(transition_matrices_pad, dim=0)

