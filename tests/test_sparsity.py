import pytest
import itertools as it

from ot_markov_distances.misc import all_equal
from ot_markov_distances import discounted_wl_infty, sinkhorn_distance
from ot_markov_distances.utils import weighted_transition_matrix, densify, reindex_cost_matrix, cost_matrix_index
from experiments.utils.data_generation import circle_graph

import torch


@pytest.fixture(params=[(2, 17, 23), (5, 9, 11), (10, 5, 7)])
def generate_data(batch_size= 2, n1 = 17, n2 = 23):
    test_sizes=[n1] * batch_size
    test_graphs = [circle_graph(size, kind="nn", k=2) for size in test_sizes]
    test_matrices = [weighted_transition_matrix(g, q=.1) for g in test_graphs]
    mx = torch.stack(test_matrices)
    test_sizes=[n2] * batch_size
    test_graphs = [circle_graph(size, kind="nn", k=3) for size in test_sizes]
    test_matrices = [weighted_transition_matrix(g, q=.1) for g in test_graphs]
    my = torch.stack(test_matrices)
    C = torch.rand((batch_size, n1, n2)) * 5

    return mx, my, C

def test_densify(generate_data):
    mx, _, _ = generate_data
    mx_index, mx_mask, mx_dense = densify(mx)

    *batch, n, n_ = mx.shape
    assert all_equal(n, n_)
    # flat stuff
    mx_flat = mx.view(-1, n)
    n_flat, _ = mx_flat.shape #n_flat = n * batch

    mx_flat_degree = (mx_flat>0).sum(-1)
    mx_max_degree = mx_flat_degree.max()

    # indices and padding masks
    nonzero_indices = torch.zeros((*batch, n, mx_max_degree), dtype=torch.long)
    nonzero_indices_mask = torch.zeros((*batch, n, mx_max_degree), 
                                       dtype=torch.bool)
    nzi_flat = nonzero_indices.view(-1, mx_max_degree)
    nzi_flat_mask = nonzero_indices_mask.view(-1, mx_max_degree)

    for i in range(n_flat):
        line, = mx_flat[i].nonzero(as_tuple=True)
        line_len = len(line)
        nzi_flat[i, :line_len] = line
        nzi_flat_mask[i, :line_len] = 1

    assert (mx_index ==  nonzero_indices).all()
    assert (mx_mask == nonzero_indices_mask).all()

    assert torch.allclose(mx_dense.sum(-1), torch.tensor(1.))
    for b, i, j in it.product(*(range(i) for i in (*batch, n, mx_max_degree))):
        assert mx_dense[b, i, j] == mx[b, i, mx_index[b, i, j]] or not mx_mask[b, i, j].item()
    for b, i, j in it.product(*(range(i) for i in (*batch, n, n))):
        assert (mx[b, i, j].item() == 0) == (
            j not in mx_index[b, i]
            or not mx_mask[b, i, list(mx_index[b, i]).index(j)]
        ) 

def test_densified_sinkhorn(generate_data):
    mx, my, C = generate_data
    mx_index, mx_mask, mx_dense = densify(mx)
    my_index, my_mask, my_dense = densify(my)

    b, n, m = C.shape
    b_, n_, dx = mx_index.shape
    b__, m_, dy = my_index.shape
    assert all_equal(b, b_, b__) and n == n_ and m == m_

    C = C.contiguous()

    C_index, C_mask = cost_matrix_index(C, mx_index, my_index, mx_mask, my_mask)
    C_dense = reindex_cost_matrix(C, C_index, C_mask)

    for bi, ni, mi, i1, i2 in it.product(range(b), range(n), range(m), range(dx), range(dy)):
        #print(bi, ni, mi, i1, i2)
        assert ((C_dense[bi, ni, mi, i1, i2] ==
                C[bi, mx_index[bi, ni, i1], my_index[bi, mi, i2]]) or
                (not C_mask[bi, ni, mi, i1, i2]))

    assert torch.allclose(sinkhorn_distance(mx_dense[:, :, None, :], my_dense[:, None, :, :], 
                                            C_dense, epsilon=.1, max_iter=1000), 
                          sinkhorn_distance(mx[:, :, None, :], my[:, None, :, :], 
                                            C[:, None, None, :, :], epsilon=.1, max_iter=1000), 
                          atol=1e-4, rtol=1e-2)


