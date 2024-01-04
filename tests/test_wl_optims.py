import pytest
import itertools as it

from ot_markov_distances.misc import all_equal
from ot_markov_distances import discounted_wl_infty, discounted_wl_k
from ot_markov_distances.utils import weighted_transition_matrix, densify, reindex_cost_matrix, cost_matrix_index
from experiments.utils.data_generation import circle_graph

import torch

@pytest.fixture(params=[(1, 2, 3, 10), (2, 17, 23, 10), (5, 9, 11, 22), (10, 5, 7, 3)])
def generate_data(batch_size= 2, n1 = 17, n2 = 23, label_size=10):
    test_sizes=[n1] * batch_size
    test_graphs = [circle_graph(size, kind="nn", k=2) for size in test_sizes]
    test_matrices = [weighted_transition_matrix(g, q=.1) for g in test_graphs]
    mx = torch.stack(test_matrices)
    test_sizes=[n2] * batch_size
    test_graphs = [circle_graph(size, kind="nn", k=3) for size in test_sizes]
    test_matrices = [weighted_transition_matrix(g, q=.1) for g in test_graphs]
    my = torch.stack(test_matrices)
    #labels for mx and my
    lx = torch.rand((batch_size, n1, label_size))
    ly = torch.rand((batch_size, n2, label_size))
    # C = torch.rand((batch_size, n1, n2)) * 5
    C = torch.square(lx[:, :, None] - ly[:, None, :]).sum(-1) 

    return mx, my, lx, ly, C


# test that discounted wl infty with all optimizations returns the same as the naive version discounted wl k with big k
def test_discounted_wl_infty_optim(generate_data):
    mx, my, lx, ly, C = generate_data

    dwl_infty = discounted_wl_infty(mx, my, C, delta=.9, sinkhorn_reg=.1)
    dwl_k = discounted_wl_k(mx, my, cost_matrix=C, k=1000, delta=.9, reg=.1)

    assert torch.allclose(dwl_infty, dwl_k, rtol=1e-2, atol=1e-2)
