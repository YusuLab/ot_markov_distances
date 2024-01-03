import pytest
import warnings
import math

import networkx as nx
import torch

from ot_markov_distances import sinkhorn_distance
from ot_markov_distances.utils import weighted_transition_matrix

from ot import emd

@pytest.mark.parametrize("n, m, p", [(4, 3, .5), (5, 4, .5), (6, 5, .5)])
def test_sinkhorn_distance(n, m, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G1 = nx.erdos_renyi_graph(n, p)
        G2 = nx.erdos_renyi_graph(m, p)
        M1 = torch.tensor(weighted_transition_matrix(G1, .1), 
                          dtype=torch.float32, )
        M2 = torch.tensor(weighted_transition_matrix(G2, .1), 
                          dtype=torch.float32)
        #l1 = torch.tensor(np.array([G1.degree(i) for i in G1.nodes]), dtype=torch.float32)
        #l2 = torch.tensor(np.array([G2.degree(i) for i in G2.nodes]), dtype=torch.float32)
        l1 = torch.randn(len(G1))
        l2 = torch.randn(len(G2))
        MX = torch.tensor(M1[None, ...], dtype=torch.float32, 
                          requires_grad=True)
        MY = torch.tensor(M2[None, ...], dtype=torch.float32, 
                          requires_grad=True)
        l1 = torch.tensor(l1[None, ...], dtype=torch.float32, 
                          requires_grad=True)
        l2 = torch.tensor(l2[None, ...], dtype=torch.float32, 
                          requires_grad=True)
    
    
    mx = MX[:, 0, :].squeeze()
    my = MY[:, 0, :].squeeze()
    c = abs(l1[:, :, None] - l2[:, None, :]).squeeze()
    epsilon=.001
    
    
    # pot = (ot_sinkhorn_log(mx, my, c, reg=epsilon) * c).sum()
    pot_emd = (emd(mx, my, c) * c).sum()
    ours, has_converged = sinkhorn_distance(MX[:, 0, :], MY[:, 0, :], 
                             abs(l1[:, :, None] - l2[:, None, :]), 
                             epsilon=epsilon, return_has_converged=True)

    # assert torch.allclose(pot, ours)
    if has_converged:
        assert torch.allclose(ours, pot_emd, 
                          atol=epsilon * (math.log(n) + math.log(m) + 1))
    else:
        warnings.warn("Sinkhorn distance did not converge")

