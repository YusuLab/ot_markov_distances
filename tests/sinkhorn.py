import warnings

import networkx as nx
import torch
from torch import Tensor

from ot_markov_distances import sinkhorn

from ot.bregman import sinkhorn_log as ot_sinkhorn_log
from ot import emd


n = 4; m = 3; p=.5
with warnings.catch_warnings():
    G1 = nx.erdos_renyi_graph(n, p)
    G2 = nx.erdos_renyi_graph(m, p)
    M1 = torch.tensor(weighted_transition_matrix(G1, .1), dtype=torch.float32, )
    M2 = torch.tensor(weighted_transition_matrix(G2, .1), dtype=torch.float32)
    #l1 = torch.tensor(np.array([G1.degree(i) for i in G1.nodes]), dtype=torch.float32)
    #l2 = torch.tensor(np.array([G2.degree(i) for i in G2.nodes]), dtype=torch.float32)
    l1 = torch.randn(len(G1))
    l2 = torch.randn(len(G2))
    MX = torch.tensor(M1[None, ...], dtype=torch.float32, requires_grad=True)
    MY = torch.tensor(M2[None, ...], dtype=torch.float32, requires_grad=True)
    l1 = torch.tensor(l1[None, ...], dtype=torch.float32, requires_grad=True)
    l2 = torch.tensor(l2[None, ...], dtype=torch.float32, requires_grad=True)


mx = MX[:, 0, :].squeeze()
my = MY[:, 0, :].squeeze()
c = abs(l1[:, :, None] - l2[:, None, :]).squeeze()
epsilon=.001


print("POT:")
print((ot_sinkhorn_log(mx, my, c, reg=epsilon) * c).sum())
print("POT EMD:")
print((emd(mx, my, c) * c).sum() )

print("ours:")
print(sinkhorn(MX[:, 0, :], MY[:, 0, :], abs(l1[:, :, None] - l2[:, None, :]), epsilon=epsilon))
