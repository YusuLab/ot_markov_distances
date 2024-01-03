from typing import Literal
import math

import networkx as nx
import sklearn.neighbors 
import torch
import numpy as np
from numpy.random import default_rng
from ot_markov_distances.utils import weighted_transition_matrix, draw_markov
rng = default_rng()

def get_oriented_circle(n, doubled_edges=1):
    G = nx.DiGraph()
    angles = np.linspace(0, 2*np.pi, n)
    x = np.cos(angles)
    y = np.sin(angles)
    coords = np.stack([x, y], axis=-1)
    for node_index, node_coords in zip(range(n), coords):
        G.add_node(node_index, attr=node_coords, pos=node_coords)
        for i in range(1, doubled_edges+1):
            G.add_edge((node_index - i) % n, node_index)
    return G

def gen_noisy_oriented_circles(n_targets, target_size, er_p=.05, doubled_edges=2):
    graphs = [get_oriented_circle(target_size, doubled_edges) for _ in range(n_targets)]
    graphs = [add_er_noise(graph, p=er_p) for graph in graphs]
    markovs = torch.stack([weighted_transition_matrix(graph, 0) for graph in graphs])
    distributions = torch.ones(n_targets, target_size) / target_size
    labels = torch.as_tensor(np.asarray(
        [[i for i in nx.get_node_attributes(graph, "attr").values()
            ] for graph in graphs ]
         ), dtype=torch.float32)
    return graphs, markovs, distributions, labels


def circle_sample(n, radius=1, noise=.01):
    #we add gaussian noise of size radius * noise
    thetas = rng.random((n,)) * 2 * np.pi
    values = np.stack((np.cos(thetas), np.sin(thetas)), axis=-1)
    values = values * radius
    values = values + rng.standard_normal(values.shape) * radius * noise
    return values

def nn_graph(samples, k) -> nx.Graph:
    graph_adj = sklearn.neighbors.kneighbors_graph(samples, k)
    G: nx.Graph = nx.from_numpy_array(graph_adj)
    for (_, data), position in zip(G.nodes(data=True), samples):
        data["pos"] = position
    return G

def distance_graph(samples, d) -> nx.Graph:
    distance_matrix = np.square(samples[:, None, :]- samples[None, :, :]).mean(-1)
    connected_matrix = distance_matrix <= d*d
    G: nx.Graph = nx.from_numpy_array(connected_matrix)
    for (_, data), position in zip(G.nodes(data=True), samples):
        data["pos"] = position
    return G

def circle_graph(n, radius=1, noise=.01 , kind:Literal["nn", "distance"]="nn", k=5):
    sample = circle_sample(n, radius, noise)

    match kind:
        case "nn":
            return nn_graph(sample, k)
        case "distance":
            return distance_graph(sample, .7 * radius)


def FGW_build_noisy_circular_graph(N=20,mu=0,sigma=0.3,structure_noise_p=0.):
    """Credit: this code is from the Fused Gromov wasserstein code base, 
    (https://github.com/tvayer/FGW)
    slightly modified

    modifications:
        - returns a networkx.Graph
        - attributes are the two coordinates
        - no "with_noise" parameter. to disable noise, just set sigma to 0.
        - also simplified edge cases with good old modulo trick

    """
    g=nx.Graph()
    g.add_nodes_from(range(N))
    for i in range(N):
        noise = rng.normal(mu, sigma, 2) 
        angle = 2 * i * np.pi / N
        g.nodes[i]["attr"] = np.array([np.cos(angle), np.sin(angle)]) + noise
        g.nodes[i]["pos"] = g.nodes[i]["attr"]
        g.add_edge(i,(i+1)%N)
        if rng.random() < structure_noise_p:
            g.add_edge(i,(i+2)%N)
    return g


def add_er_noise(G, p):
    """ Adds Erdos-Renyi type noise to a graph G
    adds or remove k random edges from graph
    """
    G = nx.convert_node_labels_to_integers(G)
    G_matrix = nx.to_numpy_array(G)
    noise = nx.to_numpy_array(nx.erdos_renyi_graph(len(G), p, directed=G.is_directed()))
    new_G_matrix = np.logical_xor(G_matrix, noise)
    new_G = nx.create_empty_copy(G)
    edges = nx.from_numpy_array(new_G_matrix, create_using=type(G)).edges()
    new_G.add_edges_from(edges)
    return new_G

def get_label_matrix(G, label="attr"):
    values = [ torch.as_tensor(v) for v in nx.get_node_attributes(G, label).values()]
    values = torch.stack(values, dim=0)
    values = values.to(torch.float32)
    return values


def graphon_sample(M):
    M_sampled = rng.random(M.shape)<M
    return nx.from_numpy_array(M_sampled)
