from typing import Literal
import networkx as nx
import sklearn.neighbors 
import numpy as np
from numpy.random import default_rng
rng = default_rng()

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
            return distance_graph(sample, .5 * radius)

