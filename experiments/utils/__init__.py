import networkx as nx
import numpy as np

def grid_init(*args):
    match args:
        case []:
            return None
        case [n0, *tail]:
            return [grid_init(*tail) for _ in range(n0)]

def my_threshold(M):
    median = np.median(M)
    #print(median)
    return 3 * median # why hard when it could be easy

def markov_threshold(M, method=my_threshold):
    threshold = method(M)
    G = nx.from_numpy_array( M> threshold, create_using=nx.DiGraph)
    return G
