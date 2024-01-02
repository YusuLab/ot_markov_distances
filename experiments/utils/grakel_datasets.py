from dataclasses import dataclass, field
from typing import Any, Literal

from unittest.mock import patch
from urllib.request import urlopen, HTTPError
import ssl
from shutil import copyfileobj

import grakel
import networkx as nx
import numpy as np

# this function doesn’t function in grakel 
# so I monkey-patch it here, correcting the bug
def _download_zip(url, output_name):
    """Download a file from a requested url and store locally.
    Parameters
    ----------
    url : str
        The url from where the file will be downloaded.
    output_name : str
        The name of the file in the local directory.
    Returns
    -------
    None.
    """
    #ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context = ssl._create_unverified_context()
    filename = output_name + ".zip"
    try:
        data_url = urlopen(url, context=context)
    except HTTPError as e:
        if e.code == 404:
            e.msg = "Dataset '%s' not found on mldata.org." % output_name
        raise

    # Store Zip File
    try:
        with open(filename, 'w+b') as zip_file:
            copyfileobj(data_url, zip_file)
    except Exception:
        os.remove(filename)
        raise
    data_url.close()
    
def grakel_to_nx(G, include_attr=True) -> list[nx.Graph]:
    nx_G: list[nx.Graph] = []
    for graph in G:
        adj_mat = graph.get_adjacency_matrix()
        nx_graph = nx.from_numpy_array(adj_mat)
        nodes = sorted(list(graph.node_labels.keys()))
        if include_attr == True:
            for i in range(adj_mat.shape[0]):
                nx_graph.nodes[i]["attr"] = graph.node_labels[nodes[i]]
        nx_G.append(nx_graph)
    return nx_G

def grakel_to_igraph(G, add_attr=False):
    lst = []
    attr_list = []
    max_nodes = 0
    szs = []
    for graph in G:
        adj_mat = graph.get_adjacency_matrix()
        igraph = ig.Graph.Adjacency(adj_mat)
        n = adj_mat.shape[0]
        if add_attr:
            nodes = sorted(list(graph.node_labels.keys()))
            attrs = []
            for i in range(len(nodes)):
                attrs.append(graph.node_labels[nodes[i]])
            igraph.vs["label"] = attrs
        if n > max_nodes:
            max_nodes = n
        lst.append(igraph)
    if add_attr:
        for graph in G:
            nodes = sorted(list(graph.node_labels.keys()))
            attrs = [0]*max_nodes
            for i in range(graph.n):
                attrs[i] = graph.node_labels[nodes[i]]
            attr_list.append(attrs)
    return lst, attr_list


@dataclass
class Dataset:
    title: str
    grakel_name: str
    grakel_ds: Any =  field(repr=False)
    graphs_nx: list[nx.Graph] = field(repr=False)
    target: np.ndarray = field(repr=False)
    attr_type: Literal["discrete", "continuous", "degree"] 
    
    
    @classmethod
    def get(cls, grakel_name, title=None, attr_type=None):
        if title is None:
            title = grakel_name
        with patch("grakel.datasets.base._download_zip", _download_zip):    
            ds = grakel.datasets.fetch_dataset(
                    grakel_name, as_graphs=True, 
                    produce_labels_nodes=attr_type == "degree", 
                    prefer_attr_nodes=attr_type == "continuous", 
            )
        G = ds.data
        nx_G = grakel_to_nx(G, include_attr=True)
        #igraphs, graph_attrs = grakel_to_igraph(G, add_attr=False)
        y = ds.target
        if attr_type is None:
            raise NotImplementedError
        return cls(title=title, 
                   grakel_name=grakel_name, 
                   grakel_ds=ds, 
                   graphs_nx = nx_G, 
                   target=y, 
                   attr_type=attr_type, 
                   )


