import itertools as it; import more_itertools as mit
import gc
import functools as ft
import logging; logger = logging.getLogger(__name__)
import os, sys

from torch.cuda import OutOfMemoryError

def append_path(s):
    s = os.path.abspath(s)
    if s in sys.path: return
    sys.path.insert(0,s)

import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm

from ot_markov_distances import discounted_wl_infty
from ot_markov_distances.wl import wl_k_sparse as wl_k
from ot_markov_distances.utils import weighted_transition_matrix

append_path(f"{os.environ['HOME']}/research/FGW/lib")
from ot_distances import Fused_Gromov_Wasserstein_distance #this is FGW code, it’s not properly packaged so we have to import it like this
from graph import Graph as FGW_Graph

def to_FGW_Graph(G, attr_name="attr"):
    for node in G.nodes:
        G.nodes[node]["attr_name"] = G.nodes[node][attr_name]
    return FGW_Graph(G)

default_wl_parameters = dict(delta=.2, 
                   x_is_sparse=True, 
                   y_is_sparse=True,
                   convergence_threshold_rtol = .01,
                   convergence_threshold_atol = 1e-3) 

default_wl_k_parameters = dict(
    sinkhorn_parameters = dict( 
        max_iter=20,
        cv_atol=1e-3, cv_rtol=1e-2
    )
)

def compute_distance_matrix( set1: list[nx.Graph], set2: list[nx.Graph], 
                            attr_type=None, 
                            device=torch.device("cpu"), q=.1, k=None, **wl_parameters,):
    del k #ignore k if it is in the params
    distance_matrix = torch.zeros(len(set1), len(set2), device=device)
    wl_parameters = {**default_wl_parameters, **wl_parameters}
    logger.info(f"wl_parameters: {wl_parameters}")

    
    if set1 is set2:
        #only compute the upper triangular part
        iterator = it.combinations(enumerate(set1), 2)
        iterator = (((clip1_i, clip1), (clip2_i, clip2))
                    for (clip1_i, clip1), (clip2_i, clip2) in iterator
                    if clip1_i != clip2_i
                    )
        n_iter = (len(set1) * (len(set1) - 1) // 2 )
    else: 
        iterator = it.product(enumerate(set1), enumerate(set2))
        n_iter = len(set1) * len(set2)
    iterator = tqdm(iterator, total = n_iter)
    
    for (clip1_i, clip1), (clip2_i, clip2) in iterator:
        #distance matrix
        attr1 = torch.as_tensor(list(nx.get_node_attributes(clip1, "attr").values()), device=device)
        attr2 = torch.as_tensor(list(nx.get_node_attributes(clip2, "attr").values()), device=device)
        if (attr_type is None and torch.is_floating_point(attr1)
            or attr_type == "continuous"):
            D = torch.cdist (attr1, attr2)
            # D = (attr1[:, None] - attr2[None, :]).square()
            # if len(D.shape)>2:
            #     D = D.mean(*range(2, len(D.shape)))
        else:
            if len(attr1.shape) > 1: attr1 = attr1.squeeze()
            if len(attr2.shape) > 1: attr2 = attr2.squeeze()
            D = (attr1[:, None] != attr2[None, :])
        D = D.to(torch.float32)

        #markov chains
        markov1 = weighted_transition_matrix(clip1, q=q)
        markov2 = weighted_transition_matrix(clip2, q=q)
        
        #distributions
        mu1 = torch.ones(1, len(clip1), device=device) / len(clip1)
        mu2 = torch.ones(1, len(clip2), device=device) / len(clip2)
        
        #distance
        distance_matrix[clip1_i, clip2_i] = \
            discounted_wl_infty(markov1.to(device)[None, ...], 
                                markov2.to(device)[None, ...], 
                                D[None, ...], 
                                muX=mu1,
                                muY=mu2,
                                **wl_parameters).square()
        
    if set1 is set2:
        distance_matrix = distance_matrix + distance_matrix.T
    return distance_matrix


def compute_distance_matrix_fgw( set1: list[nx.Graph], set2: list[nx.Graph], 
                            attr_type=None, 
                            **fgw_parameters,):
    distance_matrix = torch.zeros(len(set1), len(set2), device="cpu")
    match attr_type:
        case None:
            raise ValueError("attr_type must be specified")
        case "continuous":
            fgw_parameters["features_metric"] = "sqeuclidean"
        case "discrete":
            fgw_parameters["features_metric"] = "dirac"

    gw = Fused_Gromov_Wasserstein_distance(**fgw_parameters)
    logger.info(f"fgw_parameters: {fgw_parameters}")

    dist=lambda G1,G2: gw.graph_d(to_FGW_Graph(G1), to_FGW_Graph(G2))

    
    if set1 is set2:
        #only compute the upper triangular part
        iterator = it.combinations(enumerate(set1), 2)
        iterator = (((clip1_i, clip1), (clip2_i, clip2))
                    for (clip1_i, clip1), (clip2_i, clip2) in iterator
                    if clip1_i != clip2_i
                    )
        n_iter = (len(set1) * (len(set1) - 1) // 2 )
    else: 
        iterator = it.product(enumerate(set1), enumerate(set2))
        n_iter = len(set1) * len(set2)
    iterator = tqdm(iterator, total = n_iter)
    
    for (clip1_i, clip1), (clip2_i, clip2) in iterator:
        try:
            distance_matrix[clip1_i, clip2_i] = dist(clip1, clip2)
        except Exception as e:
            logger.error(f"Error computing distance between {clip1_i} and {clip2_i}: {e}")
            distance_matrix[clip1_i, clip2_i] = np.nan
        
    if set1 is set2:
        distance_matrix = distance_matrix + distance_matrix.T
    return distance_matrix


def compute_distance_matrix_dwl( set1: list[nx.Graph], set2: list[nx.Graph], 
                            attr_type=None, 
                            device=torch.device("cpu"), q=.1, k=2, **wl_parameters,):
    wl_parameters = {**default_wl_k_parameters, **wl_parameters}
    distance_matrix = torch.zeros(len(set1), len(set2), device=device)
    logger.info(f"wl_parameters: {wl_parameters}")

    
    if set1 is set2:
        #only compute the upper triangular part
        iterator = it.combinations(enumerate(set1), 2)
        iterator = (((clip1_i, clip1), (clip2_i, clip2))
                    for (clip1_i, clip1), (clip2_i, clip2) in iterator
                    if clip1_i != clip2_i
                    )
        n_iter = (len(set1) * (len(set1) - 1) // 2 )
    else: 
        iterator = it.product(enumerate(set1), enumerate(set2))
        n_iter = len(set1) * len(set2)
    iterator = tqdm(iterator, total = n_iter)
    
    for (clip1_i, clip1), (clip2_i, clip2) in iterator:
        #distance matrix
        attr1 = torch.as_tensor(list(nx.get_node_attributes(clip1, "attr").values()), device=device)
        attr2 = torch.as_tensor(list(nx.get_node_attributes(clip2, "attr").values()), device=device)
        if (attr_type is None and torch.is_floating_point(attr1)
            or attr_type == "continuous"):
            D = torch.cdist (attr1, attr2)
        else:
            if len(attr1.shape) > 1: attr1 = attr1.squeeze()
            if len(attr2.shape) > 1: attr2 = attr2.squeeze()
            D = (attr1[:, None] != attr2[None, :])
        D = D.to(torch.float32)

        #markov chains
        markov1 = weighted_transition_matrix(clip1, q=q)
        markov2 = weighted_transition_matrix(clip2, q=q)
        
        #distributions
        mu1 = torch.ones(1, len(clip1), device=device) / len(clip1)
        mu2 = torch.ones(1, len(clip2), device=device) / len(clip2)
        
        try:
            #distance
            distance_matrix[clip1_i, clip2_i] = \
                wl_k(markov1.to(device)[None, ...], 
                    markov2.to(device)[None, ...], 
                    cost_matrix=D[None, ...], 
                    k=k,
                    muX=mu1,
                    muY=mu2,
                    **wl_parameters)
        except OutOfMemoryError:
            logger.warn("Out of memory error, computing this one on cpu")
            gc.collect()
            torch.cuda.empty_cache()
            distance_matrix[clip1_i, clip2_i] = \
                wl_k(markov1.to("cpu")[None, ...], 
                    markov2.to("cpu")[None, ...], 
                    cost_matrix=D.to("cpu")[None, ...], 
                    k=k,
                    muX=mu1.to("cpu"),
                    muY=mu2.to("cpu"),
                    **wl_parameters).to(device)


        
    if set1 is set2:
        distance_matrix = distance_matrix + distance_matrix.T
    return distance_matrix
