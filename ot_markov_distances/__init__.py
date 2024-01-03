"""
This module contains the implementation of the discounted wl distance
for the paper "Distances for Markov Chains, and Their Differentiation" 
:cite:`brugere2023distances`
"""
from .sinkhorn import sinkhorn as sinkhorn_distance
from .wl import wl_k, markov_measure
from .discounted_wl import discounted_wl_infty, discounted_wl_k

__all__: list[str] = [
    "sinkhorn_distance",
    "wl_k", 
    "markov_measure", 
    "discounted_wl_infty",
    "discounted_wl_k"
]
