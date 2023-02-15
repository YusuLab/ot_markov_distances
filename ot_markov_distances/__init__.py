from .sinkhorn import sinkhorn as sinkhorn_distance
from .wl import wl_k, markov_measure
from .discounted_wl import wl_reg_infty, wl_delta_k

__all__: list[str] = [
    "sinkhorn_distance",
    "wl_k", 
    "markov_measure", 
    "wl_reg_infty", 
    "wl_delta_k"
]
