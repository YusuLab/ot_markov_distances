import os
import torch
from .modules import ParametricMarkovMatrixWithLabels, ParametricMarkovMatrixWithMatchings

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from ot_markov_distances.discounted_wl import discounted_wl_infty

def opt_loop_batched(target_matrices, 
             n_steps=300, projection_size=5, 
             run_name="barycenter", 
            device = torch.device("cuda:0"), 
             lr=1e-2, 
             weight_decay=0,
             heat=1.,
             matching_heat=1.,
             wl_parameters = dict(delta=.5, sinkhorn_reg=.01, x_is_sparse=False, y_is_sparse=True),
             labels = None,
            time_factor=None, 
            ):
    
    n_targets = len(target_matrices)
    target_sizes = [matrix.shape[0] for matrix in target_matrices]
    target_matrices = [matrix.to(device) for matrix in target_matrices]
    target_matrices_batched = torch.stack(target_matrices, dim=0)
    
    if labels is None:
        projection = ParametricMarkovMatrixWithMatchings(projection_size, 
                                                         *target_sizes, 
                                                         heat=heat,
                                                        matching_heat=matching_heat).to(device)
    else:
        projection = ParametricMarkovMatrixWithLabels(projection_size, *labels, heat=heat, time_factor=time_factor).to(device)
                                        

    target_measures = [
        torch.ones(size, device=device, requires_grad=False) / size
        for size in target_sizes]
    target_measures_batched = torch.stack(target_measures, dim=0)
    
    projection_measure = torch.ones((n_targets, projection_size), device=device, requires_grad=False) / projection_size


    optim = torch.optim.Adam(projection.parameters(), lr=lr, weight_decay=weight_decay)

    writer = SummaryWriter(f"{os.environ['HOME']}/tensorboard/{run_name}")
    parameter_values = []
    losses = []

    for step in trange(n_steps):
        optim.zero_grad()

        M, *Ds = projection()
        batched_Ds = torch.stack(Ds, dim=0)
        batched_M = M.expand(n_targets, -1, -1)
        
        loss = discounted_wl_infty(batched_M, 
                            target_matrices_batched, 
                            batched_Ds, 
                            muX=projection_measure,
                            muY=target_measures_batched,
                            **wl_parameters).square().mean(0)
        loss.backward()
        optim.step()


        writer.add_scalar("loss", loss, step) 
        losses.append(loss.item())
        with torch.no_grad():
            markov, *matchings = projection.get()
            parameter_values.append(
                (markov.numpy(force=True), 
                 *(matching.detach().cpu() for matching in matchings)
                )
            )
            
    return projection, parameter_values, losses

