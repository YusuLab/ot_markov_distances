"""
Differentiable sinkhorn divergence.
This module provides a pytorch/autograd compatible implementation of sinkhorn
divergence, 
using the method described in :cite:`feydyInterpolatingOptimalTransport2019`
"""
from typing import Optional,  overload, Literal
from math import ceil

import torch
from torch import Tensor

from . import utils

__all__ = ["sinkhorn_internal", "sinkhorn"]

@overload
def sinkhorn_internal(a: Tensor, b: Tensor, C: Tensor, 
                      epsilon: float, k: int=100, *, 
                      check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
                      return_has_converged:Literal[False] = False) -> tuple[Tensor, Tensor, Tensor]:
    ...

@overload
def sinkhorn_internal(a: Tensor, b: Tensor, C: Tensor, 
                      epsilon: float, k: int=100, *,
                      check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
                      return_has_converged:Literal[True]) -> tuple[Tensor, Tensor, Tensor, bool]:
    ...

def sinkhorn_internal(a: Tensor, b: Tensor, C: Tensor, 
                      epsilon: float, k: int=100, *, 
                      check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
                      return_has_converged:Literal[True, False] = False):
    r"""Same as sinkhorn, but returns f, g and P instead of the result
    Beware, this function does not have the shortcut differentiation
    
    (It can still be differentiated by autograd though)

    Args:
        a: (\*batch, n) vector of the first distribution
        b: (\*batch, m) vector of the second distribtion
        C: (\*batch, n, m) cost matrix
        epsilon: regularisation term for sinkhorn
        k: max number of sinkhorn iterations (default 100)
        check_convergence_interval: if int, check for convergence every 
            ``check_convergence_interval``. 
            If float, check for convergence every ``check_convergence_interval * k``. 
            If 0, never check for convergence 
            (apart from the last iteration if ``return_has_converged==True``)
            If convergence is reached early, the algorithm returns.
        cv_atol, cv_rtol: absolute and relative tolerance for the converegence criterion
        return_has_converged: whether to return a boolean indicating whether the 
            algorithm has converged. Setting this to True means that the  function
            will always check for convergence at the last iteration 
            (regardless of the value of ``check_convergence_interval``)
    Returns:
        f: Tensor (\*batch, n)
        g: Tensor (\*batch, m)
        log_P: Tensor (\*batch, n, m)
        has_converged: bool only returened if ``return_has_converged`` is True. 
            Indicates whether the algorithm has converged
    """
    *batch, n = a.shape
    *batch_, m = b.shape
    *batch__, n_, m_ = C.shape
    batch = torch.broadcast_shapes(batch, batch_, batch__)
    device = a.device
    dtype=a.dtype
    assert n == n_
    assert m == m_
    steps_at_which_to_check_for_convergence:list[bool]
    match check_convergence_interval:
        case float():
            check_convergence_interval = ceil(check_convergence_interval * k)
            steps_at_which_to_check_for_convergence = \
                    [(step - 1) % check_convergence_interval == 0 for step in range(k)]
        case 0:
            steps_at_which_to_check_for_convergence = [False] * k 
        case int():
            steps_at_which_to_check_for_convergence = \
                    [(step - 1) % check_convergence_interval == 0 for step in range(k)]
        case _: raise ValueError("check_convergence_interval must be a float or an int")
    if return_has_converged:
        steps_at_which_to_check_for_convergence[-1] = True # to be able to say for sure whether
        # we have converged

    log_a = a.log()[..., :, None]
    log_b = b.log()[..., None, :]
    mC_eps = - C / epsilon

    f_eps = torch.randn((*batch, n, 1), device=device, dtype=dtype)
    #f over epsilon + log_a 
    #batch, n
    g_eps = torch.randn((*batch, 1, m), device=device, dtype=dtype)
    #g over epsilon + log_b
    #batch, m
    has_converged: bool = True
    for _,  should_check_convergence in zip(range(k), steps_at_which_to_check_for_convergence):
        if should_check_convergence:
            f_eps_old = f_eps
            g_eps_old = g_eps

        f_eps =  - torch.logsumexp(mC_eps + g_eps + log_b, dim=-1, keepdim=True)
        g_eps =  - torch.logsumexp(mC_eps + f_eps + log_a, dim=-2, keepdim=True)

        if (should_check_convergence 
            and torch.allclose(f_eps, f_eps_old, atol=cv_atol, rtol=cv_rtol) #type:ignore
            and torch.allclose(g_eps, g_eps_old, atol=cv_atol, rtol=cv_rtol)): #type:ignore
            break
    else:
        has_converged=False
    log_P = mC_eps + f_eps + g_eps + log_a + log_b
    # Note: we dont actually need save_for_backwards here
    # we could say ctx.f_eps = f_eps etc.
    # save_for_backwards does sanity checks 
    # such as checking that the saved variables do not get changed inplace
    # (if you have output them, which is not the case)
    # see https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483/5
    f = epsilon * f_eps.squeeze(-1)
    g = epsilon * g_eps.squeeze(-2)
    if return_has_converged:
        return f, g, log_P, has_converged
    return f, g, log_P
    #print(res, (f.squeeze(-1) * a).sum(-1) + (g.squeeze(-2) * b).sum(-1))


class Sinkhorn(torch.autograd.Function):
    """Computes Sinkhorn divergence"""
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, C: Tensor, 
        epsilon: float, k: int=100, 
        check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
        return_has_converged:Literal[True, False] = False
                ) -> Tensor | tuple[Tensor, bool]:
        r"""Batched version of sinkhorn distance

        It is computed as in [@feydyInterpolatingOptimalTransport2019, Property 1]

        The 3 batch dims will be broadcast to each other. 
        Every steps is only broadcasted torch operations,
        so it should be reasonably fast on gpu

        Args:
            a: (\*batch, n) First distribution. 
            b: (\*batch, m) Second distribution
            C: (\*batch, n, m) Cost matrix
            epsilon: Regularization parameter
            k: max number of sinkhorn iterations (default 100)
            check_convergence_interval: if int, check for convergence every 
                ``check_convergence_interval``. 
                If float, check for convergence every ``check_convergence_interval * k``. 
                If 0, never check for convergence 
                (apart from the last iteration if ``return_has_converged==True``)
                If convergence is reached early, the algorithm returns.
            cv_atol, cv_rtol: absolute and relative tolerance for the converegence criterion
            return_has_converged: whether to return a boolean indicating whether the 
                algorithm has converged. Setting this to True means that the  function
                will always check for convergence at the last iteration 
                (regardless of the value of ``check_convergence_interval``)

        Returns:
            divergence: (\*batch) :math:`\text{divergence}[*i] = OT^\epsilon(a[*i], b[*i], C[*i])`
        """

        with torch.no_grad():
            f, g, log_P, *has_converged = sinkhorn_internal(
                    a, b, C, epsilon, k, 
                check_convergence_interval=check_convergence_interval,
                cv_atol=cv_atol, cv_rtol=cv_rtol, 
                return_has_converged=return_has_converged)#type:ignore
            res = (f * a).sum(-1) + (g * b).sum(-1)

        # Note: we dont actually need save_for_backwards here
        # we could say ctx.f_eps = f_eps etc.
        # save_for_backwards does sanity checks 
        # such as checking that the saved variables do not get changed inplace
        # (if you have output them, which is not the case)
        # see https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483/5
        ctx.save_for_backward(f, g, log_P)
        ctx.epsilon = epsilon

        if has_converged: 
            return res, *has_converged
        return res
    
    @staticmethod
    def backward(ctx, grad_output):
        r"""We use the fact that the primal solution $P$
        and the dual solutions $f, g$
        are the gradients for respectively the cost matrix $C$, 
        and the input distributions $\mu_x$ and $\mu_y$
        :cite:`peyreComputationalOT2018`

        This allows us to shortcut the backward pass. 
        Note that :cite:t:`peyreComputationalOT2018` discourage doing this
        if the sinkhorn iterations do not converge
        """
        #output gradient d _ res y . so 
        # grad_output (*batch, 1) (or maybe just (*batch)
        # f_eps (*batch, n, 1)
        # g_eps (*batch, 1, m)
        # log_P (*batch, n, m)
        f, g, log_P = ctx.saved_tensors
        # epsilon = ctx.epsilon

        d_a = f * grad_output[..., None]
        d_a = d_a - d_a.mean(-1, keepdim=True) # so that the gradient
        #maintains the space of distributions
        d_b = g * grad_output[..., None]
        #d_b = d_b.squeeze(-2)
        d_b = d_b - d_b.mean(-1, keepdim=True) # idem
        d_C = log_P.exp() * grad_output[..., None, None]
        
        return d_a, d_b, d_C, *([None] * 6)
        

def sinkhorn(a: Tensor, b: Tensor, C: Tensor, 
             epsilon: float, max_iter: int=100, 
            *,  
            check_convergence_interval:int | float=.1, cv_atol=1e-4, cv_rtol=1e-5,
            return_has_converged:Literal[True, False] = False
             ) -> Tensor:
    r"""Differentiable sinkhorn distance

    This is a pytorch implementation of sinkhorn, 
    batched (over ``a``, ``b`` and ``C``) 

    It is compatible with pytorch autograd gradient computations.

    See the documentation of :class:`Sinkhorn` for details.

    Args:
        a: (\*batch, n) vector of the first distribution
        b: (\*batch, m) vector of the second distribtion
        C: (\*batch, n, m) cost matrix
        epsilon: regularisation term for sinkhorn
        max_iter: max number of sinkhorn iterations (default 100)
        check_convergence_interval: if int, check for convergence every 
            ``check_convergence_interval``. 
            If float, check for convergence every ``check_convergence_interval * max_iter``. 
            If 0, never check for convergence 
            (apart from the last iteration if ``return_has_converged==True``)
            If convergence is reached early, the algorithm returns.
        cv_atol, cv_rtol: absolute and relative tolerance for the converegence criterion
        return_has_converged: whether to return a boolean indicating whether the 
            algorithm has converged. Setting this to True means that the  function
            will always check for convergence at the last iteration 
            (regardless of the value of ``check_convergence_interval``)

    Returns:
        Tensor: (\*batch). result of the sinkhorn computation
    """
    return Sinkhorn.apply(a, b, C, epsilon, max_iter, 
            check_convergence_interval, cv_atol, cv_rtol, return_has_converged)

