"""
Differentiable sinkhorn divergence.
This module provides a pytorch/autograd compatible implementation of sinkhorn
divergence, 
using the method described in :cite:`feydyInterpolatingOptimalTransport2019`
much of the code is inspired from the version in the POT library :cite:`flamary2021pot`.
"""


import torch
from torch import Tensor

class Sinkhorn(torch.autograd.Function):
    """Computes Sinkhorn divergence"""
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, C: Tensor, 
        epsilon: float, k: int=100):
        r"""Batched version of sinkhorn distance

        The 3 batch dims will be broadcast to each other. 
        Every steps is only broadcasted torch operations,
        so it should be reasonably fast on gpu

        Args:
            a: (*batch, n) First distribution. 
            b: (*batch, m) Second distribution
            C: (*batch, n, m) Cost matrix
            epsilon: Regularization parameter
            k: number of iteration (this version does not check for convergence)
            return_solutions: whether to return P, f and g

        Returns:
            divergence: (*batch) $divergence[*i] = OT^\epsilon(a[*i], b[*i], C[*i])$

            (f_eps, g_eps, log_P): only if return_solutions is True. 
            f_eps is f/epsilon, g_eps is g/epsilon where f and g are the dual solutions. 
            log_P is log(P) where P is the optimal transport
        """

        *batch, n = a.shape
        *batch_, m = b.shape
        *batch__, n_, m_ = C.shape
        batch = torch.broadcast_shapes(batch, batch_, batch__)
        assert n == n_
        assert m == m_
        with torch.no_grad():
            log_a = a.log()[..., :, None]
            log_b = b.log()[..., None, :]
            mC_eps = - C / epsilon

            f_eps = torch.randn((*batch, n, 1))#f over epsilon + log_a 
            #batch, n
            g_eps = torch.randn((*batch, 1, m))#g over epsilon + log_b
            #batch, m
            for _ in range(k):
                f_eps = log_a - torch.logsumexp(mC_eps + g_eps, dim=-1, keepdim=True)
                g_eps = log_b - torch.logsumexp(mC_eps + f_eps, dim=-2, keepdim=True)
            log_P = mC_eps + f_eps + g_eps
            #res = (C.log() + log_P).exp().sum((-1, -2))

            # Note: we dont actually need save_for_backwards here
            # we could say ctx.f_eps = f_eps etc.
            # save_for_backwards does sanity checks 
            # such as checking that the saved variables do not get changed inplace
            # (if you have output them, which is not the case)
            # see https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483/5
            #print(f_eps * epsilon)
            #f = epsilon *  (f_eps - log_a)
            #g = epsilon * (g_eps - log_b)
            f = - epsilon * torch.logsumexp(mC_eps + g_eps, dim=-1, keepdim=True).squeeze(-1)
            g = - epsilon * torch.logsumexp(mC_eps + f_eps, dim=-2, keepdim=True).squeeze(-2)
            #print(res, (f.squeeze(-1) * a).sum(-1) + (g.squeeze(-2) * b).sum(-1))
            res = (f * a).sum(-1) + (g * b).sum(-1)
            ctx.save_for_backward(f, g, log_P)
            ctx.epsilon = epsilon

        return res
    
    @staticmethod
    def backward(ctx, grad_output):
        r"""We use the fact that the primal solution $P$
        and the dual solutions $f, g$
        are the gradients for respectively the cost matrix $C$, 
        and the input distributions $\mu_x$ and $\mu_y$
        :cite:`peyreComputationalOT2018`

        This allows us to shortcut the backward pass. 
        Note that `citet`:peyreComputationalOT2018: discourage doing this
        if the sinkhorn iterations do not converge
        """
        #output gradient d _ res y . so 
        # grad_output (*batch, 1) (or maybe just (*batch)
        # f_eps (*batch, n, 1)
        # g_eps (*batch, 1, m)
        # log_P (*batch, n, m)
        f, g, log_P = ctx.saved_tensors
        epsilon = ctx.epsilon

        d_a = f * grad_output[..., None]
        d_a = d_a - d_a.mean(-1, keepdim=True) # so that the gradient
        #maintains the space of distributions
        d_b = g * grad_output[..., None]
        #d_b = d_b.squeeze(-2)
        d_b = d_b - d_b.mean(-1, keepdim=True) # idem
        d_C = log_P.exp() * grad_output[..., None, None]
        
        return d_a, d_b, d_C, None, None
        

def sinkhorn(a: Tensor, b: Tensor, C: Tensor, 
             epsilon: float, k: int=100) -> Tensor:
    """Differentiable sinkhorn distance

    This is a pytorch implementation of sinkhorn, 
    batched (over `a`, `b` and `C`) 

    It is compatible with pytorch autograd gradient computations.

    See the documentation of :class:`Sinkhorn` for details.

    Args:
        a: (*batch, n) vector of the first distribution
        b: (*batch, m) vector of the second distribtion
        C: (*batch, n, m) cost matrix
        epsilon: regularisation term for sinkhorn
        k: number of sinkhorn iterations (default 100)

    Returns:
        Tensor: (*batch). result of the sinkhorn computation
    """
    return Sinkhorn.apply(a, b, C, epsilon, k)
