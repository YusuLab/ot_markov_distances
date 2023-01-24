import warnings

from ml_lib.misc import all_equal, debug_time
from ml_lib.pipeline.annealing_scheduler import get_scheduler
import torch
from torch import Tensor
from torch.nn import functional as F

from .sinkhorn import sinkhorn_internal, sinkhorn
from .utils import markov_measure, double_last_dimension

class DiscountedWlCostMatrix(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, MX: Tensor, MY: Tensor, 
        distance_matrix: Tensor,
        delta: float = .4,
        sinkhorn_reg: float=.01,
        max_iter: int = 50,
        convergence_threshold_rtol = .005,
        convergence_threshold_atol = 1e-6,
        sinkhorn_iter: int= 100,
        return_differences: bool=False,
        sinkhorn_iter_scheduler="constant"
        ):
        """computes the regularized WL distance

        computes the regularized WL distance between two markov transition matrices 
        (represented as torch tensor)

        Batched over first dimension (b)
        
        delta can be a torch tensor (or a simple float). 
        please don’t modify it inplace between forward and backward, I don't check for that

        Args:
            MX: (b, n, n) first transition tensor
            MY: (b, m, m) second transition tensor
            l1: (b, n,) label values for the first space
            l2: (b, m,) label values for the second space
            k: number of steps (k parameter for the WL distance)
            muX: stationary distribution for MX (if omitted, will be recomuputed)
            muY: stationary distribution for MY (if omitted, will be recomuputed)
            reg: regularization parameter for sinkhorn
            delta: regularization parameter for WL
            sinkhorn_iter: number of sinkhorn iterations for a step
        """
        b, n, n_ = MX.shape
        b_, m, m_ = MY.shape
        b__, n__, m__  = distance_matrix.shape
        assert all_equal(n, n_, n__) and all_equal(m, m_, m__) and all_equal(b, b_, b__)
        assert max_iter >= 1, "Can’t really converge without iterating"
        one_minus_delta = 1 - delta
        scheduler = get_scheduler(sinkhorn_iter_scheduler, T_0=max_iter, beta_0=sinkhorn_iter)
        
        differences = []
        debug_time()
        with torch.no_grad():
            cost_matrix = delta * distance_matrix

            for _ in range(max_iter):
                # sinkhorn pass
                f, g, log_P = sinkhorn_internal(
                        MX[:, :, None, :], # b, n, 1, n
                        MY[:, None, :, :], # b, 1, m, m
                        cost_matrix[:, None, None, :, :], # b, 1, 1, n, m
                        epsilon=sinkhorn_reg, 
                        k= int(scheduler.step()) + 1
                )
                sinkhorn_result = (f * MX[:, :, None, :]).sum(-1) + (g*MY[:, None, :, :]).sum(-1)
    
                # update
                new_cost_matrix = delta * distance_matrix \
                    + one_minus_delta * sinkhorn_result # b, n, m
                
                # stop condition 
                if torch.allclose(cost_matrix, new_cost_matrix, 
                            rtol=convergence_threshold_rtol,
                            atol=convergence_threshold_atol):
                    break
                differences.append(F.mse_loss(cost_matrix, new_cost_matrix))
                cost_matrix = new_cost_matrix 
                debug_time("iteration")
            else:
                warnings.warn("regularized WL did not converge")

        ctx.save_for_backward(f, g, log_P) #type:ignore
        ctx.delta = delta
        ctx.one_minus_delta = one_minus_delta
        if return_differences:
            return cost_matrix, differences
        return cost_matrix
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        See the paper for details on how the backward pass is computed.
        
        Args:
            grad_output: (b,)
        """
        f, g, log_P = ctx.saved_tensors
        b, n, m, _, _ = log_P.shape
        delta = ctx.delta
        one_minus_delta = ctx.one_minus_delta
        device = log_P.device
        with torch.no_grad():
            P = log_P.exp() #b, n, m, n, m
            # f  (b, n, m, n)
            # F = torch.einsum("bijl->bijil", f) #doesnt work but that’s the idea
            F = torch.permute(f, (0, 2, 3, 1)) #bijl -> bjli
            F = double_last_dimension(F) #bjlii
            F = torch.permute(F, (0, 3, 1, 4, 2)) #bijil
            # (n, m, n, n)
            # g (b, n, m, m)
            # G = torch.einsum("bijl->bijjl", g) #(n, m, m, m)
            G = torch.permute(g, (0, 1, 3, 2)) #bijl -> bilj
            G = double_last_dimension(G) #biljj
            G = torch.permute(G, (0, 1, 3, 4, 2)) #bijjl 
        
            mymatrix = torch.eye(n*m, device=device)[None, ...] - one_minus_delta * P.reshape(b, n*m, n*m) 
            # b, n*m, n*m
            
            #print(torch.det(mymatrix))
            #print(mymatrix)
            #see paper for a proof this is diagonally dominant, thus invertible
            Delta = delta * torch.inverse(mymatrix)
            Gamma = one_minus_delta * torch.linalg.solve(mymatrix, F.reshape(b, n*m, n*n))
            Theta = one_minus_delta * torch.linalg.solve(mymatrix, G.reshape(b, n*m, m*m))
            
            Delta = Delta.reshape(b, n, m, n, m)
            Gamma = Gamma.reshape(b, n, m, n, n) 
            Theta = Theta.reshape(b, n, m, m, m)

            d_cost_matrix = torch.einsum("bijkl,bij->bkl", Delta, grad_output)
            d_mx = torch.einsum("bijkl,bij->bkl", Gamma, grad_output)
            d_my = torch.einsum("bijkl,bij->bkl", Theta, grad_output)
                        
            d_mx = d_mx - d_mx.mean(-1, keepdims=True)# normalize the markov gradients to stay in the markov space
            d_my = d_my - d_my.mean(-1, keepdims=True)        
        return (d_mx, d_my, d_cost_matrix, *[None]*8 )

def wl_reg_cost_matrix(MX: Tensor, MY: Tensor, 
        distance_matrix: Tensor,
        delta: float = .4,
        sinkhorn_reg: float=.01,
        max_iter: int = 50,
        convergence_threshold_rtol = .005,
        convergence_threshold_atol = 1e-6,
        sinkhorn_iter: int= 100,
        return_differences: bool=False,
        sinkhorn_iter_scheduler="constant"
        ):
    return DiscountedWlCostMatrix.apply(MX, MY, 
        distance_matrix,
        delta,
        sinkhorn_reg,
        max_iter,
        convergence_threshold_rtol,
        convergence_threshold_atol,
        sinkhorn_iter,
        return_differences,
        sinkhorn_iter_scheduler)


def wl_reg_infty(MX: Tensor, MY: Tensor, 
        distance_matrix: Tensor,
        muX: Tensor | None = None,
        muY: Tensor | None = None, 
        delta: float = .4,
        sinkhorn_reg: float=.01,
        max_iter: int = 50,
        convergence_threshold_rtol = .005,
        convergence_threshold_atol = 1e-6,
        sinkhorn_iter: int= 100,
        return_differences: bool=False,
        sinkhorn_iter_scheduler="constant"
        ):
    cost_matrix =  wl_reg_cost_matrix(MX, MY, 
        distance_matrix,
        delta,
        sinkhorn_reg,
        max_iter,
        convergence_threshold_rtol,
        convergence_threshold_atol,
        sinkhorn_iter,
        return_differences,
        sinkhorn_iter_scheduler)
    if muX is None: 
        muX = markov_measure(MX)
    if muY is None:
        muY = markov_measure(MY)

    return sinkhorn(muX, muY, cost_matrix, sinkhorn_reg)

def wl_delta_k(MX: Tensor, MY: Tensor, 
        l1: Tensor, l2: Tensor,
        k: int,
        muX: Tensor | None = None,
        muY: Tensor | None = None, 
        reg: float=.0001, 
        delta: float = .1,
        sinkhorn_iter: int= 100,
        return_differences:bool= False,
        ):
    """computes the discounted WL distance

    computes the WL-delta distance between two markov transition matrices 
    (represented as torch tensor)

    Batched over first dimension (b)

    Args:
        MX: (b, n, n) first transition tensor
        MY: (b, m, m) second transition tensor
        l1: (b, n,) label values for the first space
        l2: (b, m,) label values for the second space
        k: number of steps (k parameter for the WL distance)
        muX: stationary distribution for MX (if omitted, will be recomuputed)
        muY: stationary distribution for MY (if omitted, will be recomuputed)
        reg: regularization parameter for sinkhorn
        sinkhorn_iter: number of sinkhorn iterations for a step
    """
    b, n, n_ = MX.shape
    b_, m, m_ = MY.shape
    assert (n==n_) and (m == m_) and (b == b_)
    one_minus_delta = 1 - delta

    distance_matrix = (l1[:, :, None] - l2[:, None, :]).abs()
    cost_matrix = delta * distance_matrix

    
    if return_differences:
        differences = []
    debug_time()
    for i in range(k):
        new_cost_matrix = delta * distance_matrix \
            + one_minus_delta * sinkhorn(
                MX[:, :, None, :], # b, n, 1, n
                MY[:, None, :, :], # b, 1, m, m
                cost_matrix[:, None, None, :, :], # b, 1, 1, n, m
                epsilon=reg, 
                k= sinkhorn_iter
        ) # b, n, m
        if return_differences:
            differences.append(F.mse_loss(new_cost_matrix, cost_matrix))
            
        cost_matrix = new_cost_matrix 
        debug_time("iteration")

    if muX is None: 
        muX = markov_measure(MX)
    if muY is None:
        muY = markov_measure(MY)
    if return_differences:
        return sinkhorn(muX, muY, cost_matrix, reg), differences
    return sinkhorn(muX, muY, cost_matrix, reg)
