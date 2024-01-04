r"""
This module contains the implementation of the discounted WL distance,
with its forward and backward pass (implemented as a torch.autograd.Function)

The depth-:math:`\infty` version can be computed with the function :func:`wl_reg_infty`.
The depth-:math:`k` version can be computed with the function :func:`wl_reg_k`.
"""

# from typing import Literal
import warnings

from .misc import all_equal, debug_time
import torch
from torch import Tensor, FloatTensor
from torch.nn import functional as F

from .sinkhorn import sinkhorn_internal, sinkhorn
from .utils import markov_measure,\
        densify, cost_matrix_index as make_cost_matrix_index,\
        reindex_cost_matrix, re_project_C, degree_markov, dummy_densify

__all__ = [
        "discounted_wl_k", 
        "discounted_wl_infty", 
        "discounted_wl_infty_cost_matrix"
]

def discounted_wl_cost_matrix_step(MX: FloatTensor, 
                                   MY:FloatTensor, cost_matrix, 
                                   distance_matrix, 
                                   sinkhorn_reg, 
                                   delta, one_minus_delta,
                                   sinkhorn_max_iter, 
                                   cost_matrix_index: Tensor|None=None,
                                   cost_matrix_mask: Tensor|None=None,
                                   save_to_ctx=None
                                   ) -> tuple[Tensor, bool]:
    """One step for the discounted WL cost matrix

    will save the 

    Args:
        MX : Fist markov matrix (either a dense or a sparse version)
        MY : Second markov matrix (either a dense or a sparse version)
        cost_matrix : current cost matrix
        sinkhorn_reg : regularization parameter for the sinkhorn algorithm
        delta : delta parameter for the WL distance
        one_minus_delta : 1 - delta
        sinkhorn_max_iter : maximum number of iterations for the sinkhorn algorithm
        cost_matrix_index: index for the cost matrix if the sparse version is used
        save_to_ctx : eventually a context the f, g and log_P will be saved to

    Returns:
        new_cost_matrix: the cost matrix after update
        sinkhorn_converged: whether the sinkhorn algorithm converged
    """
    b, n, one, dx = MX.shape
    b_, one_, m, dy = MY.shape
    b__, n_, m_ = cost_matrix.shape
    assert all_equal(b, b_, b__) and n == n_ and m == m_ \
            and all_equal(one, one_, 1)

    x_sparse = n != dx
    y_sparse = m != dy

    if cost_matrix_index is not None:
        cost_matrix_sparse = reindex_cost_matrix(cost_matrix, 
                                                 cost_matrix_index, 
                                                 cost_matrix_mask) 
        # b, n, m, dx, dy
    else:
        assert not x_sparse and not y_sparse
        cost_matrix_sparse = cost_matrix[:, None, None, :, :]
    
    
    f, g, log_P, sinkhorn_converged = sinkhorn_internal(
            MX, # b, n, 1, dx
            MY, # b, 1, m, dy
            cost_matrix_sparse, # b, n, m, dx, dy
            epsilon=sinkhorn_reg, 
            k= sinkhorn_max_iter, 
            return_has_converged=True
    )

    sinkhorn_result = ((f*MX).sum(-1)
                    + (g*MY).sum(-1))

    new_cost_matrix = delta * distance_matrix \
        + one_minus_delta * sinkhorn_result # b, n, m

    if save_to_ctx is not None:
        if not x_sparse:
            save_to_ctx.f = f
        if not y_sparse:
            save_to_ctx.g = g
        save_to_ctx.log_P = log_P
        
    return new_cost_matrix, sinkhorn_converged


class DiscountedWlCostMatrix(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, MX: FloatTensor, MY: FloatTensor, 
        distance_matrix: FloatTensor,
        delta: float = .4,
        sinkhorn_reg: float=.01,
        max_iter: int = 50,
        convergence_threshold_rtol = .005,
        convergence_threshold_atol = 1e-6,
        max_sinkhorn_iter: int= 100,
        return_differences: bool=False,
        sinkhorn_iter_schedule = 10,
        x_is_sparse:bool|None=None,
        y_is_sparse:bool|None=None,
        ):
        """This is an internal function, you should not have to call it directly

        See :func:`discounted_wl_infty_cost_matrix` or :func:`discounted_wl_infty` 
        for the public API

        computes the regularized WL distance cost matrix between two markov transition matrices 
        (represented as torch tensor)

        Batched over first dimension (b)
        
        delta can be a torch tensor (or a simple float). 
        please don’t modify it inplace between forward and backward, I don't check for that

        Args:
            MX: (b, n, n) first transition tensor
            MY: (b, m, m) second transition tensor
            muX: stationary distribution for MX (if omitted, will be recomuputed)
            muY: stationary distribution for MY (if omitted, will be recomuputed)
            reg: regularization parameter for sinkhorn
            delta: regularization parameter for WL
            sinkhorn_iter: number of sinkhorn iterations for a step
            x_is_sparse: whether to use the accelerated algorithm, 
                    considering MX is sparse 
                    (default: compute the degree, 
                    and check whether it lower that 2/3 n.
                    If so, consider MX sparse)
            y_is_sparse: whether to use the accelerated algorithm, 
                    considering MY is sparse 
                    (default: compute the degree, 
                    and check whether it lower that 2/3 m.
                    if so, consider MY sparse)

        """
        with torch.no_grad():
            # initialization
            b, n, n_ = MX.shape
            b_, m, m_ = MY.shape
            b__, n__, m__  = distance_matrix.shape
            assert all_equal(n, n_, n__) \
                    and all_equal(m, m_, m__) and all_equal(b, b_, b__)
            assert max_iter >= 1, "Can’t really converge without iterating"

            one_minus_delta = 1 - delta
            
            differences = []
            debug_time()

            cost_matrix = (delta * distance_matrix).contiguous()
            if x_is_sparse is None: x_is_sparse = degree_markov(MX) < 2/3 * n
            if y_is_sparse is None: y_is_sparse = degree_markov(MY) < 2/3 * m

            if x_is_sparse: mx_index, mx_mask, mx_dense = densify(MX) 
            else: mx_index, mx_mask, mx_dense = dummy_densify(MX)
            if y_is_sparse: my_index, my_mask, my_dense = densify(MY)
            else: my_index, my_mask, my_dense = dummy_densify(MY)
            mx_dense = mx_dense[:, :, None, :] # b, n, 1, dx
            my_dense = my_dense[:, None, :, :] # b, 1, m, dy

            if x_is_sparse or y_is_sparse:
                c_index, c_mask = make_cost_matrix_index(
                        cost_matrix, mx_index, my_index, mx_mask, my_mask)
            else:
                c_index, c_mask = None, None

            if sinkhorn_iter_schedule != 0:
                sinkhorn_iter = 1
            else: sinkhorn_iter = max_sinkhorn_iter


            for _ in range(max_iter):
                # sinkhorn pass
                new_cost_matrix, sinkhorn_converged = discounted_wl_cost_matrix_step(
                        mx_dense, my_dense, cost_matrix, 
                        distance_matrix,
                        sinkhorn_reg, delta, one_minus_delta,
                        sinkhorn_max_iter=sinkhorn_iter, 
                        cost_matrix_index = c_index, 
                        cost_matrix_mask = c_mask, 
                        save_to_ctx=ctx
                        )
                # update sinkhorn iter
                if sinkhorn_iter_schedule != 0 and not sinkhorn_converged:
                    sinkhorn_iter = min(sinkhorn_iter + sinkhorn_iter_schedule, max_sinkhorn_iter)
                # stop condition 
                if (sinkhorn_converged or sinkhorn_iter == max_sinkhorn_iter) \
                        and torch.allclose(cost_matrix, new_cost_matrix, 
                            rtol=convergence_threshold_rtol,
                            atol=convergence_threshold_atol):
                    break
                differences.append(F.mse_loss(cost_matrix, new_cost_matrix))
                cost_matrix[...] = new_cost_matrix #type: ignore
                debug_time("iteration")
            else:
                warnings.warn("regularized WL did not converge")

        # ctx.save_for_backward(f, g, log_P) #type:ignore
        ctx.x_is_sparse = x_is_sparse; ctx.y_is_sparse = y_is_sparse
        ctx.delta = delta; ctx.one_minus_delta = one_minus_delta
        ctx.c_index = c_index; ctx.c_mask = c_mask
        if return_differences: return cost_matrix, differences
        return cost_matrix
        
    @staticmethod
    def backward(ctx, grad_output):
        r"""We use a simplified version of the formulae in the paper
        
        see :doc:`notes_gradient` for more details (in the doc).
        Args:
            grad_output: (b, n, m)
        """
        with torch.no_grad():
            # recover saved stuff
            log_P = ctx.log_P
            c_index, c_mask = ctx.c_index, ctx.c_mask
            b, n, m, _, _ = log_P.shape
            delta = ctx.delta
            one_minus_delta = ctx.one_minus_delta
            device = log_P.device
            log_P = log_P

            # what do we need to do?
            mx_needs_grad, my_needs_grad, c_needs_grad, *rest_needs_grad = \
                ctx.needs_input_grad

            assert not any(rest_needs_grad), "required grad on an unsupported variable"
            number_of_nograd_parameters = len(rest_needs_grad)

            # compute 
            P = log_P.exp() #b, n, m, n, m
            P = re_project_C(P, c_index, c_mask)
        
            K = torch.eye(n*m, device=device)[None, ...] \
                    - one_minus_delta * P.reshape(b, n*m, n*m) 
            # K = I - (1- delta) P
            # b, n*m, n*m
            #see paper for a proof this is diagonally dominant, thus invertible
            K_Tm1_grad = torch.linalg.solve(K.transpose(-1, -2), 
                                            grad_output.reshape(b, n*m))
            K_Tm1_grad = K_Tm1_grad.reshape(b, n, m)
            # (K^{T, -1} @ grad)
            # remark that now
            # d_c = delta * K_Tm1_grad
            # d_mx = (1 - delta) * F.T @ K_T_
            

            if mx_needs_grad:
                assert not ctx.x_is_sparse, "can’t differentiate on sparse kernel"
                f = ctx.f
                # f  (b, n, m, n)
                # F = torch.einsum("bijl->bijil", f)
                #doesnt work but that’s the idea
                # F = torch.permute(f, (0, 2, 3, 1)) #bijl -> bjli
                # F = double_last_dimension(F) #bjlii
                # F = torch.permute(F, (0, 3, 1, 4, 2)) #bijil
                # Gamma = one_minus_delta * \
                #         torch.linalg.solve(mymatrix, F.reshape(b, n*m, n*n))
                # Gamma = Gamma.reshape(b, n, m, n, n) 
                # d_mx = torch.einsum("bijkl,bij->bkl", Gamma, grad_output)
                d_mx = one_minus_delta\
                        * torch.einsum("bkjl,bkj->bkl", f, K_Tm1_grad)
                d_mx = d_mx - d_mx.mean(-1, keepdims=True)
                # normalize the markov gradients to stay in the markov space
            else:
                d_mx = None
            if my_needs_grad:
                assert not ctx.y_is_sparse, "can’t differentiate on sparse kernel"
                g = ctx.g
                # g (b, n, m, m)
                # G = torch.einsum("bijl->bijjl", g) #(n, m, m, m)
                # G = torch.permute(g, (0, 1, 3, 2)) #bijl -> bilj
                # G = double_last_dimension(G) #biljj
                # G = torch.permute(G, (0, 1, 3, 4, 2)) #bijjl 
                # Theta = one_minus_delta * \
                #         torch.linalg.solve(mymatrix, G.reshape(b, n*m, m*m))
                # Theta = Theta.reshape(b, n, m, m, m)
                # d_my = torch.einsum("bijkl,bij->bkl", Theta, grad_output)
                d_my = one_minus_delta\
                        * torch.einsum("bikl,bik->bkl", g, K_Tm1_grad)
                d_my = d_my - d_my.mean(-1, keepdims=True)        
                # normalize the markov gradients to stay in the markov space
            else:
                d_my = None
            
            if c_needs_grad:
                # Delta = delta * torch.inverse(mymatrix)
                # Delta = Delta.reshape(b, n, m, n, m)
                # d_cost_matrix = \
                #     torch.einsum("bijkl,bij->bkl", Delta, grad_output)
                d_cost_matrix = delta * K_Tm1_grad
            else:
                d_cost_matrix = None

                        
        return (d_mx, d_my, d_cost_matrix, *[None]*number_of_nograd_parameters )

def discounted_wl_infty_cost_matrix(
        MX: Tensor, MY: Tensor, 
        distance_matrix: Tensor,
        delta: float = .4,
        sinkhorn_reg: float=.01,
        max_iter: int = 50,
        convergence_threshold_rtol: float = .005,
        convergence_threshold_atol: float = 1e-6,
        sinkhorn_iter: int= 100,
        return_differences: bool=False,
        sinkhorn_iter_schedule: int=10, 
        x_is_sparse:bool|None=None,
        y_is_sparse:bool|None=None,
        ):
    return DiscountedWlCostMatrix.apply(
        MX, MY, 
        distance_matrix,
        delta,
        sinkhorn_reg,
        max_iter,
        convergence_threshold_rtol,
        convergence_threshold_atol,
        sinkhorn_iter,
        return_differences,
        sinkhorn_iter_schedule, 
        x_is_sparse, 
        y_is_sparse,
        )


def discounted_wl_infty(
        MX: Tensor, MY: Tensor, 
        distance_matrix: Tensor,
        muX: Tensor | None = None,
        muY: Tensor | None = None, 
        delta: float = .4,
        sinkhorn_reg: float=.01,
        max_iter: int = 50,
        convergence_threshold_rtol:float = .005,
        convergence_threshold_atol:float = 1e-6,
        sinkhorn_iter: int= 100,
        sinkhorn_iter_schedule: int =10,
        x_is_sparse:bool|None=None,
        y_is_sparse:bool|None=None,
        ):
    """Discounted WL infinity distance

    Computes the discounted WL infinity distance
    between ``(MX, muX)`` and ``(MY, muY)``
    with cost matrix ``distance_matrix`` and discount factor ``delta``.

    Args:
        MX: (b, n, n) first transition tensor
        MY: (b, m, m) second transition tensor
        distance_matrix: [TODO:description]
        muX: initial distribution for MX (if omitted, the stationary distribution will be used instead)
        muY: initial distribution for MY (if omitted, the stationary distribution will be used instead)        
        delta: discount factor
        sinkhorn_reg: regularization parameter for the sinkhorn algorithm
        max_iter: maximum number of iterations.
        convergence_threshold_rtol : relative tolerance for convergence criterion (see ``torch.allclose``)
        convergence_threshold_atol : absolute tolerance for convergence criterion (see ``torch.allclose``)
        sinkhorn_iter: maximum number of sinkhorn iteration
        sinkhorn_iter_schedule ([TODO:type]): [TODO:description]
        x_is_sparse: whether to use the accelerated algorithm, 
                considering MX is sparse 
                (default: compute the degree, 
                and check whether it lower that 2/3 n.
                If so, consider MX sparse)
        y_is_sparse: whether to use the accelerated algorithm, 
                considering MY is sparse 
                (default: compute the degree, 
                and check whether it lower that 2/3 m.
                if so, consider MY sparse)
    """
    cost_matrix =  discounted_wl_infty_cost_matrix(
        MX=MX, 
        MY=MY, 
        distance_matrix=distance_matrix,
        delta=delta,
        sinkhorn_reg=sinkhorn_reg,
        max_iter=max_iter,
        convergence_threshold_rtol=convergence_threshold_rtol,
        convergence_threshold_atol=convergence_threshold_atol,
        sinkhorn_iter=sinkhorn_iter,
        return_differences=False, 
        sinkhorn_iter_schedule=sinkhorn_iter_schedule, 
        x_is_sparse=x_is_sparse, 
        y_is_sparse=y_is_sparse,
        )
    if muX is None: 
        muX = markov_measure(MX)
    if muY is None:
        muY = markov_measure(MY)

    return sinkhorn(muX, muY, cost_matrix, sinkhorn_reg)

def discounted_wl_k(MX: Tensor, MY: Tensor, 
        l1: Tensor|None = None, 
        l2: Tensor|None = None,
        *, 
        cost_matrix: Tensor | None = None,
        delta: Tensor|float = .4, 
        k: int,
        muX: Tensor | None = None,
        muY: Tensor | None = None, 
        reg: float=.1, 
        sinkhorn_parameters: dict = dict(), 
        return_differences: bool=False, 
        ):
    r"""computes the discounted WL distance

    computes the WL-delta distance between two markov transition matrices 
    (represented as torch tensor)

    This function does not have the backward pass mentioned in the paper,
    because that formula is only valid for the case :math:`k=\infty`

    Batched over first dimension (b)

    Args:
        MX: (b, n, n) first transition tensor
        MY: (b, m, m) second transition tensor
        l1: (b, n,) label values for the first space
        l2: (b, m,) label values for the second space
        cost_matrix: (b, n, m) allows specifying the cost matrix instead
        k: number of steps (k parameter for the WL distance)
        muX: distribution for MX (if omitted, the stationary distrubution will be used)
        muY: distribution for MY (if omitted, the stationary distrubution will be used)
        reg: regularization parameter for sinkhorn
    """
    b, n, n_ = MX.shape
    b_, m, m_ = MY.shape
    assert (n==n_) and (m == m_) and (b == b_)
    one_minus_delta = 1 - delta

    if cost_matrix is None:
        assert (l1 is not None) and (l2 is not None)
        cost_matrix = (l1[:, :, None] - l2[:, None, :]).abs()
    distance_matrix = cost_matrix
    delta_distance_matrix = delta * distance_matrix
    cost_matrix = delta_distance_matrix

    
    if return_differences:
        differences = []
    else:
        differences = None
    debug_time()
    for i in range(k):
        new_cost_matrix = delta_distance_matrix \
            + one_minus_delta * sinkhorn(
                MX[:, :, None, :], # b, n, 1, n
                MY[:, None, :, :], # b, 1, m, m
                cost_matrix[:, None, None, :, :], # b, 1, 1, n, m
                epsilon=reg, 
                **sinkhorn_parameters
        ) # b, n, m
        if return_differences:
            assert differences is not None
            differences.append(F.mse_loss(new_cost_matrix, cost_matrix))
            
        cost_matrix = new_cost_matrix 
        debug_time("iteration")

    if muX is None: 
        muX = markov_measure(MX)
    if muY is None:
        muY = markov_measure(MY)
    if return_differences:
        assert differences is not None
        return sinkhorn(muX, muY, cost_matrix, reg), differences
    return sinkhorn(muX, muY, cost_matrix, reg)
