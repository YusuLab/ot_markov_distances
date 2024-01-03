Notes on Gradient computation
=============================

We compute the gradient using the technique described in :cite:t:`brugere2023distances`.

But we push the formulae a little further to simplify the computation:


Remark that , with the notations of the paper, 
denoting

.. math::
   \begin{gather}
   K = I_{nm} - (1 - \delta)P\\
   \Delta := \left(\Delta_{ij}^{kl}\right){}_{1 \leq i \leq n, 1 \leq j \leq m }^{ 1 \leq k \leq n, 1 \leq l \leq m}, \,
   \Gamma := \left(\Gamma_{ij}^{kk'}\right){}_{1 \leq i \leq n, 1 \leq j \leq m }^{ 1 \leq k \leq n, 1 \leq k' \leq n}, 
   \Theta := \left(\Theta_{ij}^{ll'}\right){}_{1 \leq i \leq n, 1 \leq j \leq m }^{ 1 \leq l \leq m, 1 \leq l' \leq m},  \\
   \text{where}~~~   
   \Delta_{ij}^{kl} := \frac{\partial C^{\epsilon,\delta, (\infty)}_{ij}}{\partial C_{kl}}, \,
   \Gamma_{ij}^{kk'} := \frac{\partial C^{\epsilon,\delta, (\infty)}_{ij}}{\partial m^{\setX}_{kk'}}, \,
   \Theta_{ij}^{ll'} := \frac{\partial C^{\epsilon,\delta, (\infty)}_{ij}}{\partial m^{\setY}_{ll'}}.\\
   \end{gather}

and denoting also 

.. math::
   \begin{gather}
   G^O_{ij} := \frac{\partial \text{loss}}{\partial C^{\epsilon,\delta, (\infty)}_{ij}}\\
   G^C_{kl} := \frac{\partial \text{loss}}{\partial C_{kl}}, \\
   G^X_{kk'} := \frac{\partial \text{loss}}{\partial m^{\setX}_{kk'}}\\
   G^Y_{ll'} := \frac{\partial \text{loss}}{\partial m^{\setY}_{ll'}}\\
   \end{gather}

Then (in matrix notation, ie with dimensions/codims flattened together)

.. math::
   \begin{gather}
   G^C = \Delta^T G^O\\
   G^X = \Gamma^T G^O\\
   G^Y = \Theta^T G^O\\
   \end{gather}

Developing

.. math::
   \begin{gather}
   G^C = \delta (K^T)^{-1} G^O\\
   G^X = (1-\delta) F^T (K^T)^{-1} G^O\\
   G^Y = (1-\delta) G^T (K^T)^{-1} G^O\\
   \end{gather}

Thus we save some compute by applying above formulae,
and computing :math:`(K^T)^{-1} G^O` only once.

Note also that :math:`(K^T)^{-1} G^O` can be computed with `torch.solve`
instead of `torch.inv` for more efficiency and stability

We call this matrix ``K_Tm1_grad`` in the implementation

Note also that :math:`F` and :math:`G` do not need to be explicitly computed:
denote by :math:`L := (K^T)^{-1} G^O`, 
then 

.. math::
   G^X_{kk'} &= (F^T L)_{kk'} \\
   &= (F^{T})_{kk'}^{ij} L_{ij} \\
   &= F^{kk'}_{ij} L_{ij} \\
   &= f^{k'}_{ij}\1_{i=k} L_{ij} \\
   &= f^{k'}_{kj} L{kj} \\

And similarly for :math:`G^Y`
