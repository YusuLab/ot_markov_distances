
Basic Usage
===========

For installation instructions, refer to the :ref:`main page of the documentation <index>`

This package contains 3 modules (plus some utility and misc. functions)

- :py:mod:`ot_markov_distances.discounted_wl` is the main module. It contains the discounted wl distance code
  - In particular :py:func:`ot_markov_distance.discounted_wl.discounted_wl_infty` computes the discounted wl-$\infty$ distance, **with support for backpropagation**.
- :py:mod:`ot_markov_distances.sinkhorn` contains our (pytorch, vectorized) implementation of the sinkhorn distance, with support for backpropagation.
- :py:mod:`ot_markov_distances.wl` contains our (pytorch) implementation of the wl distance by :cite:t:`chen2022weisfeilerlehman`


Example
-------

Consider reading the experiments code for more in-depth examples. 

In particular the loop in ``experiments/utils/optimization_loop.py`` shows how one can use the discounted WL distance as an optimization loss.

**Computation of the distance between two graphs**

.. code:: python

   import networkx as nx
   from ot_markov_distances.discounted_wl import discounted_wl_infty
   from ot_markov_distances.utils import weighted_transition_matrix, uniform_distribution

   # generate dummy graphs
   G1 = nx.erdos_renyi_graph(30, .1, directed=True)
   G2 = nx.erdos_renyi_graph(20, .15, directed=True)

   # make them into markov transition matrices (with self-transition probability q=0.)
   # the unsqueeze call is to create the batch dimension
   M1 = weighted_transition_matrix(G1, q=0.).unsqueeze(0)
   M2 = weighted_transition_matrix(G2, q=0.).unsqueeze(0)

   # use the uniform distribution. This has given the stablest results in our experiments 
   # as opposed to the default, the stationary distribution of the markov chain
   mu1 = torch.ones(1, 30,) /  30
   mu2 = torch.ones(1, 20,) /  20

   # and a random distance matrix (simulated here as the distance between random labels of size 5)
   C = torch.cdist(torch.rand(1, 30, 5), torch.rand(1, 20, 5))

   distance = discounted_wl_infty(M1, M2, C, muX=mu1, muY=mu2)




   
