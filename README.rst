ot_markov_distances
===================

|github workflow badge| |codecov|

Distances on graphs based on optimal transport

This is the implementation code for :cite:t:`brugere2023distances`.

::

   Brugere, T., Wan, Z., & Wang, Y. (2023). Distances for Markov Chains, and Their Differentiation. ArXiv, abs/2302.08621.

Setup
-----

Dependencies
~~~~~~~~~~~~

.. note::
   The main branch uses ``cuda11.8`` in its dependencies. If for some
   reason you need to use ``cuda12``, clone the ``cuda12`` branch
   instead

This package manages its dependencies via
`poetry <https://python-poetry.org/>`__. I recommend you install it
(otherwise if you prefer to manage them manually, a list of the
dependencies is available in the file ``pyproject.toml``)

When you have ``poetry``, you can add dependencies using our makefile

.. code:: console

   $ make .make/deps

or directly with poetry

.. code:: console

   $ poetry install

Project structure
~~~~~~~~~~~~~~~~~

.. code:: console

   .
   ├── docs    #contains the generated docs (after typing make)
   │   ├── build
   │   │   └── html            #Contains the html docs in readthedocs format
   │   └── source
   ├── experiments             #contains jupyter notebooks with the experiments
   │   └── utils               #contains helper code for the experiments
   ├── ot_markov_distances     #contains reusable library code for computing and differentiating the discounted WL distance
   │   ├── discounted_wl.py    # implementation of our discounted WL distance
   │   ├── __init__.py
   │   ├── sinkhorn.py         # implementation of the sinkhorn distance
   │   ├── utils.py            # utility functions
   │   └── wl.py               #implementation of the wl distance by Chen et al.
   ├── staticdocs #contains the static source for the docs
   │   ├── build
   │   └── source 
   └── tests #contains sanity checks

Documentation
-------------

.. warning::
   Do not edit the documentation directly in the ``docs/`` folder,
   that folder is wiped every time the documentation is built. The
   static parts of the documentation can be edited in ``staticdocs/``.

You can build documentation and run tests using

.. code:: console

   $ make

Alternatively, you can build only the documentation using

.. code:: console

   $ make .make/build-docs

The documentation will be available in ``docs/build/html`` in the
readthedocs format

Running Experiments
-------------------

Running experiments requires installing development dependencies. This can be done by running

.. code:: console

   $ make .make/dev-deps

or alternatively

.. code:: console

   $ poetry install --with dev


`Experiments <experiments>`__ can be found in the ``experiments/``
directory (see `Project structure <#project-structure>`__ ).

The Barycenter and Coarsening experiments can be found in
``experiments/Barycenter.ipynb`` and ``experiments/Coarsening.ipynb``.

.. |github workflow badge| image:: https://github.com/YusuLab/ot_markov_distances/actions/workflows/testing-and-docs.yml/badge.svg
.. |codecov| image:: https://codecov.io/gh/YusuLab/ot_markov_distances/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/YusuLab/ot_markov_distances
