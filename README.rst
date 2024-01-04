ot_markov_distances
===================

|github workflow badge| |codecov| |doc badge| |pytorch badge| 

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

The ``TUDataset`` package is also needed to run the classification experiment, but it is not available via ``pip`` / ``poetry``. 
To install it, follow the instruction in `the tudataset repo`_, 
including the "Compilation of kernel baselines" section, and add the directory where you downloaded it to your ``$PYTHONPATH``.
eg:

.. code:: console

   $ export PYTHONPATH="/path/to/tudataset:$PYTHONPATH"


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

The documentation is available online: `read the documentation <tristan.bruge.re/documentation/ot_markov_distances>`_

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

The performance graphs are computed in  ``experiments/Performance.ipynb``

Classification experiment
~~~~~~~~~~~~~~~~~~~~~~~~~

The Classification experiment (see the first paragraph of section 6 in the paper) is not in a jupyter notebook, but accessible via a command line. 

As an additional dependency it needs ``tudataset``, which is not installable via ``pip``. To install it follow the instructions in `the tudataset repo`_.
, including the "Compilation of kernel baselines" section, and add the directory where you downloaded it to your ``$PYTHONPATH``.


Now you can run the classification experiment using the command

.. code:: console

   $ poetry run python -m experiments.classification
   usage: python -m experiments.classification [-h] {datasets_info,distances,eval} ...

   Run classification experiments on graph datasets

   positional arguments:
     {datasets_info,distances,eval}
       datasets_info       Print information about given datasets
       distances           Compute distance matrices for given datasets
       eval                Evaluate a kernel based on distance matrix

   options:
     -h, --help            show this help message and exit

The yaml file containing dataset information that should be passed to the command line is in ``experiments/grakel_datasets.yaml``. 
Modifying this file should allow running the experiment on different datasets.

.. _`the tudataset repo`: https://github.com/chrsmrrs/tudataset

.. |github workflow badge| image:: https://github.com/YusuLab/ot_markov_distances/actions/workflows/testing-and-docs.yml/badge.svg
.. |codecov| image:: https://codecov.io/gh/YusuLab/ot_markov_distances/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/YusuLab/ot_markov_distances
.. |pytorch badge| image:: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
.. |doc badge| image:: https://img.shields.io/badge/documentation-green?style=for-the-badge&logo=readme&logoColor=black
   :target: https://tristan.bruge.re/documentation/ot_markov_distances

