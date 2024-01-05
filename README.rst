ot_markov_distances
===================

|github workflow badge| |codecov| |doc badge| |pytorch badge| 

Differentiable distances on graphs based on optimal transport

This is the implementation code for 

::

   Brugere, T., Wan, Z., & Wang, Y. (2023). Distances for Markov Chains, and Their Differentiation. ArXiv, abs/2302.08621.

Setup
-----

Installing as a library
~~~~~~~~~~~~~~~~~~~~~~~

The ``ot_markov_distances`` package can be installed with the following command:

.. code:: console

   pip install ot-markov-distances

If for some reason you need to use ``cuda11.8`` (ie you are installing ``torch+cuda118``)
then use the following command instead

.. code:: console

   pip install git+https://github.com/YusuLab/ot_markov_distances@cuda118

Dependencies
~~~~~~~~~~~~

Python version
^^^^^^^^^^^^^^

This project requires ``python 3.10`` *a minima*. 
If your python version is prior to ``3.10``, 
you need to update (or to create a new ``conda`` environment) 
to a version above (latest release at the time of writing is ``3.12``)


Python dependencies 
^^^^^^^^^^^^^^^^^^^

.. note::
   The main branch uses the default (``cuda12``) version of torch 
   in its dependencies. If for some
   reason you need to use ``cuda11.8``, clone the ``cuda118`` branch
   instead

This package manages its dependencies via
`poetry <https://python-poetry.org/>`__. I recommend you install it
(otherwise if you prefer to manage them manually, a list of the
dependencies is available in the file ``pyproject.toml``)

When you have ``poetry``, you can add dependencies using our makefile

.. note::
   If you want to create a virtual environment for this project 
   (as opposed to using the one you are currently in)
   you can use the command ``poetry env use python3.12``
   (or other python version)

.. code:: console

   $ make .make/deps

or directly with poetry

.. code:: console

   $ poetry install


TUDataset
^^^^^^^^^

*If you are planning to reproduce the classification experiment.*

The ``TUDataset`` package is also needed to run the classification experiment, 
but it is not available via ``pip`` / ``poetry``. 
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

The documentation is available online: `read the documentation <http://tristan.bruge.re/documentation/ot_markov_distances>`_

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

FAQ
---

I have a question about the paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case just send me an email through the email address mentioned in the paper.

I have noticed a bug in the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please use the Github "Issues" feature to open a ticket, and post a description of the bug, the error message and a
`minimal reproducible example <https://en.wikipedia.org/wiki/Minimal_reproducible_example>`_ . I’ll try to fix it.

Or if you have fixed it, you can submit a Pull Request directly

I cannot install the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you followed all the instructions correctly, please create a ticket using Github Issues.


Why do you need ``python3.10`` ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because I am using `structural pattern matching <https://peps.python.org/pep-0634/>`_, and some typing features such as `this one <https://peps.python.org/pep-0604/>`_ .



.. _`the tudataset repo`: https://github.com/chrsmrrs/tudataset

.. |github workflow badge| image:: https://github.com/YusuLab/ot_markov_distances/actions/workflows/testing-publish.yml/badge.svg
.. |codecov| image:: https://codecov.io/gh/YusuLab/ot_markov_distances/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/YusuLab/ot_markov_distances
.. |pytorch badge| image:: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
.. |doc badge| image:: https://img.shields.io/badge/documentation-green?style=for-the-badge&logo=readme&logoColor=black
   :target: https://tristan.bruge.re/documentation/ot_markov_distances

