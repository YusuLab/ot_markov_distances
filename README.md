# ot_markov_distances

Distances on graphs based on optimal transport 

This is the implementation code for {cite}`brugère2023distances`

```
Brugere, T., Wan, Z., & Wang, Y. (2023). Distances for Markov Chains, and Their Differentiation. ArXiv, abs/2302.08621.
```

## Setup

### Dependencies

This package manages its dependencies via [poetry](https://python-poetry.org/). 
I recommend you install it (otherwise if you prefer to manage them manually, a list of the dependencies is available in the file `pyproject.toml`)

When you have `poetry`, you can add dependencies using our makefile

```console
$ make .make/deps
```

or directly with poetry

```console
$ poetry install
```

### Project structure

```console
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
```


## Documentation

```{warning}
Do not edit the documentation directly in the `docs/` folder, 
that folder is wiped every time the documentation is built. 
The static parts of the documentation can be edited in `staticdocs/`.
```

You can build documentation and run tests using

```console
$ make
```

Alternatively, you can build only the documentation using

```console
$ make .make/build-docs
```

The documentation will be available in `docs/build/html` in the readthedocs format



