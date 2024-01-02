"""
Part of this code is adapted from https://github.com/chens5/WL-distance/blob/main/experiments/svm.py
"""

# ------------------ Imports ------------------
from typing import Literal, Any
import argparse
from argparse import ArgumentParser
from contextlib import contextmanager
import itertools as it; import more_itertools as mit
import functools as ft
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import default_rng; rng=default_rng()
import seaborn as sns
import torch
from torch import Tensor
import torch.utils.data


def append_path(s):
    s = os.path.abspath(s)
    if s in sys.path: return
    sys.path.insert(0,s)
append_path("")
append_path(f"{os.environ['HOME']}/software/tudataset")

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import grakel
import grakel.datasets
from tud_benchmark import kernel_baselines#the grakel kernels bug with continuous features
from tud_benchmark.auxiliarymethods import datasets as tud_benchmarks_dp
import networkx as nx
import time
import ot
import sys
# import wwl
# import igraph as ig
import cProfile
import re
import multiprocessing as mp
import matplotlib.pyplot as plt
import yaml


from .utils.grakel_datasets import Dataset
from .utils.distance_matrix import compute_distance_matrix as wl_distance_matrix, compute_distance_matrix_fgw as fgw_distance_matrix, compute_distance_matrix_dwl
from .utils.ksvm_utils import KreinSVC

# ------------------ utils ------------------

@contextmanager
def cwd(path):
    """Context manager to change the working directory, 
    gotten from https://stackoverflow.com/a/37996581/4948719"""
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def datasets_from_yaml(yaml_path: Path) -> list[Dataset]:
    with open(yaml_path, "r") as f:
        datasets = yaml.safe_load(f)
    return [Dataset.get(**ds) for ds in datasets]


# ------------------ datasets_info ------------------

def datasets_info(yaml_path: Path):
    datasets = datasets_from_yaml(yaml_path)
    for ds in datasets:
        print("")
        print(ds)
        print(f"labels: {np.unique(ds.target)}")
        print(f"n_samples: {len(ds.target)}")
        ex_graph = ds.graphs_nx[rng.integers(len(ds.graphs_nx))]
        ex_node = ex_graph.nodes[rng.integers(len(ex_graph.nodes))]
        ex_feature = ex_node['attr']
        try:
            shape = len(ex_feature)
        except:
            shape = "scalar"
        print(f"n_features: {shape}")
        print(f"example_feature: {ex_feature}")

# ------------------ distance matrices ------------------

def compute_distance_matrices(datasets_yaml: Path, output_file: Path, 
                model: Literal["WL_DELTA"], other_params: dict[str, Any]):

    datasets = datasets_from_yaml(datasets_yaml)
    matrices = dict()
    match model:
        case "WL_DELTA":
            for ds in datasets:
                matrices[ds.title] = wl_distance_matrix(
                        ds.graphs_nx, ds.graphs_nx,
                        attr_type=ds.attr_type,
                        **other_params).cpu()
        case "DWL":
            for ds in datasets:
                matrices[ds.title] = compute_distance_matrix_dwl(
                        ds.graphs_nx, ds.graphs_nx,
                        attr_type=ds.attr_type,
                        **other_params).cpu()
        case "WL"|"WLOA"|"GRAPHLET":
            n_iter  = other_params.get("n_iter", 5)
            match model:
                case "WL":
                    #kernel = grakel.WeisfeilerLehman(**other_params, normalize=True)
                    kernel = kernel_baselines.compute_wl_1_dense
                case "WLOA":
                    #kernel = grakel.WeisfeilerLehmanOptimalAssignment(**other_params, normalize=True)
                    kernel = kernel_baselines.compute_wloa_dense
                case "GRAPHLET":
                    kernel = kernel_baselines.compute_graphlet_dense
            for ds in datasets:
                # matrices[ds.title] = torch.as_tensor(kernel.fit_transform(ds.grakel_ds.data))
                with cwd(f"{os.environ['HOME']}/software/tudataset/tud_benchmark"):
                    # function only works if the current working directory is tud_benchmark
                    tud_benchmarks_dp.get_dataset(ds.grakel_name)
                    if model == "GRAPHLET":
                        gram_matrix = kernel(
                                ds.grakel_name, 
                                ds.attr_type != "degree", False )
                    else:
                        gram_matrix = kernel(
                                ds.grakel_name, 
                                n_iter, ds.attr_type != "degree", False )
                    matrices[ds.title] = torch.as_tensor(gram_matrix)
        case "WWL":
            raise NotImplementedError()
        case "FGW":
            for ds in datasets:
                matrices[ds.title] = fgw_distance_matrix(
                        ds.graphs_nx, ds.graphs_nx,
                        attr_type=ds.attr_type,
                        **other_params).cpu()
        case _:
            raise ValueError(f"Unknown model: {model}")

    torch.save(matrices, output_file)

# ------------------ classification ------------------

@ignore_warnings(category=ConvergenceWarning)
def choose_parameters(D, y, 
                      gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000] , 
                      Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000] , 
                      cv=5,
                      model="svm"):
    # stolen from https://github.com/chens5/WL-distance/blob/6733ed1859a491a29ea8b813ac4ba1903357048f/experiments/svm.py#L181C1-L211C33 
    cv = StratifiedKFold(n_splits=cv)
    results = []
    param_pairs = []
    for g in gammas:
        for c in Cs:
            param_pairs.append((g, c))

    for train_index, test_index in cv.split(D, y):
        split_results = []
        for i in range(len(gammas)):
            for j in range(len(Cs)):
                g = gammas[i]
                c = Cs[j]
                D_train = D[train_index][:, train_index]
                D_test = D[test_index][:, train_index]
                y_train = y[train_index]
                y_test = y[test_index]
                median = np.median(D_train)
                if model == "svm":
                    K_train = np.exp(-D_train / (median * g))
                    K_test = np.exp(-D_test / (median * g))
                    clf = SVC(kernel='precomputed', C = c, max_iter=1000)
                    clf.fit(K_train, y_train)
                    y_pred = clf.predict(K_test)
                elif model == "krein-svm":
                    clf = KreinSVC(kernel='precomputed', C = c, max_iter=1000, 
                                   psd_gamma=g * median)
                    clf.fit(D_train, y_train)
                    y_pred = clf.predict(D_test)
                split_results.append(accuracy_score(y_test, y_pred))
        results.append(split_results)

    results = np.array(results)
    fin_results = results.mean(axis=0)
    best_idx = np.argmax(fin_results)
    return param_pairs[best_idx]

def test_knn(D_train, D_test, y_train, y_test, k=1, is_gram=False, **kwargs):
    neigh = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    neigh.fit(D_train, y_train)
    y_pred = neigh.predict(D_test)
    return accuracy_score(y_test, y_pred)

def test_svm(D_train, D_test, y_train, y_test, is_gram=False, **kwargs):
    if not is_gram:
        gamma, C = choose_parameters(D_train, y_train)
        median = np.median(D_train)
        kernel_train = np.exp(- D_train / (gamma * median))
        kernel_test = np.exp( - D_test / (gamma * median))
    else:
        kernel_train = D_train
        kernel_test = D_test
    clf = SVC(kernel="precomputed")
    clf.fit(kernel_train, y_train)
    y_pred = clf.predict(kernel_test)
    return accuracy_score(y_test, y_pred)

def test_krein_svc(D_train, D_test, y_train, y_test, is_gram=False, **kwargs):
    assert not is_gram, "Krein kernel only works with distance matrices"
    # grid search
    # gamma, C = choose_parameters(D_train, y_train, model="krein-svm")
    # median = np.median(D_train)
    # clf = KreinSVC(C=C, psd_gamma=gamma * median)
    clf = KreinSVC()
    clf.fit(D_train, y_train, )
    y_pred = clf.predict(D_test)
    return accuracy_score(y_test, y_pred)

def test_model(distances, gt, model="knn", **kwargs):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    if not kwargs.get("is_gram", False):
        # set negative values to NaN
        # they’re probably due to numerical errors
        distances[distances < 0] = np.nan

    # fill nans with +inf -> doesn_t work, use very large number instead
    distances = np.nan_to_num(distances, nan=1e10)
    for train_indices, test_indices in kfold.split(distances, gt):
        D_train = distances[train_indices][:, train_indices]
        D_test = distances[test_indices][:, train_indices]
        y_train = gt[train_indices]
        y_test = gt[test_indices]

        match model:
            case "knn":
                accuracies.append(test_knn(D_train, D_test, y_train, y_test, **kwargs))
            case "svm":
                accuracies.append(test_svm(D_train, D_test, y_train, y_test, **kwargs))
            case "krein-svm":
                accuracies.append(test_krein_svc(D_train, D_test, y_train, y_test, **kwargs))
    return np.mean(accuracies), np.std(accuracies)

def classification(datasets_yaml: Path, distances_file: Path, **kwargs):
    datasets = datasets_from_yaml(datasets_yaml)
    distances = torch.load(distances_file)
    for ds in datasets:
        print("")
        print(ds.title)
        gt = ds.target
        ds_distances = distances[ds.title].numpy()
        # knn_acc, knn_std = test_model(ds_distances, gt, "knn", **kwargs)
        acc, std= test_model(ds_distances, gt, **kwargs)
        print(f"{kwargs['model']}: {acc:.1%} ± {std:.1%}")
        # print(f"svm: {svm_acc:.3f} ± {svm_std:.3f}")


# ------------------ Usual command line boilerplate ------------------

def argument_parser(parser=None):
    if parser is None:
        parser = ArgumentParser(
                prog="python -m experiments.classification",
                description="Run classification experiments on graph datasets",
                )

    subparsers = parser.add_subparsers(dest="command")

    # ------------------ Command: datasets_info ------------------
    parser_datasets_info = subparsers.add_parser(
            "datasets_info",
            help="Print information about given datasets",
    )
    parser_datasets_info.add_argument(
            "datasets",
            type=Path,
            help="Path to the yaml file containing dataset information"
    )

    # ------------------ Command: compute_distance_matrices ------------------
    parser_compute_distance_matrices = subparsers.add_parser(
            "distances",
            help="Compute distance matrices for given datasets",
    )
    parser_compute_distance_matrices.add_argument(
            "datasets",
            type=Path,
            help="Path to the yaml file containing dataset information"
    )

    parser_compute_distance_matrices.add_argument(
            "output_file",
            type=Path,
            help="Path to the output file"
    )

    parser_compute_distance_matrices.add_argument(
            "--model",
            type=str,
            default="WL_DELTA",
            choices=["WL_DELTA", "WL", "WLOA", "DWL", "GRAPHLET", "FGW"],
            help="Model to use"
    )

    parser_compute_distance_matrices.add_argument(
            "--delta",
            type=float,
            default=None,
            help="Delta parameter for WL_DELTA"
    )

    parser_compute_distance_matrices.add_argument(
            "--q",
            type=float,
            default=None,
            help="q parameter for Markov based kernels"
    )

    parser_compute_distance_matrices.add_argument(
            "--epsilon",
            type=float,
            default=None,
            help="epsilon parameter for sinkhorn", 
    )

    parser_compute_distance_matrices.add_argument(
            "--n_iter",
            type=int,
            default=None,
            help="Number of iterations for models that need it", 
    )

    parser_compute_distance_matrices.add_argument(
            "--device",
            type=torch.device,
            default=None,
            help="Device to use for computation"
    )   

    parser_compute_distance_matrices.add_argument(
            "--k",
            type=int,
            default=2, 
            help="K parameter for dWL-k"
    )

    # ------------------ Command: eval ------------------
    parser_eval = subparsers.add_parser(
            "eval",
            help="Evaluate a kernel based on distance matrix",
    )

    parser_eval.add_argument(
            "datasets",
            type=Path,
            help="Path to the yaml file containing dataset information"
    )

    parser_eval.add_argument(
            "distance_matrices",
            type=Path,
            help="Path to the distance matrices file"
    )

    parser_eval.add_argument(
            "--is-gram",
            type=bool,
            default=False,
            action=argparse.BooleanOptionalAction, 
            help="Whether the distance matrices are already gram matrices"
    )   

    parser_eval.add_argument(
            "--model",
            type=str,
            default="knn",
            choices=["knn", "svm", "krein-svm"],
            help="Model to use"
    )

    parser_eval.add_argument(
            "--gamma",
            type=float,
            default=None,
            help="Gamma parameter for SVM"
    )

    
    return parser

if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    match args.command:
        case "datasets_info":
            datasets_info(args.datasets)

        case "distances":
            other_params = dict()
            if args.delta is not None: other_params["delta"] = args.delta
            if args.q is not None: other_params["q"] = args.q
            if args.epsilon is not None: other_params["sinkhorn_reg"] = args.epsilon
            if args.k is not None: other_params["k"] = args.k
            if args.device is not None: other_params["device"] = args.device
            with torch.no_grad():
                compute_distance_matrices(args.datasets, 
                                          args.output_file, args.model, other_params) 
        case "eval":
            other_params = dict()
            if args.gamma is not None: other_params["gamma"] = args.gamma
            with torch.no_grad():
                classification(args.datasets, args.distance_matrices, 
                               model=args.model, 
                               is_gram=args.is_gram,  **other_params)
        case _:
            parser.print_help()
            sys.exit(1)
