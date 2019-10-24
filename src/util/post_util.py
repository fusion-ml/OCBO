"""
Utilities for analyzing previous runs.
"""
from __future__ import division
import os
import pickle as pkl

import numpy as np

def load_results(run_path):
    """Load in runs from a path.
    Args:
        run_path: Path to the directory with the data.
    Returns: Options and list of results.
    """
    options = None
    f_infos = None
    data = []
    for filename in os.listdir(run_path):
        path = os.path.join(run_path, filename)
        if 'trial' in filename:
            with open(path, 'r') as f:
                data.append(pkl.load(f))
        elif 'run_info' in filename:
            with open(path, 'r') as f:
                options = pkl.load(f)
        elif 'function' in filename:
            with open(path, 'r') as f:
                f_infos = pkl.load(f)
    return options, data, f_infos

def get_total_best_mats(results, use_risk_neutral=False):
    """Assemble matrix of total bests where each row is single run.
    Args:
        results: List of dictionary mapping method -> Namespaces.
    Returns: Dictionary mapping method -> ndarray.
    """
    to_return = {}
    for run_dict in results:
        for method, res in list(run_dict.items()):
            val = res.rn_total if use_risk_neutral else res.total_best
            if method in to_return:
                to_return[method].append(val)
            else:
                to_return[method] = [val]
    for method, vals in (to_return.items()):
        to_return[method] = np.asarray(vals)
    return to_return

def get_choice_timelines(results):
    """Assemble matrix of choice timelines.
    Args:
        results: List of dictionary mapping method -> Namespaces.
    Returns: Dictionary mapping method -> ndarray.
    """
    to_return = {}
    for run_dict in results:
        for method, res in (run_dict.items()):
            if method in to_return:
                to_return[method].append(res.choice_timeline)
            else:
                to_return[method] = [res.choice_timeline]
    for method, vals in (to_return.items()):
        to_return[method] = np.asarray(vals)
    return to_return

def get_best_for_function(f_name, results):
    """Assemble matrix of bests for function where each row is single run.
    Args:
        results: List of dictionary mapping method -> Namespaces.
    Returns: Dictionary mapping method -> ndarray.
    """
    to_return = {}
    for run_dict in results:
        for method, res in (run_dict.items()):
            if method in to_return:
                to_return[method].append(res.curr_best[f_name])
            else:
                to_return[method] = [res.curr_best[f_name]]
    for method, vals in (to_return.items()):
        max_length = max([len(v) for v in vals])
        same_length = []
        for v in vals:
            diff_length = max_length - len(v)
            if diff_length > 0:
                v += [v[-1]] * diff_length
            same_length.append(list(zip(*v))[0])
        to_return[method] = np.asarray(same_length)
    return to_return

def get_query_proportions(results, f_names):
    """Get proportions of how much each function was queried.
    Args:
        results: List of dictionary mapping method -> Namespaces.
        f_names: Name of functions.
    Returns: Dictionary mapping method -> ndarray.
    """
    to_return = {}
    for run_dict in results:
        for method, res in (run_dict.items()):
            props = []
            for name in f_names:
                props.append(len(res.curr_best[name]))
            props = np.asarray(props, dtype=np.float64)
            props /= sum(props)
            if method in to_return:
                to_return[method].append(props)
            else:
                to_return[method] = [props]
    for method, vals in (to_return.items()):
        to_return[method] = np.asarray(vals)
    return to_return

def best_vals_to_regrets(f_infos, best_mats):
    """Convert best values found to simple regret.
    Args:
        f_infos: List of Namespaces for functions to look at.
        best_mats: Dict method -> matrix of best values found
            (row corresponds to run).
    Returns: Dictionary mapping method -> transformed matrix.
    """
    max_val = 0
    for f_info in f_infos:
        max_val += f_info.max_val
    to_return = {}
    for method, mat in (best_mats.items()):
        to_return[method] = max_val - mat
    return to_return

