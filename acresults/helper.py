import numpy as np
import pandas as pd
import os
from datetime import date
from functools import reduce

from . import constants


def lmap(f, lst):
    return list(map(f, lst))


def lfilter(pred, lst):
    return list(filter(pred, lst))


def starts_with(this, text):
    return this == text[:len(this)]


def get_result_fnames(project_name, aquacrop_dir):
    results_dir = os.path.join(aquacrop_dir, "OUTP")
    names_with_OUT = {name: (lambda name: f"{project_name}{name}.OUT")(
        name) for name in constants.RESULTS_EXTENSIONS}
    return {name: (lambda name: os.path.join(results_dir, name))(names_with_OUT[name]) for name in names_with_OUT}


def weibull(arr):
    """returns the corresponding probability using _weibull formula."""
    n = arr.size
    return np.array([((r + 1) / (n + 1), v) for r, v in enumerate(arr)]).T


def rev_sort(arr):
    return np.flipud(np.sort(arr))


def non_zero(arr):
    return np.array(list(filter(lambda x: not np.isclose(x, 0), arr)))


def p_zero(arr):
    return len(list(filter(lambda x: np.isclose(x, 0), arr))) / arr.size


def p_local(p_z, p_actual):
    return (p_actual - p_z) / (1 - p_z)

def to_lower(s):
    return s.lower()

def is_in(result, var):
    return var.strip().lower() in lmap(to_lower, result.get_names())

def extract(result, var):
    index = find(var, result.get_names())
    real_name = result.get_names()[index]
    return result.get_column(real_name)
    
def find(name, list_names):
    """Finds `name` from the `list_names` by ignoring the case of all the names.

    Returns:
        index
    """
    lower_names = lmap(to_lower, list_names)
    return lower_names.index(name.lower())

def concat(results):
    lengths = lmap(lambda res: len(res), results)
    results_ = [
        res.df.assign(project_name=[res.project_name] * n)
            for res, n in zip(results, lengths)
    ]
    return reduce(lambda acc, a: pd.concat([acc, a], ignore_index=True), results_)
    