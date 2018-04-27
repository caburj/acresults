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


def to_dataframe(lines):
    columns = lines[0].strip().split()
    values = lmap(lambda line: lmap(float, line.strip().split()), lines[2:])
    units = ["-"] * 5 + lines[1].strip().split()
    res = pd.DataFrame(values, columns=columns)
    date_index = [date(*map(int, (y, m, d)))
                  for y, m, d in zip(res.Year, res.Month, res.Day)]
    res.index = date_index
    res.units = units
    return res


def get_result_fnames(project_name, aquacrop_dir):
    results_dir = os.path.join(aquacrop_dir, "OUTP")
    names_with_OUT = {name: (lambda name: f"{project_name}{name}.OUT")(
        name) for name in constants.RESULTS_EXTENSIONS}
    return {name: (lambda name: os.path.join(results_dir, name))(names_with_OUT[name]) for name in names_with_OUT}


def load_Run(fname):
    with open(fname) as f:
        for i in range(3):
            f.readline()
        column_names = f.readline().strip().split()
        units = ["-"] * 4 + f.readline().strip().split() + ["-"] * 3
        remaining_lines = lmap(lambda s: s.strip(), f.readlines())
        index_of_Legend = remaining_lines.index("Legend")
        runs = lmap(lambda line: lmap(float, line.split()),
                    remaining_lines[:index_of_Legend - 1])
        res = pd.DataFrame(runs, columns=column_names)
        res.units = units
        return res


def load(fname):
    with open(fname) as f:
        lines = f.readlines()
        index_of_Legend = lines.index("Legend\n")
        asterisk_locs = [i for (i, truth) in enumerate(
            map(lambda line: starts_with("**", line), lines)) if truth]
        groups = {int(lines[i][15:19]): lines[i + 1:j - 1]
                  for i, j in zip(asterisk_locs, asterisk_locs[1:] + [index_of_Legend])}
        return {i: to_dataframe(groups[i]) for i in groups}


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

def is_in(dct, var):
    names = list(dct[1].columns)
    return to_lower(var.strip()) in lmap(to_lower, names)

def extract(dct, var):
    names = list(dct[1].columns)
    index = lmap(to_lower, names).index(to_lower(var))
    var_ = names[index]
    return {key: dct[key][[var_]].reset_index()[var_] for key in dct}