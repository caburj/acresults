import numpy as np
import pandas as pd
import os
from datetime import date
from functools import reduce

from . import constants

class Run:
    """This is the abstraction of each run results.

    This can be used to represent the `Run` result.
    """

    def __init__(self, df, names, units, descriptions):
        self._df = df
        self._names = names
        self._units = units
        self._descriptions = descriptions

    def get_df(self):
        return self._df

    def get_names(self):
        return self._names

    def get_units(self):
        return self._units

    def get_descriptions(self):
        return self._descriptions

    def get_column(self, name):
        index = find(name, self.get_names())
        real_name = self.get_names()[index]
        unit = self.get_units()[real_name]
        try:
            description = self.get_descriptions()[real_name]
        except Exception:
            description = ''
        column = self.get_df()[real_name]
        column.name = real_name
        column.unit = unit
        column.description = description
        return column

class Result:
    """Class for each of the result files.
    
    """
    def __init__(self, runs):
        self._runs = runs
    
    def get_runs(self):
        return self._runs

    def get_run(self, number):
        return self.get_runs()[number]

    def get_names(self):
        return self.get_run(1).get_names()

    def get_units(self):
        return self.get_run(1).get_units()

    def get_descriptions(self):
        return self.get_run(1).get_descriptions()

    def get_column(self, name, DAP_as_index=False):
        runs = self.get_runs()
        if DAP_as_index:
            def re_index(series):
                name = series.name
                series_ = series.reset_index()
                series_.index = series_.index + 1
                return series_[name]
            return {k: re_index(runs[k].get_column(name)) for k in runs}
        return {k: runs[k].get_column(name) for k in runs}


def lmap(f, lst):
    return list(map(f, lst))


def lfilter(pred, lst):
    return list(filter(pred, lst))


def starts_with(this, text):
    return this == text[:len(this)]


def to_dataframe(lines):
    columns = lines[0].strip().split()
    values = lmap(lambda line: lmap(float, line.strip().split()), lines[2:])
    res = pd.DataFrame(values, columns=columns)
    date_index = [date(*map(int, (y, m, d)))
                  for y, m, d in zip(res.Year, res.Month, res.Day)]
    res.index = date_index
    return res


def get_result_fnames(project_name, aquacrop_dir):
    results_dir = os.path.join(aquacrop_dir, "OUTP")
    names_with_OUT = {name: (lambda name: f"{project_name}{name}.OUT")(
        name) for name in constants.RESULTS_EXTENSIONS}
    return {name: (lambda name: os.path.join(results_dir, name))(names_with_OUT[name]) for name in names_with_OUT}


def load_summary(fname):
    with open(fname) as f:
        for i in range(3):
            f.readline()
        names = f.readline().strip().split()
        units = ["-"] * 4 + f.readline().strip().split() + ["-"] * 3
        remaining_lines = lmap(lambda s: s.strip(), f.readlines())
        index_of_Legend = remaining_lines.index("Legend")
        summary = lmap(lambda line: lmap(float, line.split()),
                    remaining_lines[:index_of_Legend - 1])
        summary_df = pd.DataFrame(summary, columns=names)
        description_lines = remaining_lines[index_of_Legend + 2:-1]
        descriptions = dict([(line[:11].strip(), line[11:].strip())
                             for line in description_lines])
        return Run(summary_df, names, dict(zip(names, units)), descriptions)


def load_result(fname):
    def _get_description(line, stage_description=[]):
        if line[:5] == "Stage":
            stage_description.append(line[14:].strip().split(": "))
            if line[15] == "4":
                return "Stage", dict(stage_description)
        return line[:13].strip(), line[14:].strip()

    with open(fname) as f:
        lines = f.readlines()
        index_of_Legend = lines.index("Legend\n")
        asterisk_locs = [i 
                         for (i, truth) in enumerate(
                             map(lambda line: starts_with("**", line), lines)) 
                         if truth]
        groups = {int(lines[i][15:19]): lines[i + 1:j - 1]
                  for i, j in zip(asterisk_locs, asterisk_locs[1:] + [index_of_Legend])}
        
        units = ['-'] * 5 + lines[5].strip().split()
        names = lines[4].strip().split()

        description_lines = lines[index_of_Legend + 2 : ]
        descriptions = dict([_get_description(line) for line in description_lines])
        runs = {i: Run(to_dataframe(groups[i]), names, dict(zip(names, units)), descriptions) 
                for i in groups}
        return Result(runs)
        

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
