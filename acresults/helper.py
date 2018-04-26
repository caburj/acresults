import numpy as np
import pandas as pd
import os

from .constants import *

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
    
    try:
        res = pd.DataFrame(values, columns=columns)
    except AssertionError:
        return None
    
    res.units = units
    return res

def get_result_fnames(project_name, aquacrop_dir):
    results_dir = os.path.join(aquacrop_dir, "OUTP")
    names_with_OUT = {name:(lambda name: f"{project_name}{name}.OUT")(name) for name in RESULTS_EXTENSIONS}
    return {name:(lambda name: os.path.join(results_dir, name))(names_with_OUT[name]) for name in names_with_OUT}

def load_Run(fname):
    with open(fname) as f:
        for i in range(3):
            f.readline()
        column_names = f.readline().strip().split()
        units = ["-"] * 4 + f.readline().strip().split() + ["-"] * 3
        remaining_lines = lmap(lambda s: s.strip(), f.readlines())
        index_of_Legend = remaining_lines.index("Legend")
        runs = lmap(lambda line: lmap(float, line.split()), remaining_lines[:index_of_Legend-1])
        res = pd.DataFrame(runs, columns=column_names)
        res.units = units
        return res
    
def load(fname):
    with open(fname) as f:
        lines = f.readlines()
        index_of_Legend = lines.index("Legend\n")
        lines_ = lines[3:index_of_Legend]
        asterisk_locs = [i for (i, truth) in enumerate(map(lambda line: starts_with("**", line), lines_)) if truth]
        groups = {int(lines_[i][15:19]):lines_[i+1:j-1] for i, j in zip(asterisk_locs, asterisk_locs[1:] + [index_of_Legend])}

    return {i: to_dataframe(groups[i]) for i in groups}