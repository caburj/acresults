import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from .helper import *
from .groupers import *

def _weibull(arr):
    """returns the corresponding probability using _weibull formula."""
    n = arr.size
    return np.array([((r + 1) / (n + 1), v) for r, v in enumerate(arr)]).T

def _rev_sort(arr):
    return np.flipud(np.sort(arr))

def _non_zero(arr):
    return np.array(list(filter(lambda x: not np.isclose(x, 0), arr)))

def _p_zero(arr):
    return len(list(filter(lambda x: np.isclose(x, 0), arr))) / arr.size

def _p_local(p_z, p_actual):
    return (p_actual - p_z) / (1 - p_z)

class FrequencyAnalysis:
    def __init__(self, arr):
        """
        Class for frequency analysis.
        """
        non_zero_arr = _rev_sort(_non_zero(arr))
        self.arr = arr
        self.non_zero_arr = non_zero_arr
        self.p_z = _p_zero(arr)
        self.mu = np.mean(non_zero_arr)
        self.std = np.std(non_zero_arr)

    def get_value(self, prob_exceedance):
        """
        Returns the value corresponding to the given probability of exceedance.
        """
        if (1 - prob_exceedance < self.p_z):
            return 0.
        return norm.ppf(_p_local(self.p_z, 1 - prob_exceedance), self.mu, self.std)

    def get_prob_exceedance(self, value):
        """
        Returns the probability of exceedance corresponding to the given value.
        """
        if value <= 0.:
            return 1
        x = norm.cdf(value, self.mu, self.std)
        return 1 - (x * (1 - self.p_z) + self.p_z)

    def plot_distribution(self, xlabel="Value"):
        x = self.non_zero_arr
        y = np.array(lmap(lambda v: self.get_prob_exceedance(v), x))
        plt.plot(x, y)
        actual_y, actual_x = _weibull(self.non_zero_arr)
        plt.plot(actual_x, np.flipud(1 -(actual_y * (1 - self.p_z) + self.p_z)), "o")
        plt.xlabel(xlabel)
        plt.ylabel("Probability of Exceedance")

def aggregate(df, grouper, using=np.sum):
    """
    Aggregates a given dataframe.

    Parameters
    ==========
    grouper: a function for grouping the data
        Choose from available: `decade_grouper`, `month_grouper`
    
    using: a function to aggregate the data
        defaults to `np.sum`
    """
    return df.groupby(grouper).aggregate(using)

def NIR_fa(Inet_result, by="decade"):
    """
    Frequency analysis of the net irrigation requirements.
    """
    inet_keys = Inet_result.keys()
    grouper = decade_grouper if by == "decade" else month_grouper
    combine = lambda acc, x: pd.concat([acc, x])
    aggregated_inet = lmap(lambda k: aggregate(Inet_result[k].Inet, grouper, np.sum), inet_keys)
    all_data = reduce(combine, aggregated_inet)
    times = list(set(lmap(lambda v: v[4:], all_data.index.values)))
    regex_times = lmap(lambda s: f"{s}$", times)
    
    return {re_time[1:-1]: FrequencyAnalysis(all_data.filter(regex=re_time).values) for re_time in regex_times}

def NIR_fa_decade(Inet_result):
    return NIR_fa(Inet_result, by="decade")


def NIR_fa_month(Inet_result):
    return NIR_fa(Inet_result, by="month")

def NIR_chart_decade(nir_fa, start, end):
    time_index = decade_range(start, end)
    wet_values = list(map(lambda x: nir_fa[x].get_value(0.8), time_index))
    normal_values = list(map(lambda x: nir_fa[x].get_value(0.5), time_index))
    dry_values = list(map(lambda x: nir_fa[x].get_value(0.2), time_index))
    return pd.DataFrame([dry_values, normal_values, wet_values], index=["Dry", "Normal", "Wet"], columns=time_index)


def NIR_chart_month(nir_fa, start, end):
    time_index = month_range(start, end)
    wet_values = list(map(lambda x: nir_fa[x].get_value(0.8), time_index))
    normal_values = list(map(lambda x: nir_fa[x].get_value(0.5), time_index))
    dry_values = list(map(lambda x: nir_fa[x].get_value(0.2), time_index))
    return pd.DataFrame([dry_values, normal_values, wet_values], index=["Dry", "Normal", "Wet"], columns=time_index)


def decade_range(start="11-D2", end="04-D2"):
    """
    Returns a list of continuous decade based on the given start and end (inclusive).

    Parameters
    ==========
    start & end: string
        f"{month:02}-D{decade}"

    Return
    ======
    list of values having the same format as the inputs.
    """
    start_ = int(start[:2]), int(start[-1])
    end_ = int(end[:2]), int(end[-1])

    new_end_ = end_
    if start_[0] > end_[0]:
        new_end_ = end_[0] + 12, end_[1]

    lst = []
    for i in range(start_[0], new_end_[0] + 1):
        for j in range(1, 4):
            if i == start_[0] and j < start_[1]:
                continue
            if i == new_end_[0] and j > new_end_[1]:
                break
            lst.append(f"{i if i <= 12 else i % 12:02}-D{j}")

    return lst

def month_range(start="11", end="05"):
    """
    Returns a list of continuous months based on the given start and end (inclusive).

    Parameters
    ==========
    start & end: string
        f"{month:02}"

    Return
    ======
    list of values having the same format as the inputs.
    """
    start_ = int(start)
    end_ = int(end)

    new_end_ = end_
    if start_ > end_:
        new_end_ = end_ + 12
        
    lst = []
    for i in range(start_, new_end_ + 1):
        lst.append(f"{i if i <= 12 else i % 12:02}")

    return lst
