import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from .helper import *

def weibull(arr):
    """returns the corresponding probability using weibull formula."""
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

class FrequencyAnalysis:
    def __init__(self, arr):
        non_zero_arr = rev_sort(non_zero(arr))
        self.arr = arr
        self.non_zero_arr = non_zero_arr
        self.p_z = p_zero(arr)
        self.mu = np.mean(non_zero_arr)
        self.std = np.std(non_zero_arr)

    def get_value(self, prob_exceedance):
        """Returns the value corresponding to the given probability of exceedance."""
        if (1 - prob_exceedance < self.p_z):
            return 0.
        return norm.ppf(p_local(self.p_z, 1 - prob_exceedance), self.mu, self.std)

    def get_prob_exceedance(self, value):
        """Returns the probability of exceedance corresponding to the given value."""
        if value <= 0.:
            return 1
        x = norm.cdf(value, self.mu, self.std)
        return 1 - (x * (1 - self.p_z) + self.p_z)

    def plot_distribution(self, xlabel="Value"):
        x = self.non_zero_arr
        y = np.array(lmap(lambda v: self.get_prob_exceedance(v), x))
        plt.plot(x, y)
        actual_y, actual_x = weibull(self.non_zero_arr)
        plt.plot(actual_x, np.flipud(1 -(actual_y * (1 - self.p_z) + self.p_z)), "o")
        plt.xlabel(xlabel)
        plt.ylabel("Probability of Exceedance")
