import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from functools import reduce

from . import helper
from . import groupers

from .helper import lmap, lfilter

__all__ = [
    "ACResult", 
    "FrequencyAnalysis",
    "NIR_chart_decade",
    "NIR_chart_month",
    "NIR_fa",
    "boxplot"
]

class ACResult:
    """The class abstracting all the results of an AquaCrop simulation.

    Method:
        get(name) : returns the results corresponding to the given result name.
            or use get_`name`.
        get_project_name() : returns the project name.
    """

    def __init__(self, project_name, aquacrop_dir):
        self.fnames = helper.get_result_fnames(project_name, aquacrop_dir)
        self.project_name = project_name
        self.clim = None
        self.compec = None
        self.compwc = None
        self.crop = None
        self.inet = None
        self.prof = None
        self.run = None
        self.salt = None
        self.wabal = None

    def get(self, name):
        """Extracts the results file with given name and returns as dict of
        pd.DataFrame or pd.DataFrame.

        Args:
            name (str) : Should be one in the following list
                [
                    "Clim", "CompEC", "CompWC", "Crop", "Inet", "Prof", "Salt",
                    "Wabal", "Run"
                ]
            run_number (int) : run number - based on the sequence of the
                simulations

        Returns:
            dict(int:pd.DataFrame) if run_number is None else
            pd.DataFrame of the result with `name`. 

        """
        if name == "Clim":
            return self.get_Clim()

        elif name == "CompEC":
            return self.get_CompEC()

        elif name == "CompWC":
            return self.get_CompWC()

        elif name == "Crop":
            return self.get_Crop()

        elif name == "Inet":
            return self.get_Inet()

        elif name == "Prof":
            return self.get_Prof()

        elif name == "Salt":
            return self.get_Salt()

        elif name == "Wabal":
            return self.get_Wabal()

        elif name == "Run":
            return self.get_Run()

        else:
            raise Exception(f"name: `{name}` is not recognized.")

    def get_project_name(self):
        return self.project_name

    def get_variable(self, variable):
        if helper.is_in(self.get("Clim"), variable):
            return helper.extract(self.get("Clim"), variable)

        elif helper.is_in(self.get("Crop"), variable):
            return helper.extract(self.get("Crop"), variable)
        
        elif helper.is_in(self.get("Inet"), variable):
            return helper.extract(self.get("Inet"), variable)

        elif helper.is_in(self.get("Prof"), variable):
            return helper.extract(self.get("Prof"), variable)

        elif helper.is_in(self.get("Salt"), variable):
            return helper.extract(self.get("Salt"), variable)

        elif helper.is_in(self.get("Wabal"), variable):
            return helper.extract(self.get("Wabal"), variable)
        
        else:
            raise Exception(f"variable = {variable}: Unable to find that variable.")

    def get_Clim(self):
        if self.clim is None:
            self.clim = helper.load(self.fnames['Clim'])
        return self.clim

    def get_CompEC(self):
        raise NotImplementedError()

        if self.compec is None:
            self.compec = helper.load(self.fnames['CompEC'])
        return self.compec

    def get_CompWC(self):
        raise NotImplementedError()

        if self.compwc is None:
            self.compwc = helper.load(self.fnames['CompWC'])
        return self.compwc

    def get_Crop(self):
        if self.crop is None:
            self.crop = helper.load(self.fnames['Crop'])
        return self.crop

    def get_Inet(self):
        if self.inet is None:
            self.inet = helper.load(self.fnames['Inet'])
        return self.inet

    def get_Prof(self):
        if self.prof is None:
            self.prof = helper.load(self.fnames['Prof'])
        return self.prof

    def get_Salt(self):
        if self.salt is None:
            self.salt = helper.load(self.fnames['Salt'])
        return self.salt

    def get_Wabal(self):
        if self.wabal is None:
            self.wabal = helper.load(self.fnames['Wabal'])
        return self.wabal

    def get_Run(self):
        if self.run is None:
            self.run = helper.load_Run(self.fnames['Run'])
        return self.run


class FrequencyAnalysis:
    def __init__(self, arr):
        """Class for frequency analysis.

        Methods:
            get_value(prob_exceedance)
                - returns the value corresponding to the given probability of 
                exceedance returns 0. if all the data is zero.

            get_prob_exceedance(value)
                - returns the probability of exceedance that corresponce to the 
                given value.
                - return 1. if the value is less than or equal to zero.

        """
        non_zero_arr = helper.rev_sort(helper.non_zero(arr))
        self.arr = arr
        self.non_zero_arr = non_zero_arr
        self.p_z = helper.p_zero(arr)
        self.mu = np.mean(non_zero_arr)
        self.std = np.std(non_zero_arr)

    def get_value(self, prob_exceedance):
        """
        Returns the value corresponding to the given probability of exceedance.
        """
        if (1 - prob_exceedance <= self.p_z):
            return 0.
        return norm.ppf(helper.p_local(self.p_z, 1 - prob_exceedance), self.mu, self.std)

    def get_prob_exceedance(self, value):
        """
        Returns the probability of exceedance corresponding to the given value.
        """
        if value <= 0.:
            return 1.
        x = norm.cdf(value, self.mu, self.std)
        return 1. - (x * (1. - self.p_z) + self.p_z)

    def plot_distribution(self, xlabel="Value"):
        x = self.non_zero_arr
        y = np.array(helper.lmap(lambda v: self.get_prob_exceedance(v), x))
        plt.plot(x, y)
        actual_y, actual_x = helper.weibull(self.non_zero_arr)
        plt.plot(actual_x, np.flipud(
            1 - (actual_y * (1 - self.p_z) + self.p_z)), "o")
        plt.xlabel(xlabel)
        plt.ylabel("Probability of Exceedance")


def aggregate(df, grouper, using=np.sum):
    """Aggregates a given dataframe.

    Arguments:
        grouper: a function for grouping the data
            Choose from available: `decade_grouper`, `month_grouper`

        using: a function to aggregate the data
            defaults to `np.sum`
    """
    return df.groupby(grouper).aggregate(using)


def NIR_fa(Inet_result, by="decade"):
    """Frequency analysis of the Net Irrigation Requirements.

    Arguments:
        Inet_result : from ACResult().get("Inet")
        by (string) : "decade" or "month" only
    """
    inet_keys = Inet_result.keys()
    grouper = groupers.decade_grouper if by == "decade" else groupers.month_grouper

    def combine(acc, x): return pd.concat([acc, x])
    aggregated_inet = helper.lmap(lambda k: aggregate(
        Inet_result[k].Inet, grouper, np.sum), inet_keys)
    all_data = reduce(combine, aggregated_inet)
    times = list(set(helper.lmap(lambda v: v[4:], all_data.index.values)))
    regex_times = helper.lmap(lambda s: f"{s}$", times)

    return {re_time[1:-1]: FrequencyAnalysis(all_data.filter(regex=re_time).values) for re_time in regex_times}


def NIR_fa_decade(Inet_result):
    return NIR_fa(Inet_result, by="decade")


def NIR_fa_month(Inet_result):
    return NIR_fa(Inet_result, by="month")


def NIR_chart_decade(Inet_result, start, end):
    """Net irrigation requirement chart.
    
    Based on probability of 0.8, 0.5 and 0.2 for wet, normal and dry conditions,
    respectively.

    Arguments:
        Inet_result : from ACResult().get_Inet()
        start (string) : start date with format->f"{month:02}-D{decade}" 
            e.g. "11-D1"
        end (string) : end date (inclusive) with format->f"{month:02}-D{decade}" 
            e.g. "05-D3"

    Returns:
        pd.DataFrame
    """
    nir_fa = NIR_fa_decade(Inet_result)
    time_index = decade_range(start, end)
    wet_values = list(map(lambda x: nir_fa[x].get_value(0.8), time_index))
    normal_values = list(map(lambda x: nir_fa[x].get_value(0.5), time_index))
    dry_values = list(map(lambda x: nir_fa[x].get_value(0.2), time_index))
    return pd.DataFrame([dry_values, normal_values, wet_values], index=["Dry", "Normal", "Wet"], columns=time_index)


def NIR_chart_month(Inet_result, start, end):
    """Net irrigation requirement chart.
    
    Based on probability of 0.8, 0.5 and 0.2 for wet, normal and dry conditions,
    respectively.

    Arguments:
        Inet_result : from ACResult().get_Inet()
        start (string) : start date with format->f"{month:02}" e.g. "11"
        end (string) : end date (inclusive) with format->f"{month:02}" e.g. "05"

    Returns:
        pd.DataFrame
    """
    nir_fa = NIR_fa_month(Inet_result)
    time_index = month_range(start, end)
    wet_values = list(map(lambda x: nir_fa[x].get_value(0.8), time_index))
    normal_values = list(map(lambda x: nir_fa[x].get_value(0.5), time_index))
    dry_values = list(map(lambda x: nir_fa[x].get_value(0.2), time_index))
    return pd.DataFrame([dry_values, normal_values, wet_values], index=["Dry", "Normal", "Wet"], columns=time_index)


def decade_range(start="11-D2", end="04-D2"):
    """Returns a list of continuous decade based on the given start and end 
    (inclusive).

    Arguments:
        start & end (string) : f"{month:02}-D{decade}"

    Returns:
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
    """Returns a list of continuous months based on the given start and end 
    (inclusive).

    Arguments:
        start & end (string) : f"{month:02}"

    Returns:
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

def boxplot(results, variable):
    """Makes boxplots of the variable of the given results.

    Only process the data from the `Run` result.

    Arguments:
        results (List[ACResults]) : list of ACResults object
        variable (string) : a variable in the Run result.
    """
    first_res = results[0].get("Run")
    column_names = list(first_res.columns)
    name_index = helper.lmap(lambda name: name.lower(), column_names).index(variable.lower())

    variable_ = column_names[name_index]
    unit = first_res.units[name_index]

    values = helper.lmap(lambda res: res.get("Run")[variable_], results)
    names = helper.lmap(lambda res: res.get_project_name(), results)
    
    plt.boxplot(values, notch=True)
    plt.xticks(range(1, len(results)+1), names)
    plt.ylabel(f"{variable_} ({unit})")
    plt.xlabel("Project Names")

def tsplot(result, variable, with_ci=True):
    """Time series plot of a variable from a ACResult object.

    Takes the mean of all the runs.

    Arguments:
        result (ACResult) : data
        variable (string) : variable to plot
        with_ci (bool) : with confidence interval?
    """
    
