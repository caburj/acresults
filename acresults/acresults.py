import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from functools import reduce
from datetime import date
import seaborn as sns
from collections import namedtuple

from . import helper
from . import groupers
from . import constants

from .helper import lmap, lfilter


__all__ = [
    "ACOutput",
    "FrequencyAnalysis",
    "NIR_chart_decade",
    "NIR_chart_month",
    "NIR_fa",
    "boxplot",
    "tsplot",
    "barplot",
    "summarize"
]


Summary = namedtuple('Summary', 'df name unit')


#############
## CLASSES ##
#############


class ACOutput:
    def __init__(self, project_name, aquacrop_dir):
        self.fnames = helper.get_result_fnames(project_name, aquacrop_dir)
        self.project_name = project_name
        self.data = {}

    def get(self, name):
        return self[name]

    def get_variable(self, var):
        res = self._find_result_with(var)

        if res is None:
            raise Exception(f"Unable to find variable='{var}'.")

        index = helper.find(var, res.names)
        real_name = res.names[index]
        unit = res.units[real_name]
        try:
            description = res.descriptions[real_name]
        except Exception:
            description = ''
        project_name = res.project_name
        df = res[['run_number', 'date', 'DAP', real_name]]
        return Variable(df, real_name, unit, description, project_name)
    
    def get_summary(self, var=None):
        if var is None:
            return self['Run']
        
        res = self['Run']
        try:
            index = helper.find(var, res.names)
        except Exception:
            raise Exception(f"Unable to find variable='{var}' in summary.")

        real_name = res.names[index]
        unit = res.units[real_name]
        try:
            description = res.descriptions[real_name]
        except Exception:
            description = ''
        project_name = res.project_name
        df = res.df[['RunNr', real_name]]
        return Variable(df, real_name, unit, description, project_name)

    def __getitem__(self, name):
        try:
            index = helper.find(name, constants.RESULTS_EXTENSIONS)
            real_name = constants.RESULTS_EXTENSIONS[index]
            if not (real_name in self.data):
                if real_name == "Run":
                    res = self._read_run(real_name, self.project_name)
                else:
                    res = self._read(real_name, self.project_name)
                self.data[real_name] = res
            return self.data[real_name]
        except ValueError:
            raise Exception(f"Cannot find name=`{name}`.")

    def __repr__(self):
        return f"ACOutput(project_name='{self.project_name}')"

    def _find_result_with(self, var):
        def is_in(var, res):
            try:
                helper.find(var, res.names)
                return True
            except Exception:
                return False
        results = [self[name] for name in constants.RESULTS_EXTENSIONS if name != 'Run']
        return reduce(lambda acc, a: a if is_in(var, a) else acc, results, None)

    def _read_run(self, name, project_name):
        with open(self.fnames[name]) as f:
            for i in range(3):
                f.readline()
            names = f.readline().strip().split()
            units = ["-"] * 4 + f.readline().strip().split() + ["-"] * 3
            remaining_lines = lmap(lambda s: s.strip(), f.readlines())
            index_of_Legend = remaining_lines.index("Legend")
            summary = lmap(lambda line: lmap(float, line.split()),
                        remaining_lines[:index_of_Legend - 1])
            summary_df = pd.DataFrame(summary, columns=names)
            description_lines = remaining_lines[index_of_Legend + 2:]
            descriptions = dict([(line[:11].strip(), line[11:].strip())
                                for line in description_lines])
            return Result(summary_df, names, dict(zip(names, units)), descriptions, project_name, name)

    def _read(self, name, project_name):
        def _get_description(line, stage_description=[]):
            if line[:5] == "Stage":
                stage_description.append(line[14:].strip().split(": "))
                if line[15] == "4":
                    return "Stage", dict(stage_description)
            return line[:13].strip(), line[14:].strip()

        def to_df(lines, run_number):
            names = lines[0].strip().split()
            values = lmap(lambda line: lmap(float, line.strip().split()), lines[2:])
            res = pd.DataFrame(values, columns=names)
            dates = [date(*map(int, (y, m, d))) for y, m, d in zip(res.Year, res.Month, res.Day)]
            return res.assign(date=dates).assign(run_number=[run_number] * len(dates))

        with open(self.fnames[name]) as f:
            lines = f.readlines()
            index_of_Legend = lines.index("Legend\n")
            asterisk_locs = [i 
                            for (i, truth) in enumerate(
                                map(lambda line: helper.starts_with("**", line), lines)) 
                            if truth]
            groups = {int(lines[i][15:19]): lines[i + 1:j - 1]
                    for i, j in zip(asterisk_locs, asterisk_locs[1:] + [index_of_Legend])}
            
            units = ['-'] * 5 + lines[5].strip().split() + ['-'] * 2
            names = lines[4].strip().split() + ['date', 'run_number']
            units_ = dict(zip(names, units))

            description_lines = lines[index_of_Legend + 2 : ]
            descriptions = dict([_get_description(line) for line in description_lines])
            res_list = [to_df(lines, n) for n, lines in groups.items()]
            df = reduce(lambda acc, a: pd.concat([acc, a]), res_list).reset_index()
            del df['index']
            return Result(df, names, units_, descriptions, project_name, name)


class Result:
    def __init__(self, df, names, units, descriptions, project_name, name):
        self.df = df
        self.names = names
        self.units = units
        self.descriptions = descriptions
        self.project_name = project_name
        self.name = name
        self._len = len(self.df)

    def __getitem__(self, name):
        return self.get_variable(name)

    def __len__(self):
        return self._len
    
    def __repr__(self):
        return f"Result(name='{self.name}', project_name='{self.project_name}')"

    def get_variable(self, name):
        try:
            index = helper.find(name, self.names)
        except Exception:
            raise Exception(f"Unable to find variable='{name}'.")

        real_name = self.names[index]
        unit = res.units[real_name]

        try:
            description = res.descriptions[real_name]
        except Exception:
            description = ''

        project_name = res.project_name
        df = res[['run_number', 'date', 'DAP', real_name]]
        return Variable(df, real_name, unit, description)

    def get_run_numbers(self):
        return list(set(self.df.run_number))

    def get_run(self, number):
        return self.df[self.df.run_number == number]


class Variable:
    def __init__(self, df, name, unit, description, project_name):
        self.df = df
        self.name = name
        self.unit = unit
        self.description = description
        self.project_name = project_name

    def __getitem__(self, name):
        return self.df[name]

    def __repr__(self):
        return f"Variable(name='{self.name}', project_name='{self.project_name}')"

    def values(self):
        return self.df[self.name].values

    def get_run_numbers(self):
        return list(set(self.df.run_number))

    def get_run(self, number, index='DAP'):
        if not (index in ['DAP', 'date']):
            index = 'DAP'
        df = self.df[self.df.run_number == number][[index, self.name]]
        return df.set_index(index)[self.name]


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


######################
## PUBLIC FUNCTIONS ##
######################


def aggregate(df, grouper, using=np.sum):
    """Aggregates a given dataframe.

    Arguments:
        grouper: a function for grouping the data
            Choose from available: `decade_grouper`, `month_grouper`

        using: a function to aggregate the data
            defaults to `np.sum`
    """
    return df.groupby(grouper).aggregate(using)


def NIR_fa(output, by="decade"):
    """Frequency analysis of the Net Irrigation Requirements.

    Arguments:
        Inet_result : from ACResult().get("Inet")
        by (string) : "decade" or "month" only
    """
    res = output['inet']
    x = dict(list(res.df.groupby('run_number')))
    inet = {i: x[i].set_index('date')['Inet'] for i in x.keys()}
    grouper = groupers.decade_grouper if by == "decade" else groupers.month_grouper

    def combine(acc, x): 
        return pd.concat([acc, x])

    aggregated_inet = helper.lmap(lambda k: aggregate(inet[k], grouper, np.sum), inet.keys())
    all_data = reduce(combine, aggregated_inet)
    times = list(set(helper.lmap(lambda v: v[4:], all_data.index.values)))
    regex_times = helper.lmap(lambda s: f"{s}$", times)

    return {re_time[1:-1]: FrequencyAnalysis(all_data.filter(regex=re_time).values) for re_time in regex_times}


def NIR_fa_decade(output):
    return NIR_fa(output, by="decade")


def NIR_fa_month(output):
    return NIR_fa(output, by="month")


def NIR_chart_decade(acresult, start, end):
    """Net irrigation requirement chart.
    
    Based on probability of 0.8, 0.5 and 0.2 for wet, normal and dry conditions,
    respectively.

    Arguments:
        acresult : from ACResult()
        start (string) : start date with format->f"{month:02}-D{decade}" 
            e.g. "11-D1"
        end (string) : end date (inclusive) with format->f"{month:02}-D{decade}" 
            e.g. "05-D3"

    Returns:
        pd.DataFrame
    """
    nir_fa = NIR_fa_decade(acresult)
    time_index = decade_range(start, end)
    wet_values = list(map(lambda x: nir_fa[x].get_value(0.8), time_index))
    normal_values = list(map(lambda x: nir_fa[x].get_value(0.5), time_index))
    dry_values = list(map(lambda x: nir_fa[x].get_value(0.2), time_index))
    return pd.DataFrame([dry_values, normal_values, wet_values], index=["Dry", "Normal", "Wet"], columns=time_index)


def NIR_chart_month(output, start, end):
    """Net irrigation requirement chart.
    
    Based on probability of 0.8, 0.5 and 0.2 for wet, normal and dry conditions,
    respectively.

    Arguments:
        output : from ACResult()
        start (string) : start date with format->f"{month:02}" e.g. "11"
        end (string) : end date (inclusive) with format->f"{month:02}" e.g. "05"

    Returns:
        pd.DataFrame
    """
    nir_fa = NIR_fa_month(output)
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


def tsplot(output, variable, with_ci=True, ci_percentage=0.9):
    """Time series plot of a variable from a ACResult object.

    Takes the mean of all the runs.

    For simplicity, last element is not included in the calculation.

    Arguments:
        acresult (ACResult) : data
        variable (string) : variable to plot
        with_ci (bool) : with confidence interval?
    """
    var_table = output.get_variable(variable)
    variable_ = var_table.name
    columns = dict(list(var_table.df.groupby('run_number')))
    values_ = pd.concat([columns[k][variable_][:-1].reset_index()[variable_] for k in columns], axis=1)
    mean = values_.mean(axis=1).values
    std = values_.std(axis=1).values
    z = norm.ppf(ci_percentage + (1 - ci_percentage) / 2)
    upper = mean + z * std
    lower = mean - z * std
    plt.plot(np.arange(1, std.size + 1), mean, label=var_table.project_name)
    plt.fill_between(np.arange(1, std.size+1), lower, upper, alpha=0.3)
    plt.xlabel("DAP")

    unit = var_table.unit

    plt.ylabel(f"{variable_} ({unit})")


def boxplot(outputs, variable, **kwargs):
    """Makes boxplots of the variable of the given outputs.

    Only process the data from the `Run` result.

    Arguments:
        outputs (List[ACOutput]) : list of ACOutput object
        variable (string) : a variable in the Run result.
    """
    first_var = outputs[0].get_summary(variable)
    name = first_var.name
    unit = first_var.unit

    values = helper.lmap(lambda output: output.get_summary(variable).df[name], outputs)
    names = helper.lmap(lambda output: output.project_name, outputs)

    results = [output.get_summary() for output in outputs]
    data = helper.concat(results)[['project_name', name]]

    sns.boxplot(x='project_name', y=name, data=data, **kwargs)
    sns.swarmplot(x='project_name', y=name, data=data, color="0.25")

    # plt.boxplot(values, notch=True, sym='+')
    # plt.xticks(range(1, len(outputs) + 1), names)
    plt.ylabel(f"{name} ({unit})")
    plt.xlabel("Project Names")


def barplot(outputs, variable, **kwargs):
    index = helper.find(variable, outputs[0]['Run'].names)
    name = outputs[0]['Run'].names[index]
    unit = outputs[0]['Run'].units[name]

    results = [output['Run'] for output in outputs]
    data = helper.concat(results)[['project_name', name]]

    sns.barplot(x="project_name", y=name, data=data, capsize=0.1, **kwargs)
    plt.ylabel(f"{name} ({unit})")
    plt.xlabel("Project Name")


def summarize(outputs, variable):
    summaries = lmap(lambda output: output.get_summary(variable), outputs)
    project_names = lmap(lambda output: output.project_name, outputs)
    means = lmap(lambda summary: np.mean(summary.values()), summaries)
    stds = lmap(lambda summary: np.std(summary.values()), summaries)
    df = pd.DataFrame([means, stds], index=['Mean', 'Std'], columns=project_names)
    name = summaries[0].name
    unit = summaries[0].unit
    return Summary(df, name, unit)