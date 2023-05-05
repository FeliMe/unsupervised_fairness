"""
This script is used to plot the results of the experiments.
The results of each experiment are in a directory that looks like the following:
experiment_dir
    run_1
        test_results.csv
    run_2
        test_results.csv
    ...
    run_n
        test_results.csv

Each results.csv file contains the results of a single run of the experiment.
"""
import math
import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def gather_data(experiment_dir: str):
    run_dirs = [os.path.join(experiment_dir, run_dir) for run_dir in os.listdir(experiment_dir)]
    dfs = []
    for run_dir in run_dirs:
        results_file = os.path.join(run_dir, 'test_results.csv')
        if not os.path.exists(results_file):
            warnings.warn(f"Results file {results_file} does not exist")
            continue
        df = pd.read_csv(results_file)
        df['run_dir'] = run_dir
        dfs.append(df)
    return pd.concat(dfs)


def plot_metric(
        experiment_dir: str,
        protected_attr: str,
        attr_key: str,
        metrics: List[str],
        xlabel: str,
        ylabel: str,
        title: str,
        plt_name: str):
    # Collect data from all runs
    data = gather_data(experiment_dir)

    # Sort data by protected attribute
    if protected_attr == 'age':
        data['train_age'] = pd.Categorical(data['train_age'], ["young", "avg", "old"])
    data.sort_values(attr_key, inplace=True)

    # Plot
    width = 0.25
    ind = np.arange(len(data[attr_key].values))
    centers = ind + (len(metrics) // 2) * width
    bars = []
    mini = math.inf
    maxi = -math.inf
    for i, metric in enumerate(metrics):
        vals = data[metric].values
        if metric + "_lower" not in data.columns or metric + "_upper" not in data.columns:
            yerr = None
            min_val = np.min(vals)
            max_val = np.max(vals)
        else:
            lower = data[metric + "_lower"].values
            upper = data[metric + "_upper"].values
            yerr = np.stack([vals - lower, upper - vals], axis=0)
            min_val = np.min(lower)
            max_val = np.max(upper)
        bars.append(plt.bar(ind + i * width, vals, width=width, yerr=yerr))
        if mini > min_val:
            mini = min_val
        if maxi < max_val:
            maxi = max_val
    left, right = plt.xlim()
    for i, metric in enumerate(metrics):
        reg_vals = np.polyfit(centers, data[metric], 1)  # linear regression
        reg_ind = np.array([left, right])
        color = list(bars[i][0].get_facecolor())  # get color of corresponding bar
        color[-1] = 0.5  # set alpha to 0.5
        plt.plot(reg_ind, np.polyval(reg_vals, reg_ind), color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, wrap=True)
    plt.xticks(centers, data[attr_key].values)
    plt.xlim(left, right)
    plt.legend(bars, metrics)
    ylim_min = max(0, mini - 0.1 * (maxi - mini))
    plt.ylim(ylim_min, plt.ylim()[1])
    plt.savefig(os.path.join(THIS_DIR, plt_name))
    plt.close()


if __name__ == '__main__':
    """ FAE RSNA """
    # FAE rsna sex
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_sex')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_cDC", "test/lungOpacity_female_cDC"],
        protected_attr='sex',
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="cDC",
        title="FAE cDC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_cDC.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_aDSC", "test/lungOpacity_female_aDSC"],
        protected_attr='sex',
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="aDSC",
        title="FAE aDSC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_aDSC.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_ap", "test/lungOpacity_female_ap"],
        protected_attr='sex',
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="ap",
        title="FAE ap on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_ap.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_tpr@0.05", "test/lungOpacity_female_tpr@0.05"],
        protected_attr='sex',
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="tpr@0.05fpr",
        title="FAE tpr@0.05 on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_tpr@0.05.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_fpr@0.95", "test/lungOpacity_female_fpr@0.95"],
        protected_attr='sex',
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95",
        title="FAE fpr@0.95tpr on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_fpr@0.95.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_anomaly_score", "test/lungOpacity_female_anomaly_score"],
        protected_attr='sex',
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly_score",
        title="FAE anomaly_score on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_anomaly_score.png"
    )
