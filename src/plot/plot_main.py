"""
This script is used to plot the results of the experiments.
The results of each experiment are in a directory that looks like the following:
experiment_dir
    run_1
        seed_1
            test_results.csv
        seed_2
            test_results.csv
        ...
    run_2
        seed_1
            test_results.csv
        seed_2
            test_results.csv
        ...
    ...
    run_n
        seed_1
            test_results.csv
        seed_2
            test_results.csv
        ...

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


def gather_data_bootstrap(experiment_dir: str):
    """Gather the data of the first random seed of each run and bootstrap the results"""
    run_dirs = [os.path.join(experiment_dir, run_dir) for run_dir in os.listdir(experiment_dir)]
    dfs = []
    for run_dir in run_dirs:
        print(f"Getting data from {run_dir}")
        run_dir = get_lowest_seed(run_dir)
        results_file = os.path.join(run_dir, 'test_results.csv')
        if not os.path.exists(results_file):
            warnings.warn(f"Results file {results_file} does not exist")
            continue
        df = pd.read_csv(results_file)
        # Average all columns that have multiple non NaN values
        for col in df.columns:
            if df[col].count() > 1:
                df[f'{col}_lower'] = df[col].quantile(0.025)
                df[f'{col}_upper'] = df[col].quantile(0.975)
                df[col] = df[col].mean()
        df = df.iloc[:1]
        df['run_dir'] = run_dir
        dfs.append(df)
    return pd.concat(dfs)


def gather_data_seeds(experiment_dir: str):
    """Gather the data of multiple random seeds"""
    run_dirs = [os.path.join(experiment_dir, run_dir) for run_dir in os.listdir(experiment_dir)]
    run_dfs = []
    print("Getting data from the following run dirs:")
    for run_dir in run_dirs:
        seed_dirs = [os.path.join(run_dir, seed_dir) for seed_dir in os.listdir(run_dir)]
        seed_dfs = []
        for seed_dir in seed_dirs:
            print(f"Getting data from {seed_dir}")
            results_file = os.path.join(seed_dir, 'test_results.csv')
            df = pd.read_csv(results_file)
            # Average all columns that have multiple non NaN values
            for col in df.columns:
                if df[col].count() > 1:
                    df[col] = df[col].mean()
            df = df.iloc[:1]
            seed_dfs.append(df)
        df = pd.concat(seed_dfs)
        df = avg_numeric_in_df(df)
        run_dfs.append(df)
    return pd.concat(run_dfs)


def get_lowest_seed(directory: str):
    """Get the subdirectory with the lowest seed number"""
    seed_dirs = [os.path.join(directory, seed_dir) for seed_dir in os.listdir(directory)]
    seed_dirs = [seed_dir for seed_dir in seed_dirs if os.path.isdir(seed_dir)]
    seed_nums = [int(seed_dir.split('_')[-1]) for seed_dir in seed_dirs]
    return seed_dirs[np.argmin(seed_nums)]


def avg_numeric_in_df(df: pd.DataFrame):
    """Average all columns that have numeric values"""
    def is_numeric(col):
        return np.issubdtype(col.dtype, np.number)
    for col in df.columns:
        if is_numeric(df[col]):
            std = df[col].std()
            df[col] = df[col].mean()
            df[f'{col}_lower'] = df[col] - 1.96 * std
            df[f'{col}_upper'] = df[col] + 1.96 * std
    df = df.iloc[:1]
    return df


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
    # data = gather_data_bootstrap(experiment_dir)
    data = gather_data_seeds(experiment_dir)

    # Sort data by protected attribute
    if protected_attr == 'age':
        data['train_age'] = pd.Categorical(data['train_age'], ["young", "avg", "old"])
    data.sort_values(attr_key, inplace=True)

    # Plot
    width = 0.25
    ind = np.arange(len(data[attr_key].values))
    centers = ind + (len(metrics) // 2) * width - width / 2
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
    # FAE rsna age
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_age')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_cDC", "test/lungOpacity_old_cDC"],
        protected_attr='age',
        attr_key='train_age',
        xlabel="age of subjects in training set",
        ylabel="cDC",
        title="FAE cDC on RSNA for training with young or old patients",
        plt_name="fae_rsna_age_cDC.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_aDSC", "test/lungOpacity_old_aDSC"],
        protected_attr='age',
        attr_key='train_age',
        xlabel="age of subjects in training set",
        ylabel="aDSC",
        title="FAE aDSC on RSNA for training with young or old patients",
        plt_name="fae_rsna_age_aDSC.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_ap", "test/lungOpacity_old_ap"],
        protected_attr='age',
        attr_key='train_age',
        xlabel="age of subjects in training set",
        ylabel="ap",
        title="FAE ap on RSNA for training with young or old patients",
        plt_name="fae_rsna_age_ap.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_tpr@0.05", "test/lungOpacity_old_tpr@0.05"],
        protected_attr='age',
        attr_key='train_age',
        xlabel="age of subjects in training set",
        ylabel="tpr@0.05fpr",
        title="FAE tpr@0.05fpr on RSNA for training with young or old patients",
        plt_name="fae_rsna_age_tpr@005fpr.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_fpr@0.95", "test/lungOpacity_old_fpr@0.95"],
        protected_attr='age',
        attr_key='train_age',
        xlabel="age of subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on RSNA for training with young or old patients",
        plt_name="fae_rsna_age_fpr@095tpr.png"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_anomaly_score", "test/lungOpacity_old_anomaly_score"],
        protected_attr='age',
        attr_key='train_age',
        xlabel="age of subjects in training set",
        ylabel="anomaly_scores",
        title="FAE anomaly_scores on RSNA for training with young or old patients",
        plt_name="fae_rsna_age_anomaly_scores.png"
    )
