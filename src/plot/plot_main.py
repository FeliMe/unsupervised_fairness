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
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from einops import repeat
from scipy import stats


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def gather_data_bootstrap(experiment_dir: str):
    """Gather the data of the first random seed of each run and bootstrap the results"""
    run_dirs = [os.path.join(experiment_dir, run_dir) for run_dir in os.listdir(experiment_dir)]
    dfs = []
    for run_dir in run_dirs:
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


def gather_data_seeds(experiment_dir: str, attr_key: str, metric_names: List[str]):
    """Gather the data of multiple random seeds
    For every metric, it returns a matrix of shape (num_runs, num_seeds)
    """
    run_dirs = [os.path.join(experiment_dir, run_dir) for run_dir in os.listdir(experiment_dir)]
    run_dfs = []
    attr_key_values = []
    for run_dir in run_dirs:
        seed_dirs = [os.path.join(run_dir, seed_dir) for seed_dir in os.listdir(run_dir)]
        seed_dfs = []
        for seed_dir in seed_dirs:
            results_file = os.path.join(seed_dir, 'test_results.csv')
            df = pd.read_csv(results_file)
            # Average all columns that have multiple non NaN values
            df = avg_numeric_in_df(df)
            seed_dfs.append(df)
        df = pd.concat(seed_dfs)
        run_dfs.append(df)
        attr_key_values.append(df[attr_key].values[0])
    # Sort by protected attribute
    run_dfs = [df for _, df in sorted(zip(attr_key_values, run_dfs))]
    attr_key_values = np.sort(np.array(attr_key_values))
    # Build results dictionary
    results = {metric: [] for metric in metric_names}
    for df in run_dfs:
        for metric in metric_names:
            results[metric].append(df[metric].values)
    results = {metric: np.stack(vals, axis=0) for metric, vals in results.items()}
    return results, attr_key_values


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
            df[col] = df[col].mean()
    df = df.iloc[:1]
    return df


def plot_metric(
        experiment_dir: str,
        attr_key: str,
        metrics: List[str],
        xlabel: str,
        ylabel: str,
        title: str,
        plt_name: str):
    """Plots the given metrics as different plots"""
    # Collect data from all runs
    # data = gather_data_bootstrap(experiment_dir)
    data, attr_key_values = gather_data_seeds(experiment_dir, attr_key, metrics)

    # Plot scatter plot
    plot_metric_scatter(
        data=data,
        attr_key_values=attr_key_values,
        metrics=metrics,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        plt_name=plt_name + '_scatter.png'
    )

    # Plot bar plot
    plot_metric_bar(
        data=data,
        attr_key_values=attr_key_values,
        metrics=metrics,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        plt_name=plt_name + '_bar.png'
    )

    # Plot box-whisker plot
    plot_metric_box_whisker(
        data=data,
        attr_key_values=attr_key_values,
        metrics=metrics,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        plt_name=plt_name + '_box.png'
    )


def plot_metric_scatter(
        data: Dict[str, np.ndarray],
        attr_key_values: np.ndarray,
        metrics: List[str],
        xlabel: str,
        ylabel: str,
        title: str,
        plt_name: str):
    """
    Plots the given metrics as a scatter plot. Each metric is plotted in a
    separate subplot. The positions on the x-axis are slightly perturbed.
    """
    # Plot preparation
    width = 0.25
    ind = np.arange(len(data[metrics[0]]))
    centers = ind + (len(metrics) // 2) * width - width / 2
    f = plt.figure(figsize=(6.4 * len(metrics), 4.8))
    f.suptitle(title, wrap=True)

    for i, metric in enumerate(metrics):
        ax = f.add_subplot(1, len(metrics), i + 1)
        ys = data[metric]
        # Repeat xs
        xs = repeat(centers, 'i -> i j', j=ys.shape[1])
        # Perturb xs
        xs = xs + np.random.uniform(-width / 2, width / 2, size=xs.shape)
        # Plot with color gradient
        for j, (xs_, ys_) in enumerate(zip(xs, ys)):
            c = mpl.cm.viridis(j / len(xs))
            ax.scatter(xs_, ys_, alpha=0.5, c=c)

        # Plot regression lines
        left, right = ax.get_xlim()
        y_mean = ys.mean(axis=1)
        reg_coefs = np.polyfit(centers, y_mean, 1)  # linear regression
        reg_ind = np.array([left, right])
        reg_vals = np.polyval(reg_coefs, reg_ind)
        ax.plot(reg_ind, reg_vals, color='black')

        # Plot standard deviation of regression lines
        all_reg_coefs = []
        for seed in range(data[metric].shape[1]):
            reg_coefs = np.polyfit(centers, ys[:, seed], 1)
            all_reg_coefs.append(reg_coefs)
        all_reg_coefs = np.stack(all_reg_coefs, axis=0)
        reg_coefs_std = all_reg_coefs.std(axis=0)
        reg_coefs_mean = all_reg_coefs.mean(axis=0)
        lower_reg_vals = np.polyval(reg_coefs_mean - reg_coefs_std, reg_ind)
        upper_reg_vals = np.polyval(reg_coefs_mean + reg_coefs_std, reg_ind)
        plt.fill_between(reg_ind, lower_reg_vals, upper_reg_vals, color='black', alpha=0.2)
        # Significance test of regression line
        _, _, _, p_value, _ = stats.linregress(centers, y_mean)
        alpha = 0.05
        is_significant = p_value < alpha
        print(f"The regression line of {metric} is significantly different from 0: {is_significant}")

        # Plot settings
        ax.set_title(metric)

        # x label and x ticks
        ax.set_xlabel(xlabel)
        ax.set_xticks(centers, attr_key_values.round(2))
        ax.set_xlim(left, right)

        # y label and y ticks
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])

    # All axes should have the same y limits
    ylim_min = min([ax.get_ylim()[0] for ax in f.axes])
    ylim_max = max([ax.get_ylim()[1] for ax in f.axes])
    for ax in f.axes:
        ax.set_ylim(ylim_min, ylim_max)

    # Save plot
    print(f"Saving plot to {plt_name}")
    plt.savefig(os.path.join(THIS_DIR, plt_name))
    plt.close()


def plot_metric_bar(
        data: Dict[str, np.ndarray],
        attr_key_values: np.ndarray,
        metrics: List[str],
        xlabel: str,
        ylabel: str,
        title: str,
        plt_name: str):
    # Prepare plot
    width = 0.25
    ind = np.arange(len(data[metrics[0]]))
    centers = ind + (len(metrics) // 2) * width - width / 2
    bars = []
    mini = math.inf
    maxi = -math.inf

    # Plot bar plots
    for i, metric in enumerate(metrics):
        mean = data[metric].mean(axis=1)
        std = data[metric].std(axis=1)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        yerr = np.stack([mean - lower, upper - mean], axis=0)
        min_val = np.min(lower)
        max_val = np.max(upper)
        bars.append(plt.bar(ind + i * width, mean, width=width, yerr=yerr,
                            ecolor='darkgray'))
        if mini > min_val:
            mini = min_val
        if maxi < max_val:
            maxi = max_val
    left, right = plt.xlim()

    # Plot regression lines
    for i, metric in enumerate(metrics):
        vals = data[metric]
        mean = vals.mean(axis=1)
        reg_coefs = np.polyfit(centers, mean, 1)  # linear regression
        reg_ind = np.array([left, right])
        reg_vals = np.polyval(reg_coefs, reg_ind)
        color = list(bars[i][0].get_facecolor())  # get color of corresponding bar
        color[-1] = 0.5  # set alpha to 0.5
        plt.plot(reg_ind, reg_vals, color=color)
        # Plot standard deviation of regression lines
        all_reg_coefs = []
        for seed in range(data[metric].shape[1]):
            reg_coefs = np.polyfit(centers, vals[:, seed], 1)
            all_reg_coefs.append(reg_coefs)
        all_reg_coefs = np.stack(all_reg_coefs, axis=0)
        reg_coefs_std = all_reg_coefs.std(axis=0)
        reg_coefs_mean = all_reg_coefs.mean(axis=0)
        lower_reg_vals = np.polyval(reg_coefs_mean - reg_coefs_std, reg_ind)
        upper_reg_vals = np.polyval(reg_coefs_mean + reg_coefs_std, reg_ind)
        plt.fill_between(reg_ind, lower_reg_vals, upper_reg_vals, color=color, alpha=0.2)
        # Significance test of regression line
        _, _, _, p_value, _ = stats.linregress(centers, mean)
        alpha = 0.05
        is_significant = p_value < alpha
        print(f"The regression line of {metric} is significantly different from 0: {is_significant}")

    # Plot settings
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, wrap=True)
    plt.xticks(centers, attr_key_values.round(2))
    plt.xlim(left, right)
    plt.legend(bars, metrics)
    ylim_min = max(0, mini - 0.1 * (maxi - mini))
    plt.ylim(ylim_min, plt.ylim()[1])
    print(f"Saving plot to {plt_name}")
    plt.savefig(os.path.join(THIS_DIR, plt_name))
    plt.close()


def plot_metric_box_whisker(
        data: Dict[str, np.ndarray],
        attr_key_values: np.ndarray,
        metrics: List[str],
        xlabel: str,
        ylabel: str,
        title: str,
        plt_name: str):
    """
    Plots the given metrics as a box and whisker plot.
    """
    # Prepare plot
    width = 0.25
    ind = np.arange(len(data[metrics[0]]))
    centers = ind + (len(metrics) // 2) * width - width / 2

    # Plot bar plots
    for i, metric in enumerate(metrics):
        ys = data[metric].T
        # Repeat xs
        positions = ind + i * width
        # Plot with diamond markers
        bplot = plt.boxplot(ys, positions=positions, widths=width, showfliers=False,
                            boxprops={'color': 'gray'},
                            whiskerprops={'color': 'gray'},
                            capprops={'color': 'gray'},
                            medianprops={'color': 'darkgray'},
                            patch_artist=True)

        # Set colors
        color = 'tab:blue' if i == 0 else 'tab:orange'
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    # Plot settings
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, wrap=True)
    plt.xticks(centers, attr_key_values.round(2))

    # Save plot
    print(f"Saving plot to {plt_name}")
    plt.savefig(os.path.join(THIS_DIR, plt_name))
    plt.close()


if __name__ == '__main__':
    """ FAE RSNA """
    # FAE rsna sex
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_sex')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_AUROC", "test/lungOpacity_female_AUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_AUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_cDC", "test/lungOpacity_female_cDC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="cDC",
        title="FAE cDC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_cDC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_aDSC", "test/lungOpacity_female_aDSC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="aDSC",
        title="FAE aDSC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_aDSC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_meanPrecision", "test/lungOpacity_female_meanPrecision"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="meanPrecision",
        title="FAE meanPrecision on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_meanPrecision"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_tpr@0.05", "test/lungOpacity_female_tpr@0.05"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="tpr@0.05fpr",
        title="FAE tpr@0.05 on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_tpr@0.05"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_fpr@0.95", "test/lungOpacity_female_fpr@0.95"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95",
        title="FAE fpr@0.95tpr on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_fpr@0.95"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_anomaly_score", "test/lungOpacity_female_anomaly_score"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly_score",
        title="FAE anomaly scores on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_anomaly_score"
    )
    # FAE rsna age
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_age')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_AUROC", "test/lungOpacity_old_AUROC"],
        attr_key='old_percent',
        xlabel="age of subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_AUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_cDC", "test/lungOpacity_old_cDC"],
        attr_key='old_percent',
        xlabel="age of subjects in training set",
        ylabel="cDC",
        title="FAE cDC on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_cDC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_aDSC", "test/lungOpacity_old_aDSC"],
        attr_key='old_percent',
        xlabel="age of subjects in training set",
        ylabel="aDSC",
        title="FAE aDSC on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_aDSC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_meanPrecision", "test/lungOpacity_old_meanPrecision"],
        attr_key='old_percent',
        xlabel="age of subjects in training set",
        ylabel="meanPrecision",
        title="FAE meanPrecision on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_meanPrecision"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_tpr@0.05", "test/lungOpacity_old_tpr@0.05"],
        attr_key='old_percent',
        xlabel="age of subjects in training set",
        ylabel="tpr@0.05fpr",
        title="FAE tpr@0.05fpr on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_tpr@005fpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_fpr@0.95", "test/lungOpacity_old_fpr@0.95"],
        attr_key='old_percent',
        xlabel="age of subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.05tpr on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_fpr@095tpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_young_anomaly_score", "test/lungOpacity_old_anomaly_score"],
        attr_key='old_percent',
        xlabel="age of subjects in training set",
        ylabel="anomaly_scores",
        title="FAE anomaly scores on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_anomaly_scores"
    )
