"""This script is used to plot the results of the experiments."""
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from einops import repeat
from matplotlib.patches import Rectangle
from scipy import stats

from src.analysis.utils import gather_data_from_anomaly_scores


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


def plot_metric_regression(
        experiment_dir: str,
        attr_key: str,
        metrics: List[str],
        subgroup_names: List[str],  # Names of the subgroups (others than the metrics subgroups)
        xlabel: str,
        ylabel: str,
        title: str,
        plt_name: str):
    """Plots the given metrics in a single plot with varying attr_key values."""
    # Collect data from all runs
    data, attr_key_values = gather_data_from_anomaly_scores(
        experiment_dir, metrics, subgroup_names, attr_key)

    # Check linear interpolation error
    check_linear_regression_error(data, metrics)

    # Plot bar plot
    # plot_metric_regression_bar(
    #     data=data,
    #     attr_key_values=attr_key_values,
    #     metrics=metrics,
    #     xlabel=xlabel,
    #     ylabel=ylabel,
    #     title="",
    #     plt_name=plt_name + '_bar.pdf'
    # )

    # Plot box-whisker plot
    plot_metric_regression_box_whisker(
        data=data,
        attr_key_values=attr_key_values,
        metrics=metrics,
        xlabel=xlabel,
        ylabel=ylabel,
        title="",
        plt_name=plt_name + '_box.pdf'
    )

    # # Plot scatter plot
    # plot_metric_regression_scatter(
    #     data=data,
    #     attr_key_values=attr_key_values,
    #     metrics=metrics,
    #     xlabel=xlabel,
    #     ylabel=ylabel,
    #     # title=title,
    #     title="",
    #     # plt_name=plt_name + '_scatter.png'
    #     plt_name=plt_name + '_scatter.pdf'
    # )


def plot_metric_regression_bar(
        data: Dict[str, np.ndarray],
        attr_key_values: np.ndarray,
        metrics: List[str],
        xlabel: str,
        ylabel: str,
        title: str,
        plt_name: str):

    # Flatten dict and turn into dataframe for seaborn
    df = []
    for subgroup in metrics:
        vals = data[subgroup]  # shape: (num_attr_key_values, num_seeds)
        for i, val in enumerate(vals):
            for seed in val:
                df.append({
                    'subgroup': subgroup.split('/')[-1].split('_')[0],
                    ylabel: seed,
                    xlabel: attr_key_values[i]
                })
    df = pd.DataFrame(df)

    # Create plot
    _, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.barplot(data=df, x=xlabel, y=ylabel, hue='subgroup', ax=ax)
    # Get colors of bar plot
    bars = [r for r in ax.get_children() if isinstance(r, Rectangle)]
    colors = [c.get_facecolor() for c in bars[:-1]]
    seen = set()
    colors = [list(x) for x in colors if not (x in seen or seen.add(x))]

    # Plot regression lines
    left, right = plt.xlim()
    reg_inds = np.linspace(left, right, num=len(attr_key_values))
    for subgroup, color in zip(metrics, colors):
        vals = data[subgroup]
        mean = vals.mean(axis=1)
        reg_coefs = np.polyfit(reg_inds, mean, 1)  # linear regression
        reg_ind = np.array([left, right])
        reg_vals = np.polyval(reg_coefs, reg_ind)
        color[-1] = 0.5  # set alpha to 0.5
        plt.plot(reg_ind, reg_vals, color=color)
        _, _, _, p_value, _ = stats.linregress(reg_inds, mean)
        alpha = 0.01
        is_significant = p_value < alpha
        print(f"The regression line of {subgroup} is significantly different from 0: {is_significant}. p={p_value}")

    # Set y limit
    mini = df[ylabel].min()
    maxi = df[ylabel].max()
    ylim_min = max(0, mini - 0.1 * (maxi - mini))
    ylim_max = min(1, maxi + 0.1 * (maxi - mini))
    plt.ylim(ylim_min, ylim_max)

    # Plot settings
    plt.title(title, wrap=True)
    plt.savefig(os.path.join(THIS_DIR, plt_name), bbox_inches='tight', format='pdf')
    plt.close()


def plot_metric_regression_box_whisker(
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

    # Flatten dict and turn into dataframe for seaborn
    df = []
    for subgroup in metrics:
        vals = data[subgroup]  # shape: (num_attr_key_values, num_seeds)
        for i, val in enumerate(vals):
            for seed in val:
                df.append({
                    'subgroup': subgroup.split('/')[-1].split('_')[0],
                    ylabel: seed,
                    xlabel: attr_key_values[i]
                })
    df = pd.DataFrame(df)

    # Create plot
    _, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.boxplot(data=df, x=xlabel, y=ylabel, hue='subgroup', ax=ax)
    # Get colors of box plot
    colors = [list(h.get_facecolor()) for h in ax.get_legend_handles_labels()[0]]

    # Plot regression lines
    left, right = plt.xlim()
    reg_inds = np.linspace(left, right, num=len(attr_key_values))
    for subgroup, color in zip(metrics, colors):
        vals = data[subgroup]
        mean = vals.mean(axis=1)
        reg_coefs = np.polyfit(reg_inds, mean, 1)  # linear regression
        reg_ind = np.array([left, right])
        reg_vals = np.polyval(reg_coefs, reg_ind)
        color[-1] = 0.5  # set alpha to 0.5
        plt.plot(reg_ind, reg_vals, color=color)
        _, _, _, p_value, _ = stats.linregress(reg_inds, mean)
        alpha = 0.01
        is_significant = p_value < alpha
        print(f"The regression line of {subgroup} is significantly different from 0: {is_significant}. p={p_value}")

    # Plot settings
    plt.title(title, wrap=True)
    plt.savefig(os.path.join(THIS_DIR, plt_name), bbox_inches='tight', format='pdf')
    plt.close()


def plot_metric_regression_scatter(
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
    mini = math.inf
    maxi = -math.inf

    for i, metric in enumerate(metrics):
        # Set colors
        if i == 0:
            color = 'tab:blue'
        elif i == 1:
            color = 'tab:orange'
        else:
            color = 'tab:green'

        ys = data[metric]
        # Repeat xs
        xs = repeat(ind + i * width, 'i -> i j', j=ys.shape[1])
        # Perturb xs
        xs = xs + np.random.uniform(-width / 4, width / 4, size=xs.shape)
        # Plot with color gradient
        for j, (xs_, ys_) in enumerate(zip(xs, ys)):
            plt.scatter(xs_, ys_, alpha=0.5, c=color)

        mean = ys.mean(axis=1)
        std = ys.std(axis=1)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        min_val = np.min(lower)
        max_val = np.max(upper)
        if mini > min_val:
            mini = min_val
        if maxi < max_val:
            maxi = max_val

        # Plot regression lines
        left, right = plt.xlim()
        y_mean = ys.mean(axis=1)
        reg_coefs = np.polyfit(centers, y_mean, 1)  # linear regression
        reg_ind = np.array([left, right])
        reg_vals = np.polyval(reg_coefs, reg_ind)
        plt.plot(reg_ind, reg_vals, color='black')

        # Plot standard deviation of regression lines
        # all_reg_coefs = []
        # for seed in range(data[metric].shape[1]):
        #     reg_coefs = np.polyfit(centers, ys[:, seed], 1)
        #     all_reg_coefs.append(reg_coefs)
        # all_reg_coefs = np.stack(all_reg_coefs, axis=0)
        # reg_coefs_std = all_reg_coefs.std(axis=0)
        # reg_coefs_mean = all_reg_coefs.mean(axis=0)
        # lower_reg_vals = np.polyval(reg_coefs_mean - reg_coefs_std, reg_ind)
        # upper_reg_vals = np.polyval(reg_coefs_mean + reg_coefs_std, reg_ind)
        # plt.fill_between(reg_ind, lower_reg_vals, upper_reg_vals, color='black', alpha=0.2)
        # Significance test of regression line
        _, _, _, p_value, _ = stats.linregress(centers, y_mean)
        alpha = 0.05
        is_significant = p_value < alpha
        print(f"The regression line of {metric} is significantly different from 0: {is_significant}")

    # x label and x ticks
    plt.xlabel(xlabel)
    plt.xticks(centers, attr_key_values.round(2))
    plt.xlim(left, right)

    plt.ylabel(ylabel)

    # All axes should have the same y limits
    ylim_min = max(0, mini - 0.1 * (maxi - mini))
    ylim_max = min(1, maxi + 0.1 * (maxi - mini))
    plt.ylim(ylim_min, ylim_max)

    # Save plot
    print(f"Saving plot to {plt_name}")
    plt.savefig(os.path.join(THIS_DIR, plt_name))
    plt.close()
    exit(1)


def check_linear_regression_error(data: Dict[str, np.ndarray], metrics: List[str]):
    """Checks how well the intermediate results can be predicted using only the
    extreme values (MAE of linear interpolation between first and last value).
    """
    for metric in metrics:
        vals = data[metric]
        first = vals[0]
        last = vals[-1]
        # Linear interpolation
        pred = np.linspace(first, last, num=vals.shape[0])
        mae = np.abs(pred - vals)[1:-1].mean()
        print(f"MAE of linear interpolation of {metric}: {mae}")


def plot_metric_single(
        experiment_dir: str,
        metrics: List[str],
        subgroup_names: List[str],  # Names of the subgroups (others than the metrics subgroups)
        ylabel: str,
        title: str,
        plt_name: str):
    """Plots the given metrics in a single plot."""
    data, _ = gather_data_from_anomaly_scores(
        experiment_dir, metrics, subgroup_names, attr_key=None)

    # Flatten dict and turn into dataframe for seaborn
    df = []
    for subgroup in metrics:
        vals = data[subgroup]  # shape: (num_attr_key_values, num_seeds)
        for i, val in enumerate(vals):
            for seed in val:
                df.append({
                    'subgroup': "_".join(subgroup.split("/")[-1].split("_")[:-1]),
                    ylabel: seed,
                })
    df = pd.DataFrame(df)

    # Plot bar plots
    _, ax = plt.subplots(figsize=(6.4 * (3 / 4), 4.8))
    sns.barplot(data=df, x='subgroup', y=ylabel, ax=ax)

    # Set ylimit
    plt.ylim(0.58, 0.78)

    # Save plot
    plt.tight_layout()
    full_name = os.path.join(THIS_DIR, plt_name + ".pdf")
    print(f"Saving plot to {full_name}")
    plt.savefig(full_name, bbox_inches='tight', format='pdf')
    plt.close()


if __name__ == '__main__':
    """ FAE MIMIC-CXR sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_sex')
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_anomaly_scores"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_fpr@0.95tpr"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_AUROC", "test/female_AUROC"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_AUROC"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_subgroupAUROC"
    )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/fpr@0.95"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_fpr@0.95tpr_total"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/AUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_AUROC_total"
    # )
    """ FAE MIMIC-CXR age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_age')
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_anomaly_scores"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_fpr@0.95tpr"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_AUROC", "test/young_AUROC"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_AUROC"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_subgroupAUROC"
    )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/fpr@0.95"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_fpr@0.95tpr_total"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/AUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_AUROC_total"
    # )
    """ FAE MIMIC-CXR race """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_race')
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/white_anomaly_score", "test/black_anomaly_score"],
        subgroup_names=["test/white", "test/black"],
        attr_key='white_percent',
        xlabel="percentage of white subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on MIMIC-CXR for different proportions of white patients in training",
        plt_name="fae_mimic-cxr_race_anomaly_scores"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/white_fpr@0.95", "test/black_fpr@0.95"],
        subgroup_names=["test/white", "test/black"],
        attr_key='white_percent',
        xlabel="percentage of white subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of white patients in training",
        plt_name="fae_mimic-cxr_race_fpr@0.95tpr"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/white_AUROC", "test/black_AUROC"],
        subgroup_names=["test/white", "test/black"],
        attr_key='white_percent',
        xlabel="percentage of white subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on MIMIC-CXR for different proportions of white patients in training",
        plt_name="fae_mimic-cxr_race_AUROC"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/white_subgroupAUROC", "test/black_subgroupAUROC"],
        subgroup_names=["test/white", "test/black"],
        attr_key='white_percent',
        xlabel="percentage of white subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on MIMIC-CXR for different proportions of white patients in training",
        plt_name="fae_mimic-cxr_race_subgroupAUROC"
    )

    """ FAE CXR14 sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_cxr14_sex')
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_anomaly_scores"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_fpr@0.95tpr"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_AUROC", "test/female_AUROC"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_AUROC"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_subgroupAUROC"
    )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/fpr@0.95"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_fpr@0.95tpr_total"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/AUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_AUROC_total"
    # )
    """ FAE CXR14 age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_cxr14_age')
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_anomaly_scores"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_fpr@0.95tpr"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_AUROC", "test/young_AUROC"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_AUROC"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_subgroupAUROC"
    )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/fpr@0.95"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_fpr@0.95tpr_total"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/AUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_AUROC_total"
    # )

    """ FAE CheXpert sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_chexpert_sex')
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on CheXpert for different proportions of male patients in training",
        plt_name="fae_chexpert_sex_anomaly_scores"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on CheXpert for different proportions of male patients in training",
        plt_name="fae_chexpert_sex_fpr@0.95tpr"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_AUROC", "test/female_AUROC"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on CheXpert for different proportions of male patients in training",
        plt_name="fae_chexpert_sex_AUROC"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
        subgroup_names=["test/male", "test/female"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on CheXpert for different proportions of male patients in training",
        plt_name="fae_chexpert_sex_subgroupAUROC"
    )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/fpr@0.95"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_fpr@0.95tpr_total"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/AUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_AUROC_total"
    # )
    """ FAE CheXpert age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_chexpert_age')
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on CheXpert for different proportions of old patients in training",
        plt_name="fae_chexpert_age_anomaly_scores"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on CheXpert for different proportions of old patients in training",
        plt_name="fae_chexpert_age_fpr@0.95tpr"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_AUROC", "test/young_AUROC"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on CheXpert for different proportions of old patients in training",
        plt_name="fae_chexpert_age_AUROC"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
        subgroup_names=["test/old", "test/young"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on CheXpert for different proportions of old patients in training",
        plt_name="fae_chexpert_age_subgroupAUROC"
    )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/fpr@0.95"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_fpr@0.95tpr_total"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/AUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_AUROC_total"
    # )

    """ FAE CheXpert race """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_chexpert_race')
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/white_anomaly_score", "test/black_anomaly_score"],
        subgroup_names=["test/white", "test/black"],
        attr_key='white_percent',
        xlabel="percentage of white subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on CheXpert for different proportions of white patients in training",
        plt_name="fae_chexpert_race_anomaly_scores"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/white_fpr@0.95", "test/black_fpr@0.95"],
        subgroup_names=["test/white", "test/black"],
        attr_key='white_percent',
        xlabel="percentage of white subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on CheXpert for different proportions of white patients in training",
        plt_name="fae_chexpert_race_fpr@0.95tpr"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/white_AUROC", "test/black_AUROC"],
        subgroup_names=["test/white", "test/black"],
        attr_key='white_percent',
        xlabel="percentage of white subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on CheXpert for different proportions of white patients in training",
        plt_name="fae_chexpert_race_AUROC"
    )
    plot_metric_regression(
        experiment_dir=experiment_dir,
        metrics=["test/white_subgroupAUROC", "test/black_subgroupAUROC"],
        subgroup_names=["test/white", "test/black"],
        attr_key='white_percent',
        xlabel="percentage of white subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on CheXpert for different proportions of white patients in training",
        plt_name="fae_chexpert_race_subgroupAUROC"
    )

    """ MIMIC-CXR - Intersectional study (age and sex) """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_intersectional_age_sex')
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     ylabel="subgroupAUROC",
    #     title="",
    #     plt_name="fae_mimic-cxr_intersectional_sex_subgroupAUROC"
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     ylabel="subgroupAUROC",
    #     title="",
    #     plt_name="fae_mimic-cxr_intersectional_age_subgroupAUROC"
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_old_subgroupAUROC", "test/male_young_subgroupAUROC"],
    #     subgroup_names=["test/male_young", "test/male_old", "test/female_young", "test/female_old"],
    #     ylabel="subgroupAUROC",
    #     title="",
    #     plt_name="fae_mimic-cxr_intersectional_age_after_male_subgroupAUROC"
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/female_old_subgroupAUROC", "test/female_young_subgroupAUROC"],
    #     subgroup_names=["test/male_young", "test/male_old", "test/female_young", "test/female_old"],
    #     ylabel="subgroupAUROC",
    #     title="",
    #     plt_name="fae_mimic-cxr_intersectional_age_after_female_subgroupAUROC"
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_young_subgroupAUROC", "test/female_young_subgroupAUROC"],
    #     subgroup_names=["test/male_young", "test/male_old", "test/female_young", "test/female_old"],
    #     ylabel="subgroupAUROC",
    #     title="",
    #     plt_name="fae_mimic-cxr_intersectional_sex_after_young_subgroupAUROC"
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_old_subgroupAUROC", "test/female_old_subgroupAUROC"],
    #     subgroup_names=["test/male_young", "test/male_old", "test/female_young", "test/female_old"],
    #     ylabel="subgroupAUROC",
    #     title="",
    #     plt_name="fae_mimic-cxr_intersectional_sex_after_old_subgroupAUROC"
    # )

    """ MIMIC-CXR - model size (balanced sex) """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_model_size')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="model size",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on MIMIC-CXR for different model sizes",
    #     plt_name="fae_mimic-cxr_model_size"
    # )

    """ MIMIC-CXR - dataset size (balanced sex) """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_dataset_size')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="number of training samples",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on MIMIC-CXR for different dataset sizes",
    #     plt_name="fae_mimic-cxr_dataset_size"
    # )
