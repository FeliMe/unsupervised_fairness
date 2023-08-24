"""This script is used to plot the results of the experiments."""
import math
import os
from typing import Dict, List, Optional, Tuple

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

    plt.rcParams["font.size"] = 20

    # Collect data from all runs
    data, attr_key_values = gather_data_from_anomaly_scores(
        experiment_dir, metrics, subgroup_names, attr_key)

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

    # In case of many x-ticks, show only every second one
    xticks = ax.get_xticklabels()
    if len(xticks) > 10:
        for i, xtick in enumerate(xticks):
            if i % 2 == 1:
                xtick.set_visible(False)

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


def plot_metric_single(
        experiment_dir: str,
        metrics: List[str],
        subgroup_names: List[List[str]],  # Names of the subgroups (others than the metrics subgroups). One list per metric
        ylabel: str,
        plt_name: str,
        figsize: Optional[Tuple[int]] = None,
        plot_groups: Optional[Tuple[int]] = None,  # Lengths of the subgroups to plot
        group_colors: Optional[List] = None):
    """Plots the given metrics in a single plot."""

    plt.rcParams["font.size"] = 13

    # Default values
    if plot_groups is None:
        plot_groups = [len(metrics)]
    if group_colors is None:
        group_colors = sns.color_palette(n_colors=len(plot_groups))

    # Assertions
    assert sum(plot_groups) == len(metrics)
    assert len(metrics) == len(subgroup_names)

    # Get group indices and grouped color palette
    plot_group_inds = []
    for i, n in enumerate(plot_groups):
        plot_group_inds += [i] * n

    # Compute metrics one by one
    data = {}
    for subgroup_names_, metric in zip(subgroup_names, metrics):
        metric_data, _ = gather_data_from_anomaly_scores(
            experiment_dir, [metric], subgroup_names_, attr_key=None)
        data = {**data, **metric_data}

    # Flatten dict and turn into dataframe for seaborn
    df = []
    current_plot_group = 1
    for plot_group, subgroup in zip(plot_group_inds, metrics):
        if plot_group == current_plot_group:
            # Add a nan value to create a gap in the plot
            df.append({
                'subgroup': " " * current_plot_group,
                ylabel: np.nan,
                'plot_group': plot_group
            })
            current_plot_group += 1
        # Add the actual data
        vals = data[subgroup]  # shape: (num_attr_key_values, num_seeds)
        subgroup = ", ".join(subgroup.split("/")[-1].split("_")[:-1])
        for val in vals:
            for seed in val:
                df.append({
                    'subgroup': subgroup,
                    ylabel: seed,
                    'plot_group': plot_group
                })
    df = pd.DataFrame(df)

    # Print diffs inside plot groups
    for plot_group in range(len(plot_groups)):
        plot_group_df = df[df['plot_group'] == plot_group]
        # Remove nan values
        plot_group_df = plot_group_df[~plot_group_df[ylabel].isna()]
        subgroups = plot_group_df['subgroup'].unique()
        val_mean = None
        for i, subgroup in enumerate(subgroups):
            cur_mean = plot_group_df[plot_group_df['subgroup'] == subgroup][ylabel].mean()
            if val_mean is None:
                val_mean = cur_mean
            else:
                print(f"Diff between {subgroups[i-1]} and {subgroup}: {val_mean - cur_mean}")

    # Plot bar plots
    _, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x='subgroup', y=ylabel, hue='plot_group', dodge=False,
                palette=group_colors, ax=ax, errwidth=1.0, capsize=0.15)
    plt.legend([], [], frameon=False)

    # Plot settings
    plt.ylim(0.52, 0.78)
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("")

    # Save plot
    plt.tight_layout()
    full_name = os.path.join(THIS_DIR, plt_name + ".pdf")
    print(f"Saving plot to {full_name}")
    plt.savefig(full_name, bbox_inches='tight', format='pdf')
    plt.close()


if __name__ == '__main__':
    """ MIMIC-CXR sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_sex')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="Anomaly scores",
    #     title="FAE anomaly scores on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="FPR@0.95TPR",
    #     title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_AUROC", "test/female_AUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_subgroupAUROC"
    # )
    """ MIMIC-CXR age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_age')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="Anomaly scores",
    #     title="FAE anomaly scores on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="FPR@0.95TPR",
    #     title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_AUROC", "test/young_AUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_subgroupAUROC"
    # )
    """ MIMIC-CXR race """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_race')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/white_anomaly_score", "test/black_anomaly_score"],
    #     subgroup_names=["test/white", "test/black"],
    #     attr_key='white_percent',
    #     xlabel="Percentage of white subjects in training set",
    #     ylabel="Anomaly scores",
    #     title="FAE anomaly scores on MIMIC-CXR for different proportions of white patients in training",
    #     plt_name="fae_mimic-cxr_race_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/white_fpr@0.95", "test/black_fpr@0.95"],
    #     subgroup_names=["test/white", "test/black"],
    #     attr_key='white_percent',
    #     xlabel="Percentage of white subjects in training set",
    #     ylabel="FPR@0.95TPR",
    #     title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of white patients in training",
    #     plt_name="fae_mimic-cxr_race_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/white_AUROC", "test/black_AUROC"],
    #     subgroup_names=["test/white", "test/black"],
    #     attr_key='white_percent',
    #     xlabel="Percentage of white subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on MIMIC-CXR for different proportions of white patients in training",
    #     plt_name="fae_mimic-cxr_race_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/white_subgroupAUROC", "test/black_subgroupAUROC"],
    #     subgroup_names=["test/white", "test/black"],
    #     attr_key='white_percent',
    #     xlabel="Percentage of white subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on MIMIC-CXR for different proportions of white patients in training",
    #     plt_name="fae_mimic-cxr_race_subgroupAUROC"
    # )

    """ CXR14 sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_cxr14_sex')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="Anomaly scores",
    #     title="FAE anomaly scores on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="FPR@0.95TPR",
    #     title="FAE fpr@0.95tpr on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_AUROC", "test/female_AUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_subgroupAUROC"
    # )
    """ CXR14 age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_cxr14_age')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="Anomaly scores",
    #     title="FAE anomaly scores on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="FPR@0.95TPR",
    #     title="FAE fpr@0.95tpr on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_AUROC", "test/young_AUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_subgroupAUROC"
    # )

    """ CheXpert sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_chexpert_sex')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="Anomaly scores",
    #     title="FAE anomaly scores on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="FPR@0.95TPR",
    #     title="FAE fpr@0.95tpr on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_AUROC", "test/female_AUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="Percentage of male subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_subgroupAUROC"
    # )
    """ CheXpert age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_chexpert_age')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="Anomaly scores",
    #     title="FAE anomaly scores on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="FPR@0.95TPR",
    #     title="FAE fpr@0.95tpr on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_AUROC", "test/young_AUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="Percentage of old subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_subgroupAUROC"
    # )

    """ CheXpert race """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_chexpert_race')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/white_anomaly_score", "test/black_anomaly_score"],
    #     subgroup_names=["test/white", "test/black"],
    #     attr_key='white_percent',
    #     xlabel="Percentage of white subjects in training set",
    #     ylabel="Anomaly scores",
    #     title="FAE anomaly scores on CheXpert for different proportions of white patients in training",
    #     plt_name="fae_chexpert_race_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/white_fpr@0.95", "test/black_fpr@0.95"],
    #     subgroup_names=["test/white", "test/black"],
    #     attr_key='white_percent',
    #     xlabel="Percentage of white subjects in training set",
    #     ylabel="FPR@0.95TPR",
    #     title="FAE fpr@0.95tpr on CheXpert for different proportions of white patients in training",
    #     plt_name="fae_chexpert_race_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/white_AUROC", "test/black_AUROC"],
    #     subgroup_names=["test/white", "test/black"],
    #     attr_key='white_percent',
    #     xlabel="Percentage of white subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CheXpert for different proportions of white patients in training",
    #     plt_name="fae_chexpert_race_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/white_subgroupAUROC", "test/black_subgroupAUROC"],
    #     subgroup_names=["test/white", "test/black"],
    #     attr_key='white_percent',
    #     xlabel="Percentage of white subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on CheXpert for different proportions of white patients in training",
    #     plt_name="fae_chexpert_race_subgroupAUROC"
    # )

    """ MIMIC-CXR - Intersectional study (age, sex, and race) """
    h = 4.2
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_intersectional_age_sex_race')
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=[
    #         ["test/male", "test/female"],
    #         ["test/male", "test/female"],
    #     ],
    #     ylabel="subgroupAUROC",
    #     plt_name="fae_mimic-cxr_intersectional_sex_subgroupAUROC",
    #     figsize=(3, h),
    #     group_colors=[sns.color_palette()[0]]
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
    #     subgroup_names=[
    #         ["test/old", "test/young"],
    #         ["test/old", "test/young"],
    #     ],
    #     ylabel="subgroupAUROC",
    #     plt_name="fae_mimic-cxr_intersectional_age_subgroupAUROC",
    #     figsize=(3, h),
    #     group_colors=[sns.color_palette()[1]]
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/white_subgroupAUROC", "test/black_subgroupAUROC"],
    #     subgroup_names=[
    #         ["test/white", "test/black"],
    #         ["test/white", "test/black"],
    #     ],
    #     ylabel="subgroupAUROC",
    #     plt_name="fae_mimic-cxr_intersectional_race_subgroupAUROC",
    #     figsize=(3, h),
    #     group_colors=[sns.color_palette()[2]]
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_old_subgroupAUROC", "test/male_young_subgroupAUROC", "test/male_white_subgroupAUROC", "test/male_black_subgroupAUROC"],
    #     subgroup_names=[
    #         ["test/male_old", "test/male_young", "test/female_old", "test/female_young"],
    #         ["test/male_old", "test/male_young", "test/female_old", "test/female_young"],
    #         ["test/male_white", "test/male_black", "test/female_white", "test/female_black"],
    #         ["test/male_white", "test/male_black", "test/female_white", "test/female_black"],
    #     ],
    #     ylabel="subgroupAUROC",
    #     plt_name="fae_mimic-cxr_intersectional_age_race_after_male_subgroupAUROC",
    #     plot_groups=(2, 2),
    #     figsize=(2.1, h),
    #     group_colors=sns.color_palette()[1:3]
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/female_old_subgroupAUROC", "test/female_young_subgroupAUROC", "test/female_white_subgroupAUROC", "test/female_black_subgroupAUROC"],
    #     subgroup_names=[
    #         ["test/male_old", "test/male_young", "test/female_old", "test/female_young"],
    #         ["test/male_old", "test/male_young", "test/female_old", "test/female_young"],
    #         ["test/male_white", "test/male_black", "test/female_white", "test/female_black"],
    #         ["test/male_white", "test/male_black", "test/female_white", "test/female_black"],
    #     ],
    #     ylabel="subgroupAUROC",
    #     plt_name="fae_mimic-cxr_intersectional_age_race_after_female_subgroupAUROC",
    #     plot_groups=(2, 2),
    #     figsize=(2.1, h),
    #     group_colors=sns.color_palette()[1:3]
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_young_subgroupAUROC", "test/female_young_subgroupAUROC", "test/young_white_subgroupAUROC", "test/young_black_subgroupAUROC"],
    #     subgroup_names=[
    #         ["test/male_old", "test/male_young", "test/female_old", "test/female_young"],
    #         ["test/male_old", "test/male_young", "test/female_old", "test/female_young"],
    #         ["test/old_white", "test/old_black", "test/young_white", "test/young_black"],
    #         ["test/old_white", "test/old_black", "test/young_white", "test/young_black"],
    #     ],
    #     ylabel="subgroupAUROC",
    #     plt_name="fae_mimic-cxr_intersectional_sex_race_after_young_subgroupAUROC",
    #     plot_groups=(2, 2),
    #     figsize=(2.1, h),
    #     group_colors=[sns.color_palette()[0], sns.color_palette()[2]]
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_old_subgroupAUROC", "test/female_old_subgroupAUROC", "test/old_white_subgroupAUROC", "test/old_black_subgroupAUROC"],
    #     subgroup_names=[
    #         ["test/male_old", "test/male_young", "test/female_old", "test/female_young"],
    #         ["test/male_old", "test/male_young", "test/female_old", "test/female_young"],
    #         ["test/old_white", "test/old_black", "test/young_white", "test/young_black"],
    #         ["test/old_white", "test/old_black", "test/young_white", "test/young_black"],
    #     ],
    #     ylabel="subgroupAUROC",
    #     plt_name="fae_mimic-cxr_intersectional_sex_race_after_old_subgroupAUROC",
    #     plot_groups=(2, 2),
    #     figsize=(2.1, h),
    #     group_colors=[sns.color_palette()[0], sns.color_palette()[2]]
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_white_subgroupAUROC", "test/female_white_subgroupAUROC", "test/old_white_subgroupAUROC", "test/young_white_subgroupAUROC"],
    #     subgroup_names=[
    #         ["test/male_white", "test/male_black", "test/female_white", "test/female_black"],
    #         ["test/male_white", "test/male_black", "test/female_white", "test/female_black"],
    #         ["test/old_white", "test/young_white", "test/old_black", "test/young_black"],
    #         ["test/old_white", "test/young_white", "test/old_black", "test/young_black"],
    #     ],
    #     ylabel="subgroupAUROC",
    #     plt_name="fae_mimic-cxr_intersectional_sex_age_after_white_subgroupAUROC",
    #     plot_groups=(2, 2),
    #     figsize=(2.1, h),
    #     group_colors=sns.color_palette()[:2]
    # )
    # plot_metric_single(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_black_subgroupAUROC", "test/female_black_subgroupAUROC", "test/old_black_subgroupAUROC", "test/young_black_subgroupAUROC"],
    #     subgroup_names=[
    #         ["test/male_white", "test/male_black", "test/female_white", "test/female_black"],
    #         ["test/male_white", "test/male_black", "test/female_white", "test/female_black"],
    #         ["test/old_white", "test/young_white", "test/old_black", "test/young_black"],
    #         ["test/old_white", "test/young_white", "test/old_black", "test/young_black"],
    #     ],
    #     ylabel="subgroupAUROC",
    #     plt_name="fae_mimic-cxr_intersectional_sex_age_after_black_subgroupAUROC",
    #     plot_groups=(2, 2),
    #     figsize=(2.1, h),
    #     group_colors=sns.color_palette()[:2]
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
