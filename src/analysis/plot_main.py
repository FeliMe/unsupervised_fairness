"""This script is used to plot the results of the experiments."""
import math
import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from einops import repeat
from scipy import stats

from src.analysis.utils import gather_data_seeds


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = "Times New Roman"


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

    # # Plot box-whisker plot
    # plot_metric_box_whisker(
    #     data=data,
    #     attr_key_values=attr_key_values,
    #     metrics=metrics,
    #     xlabel=xlabel,
    #     ylabel=ylabel,
    #     title=title,
    #     plt_name=plt_name + '_box.png'
    # )

    # # Plot scatter plot
    # plot_metric_scatter(
    #     data=data,
    #     attr_key_values=attr_key_values,
    #     metrics=metrics,
    #     xlabel=xlabel,
    #     ylabel=ylabel,
    #     title=title,
    #     plt_name=plt_name + '_scatter.png'
    # )


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

    # Plot regression lines
    left, right = plt.xlim()
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
        # all_reg_coefs = []
        # for seed in range(data[metric].shape[1]):
        #     reg_coefs = np.polyfit(centers, vals[:, seed], 1)
        #     all_reg_coefs.append(reg_coefs)
        # all_reg_coefs = np.stack(all_reg_coefs, axis=0)
        # reg_coefs_std = all_reg_coefs.std(axis=0)
        # reg_coefs_mean = all_reg_coefs.mean(axis=0)
        # lower_reg_vals = np.polyval(reg_coefs_mean - reg_coefs_std, reg_ind)
        # upper_reg_vals = np.polyval(reg_coefs_mean + reg_coefs_std, reg_ind)
        # plt.fill_between(reg_ind, lower_reg_vals, upper_reg_vals, color=color, alpha=0.2)
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
    boxes = []

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
        boxes.append(bplot)

        # Set colors
        if i == 0:
            color = 'tab:blue'
        elif i == 1:
            color = 'tab:orange'
        else:
            color = 'tab:green'
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    # Plot regression lines
    left, right = plt.xlim()
    for i, metric in enumerate(metrics):
        vals = data[metric]
        mean = vals.mean(axis=1)
        reg_coefs = np.polyfit(centers, mean, 1)  # linear regression
        reg_ind = np.array([left, right])
        reg_vals = np.polyval(reg_coefs, reg_ind)
        color = list(boxes[i]['boxes'][0].get_facecolor())  # get color of corresponding box
        color[-1] = 0.5  # set alpha to 0.5
        plt.plot(reg_ind, reg_vals, color=color)

    # Plot settings
    plt.legend([bplot["boxes"][0] for bplot in boxes], metrics)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, wrap=True)
    plt.xticks(centers, attr_key_values.round(2))

    # Save plot
    print(f"Saving plot to {plt_name}")
    plt.savefig(os.path.join(THIS_DIR, plt_name))
    plt.close()


if __name__ == '__main__':
    """ FAE RSNA sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_sex')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_anomaly_score", "test/lungOpacity_female_anomaly_score"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly score",
        title="FAE anomaly scores on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_anomaly_score"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_fpr@0.95", "test/lungOpacity_female_fpr@0.95"],
        # metrics=["test/lungOpacity_male_fpr@0.95", "test/lungOpacity_fpr@0.95", "test/lungOpacity_female_fpr@0.95"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95",
        title="FAE fpr@0.95tpr on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_fpr@0.95"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_AUROC", "test/lungOpacity_female_AUROC"],
        # metrics=["test/lungOpacity_male_AUROC", "test/lungOpacity_AUROC", "test/lungOpacity_female_AUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_AUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_subgroupAUROC", "test/lungOpacity_female_subgroupAUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_subgroupAUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_male_tpr@0.05", "test/lungOpacity_female_tpr@0.05"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="tpr@0.05",
        title="FAE tpr@0.05fpr on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_tpr@0.05"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_fpr@0.95"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95",
        title="FAE fpr@0.95tpr on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_fpr@0.95_total"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_AUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_AUROC_total"
    )
    """ FAE rsna age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_age')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_old_anomaly_score", "test/lungOpacity_young_anomaly_score"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_anomaly_scores"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_old_fpr@0.95", "test/lungOpacity_young_fpr@0.95"],
        # metrics=["test/lungOpacity_old_fpr@0.95", "test/lungOpacity_fpr@0.95", "test/lungOpacity_young_fpr@0.95"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_fpr@0.95tpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_old_AUROC", "test/lungOpacity_young_AUROC"],
        # metrics=["test/lungOpacity_old_AUROC", "test/lungOpacity_AUROC", "test/lungOpacity_young_AUROC"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_AUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_old_subgroupAUROC", "test/lungOpacity_young_subgroupAUROC"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_subgroupAUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_old_tpr@0.05", "test/lungOpacity_young_tpr@0.05"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="tpr@0.05fpr",
        title="FAE tpr@0.05fpr on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_tpr@0.05fpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_fpr@0.95"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_fpr@0.95tpr_total"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/lungOpacity_AUROC"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on RSNA for different proportions of old patients in training",
        plt_name="fae_rsna_age_AUROC_total"
    )
    """ FAE CamCAN """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_camcan_age')
    # plot_metric(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="anomaly scores",
    #     title="FAE anomaly scores on CamCAN for different proportions of old patients in training",
    #     plt_name="fae_camcan_age_anomaly_scores"
    # )
    """ FAE CamCAN/BraTS """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_camcan-brats_age')
    # plot_metric(
    #     experiment_dir=experiment_dir,
    #     # metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
    #     metrics=["test/old_fpr@0.95", "test/fpr@0.95", "test/young_fpr@0.95"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on BraTS for different proportions of old patients in training",
    #     plt_name="fae_brats_age_fpr@0.95tpr"
    # )
    # plot_metric(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="anomaly scores",
    #     title="FAE anomaly scores on BraTS for different proportions of old patients in training",
    #     plt_name="fae_brats_age_anomaly_scores"
    # )
    # plot_metric(
    #     experiment_dir=experiment_dir,
    #     # metrics=["test/old_AUROC", "test/young_AUROC"],
    #     metrics=["test/old_AUROC", "test/AUROC", "test/young_AUROC"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on BraTS for different proportions of old patients in training",
    #     plt_name="fae_brats_age_AUROC"
    # )
    """ FAE MIMIC-CXR sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_sex')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_anomaly_scores"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
        # metrics=["test/male_fpr@0.95", "test/fpr@0.95", "test/female_fpr@0.95"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_fpr@0.95tpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_AUROC", "test/female_AUROC"],
        # metrics=["test/male_AUROC", "test/AUROC", "test/female_AUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_AUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_subgroupAUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_tpr@0.05", "test/female_tpr@0.05"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="tpr@0.05fpr",
        title="FAE tpr@0.05fpr on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_tpr@0.05fpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/fpr@0.95"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_fpr@0.95tpr_total"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/AUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on MIMIC-CXR for different proportions of male patients in training",
        plt_name="fae_mimic-cxr_sex_AUROC_total"
    )
    """ FAE MIMIC-CXR age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_age')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_anomaly_scores"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
        # metrics=["test/old_fpr@0.95", "test/fpr@0.95", "test/young_fpr@0.95"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_fpr@0.95tpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_AUROC", "test/young_AUROC"],
        # metrics=["test/old_AUROC", "test/AUROC", "test/young_AUROC"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_AUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_subgroupAUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_tpr@0.05", "test/young_tpr@0.05"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="tpr@0.05fpr",
        title="FAE tpr@0.05fpr on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_tpr@0.05fpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/fpr@0.95"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_fpr@0.95tpr_total"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/AUROC"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on MIMIC-CXR for different proportions of old patients in training",
        plt_name="fae_mimic-cxr_age_AUROC_total"
    )
    """ FAE CXR14 sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_cxr14_sex')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_anomaly_scores"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
        # metrics=["test/male_fpr@0.95", "test/fpr@0.95", "test/female_fpr@0.95"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_fpr@0.95tpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_AUROC", "test/female_AUROC"],
        # metrics=["test/male_AUROC", "test/AUROC", "test/female_AUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_AUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_subgroupAUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_tpr@0.05", "test/female_tpr@0.05"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="tpr@0.05fpr",
        title="FAE tpr@0.05fpr on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_tpr@0.05fpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/fpr@0.95"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_fpr@0.95tpr_total"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/AUROC"],
        attr_key='male_percent',
        xlabel="percentage of male subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on CXR14 for different proportions of male patients in training",
        plt_name="fae_cxr14_sex_AUROC_total"
    )
    """ FAE CXR14 age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_cxr14_age')
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="anomaly scores",
        title="FAE anomaly scores on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_anomaly_scores"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
        # metrics=["test/old_fpr@0.95", "test/fpr@0.95", "test/young_fpr@0.95"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_fpr@0.95tpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_AUROC", "test/young_AUROC"],
        # metrics=["test/old_AUROC", "test/AUROC", "test/young_AUROC"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_AUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="subgroupAUROC",
        title="FAE subgroupAUROC on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_subgroupAUROC"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/old_tpr@0.05", "test/young_tpr@0.05"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="tpr@0.05fpr",
        title="FAE tpr@0.05fpr on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_tpr@0.05fpr"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/fpr@0.95"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="fpr@0.95tpr",
        title="FAE fpr@0.95tpr on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_fpr@0.95tpr_total"
    )
    plot_metric(
        experiment_dir=experiment_dir,
        metrics=["test/AUROC"],
        attr_key='old_percent',
        xlabel="percentage of old subjects in training set",
        ylabel="AUROC",
        title="FAE AUROC on CXR14 for different proportions of old patients in training",
        plt_name="fae_cxr14_age_AUROC_total"
    )
