"""This script is used to plot the results of the experiments."""
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from einops import repeat
import matplotlib.colors as mcolors
from scipy import stats

from src.analysis.utils import gather_data_seeds, gather_data_from_anomaly_scores


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
    # Collect data from all runs
    # data = gather_data_bootstrap(experiment_dir)
    data, attr_key_values = gather_data_from_anomaly_scores(
        experiment_dir, metrics, subgroup_names, attr_key)
    # data, attr_key_values = gather_data_seeds(experiment_dir, attr_key, metrics)

    # Check linear interpolation error
    check_linear_regression_error(data, metrics)

    # Plot bar plot
    plot_metric_regression_bar(
        data=data,
        attr_key_values=attr_key_values,
        metrics=metrics,
        xlabel=xlabel,
        ylabel=ylabel,
        # title=title,
        title="",
        # plt_name=plt_name + '_bar.png'
        plt_name=plt_name + '_bar.pdf'
    )

    # # Plot box-whisker plot
    # plot_metric_regression_box_whisker(
    #     data=data,
    #     attr_key_values=attr_key_values,
    #     metrics=metrics,
    #     xlabel=xlabel,
    #     ylabel=ylabel,
    #     title=title,
    #     plt_name=plt_name + '_box.png'
    # )

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
    # Prepare plot
    width = 0.25
    ind = np.arange(len(data[metrics[0]]))
    centers = ind + (len(metrics) // 2) * width - width / 2
    bars = []
    mini = math.inf
    maxi = -math.inf

    # Plot bar plots
    diff = None
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

        if diff is None:
            diff = mean
        else:
            print(f"Diff between {metrics[0]} and {metrics[1]} for {attr_key_values}: {mean - diff}")

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
        alpha = 0.01
        is_significant = p_value < alpha
        print(f"The regression line of {metric} is significantly different from 0: {is_significant}. p={p_value}")

    # Plot settings
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, wrap=True)
    plt.xticks(centers, attr_key_values.round(2) if isinstance(attr_key_values[0], float) else attr_key_values)
    plt.xlim(left, right)
    # plt.legend(bars, metrics)
    if "lungOpacity" in metrics[0]:
        plt.legend(bars, [metric.split('/')[-1].split('_')[1] for metric in metrics])
    else:
        plt.legend(bars, [metric.split('/')[-1].split('_')[0] for metric in metrics])
    ylim_min = max(0, mini - 0.1 * (maxi - mini))
    ylim_max = min(1, maxi + 0.1 * (maxi - mini))
    plt.ylim(ylim_min, ylim_max)
    # plt.ylim(ylim_min, plt.ylim()[1])
    print(f"Saving plot to {plt_name}")
    plt.savefig(os.path.join(THIS_DIR, plt_name))
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

    plt.figure(figsize=(6.4 * (3 / 4), 4.8))

    # Plot bar plots
    vals = np.concatenate([data[metric] for metric in metrics])
    mean = vals.mean(axis=1)
    print(f"Diff between {metrics[0]} and {metrics[1]}: {mean[0] - mean[1]}")
    std = vals.std(axis=1)
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    yerr = np.stack([mean - lower, upper - mean], axis=0)
    plt.bar(np.arange(len(mean)), mean, color=mcolors.TABLEAU_COLORS, yerr=yerr, ecolor='darkgray')

    mini = np.min(lower)
    maxi = np.max(upper)

    # labels and ticks
    plt.ylabel(ylabel)
    # plt.xticks(np.arange(len(vals)), metrics)
    plt.xticks(np.arange(len(vals)), ["_".join(m.split("/")[-1].split("_")[:-1]) for m in metrics])

    # Set ylimit
    ylim_min = max(0, mini - 0.1 * (maxi - mini))
    ylim_max = min(1, maxi + 0.1 * (maxi - mini))
    # plt.ylim(ylim_min, ylim_max)
    plt.ylim(0.58, 0.78)

    plt.tight_layout()

    # Save plot
    full_name = os.path.join(THIS_DIR, plt_name + ".pdf")
    print(f"Saving plot to {full_name}")
    plt.savefig(full_name)
    plt.close()


if __name__ == '__main__':
    """ FAE RSNA sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_sex')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_male_anomaly_score", "test/lungOpacity_female_anomaly_score"],
    #     subgroup_names=["test/lungOpacity_male", "test/lungOpacity_female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="anomaly score",
    #     title="FAE anomaly scores on RSNA for different proportions of male patients in training",
    #     plt_name="fae_rsna_sex_anomaly_score"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_male_fpr@0.95", "test/lungOpacity_female_fpr@0.95"],
    #     subgroup_names=["test/lungOpacity_male", "test/lungOpacity_female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="fpr@0.95",
    #     title="FAE fpr@0.95tpr on RSNA for different proportions of male patients in training",
    #     plt_name="fae_rsna_sex_fpr@0.95"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_male_AP", "test/lungOpacity_female_AP"],
    #     subgroup_names=["test/lungOpacity_male", "test/lungOpacity_female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="Average Precision",
    #     title="FAE AP on RSNA for different proportions of male patients in training",
    #     plt_name="fae_rsna_sex_AP"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_male_AUROC", "test/lungOpacity_female_AUROC"],
    #     subgroup_names=["test/lungOpacity_male", "test/lungOpacity_female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on RSNA for different proportions of male patients in training",
    #     plt_name="fae_rsna_sex_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_male_subgroupAUROC", "test/lungOpacity_female_subgroupAUROC"],
    #     subgroup_names=["test/lungOpacity_male", "test/lungOpacity_female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on RSNA for different proportions of male patients in training",
    #     plt_name="fae_rsna_sex_subgroupAUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_fpr@0.95"],
    #     subgroup_names=["test/lungOpacity_male", "test/lungOpacity_female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="fpr@0.95",
    #     title="FAE fpr@0.95tpr on RSNA for different proportions of male patients in training",
    #     plt_name="fae_rsna_sex_fpr@0.95_total"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_AUROC"],
    #     subgroup_names=["test/lungOpacity_male", "test/lungOpacity_female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on RSNA for different proportions of male patients in training",
    #     plt_name="fae_rsna_sex_AUROC_total"
    # )
    """ FAE rsna age """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_rsna_age')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_old_anomaly_score", "test/lungOpacity_young_anomaly_score"],
    #     subgroup_names=["test/lungOpacity_old", "test/lungOpacity_young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="anomaly scores",
    #     title="FAE anomaly scores on RSNA for different proportions of old patients in training",
    #     plt_name="fae_rsna_age_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_old_fpr@0.95", "test/lungOpacity_young_fpr@0.95"],
    #     subgroup_names=["test/lungOpacity_old", "test/lungOpacity_young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on RSNA for different proportions of old patients in training",
    #     plt_name="fae_rsna_age_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_old_AP", "test/lungOpacity_young_AP"],
    #     subgroup_names=["test/lungOpacity_old", "test/lungOpacity_young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="Average Precision",
    #     title="FAE AP on RSNA for different proportions of old patients in training",
    #     plt_name="fae_rsna_age_AP"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_old_AUROC", "test/lungOpacity_young_AUROC"],
    #     subgroup_names=["test/lungOpacity_old", "test/lungOpacity_young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on RSNA for different proportions of old patients in training",
    #     plt_name="fae_rsna_age_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_old_subgroupAUROC", "test/lungOpacity_young_subgroupAUROC"],
    #     subgroup_names=["test/lungOpacity_old", "test/lungOpacity_young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on RSNA for different proportions of old patients in training",
    #     plt_name="fae_rsna_age_subgroupAUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_fpr@0.95"],
    #     subgroup_names=["test/lungOpacity_old", "test/lungOpacity_young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on RSNA for different proportions of old patients in training",
    #     plt_name="fae_rsna_age_fpr@0.95tpr_total"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/lungOpacity_AUROC"],
    #     subgroup_names=["test/lungOpacity_old", "test/lungOpacity_young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on RSNA for different proportions of old patients in training",
    #     plt_name="fae_rsna_age_AUROC_total"
    # )
    """ FAE MIMIC-CXR sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_sex')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="anomaly scores",
    #     title="FAE anomaly scores on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_AP", "test/female_AP"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="Average Precision",
    #     title="FAE AP on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_AP"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_AUROC", "test/female_AUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on MIMIC-CXR for different proportions of male patients in training",
    #     plt_name="fae_mimic-cxr_sex_subgroupAUROC"
    # )
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
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="anomaly scores",
    #     title="FAE anomaly scores on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_AP", "test/young_AP"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="Average Precision",
    #     title="FAE AP on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_AP"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_AUROC", "test/young_AUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on MIMIC-CXR for different proportions of old patients in training",
    #     plt_name="fae_mimic-cxr_age_subgroupAUROC"
    # )
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
    """ FAE CXR14 sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_cxr14_sex')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="anomaly scores",
    #     title="FAE anomaly scores on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_AP", "test/female_AP"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="Average Precsision",
    #     title="FAE AP on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_AP"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_AUROC", "test/female_AUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on CXR14 for different proportions of male patients in training",
    #     plt_name="fae_cxr14_sex_subgroupAUROC"
    # )
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
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="anomaly scores",
    #     title="FAE anomaly scores on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_AP", "test/young_AP"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="Average Precision",
    #     title="FAE AP on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_AP"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_AUROC", "test/young_AUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on CXR14 for different proportions of old patients in training",
    #     plt_name="fae_cxr14_age_subgroupAUROC"
    # )
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

    """ FAE CheXpert sex """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_chexpert_sex')
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_anomaly_score", "test/female_anomaly_score"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="anomaly scores",
    #     title="FAE anomaly scores on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_fpr@0.95", "test/female_fpr@0.95"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_AUROC", "test/female_AUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
    #     subgroup_names=["test/male", "test/female"],
    #     attr_key='male_percent',
    #     xlabel="percentage of male subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on CheXpert for different proportions of male patients in training",
    #     plt_name="fae_chexpert_sex_subgroupAUROC"
    # )
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
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_anomaly_score", "test/young_anomaly_score"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="anomaly scores",
    #     title="FAE anomaly scores on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_anomaly_scores"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_fpr@0.95", "test/young_fpr@0.95"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="fpr@0.95tpr",
    #     title="FAE fpr@0.95tpr on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_fpr@0.95tpr"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_AUROC", "test/young_AUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="AUROC",
    #     title="FAE AUROC on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_AUROC"
    # )
    # plot_metric_regression(
    #     experiment_dir=experiment_dir,
    #     metrics=["test/old_subgroupAUROC", "test/young_subgroupAUROC"],
    #     subgroup_names=["test/old", "test/young"],
    #     attr_key='old_percent',
    #     xlabel="percentage of old subjects in training set",
    #     ylabel="subgroupAUROC",
    #     title="FAE subgroupAUROC on CheXpert for different proportions of old patients in training",
    #     plt_name="fae_chexpert_age_subgroupAUROC"
    # )
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
