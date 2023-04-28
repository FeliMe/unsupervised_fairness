import math
import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def filter_experiment(
        df: pd.DataFrame,
        model_type: str,
        dataset: str,
        protected_attr: str,
        conditioned: bool):
    data = df[(df.model_type == model_type) & (df.dataset == dataset) & (df.protected_attr == protected_attr)]
    if conditioned:
        data = data[data.Tags.str.contains('conditioned')]
    else:
        data = data[~data.Tags.str.contains('conditioned')]
    return data


def plot_metric(
        df: pd.DataFrame,
        model_type: str,
        dataset: str,
        protected_attr: str,
        attr_key: str,
        conditioned: bool,
        metrics: List[str],
        xlabel: str,
        ylabel: str,
        title: str,
        plt_name: str):
    data = filter_experiment(
        df,
        model_type=model_type,
        dataset=dataset,
        protected_attr=protected_attr,
        conditioned=conditioned
    )
    if len(data) == 0:
        warnings.warn(f"No data found for {model_type} {dataset} {protected_attr} conditioned={conditioned} {metrics}")
        return
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
        bars.append(plt.bar(ind + i * width, vals, width=width))
        if mini > np.min(vals):
            mini = np.min(vals)
        if maxi < np.max(vals):
            maxi = np.max(vals)
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
    df = pd.read_csv(os.path.join(THIS_DIR, 'data.csv'))
    # FAE camcan
    plot_metric(
        df,
        model_type='FAE',
        dataset='camcan',
        metrics=["test/young_anomaly_scores", "test/avg_anomaly_scores", "test/old_anomaly_scores"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="anomaly score",
        title="FAE anomaly scores on CamCAN when training on each age bin respectively",
        plt_name="fae_camcan_age_anomaly_scores.png"
    )

    """ FAE RSNA """
    # FAE camcan/brats
    plot_metric(
        df,
        model_type='FAE',
        dataset='camcan/brats',
        metrics=["test/young_anomaly_scores", "test/avg_anomaly_scores", "test/old_anomaly_scores"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="anomaly score",
        title="FAE anomaly scores on BraTS when training on each age bin respectively",
        plt_name="fae_brats_age_anomaly_scores.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='camcan/brats',
        metrics=["test/young_ap", "test/avg_ap", "test/old_ap"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="average precision",
        title="FAE average precision on BraTS when training on each age bin respectively",
        plt_name="fae_brats_age_ap.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='camcan/brats',
        metrics=["test/young_cDC", "test/avg_cDC", "test/old_cDC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="cDC",
        title="FAE cDC on BraTS when training on each age bin respectively",
        plt_name="fae_brats_age_cDC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='camcan/brats',
        metrics=["test/young_aDSC", "test/avg_aDSC", "test/old_aDSC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="aDSC",
        title="FAE aDSC on BraTS when training on each age bin respectively",
        plt_name="fae_brats_age_aDSC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='camcan/brats',
        metrics=["test/young_tpr@0.05", "test/avg_tpr@0.05", "test/old_tpr@0.05"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="TPR@FPR=0.05",
        title="FAE TPR@FPT=0.05 on BraTS when training on each age bin respectively",
        plt_name="fae_brats_age_tpr@fpr=0-05.png"
    )

    # FAE rsna age
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_anomaly_scores", "test/lungOpacity_avg_anomaly_scores", "test/lungOpacity_old_anomaly_scores"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="anomaly score",
        title="FAE anomaly scores on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_anomaly_scores.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_ap", "test/lungOpacity_avg_ap", "test/lungOpacity_old_ap"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="average precision",
        title="FAE average precision on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_ap.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_cDC", "test/lungOpacity_avg_cDC", "test/lungOpacity_old_cDC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="cDC",
        title="FAE cDC on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_cDC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_aDSC", "test/lungOpacity_avg_aDSC", "test/lungOpacity_old_aDSC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="aDSC",
        title="FAE aDSC on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_aDSC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_tpr@0.05", "test/lungOpacity_avg_tpr@0.05", "test/lungOpacity_old_tpr@0.05"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="TPR@FPR=0.05",
        title="FAE TPR@FPT=0.05 on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_tpr@fpr=0-05.png"
    )

    # FAE rsna sex
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_anomaly_scores", "test/lungOpacity_female_anomaly_scores"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly score",
        title="FAE anomaly scores on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_anomaly_scores.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_ap", "test/lungOpacity_female_ap"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="average precision",
        title="FAE average precision on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_ap.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_cDC", "test/lungOpacity_female_cDC"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="cDC",
        title="FAE cDC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_cDC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_aDSC", "test/lungOpacity_female_aDSC"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="aDSC",
        title="FAE aDSC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_aDSC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_tpr@0.05", "test/lungOpacity_female_tpr@0.05"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="TPR@FPR=0.05",
        title="FAE TPR@FPT=0.05 on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_tpr@fpr=0-05.png"
    )

    """ FAE RSNA conditioned """
    # FAE rsna age
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_anomaly_scores", "test/lungOpacity_avg_anomaly_scores", "test/lungOpacity_old_anomaly_scores"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="anomaly score",
        title="FAE anomaly scores on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_anomaly_scores.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_ap", "test/lungOpacity_avg_ap", "test/lungOpacity_old_ap"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="average precision",
        title="FAE average precision on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_ap.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_cDC", "test/lungOpacity_avg_cDC", "test/lungOpacity_old_cDC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="cDC",
        title="FAE cDC on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_cDC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_aDSC", "test/lungOpacity_avg_aDSC", "test/lungOpacity_old_aDSC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="aDSC",
        title="FAE aDSC on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_aDSC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_young_tpr@0.05", "test/lungOpacity_avg_tpr@0.05", "test/lungOpacity_old_tpr@0.05"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="TPR@FPR=0.05",
        title="FAE TPR@FPT=0.05 on RSNA when training on each age bin respectively",
        plt_name="fae_rsna_age_tpr@fpr=0-05.png"
    )

    # FAE rsna sex
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_anomaly_scores", "test/lungOpacity_female_anomaly_scores"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly score",
        title="FAE anomaly scores on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_anomaly_scores.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_ap", "test/lungOpacity_female_ap"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="average precision",
        title="FAE average precision on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_ap.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_cDC", "test/lungOpacity_female_cDC"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="cDC",
        title="FAE cDC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_cDC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_aDSC", "test/lungOpacity_female_aDSC"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="aDSC",
        title="FAE aDSC on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_aDSC.png"
    )
    plot_metric(
        df,
        model_type='FAE',
        dataset='rsna',
        metrics=["test/lungOpacity_male_tpr@0.05", "test/lungOpacity_female_tpr@0.05"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="TPR@FPR=0.05",
        title="FAE TPR@FPT=0.05 on RSNA for different proportions of male patients in training",
        plt_name="fae_rsna_sex_tpr@fpr=0-05.png"
    )

    # ResNet18 rsna age
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_anomaly_scores", "test/lungOpacity_avg_anomaly_scores", "test/lungOpacity_old_anomaly_scores"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="anomaly score",
        title="ResNet18 anomaly scores on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_anomaly_scores.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_ap", "test/lungOpacity_avg_ap", "test/lungOpacity_old_ap"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="average precision",
        title="ResNet18 average precision on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_ap.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_cDC", "test/lungOpacity_avg_cDC", "test/lungOpacity_old_cDC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="cDC",
        title="ResNet18 cDC on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_cDC.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_aDSC", "test/lungOpacity_avg_aDSC", "test/lungOpacity_old_aDSC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="aDSC",
        title="ResNet18 aDSC on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_aDSC.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_tpr@0.05", "test/lungOpacity_avg_tpr@0.05", "test/lungOpacity_old_tpr@0.05"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="TPR@FPR=0.05",
        title="ResNet18 TPR@FPT=0.05 on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_tpr@fpr=0-05.png"
    )

    # ResNet18 rsna age
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_anomaly_scores", "test/lungOpacity_avg_anomaly_scores", "test/lungOpacity_old_anomaly_scores"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="anomaly score",
        title="ResNet18 anomaly scores on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_anomaly_scores.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_ap", "test/lungOpacity_avg_ap", "test/lungOpacity_old_ap"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="average precision",
        title="ResNet18 average precision on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_ap.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_cDC", "test/lungOpacity_avg_cDC", "test/lungOpacity_old_cDC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="cDC",
        title="ResNet18 cDC on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_cDC.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_aDSC", "test/lungOpacity_avg_aDSC", "test/lungOpacity_old_aDSC"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="aDSC",
        title="ResNet18 aDSC on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_aDSC.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_young_tpr@0.05", "test/lungOpacity_avg_tpr@0.05", "test/lungOpacity_old_tpr@0.05"],
        protected_attr='age',
        attr_key='train_age',
        conditioned=False,
        xlabel="train age",
        ylabel="TPR@FPR=0.05",
        title="ResNet18 TPR@FPT=0.05 on RSNA when training on each age bin respectively",
        plt_name="resnet18_rsna_age_tpr@fpr=0-05.png"
    )

    # ResNet18 rsna sex
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_male_anomaly_scores", "test/lungOpacity_female_anomaly_scores"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="anomaly score",
        title="ResNet18 anomaly scores on RSNA for different proportions of male patients in training",
        plt_name="resnet18_rsna_sex_anomaly_scores.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_male_ap", "test/lungOpacity_female_ap"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="average precision",
        title="ResNet18 average precision on RSNA for different proportions of male patients in training",
        plt_name="resnet18_rsna_sex_ap.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_male_cDC", "test/lungOpacity_female_cDC"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="cDC",
        title="ResNet18 cDC on RSNA for different proportions of male patients in training",
        plt_name="resnet18_rsna_sex_cDC.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_male_aDSC", "test/lungOpacity_female_aDSC"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="aDSC",
        title="ResNet18 aDSC on RSNA for different proportions of male patients in training",
        plt_name="resnet18_rsna_sex_aDSC.png"
    )
    plot_metric(
        df,
        model_type='ResNet18',
        dataset='rsna',
        metrics=["test/lungOpacity_male_tpr@0.05", "test/lungOpacity_female_tpr@0.05"],
        protected_attr='sex',
        attr_key='male_percent',
        conditioned=False,
        xlabel="percentage of male subjects in training set",
        ylabel="TPR@FPR=0.05",
        title="ResNet18 TPR@FPT=0.05 on RSNA for different proportions of male patients in training",
        plt_name="resnet18_rsna_sex_tpr@fpr=0-05.png"
    )
    i = 1
