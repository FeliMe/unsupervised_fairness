"""
This file computes statistics of anomalous samples for RSNA and BraTS.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = "Times New Roman"


def plot_anomaly_scores_brats(results_file: str):
    """Plots histograms of the normal and anomal samples of young and old
    patients in BraTS

    :param results_file: path to the csv files that stores the anomaly scores,
                         labels, and subgroup information
    """
    df = pd.read_csv(results_file)

    df_n = df[df['label'] == 0]
    df_a = df[df['label'] == 1]
    # Normal young
    scores_normal_young = df_n[df_n['subgroup_name'] == 'test/young']['anomaly_score'].values
    # Normal old
    scores_normal_old = df_n[df_n['subgroup_name'] == 'test/old']['anomaly_score'].values
    # Anomal young
    scores_anomal_young = df_a[df_a['subgroup_name'] == 'test/young']['anomaly_score'].values
    # Anomal old
    scores_anomal_old = df_a[df_a['subgroup_name'] == 'test/old']['anomaly_score'].values

    print("BraTS")
    print(f"Normal young: {scores_normal_young.shape}")
    print(f"Normal old: {scores_normal_old.shape}")
    print(f"Anomal young: {scores_anomal_young.shape}")
    print(f"Anomal old: {scores_anomal_old.shape}")

    # Create a plot with 2 subplots
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot histograms
    ax1.hist(scores_normal_young, bins=10, alpha=0.5, label='Normal young',
             weights=np.ones(len(scores_normal_young)) / len(scores_normal_young))
    ax1.hist(scores_anomal_young, bins=10, alpha=0.5, label='Anomal young',
             weights=np.ones(len(scores_anomal_young)) / len(scores_anomal_young))
    ax2.hist(scores_normal_old, bins=10, alpha=0.5, label='Normal old',
             weights=np.ones(len(scores_normal_old)) / len(scores_normal_old))
    ax2.hist(scores_anomal_old, bins=10, alpha=0.5, label='Anomal old',
             weights=np.ones(len(scores_anomal_old)) / len(scores_anomal_old))

    # Plot settings
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Anomaly score')
    ax2.set_xlabel('Anomaly score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Young patients')
    ax2.set_title('Old patients')
    f.suptitle('Anomaly scores of normal and anomalous samples in BraTS', wrap=True)

    # Save figure
    plt.savefig(os.path.join(THIS_DIR, 'brats_anomaly_scores.png'))


def plot_anomaly_scores_rsna(results_file: str):
    """Plots histograms of the normal and anomal samples of young and old
    patients in RSNA

    :param results_file: path to the csv files that stores the anomaly scores,
                         labels, and subgroup information
    """
    df = pd.read_csv(results_file)

    df_n = df[df['label'] == 0]
    df_a = df[df['label'] == 1]
    # Normal young
    scores_normal_young = df_n[df_n['subgroup_name'] == 'test/lungOpacity_young']['anomaly_score'].values
    # Normal old
    scores_normal_old = df_n[df_n['subgroup_name'] == 'test/lungOpacity_old']['anomaly_score'].values
    # Anomal young
    scores_anomal_young = df_a[df_a['subgroup_name'] == 'test/lungOpacity_young']['anomaly_score'].values
    # Anomal old
    scores_anomal_old = df_a[df_a['subgroup_name'] == 'test/lungOpacity_old']['anomaly_score'].values

    print("RSNA")
    print(f"Normal young: {scores_normal_young.shape}")
    print(f"Normal old: {scores_normal_old.shape}")
    print(f"Anomal young: {scores_anomal_young.shape}")
    print(f"Anomal old: {scores_anomal_old.shape}")

    # Create a plot with 2 subplots
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot histograms
    ax1.hist(scores_normal_young, bins=10, alpha=0.5, label='Normal young',
             weights=np.ones(len(scores_normal_young)) / len(scores_normal_young))
    ax1.hist(scores_anomal_young, bins=10, alpha=0.5, label='Anomal young',
             weights=np.ones(len(scores_anomal_young)) / len(scores_anomal_young))
    ax2.hist(scores_normal_old, bins=10, alpha=0.5, label='Normal old',
             weights=np.ones(len(scores_normal_old)) / len(scores_normal_old))
    ax2.hist(scores_anomal_old, bins=10, alpha=0.5, label='Anomal old',
             weights=np.ones(len(scores_anomal_old)) / len(scores_anomal_old))

    # Plot settings
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Anomaly score')
    ax2.set_xlabel('Anomaly score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Young patients')
    ax2.set_title('Old patients')
    f.suptitle('Anomaly scores of normal and anomalous samples in RSNA', wrap=True)

    # Save figure
    plt.savefig(os.path.join(THIS_DIR, 'rsna_anomaly_scores.png'))


if __name__ == '__main__':
    plot_anomaly_scores_rsna(
        os.path.join(THIS_DIR, '../../logs/FAE_rsna_age/old_percent_050/seed_1/anomaly_scores.csv'))
    plot_anomaly_scores_brats(
        os.path.join(THIS_DIR, '../../logs/FAE_camcan-brats_age/old_percent_050/seed_1/anomaly_scores.csv'))
