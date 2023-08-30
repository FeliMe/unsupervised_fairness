"""This script is used to check for statistical significance"""
import os
from typing import Dict, Tuple

import numpy as np
from scipy import stats

from src.analysis.utils import gather_data_seeds


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_metric(
        experiment_dir: str,
        attr_key: str,
        metrics: Tuple[str, str]):
    """Perform different statistical tests on data"""
    data, attr_key_values = gather_data_seeds(experiment_dir, attr_key, metrics)
    test_metric_balanced(data, metrics)
    check_pearsson_correlation_coefficient(data, metrics, attr_key_values)
    experiment = experiment_dir.split('/')[-1]
    compute_mae(data, metrics, experiment)


def test_metric_balanced(
        data: Dict[str, np.ndarray],
        metrics: Tuple[str, str],
        alpha: float = 0.01):
    """Perform Welch's t-test on balanced data"""
    vals1 = data[metrics[0]]  # Shape: (num_attr_key_values, num_seeds)
    vals2 = data[metrics[1]]  # Shape: (num_attr_key_values, num_seeds)

    middle_key_vals1 = vals1[vals1.shape[0] // 2]  # Shape: (num_seeds,)
    middle_key_vals2 = vals2[vals1.shape[0] // 2]  # Shape: (num_seeds,)

    # Two-sided test
    _, p = stats.ttest_ind(middle_key_vals1, middle_key_vals2, equal_var=False)
    print(f"Two-sided Welch's t-test for {metrics[0]} and {metrics[1]}: p={p}")
    if p < alpha:
        print(f'p < {alpha}: The metrics are significantly different')
    else:
        print(f'p >= {alpha}: No statistical difference')
    # Test if metric[0] is significantly larger than metric[1]
    _, p = stats.ttest_ind(middle_key_vals1, middle_key_vals2, alternative='greater', equal_var=False)
    print(f"One-sided Welch's t-test for {metrics[0]} > {metrics[1]}: p={p}")
    if p < alpha:
        print(f'p < {alpha}: {metrics[0]} is significantly larger than {metrics[1]}')
    else:
        print(f'p >= {alpha}: No statistical difference')
    # Test if metric[1] is significantly larger than metric[0]
    _, p = stats.ttest_ind(middle_key_vals2, middle_key_vals1, alternative='greater', equal_var=False)
    print(f"One-sided Welch's t-test for {metrics[1]} > {metrics[0]}: p={p}")
    if p < alpha:
        print(f'p < {alpha}: {metrics[1]} is significantly larger than {metrics[0]}')
    else:
        print(f'p >= {alpha}: No statistical difference')

    print("")


def compute_mae(data: Dict[str, np.ndarray],
                metrics: Tuple[str, str],
                experiment: str):
    """Compute the mean absolute error between the metrics and a linear model"""
    all_maes = []
    for metric in metrics:
        vals = data[metric]  # Shape: (num_attr_key_values, num_seeds)
        n_vals, n_seeds = vals.shape
        pred = np.linspace(vals[0], vals[-1], num=n_vals)
        maes = np.abs(pred - vals)[1:-1].mean(0)
        all_maes.append(maes)
        print(f"MAE of linear interpolation of {metric}: {maes.mean():.6f}, std: {maes.std():.6f}")
    os.makedirs("logs/maes", exist_ok=True)
    np.save(f"logs/maes/{experiment}_maes.npy", np.concatenate(all_maes))


def check_pearsson_correlation_coefficient(data: Dict[str, np.ndarray],
                                           metrics: Tuple[str, str],
                                           attr_key_values: np.ndarray):
    """Checks the pearsson correlation coefficient between the results."""
    for metric in metrics:
        y = data[metric]
        x = attr_key_values[:, None].repeat(y.shape[1], axis=1)
        corr, p_value = stats.pearsonr(x.flatten(), y.flatten())
        print(f"Pearsson correlation coefficient of {metric}: {corr}")


if __name__ == '__main__':
    """ CXR14 """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_cxr14_sex')
    print("CXR14 sex")
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
        attr_key='male_percent',
    )
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_cxr14_age')
    print("CXR14 age")
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/young_subgroupAUROC", "test/old_subgroupAUROC"],
        attr_key='old_percent',
    )

    """ MIMIC-CXR """
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_sex')
    print("MIMIC-CXR sex")
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
        attr_key='male_percent',
    )
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_age')
    print("MIMIC-CXR age")
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/young_subgroupAUROC", "test/old_subgroupAUROC"],
        attr_key='old_percent',
    )
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_mimic-cxr_race')
    print("MIMIC-CXR race")
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/white_subgroupAUROC", "test/black_subgroupAUROC"],
        attr_key='white_percent',
    )

    """ CheXpert """
    print("CheXpert sex")
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_chexpert_sex')
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/male_subgroupAUROC", "test/female_subgroupAUROC"],
        attr_key='male_percent',
    )
    experiment_dir = os.path.join(THIS_DIR, '../../logs/FAE_chexpert_age')
    print("CheXpert age")
    test_metric(
        experiment_dir=experiment_dir,
        metrics=["test/young_subgroupAUROC", "test/old_subgroupAUROC"],
        attr_key='old_percent',
    )
