"""
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
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.utils.metrics import build_metrics


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


def gather_data_from_anomaly_scores(
        experiment_dir: str,
        metric_names: List[str],
        subgroup_names: List[str],
        attr_key: Optional[str] = None):
    """Gather the data of multiple random seeds
    Loads the anomaly-scores.csv file instead of the test_results.csv file
    and computes the metrics on the fly
    """
    run_dirs = [os.path.join(experiment_dir, run_dir) for run_dir in os.listdir(experiment_dir)]
    run_dirs = sorted(run_dirs)
    run_dfs = []
    attr_key_values = []
    for run_dir in run_dirs:
        seed_dirs = [os.path.join(run_dir, seed_dir) for seed_dir in os.listdir(run_dir)]
        seed_dfs = []
        for seed_dir in seed_dirs:
            scores_file = os.path.join(seed_dir, 'anomaly_scores.csv')
            df = pd.read_csv(scores_file)
            results = compute_metrics_from_scores_file(df, subgroup_names)
            seed_df = pd.DataFrame(results, index=[0])
            if attr_key is not None:
                seed_df[attr_key] = df[attr_key].values[0]
            # For model size experiment:
            if 'model_size' in run_dir:
                seed_df[attr_key] = run_dir[run_dir.find('model_size_') + len('model_size_'):]
            seed_dfs.append(seed_df)
        df = pd.concat(seed_dfs)
        run_dfs.append(df)
        if attr_key is not None:
            attr_key_values.append(df[attr_key].values[0])
    # Sort by protected attribute
    if attr_key is not None:
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


def compute_metrics_from_scores_file(df: pd.DataFrame, subgroup_names: List[str]):
    """Compute the metrics from the anomaly scores"""
    subgroup_mapping = {subgroup_name: i for i, subgroup_name in enumerate(subgroup_names)}
    metrics = build_metrics(subgroup_names)
    for subgroup_name in subgroup_names:
        subgroup_df = df[df.subgroup_name == subgroup_name]
        anomaly_scores = torch.tensor(subgroup_df.anomaly_score.values)
        labels = torch.tensor(subgroup_df.label.values)
        subgroup = torch.tensor([subgroup_mapping[subgroup_name]] * len(labels))
        metrics.update(subgroup, anomaly_scores, labels)
    return {k: v.item() for k, v in metrics.compute().items()}
