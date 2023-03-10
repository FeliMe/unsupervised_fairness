from collections import defaultdict
from typing import List

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch import Tensor
from torchmetrics import Metric, MetricCollection


def build_metrics(subgroup_names: List[str]) -> MetricCollection:
    classification_metrics = MetricCollection({
        'ap': AveragePrecision(subgroup_names),
        'cDC': cDC(subgroup_names),
        'aDSC': AverageDSC(subgroup_names),
        'tpr@5fpr': TPR_at_FPR(subgroup_names, xfpr=0.05),
        'fpr@95tpr': FPR_at_TPR(subgroup_names, xtpr=0.95),
    })
    return classification_metrics


class AveragePrecision(Metric):
    """
    Computes the average precision score for subgroups of the data but
    min and max scores are taken from the entire dataset.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List

    def update(self, preds: Tensor, targets: Tensor):
        """
        preds: Tensor of anomaly scores and sub-group labels of shape [b, 2]
               (anomaly score, group-label)
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets)
        self.preds.append(preds)
        self.targets.append(targets)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    def compute_subgroup(self, subgroup: int):
        preds = torch.cat(self.preds)  # [N, 2]
        targets = torch.cat(self.targets)  # [N]
        if targets.sum() == 0:
            return 0
        # Compute min and max score and thresholds for whole dataset
        min_score = preds[:, 0].min()
        max_score = preds[:, 0].quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute average precision for subgroup
        targets = targets[preds[:, 1] == subgroup]  # [N_s]
        preds = preds[preds[:, 1] == subgroup, 0]  # [N_s]
        preds_bin = (preds > thresholds[:, None]).long()  # [N_s, n_thresholds]
        tp = (preds_bin * targets).sum(1)  # [n_thredholds]
        fp = (preds_bin * (1 - targets)).sum(1)  # [n_thresholds]
        precisions = tp / (tp + fp + 1e-8)  # [n_thresholds]
        return precisions.mean()

    def compute(self):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            res[f'{subgroup}_ap'] = self.compute_subgroup(i)
        return res


class TPR_at_FPR(Metric):
    """True positive rate at x% FPR."""
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str], xfpr: float = 0.05):
        super().__init__()
        assert 0 <= xfpr <= 1
        self.xfpr = xfpr
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    def update(self, preds: Tensor, targets: Tensor):
        """
        preds: Tensor of anomaly scores and sub-group labels of shape [b, 2]
               (anomaly score, group-label)
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets)
        self.preds.append(preds)
        self.targets.append(targets)

    def compute_subgroup(self, subgroup: int):
        preds = torch.cat(self.preds)  # [N, 2]
        targets = torch.cat(self.targets)  # [N]
        if targets.sum() == 0:
            return 0
        # Compute FPR threshold for total dataset
        fpr, _, thresholds = roc_curve(targets, preds[:, 0], pos_label=1)
        threshold = thresholds[np.argwhere(fpr < self.xfpr)[-1, 0]]
        # Compute TPR for subgroup
        targets = targets[preds[:, 1] == subgroup]  # [N_s]
        preds = preds[preds[:, 1] == subgroup, 0]  # [N_s]
        preds_binary = (preds > threshold).long()  # [N_s]
        tpr = (preds_binary * targets).sum() / (targets.sum() + 1e-8)
        return tpr

    def compute(self):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            res[f'{subgroup}_tpr@{self.xfpr}'] = self.compute_subgroup(i)
        return res


class FPR_at_TPR(Metric):
    """False positive rate at x% TPR."""
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str], xtpr: float = 0.95):
        super().__init__()
        assert 0 <= xtpr <= 1
        self.xtpr = xtpr
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List

    def update(self, preds: Tensor, targets: Tensor):
        assert len(preds) == len(targets)
        self.preds.append(preds)
        self.targets.append(targets)

    def compute_subgroup(self, subgroup: int):
        preds = torch.cat(self.preds)  # [N, 2]
        targets = torch.cat(self.targets)  # [N]
        if targets.sum() == 0:
            return 0
        # Compute TPR threshold for total dataset
        _, tpr, thresholds = roc_curve(targets, preds[:, 0], pos_label=1)
        threshold = thresholds[np.argwhere(tpr > self.xtpr)[0, 0]]
        # Compute TPR for subgroup
        targets = targets[preds[:, 1] == subgroup]  # [N_s]
        preds = preds[preds[:, 1] == subgroup, 0]  # [N_s]
        preds_binary = (preds > threshold).long()  # [N_s]
        fpr = (preds_binary * (1 - targets)).sum() / ((1 - targets).sum() + 1e-8)
        return fpr

    def compute(self):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            res[f'{subgroup}_fpr@{self.xtpr}'] = self.compute_subgroup(i)
        return res


class cDC(Metric):
    """Continuous Dice coefficient as in: https://arxiv.org/pdf/1906.11031.pdf"""
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List

    def update(self, preds: Tensor, targets: Tensor):
        assert len(preds) == len(targets)
        self.preds.append(preds)
        self.targets.append(targets)

    def compute_subgroup(self, subgroup: int):
        preds = torch.cat(self.preds)  # [N, 2]
        targets = torch.cat(self.targets)  # [N]
        if targets.sum() == 0:
            return 0
        # Normalize preds for full dataset
        min_score = preds[:, 0].min()
        max_score = preds[:, 0].quantile(0.99, interpolation='lower')  # Ignore outliers
        preds[:, 0] = (preds[:, 0] - min_score) / (max_score - min_score)
        # Filter relevant subgroup
        targets = targets[preds[:, 1] == subgroup]  # [N_s]
        preds = preds[preds[:, 1] == subgroup, 0]  # [N_s]
        # Compute cDC
        anb = (targets * preds).sum()  # Eq. 2
        a = targets.sum()  # Eq. 3
        b = preds.sum()  # Eq. 4
        c = (targets * preds).sum() / targets[preds > 0].sum()  # Eq. 6
        cDC = anb / (c * a + b)  # Eq. 5
        return cDC

    def compute(self):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            res[f'{subgroup}_cDC'] = self.compute_subgroup(i)
        return res


class AverageDSC(Metric):
    """
    Computes the average Dice similarity coefficient for subgroups of the data
    but min and max scores are taken from the entire dataset.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List

    def update(self, preds: Tensor, targets: Tensor):
        """
        preds: Tensor of anomaly scores and sub-group labels of shape [b, 2]
               (anomaly score, group-label)
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets)
        self.preds.append(preds)
        self.targets.append(targets)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    def compute_subgroup(self, subgroup: int):
        preds = torch.cat(self.preds)  # [N, 2]
        targets = torch.cat(self.targets)  # [N]
        if targets.sum() == 0:
            return 0
        # Compute min and max score and thresholds for whole dataset
        min_score = preds[:, 0].min()
        max_score = preds[:, 0].quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute average DSC for subgroup
        targets = targets[preds[:, 1] == subgroup]  # [N_s]
        preds = preds[preds[:, 1] == subgroup, 0]  # [N_s]
        preds_bin = (preds > thresholds[:, None]).long()  # [N_s, n_thresholds]
        tp = (preds_bin * targets).sum(1)  # [n_thredholds]
        p = preds_bin.sum(1)
        t = targets.sum()
        DSCs = (2 * tp) / (p + t + 1e-8)  # [n_thresholds]
        return DSCs.mean()

    def compute(self):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            res[f'{subgroup}_aDSC'] = self.compute_subgroup(i)
        return res


class AvgMeter:
    def __init__(self):
        self.reset()
        self.value: float
        self.n: int

    def reset(self):
        self.value = 0.0
        self.n = 0

    def add(self, value):
        self.value += value
        self.n += 1

    def compute(self):
        return self.value / self.n


class AvgDictMeter:
    def __init__(self):
        self.reset()
        self.values: dict
        self.n: int

    def reset(self):
        self.values = defaultdict(float)
        self.n = 0

    def add(self, values: dict):
        for key, value in values.items():
            self.values[key] += value
        self.n += 1

    def compute(self):
        return {key: value / self.n for key, value in self.values.items()}
