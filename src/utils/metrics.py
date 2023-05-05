from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities.data import _flatten_dict


class MyMetricCollection(MetricCollection):
    def __init__(
            self,
            metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
    ) -> None:
        super().__init__(metrics)

    def compute(self, **kwargs) -> Dict[str, Any]:
        """Compute the result for each metric in the collection."""
        res = {k: m.compute(**kwargs) for k, m in self.items(keep_base=True, copy_state=False)}
        res = _flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}


def build_metrics(subgroup_names: List[str]) -> MyMetricCollection:
    classification_metrics = MyMetricCollection({
        'ap': AveragePrecision(subgroup_names),
        'cDC': cDC(subgroup_names),
        'aDSC': AverageDSC(subgroup_names),
        'tpr@5fpr': TPR_at_FPR(subgroup_names, xfpr=0.05),
        'fpr@95tpr': FPR_at_TPR(subgroup_names, xtpr=0.95),
        'anomaly_score': AnomalyScore(subgroup_names),
    })
    return classification_metrics


class AnomalyScore(Metric):
    """
    Just a wrapper to bootstrap the anomaly score if necessary
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

    def update(self, preds: Tensor, targets: Optional[Tensor] = None):
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

    def compute_subgroup(self, subgroup: int, do_bootstrap: bool):
        preds = torch.cat(self.preds)  # [N, 2]
        preds = preds[preds[:, 1] == subgroup, 0]  # [N_s]
        if do_bootstrap:
            fake_targets = torch.zeros_like(preds)
            anomaly_score, lower, upper = bootstrap(preds, fake_targets, lambda x, y: x.mean())
            return anomaly_score, lower, upper
        else:
            anomaly_score = preds.mean()
            return anomaly_score

    def compute(self, bootstrap: bool = False, **kwargs):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            if bootstrap:
                anomaly_score, lower, upper = self.compute_subgroup(i, bootstrap)
                res[f'{subgroup}_anomaly_score'] = anomaly_score
                res[f'{subgroup}_anomaly_score_lower'] = lower
                res[f'{subgroup}_anomaly_score_upper'] = upper
            else:
                res[f'{subgroup}_anomaly_score'] = self.compute_subgroup(i, bootstrap)
        return res


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

    @staticmethod
    def compute_ap(preds_bin: Tensor, targets: Tensor):
        tp = (preds_bin * targets).sum(0)  # [n_thresholds]
        fp = (preds_bin * (1 - targets)).sum(0)  # [n_thresholds]
        precisions = tp / (tp + fp + 1e-8)  # [n_thresholds]
        return precisions.mean()

    def compute_subgroup(self, subgroup: int, do_bootstrap: bool):
        preds = torch.cat(self.preds)  # [N, 2]
        targets = torch.cat(self.targets)  # [N]
        if targets.sum() == 0:
            return 0
        # Compute min and max score and thresholds for whole dataset
        min_score = preds[:, 0].min()
        max_score = preds[:, 0].quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute average precision for subgroup
        targets = targets[preds[:, 1] == subgroup, None]  # [N_s, 1]
        preds = preds[preds[:, 1] == subgroup, 0]  # [N_s]
        preds_bin = (preds[:, None] > thresholds).long()  # [N_s, n_thresholds]
        if do_bootstrap:
            ap, lower, upper = bootstrap(preds_bin, targets, self.compute_ap)
            lower = lower.clamp_min(0)
            upper = upper.clamp_max(1)
            return ap, lower, upper
        else:
            ap = self.compute_ap(preds_bin, targets)
            return ap

    def compute(self, bootstrap: bool = False, **kwargs):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            if bootstrap:
                ap, lower, upper = self.compute_subgroup(i, bootstrap)
                res[f'{subgroup}_ap'] = ap
                res[f'{subgroup}_ap_lower'] = lower
                res[f'{subgroup}_ap_upper'] = upper
            else:
                res[f'{subgroup}_ap'] = self.compute_subgroup(i, bootstrap)
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

    @staticmethod
    def compute_tpr(preds_bin: Tensor, targets: Tensor):
        tpr = (preds_bin * targets).sum() / (targets.sum() + 1e-8)
        return tpr

    def update(self, preds: Tensor, targets: Tensor):
        """
        preds: Tensor of anomaly scores and sub-group labels of shape [b, 2]
               (anomaly score, group-label)
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets)
        self.preds.append(preds)
        self.targets.append(targets)

    def compute_subgroup(self, subgroup: int, do_bootstrap: bool):
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
        preds_bin = (preds > threshold).long()  # [N_s]
        if do_bootstrap:
            tpr, lower, upper = bootstrap(preds_bin, targets, self.compute_tpr)
            lower = lower.clamp_min(0)
            upper = upper.clamp_max(1)
            return tpr, lower, upper
        else:
            tpr = self.compute_tpr(preds_bin, targets)
            return tpr

    def compute(self, bootstrap: bool = False, **kwargs):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            if bootstrap:
                tpr, lower, upper = self.compute_subgroup(i, bootstrap)
                res[f'{subgroup}_tpr@{self.xfpr}'] = tpr
                res[f'{subgroup}_tpr@{self.xfpr}_lower'] = lower
                res[f'{subgroup}_tpr@{self.xfpr}_upper'] = upper
            else:
                res[f'{subgroup}_tpr@{self.xfpr}'] = self.compute_subgroup(i, bootstrap)
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

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_fpr(preds_bin: Tensor, targets: Tensor):
        fpr = (preds_bin * (1 - targets)).sum() / ((1 - targets).sum() + 1e-8)
        return fpr

    def compute_subgroup(self, subgroup: int, do_bootstrap: bool):
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
        preds_bin = (preds > threshold).long()  # [N_s]
        if do_bootstrap:
            fpr, lower, upper = bootstrap(preds_bin, targets, self.compute_fpr)
            lower = lower.clamp_min(0)
            upper = upper.clamp_max(1)
            return fpr, lower, upper
        else:
            fpr = self.compute_fpr(preds_bin, targets)
            return fpr

    def compute(self, bootstrap: bool = False, **kwargs):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            if bootstrap:
                tpr, lower, upper = self.compute_subgroup(i, bootstrap)
                res[f'{subgroup}_fpr@{self.xtpr}'] = tpr
                res[f'{subgroup}_fpr@{self.xtpr}_lower'] = lower
                res[f'{subgroup}_fpr@{self.xtpr}_upper'] = upper
            else:
                res[f'{subgroup}_fpr@{self.xtpr}'] = self.compute_subgroup(i, bootstrap)
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

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_cDC(preds: Tensor, targets: Tensor):
        anb = (targets * preds).sum()  # Eq. 2
        a = targets.sum()  # Eq. 3
        b = preds.sum()  # Eq. 4
        c = (targets * preds).sum() / targets[preds > 0].sum()  # Eq. 6
        cDC = anb / (c * a + b)  # Eq. 5
        return cDC

    def compute_subgroup(self, subgroup: int, do_bootstrap: bool):
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
        if do_bootstrap:
            cDC, lower, upper = bootstrap(preds, targets, self.compute_cDC)
            lower = lower.clamp_min(0)
            upper = upper.clamp_max(1)
            return cDC, lower, upper
        else:
            cDC = self.compute_cDC(preds, targets)
            return cDC

    def compute(self, bootstrap: bool = False, **kwargs):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            if bootstrap:
                cDC, lower, upper = self.compute_subgroup(i, bootstrap)
                res[f'{subgroup}_cDC'] = cDC
                res[f'{subgroup}_cDC_lower'] = lower
                res[f'{subgroup}_cDC_upper'] = upper
            else:
                res[f'{subgroup}_cDC'] = self.compute_subgroup(i, bootstrap)
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

    @staticmethod
    def compute_aDSC(preds_bin: Tensor, targets: Tensor):
        tp = (preds_bin * targets).sum(1)  # [n_thredholds]
        p = preds_bin.sum(1)
        t = targets.sum()
        DSCs = (2 * tp) / (p + t + 1e-8)  # [n_thresholds]
        return DSCs.mean()

    def compute_subgroup(self, subgroup: int, do_bootstrap: bool):
        preds = torch.cat(self.preds)  # [N, 2]
        targets = torch.cat(self.targets)  # [N]
        if targets.sum() == 0:
            return 0
        # Compute min and max score and thresholds for whole dataset
        min_score = preds[:, 0].min()
        max_score = preds[:, 0].quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute average DSC for subgroup
        targets = targets[preds[:, 1] == subgroup, None]  # [N_s, 1]
        preds = preds[preds[:, 1] == subgroup, 0]  # [N_s]
        preds_bin = (preds[:, None] > thresholds).long()  # [N_s, n_thresholds]
        if do_bootstrap:
            aDSC, lower, upper = bootstrap(preds_bin, targets, self.compute_aDSC)
            lower = lower.clamp_min(0)
            upper = upper.clamp_max(1)
            return aDSC, lower, upper
        else:
            aDSC = self.compute_aDSC(preds_bin, targets)
            return aDSC

    def compute(self, bootstrap: bool = False, **kwargs):
        res = {}
        for i, subgroup in enumerate(self.subgroup_names):
            if bootstrap:
                cDC, lower, upper = self.compute_subgroup(i, bootstrap)
                res[f'{subgroup}_aDSC'] = cDC
                res[f'{subgroup}_aDSC_lower'] = lower
                res[f'{subgroup}_aDSC_upper'] = upper
            else:
                res[f'{subgroup}_aDSC'] = self.compute_subgroup(i, bootstrap)
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


def bootstrap(preds: Tensor, targets: Tensor, metric_fn: Callable, n_bootstrap: int = 250):
    """Computes the confidence interval of a metric using bootstrapping
    of the predictions.

    :param preds: Tensor of predicted values of shape [b]
    :param targets: Tensor of target values of shape [b]
    :param metric_fn: Metric function that takes preds and targets as input
    :param n_bootstrap: Number of bootstrap iterations
    """
    b = len(preds)
    rng = torch.Generator().manual_seed(2147483647)
    idx = torch.arange(b)

    metrics = []
    for _ in range(n_bootstrap):
        pred_idx = idx[torch.randint(b, size=(b,), generator=rng)]  # Sample with replacement
        metric_boot = metric_fn(preds[pred_idx], targets[pred_idx])
        metrics.append(metric_boot)

    metrics = torch.stack(metrics)
    mean = metrics.mean()
    lower = torch.quantile(metrics, 0.025, interpolation='lower')
    upper = torch.quantile(metrics, 0.975, interpolation='higher')
    return mean, lower, upper
