"""
Bootstrapping method from https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
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
        'AUROC': AUROC(subgroup_names),
        'subgroupAUROC': SubgroupAUROC(subgroup_names),
        'AveragePrecision': AveragePrecision(subgroup_names),
        'meanPrecision': MeanPrecision(subgroup_names),
        'cDC': cDC(subgroup_names),
        'aDSC': AverageDSC(subgroup_names),
        'tpr@5fpr': TPR_at_FPR(subgroup_names, xfpr=0.05),
        'fpr@95tpr': FPR_at_TPR(subgroup_names, xtpr=0.95),
        'avg_anomaly_score': AvgAnomalyScore(subgroup_names),
        'DSC@EER': DSC_at_EER(subgroup_names),
        'upperDSC': UpperDSC(subgroup_names),
    })
    return classification_metrics


class AvgAnomalyScore(Metric):
    """
    Just a wrapper to bootstrap the anomaly score if necessary
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Optional[Tensor] = None):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(subgroups)
        self.preds.append(preds)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, subgroups: Tensor, subgroup: int):
        anomaly_score = preds[subgroups == subgroup].mean()
        if anomaly_score.isnan():
            anomaly_score = torch.tensor(0.)
        return anomaly_score

    @staticmethod
    def compute_overall(preds: Tensor):
        anomaly_score = preds.mean()
        if anomaly_score.isnan():
            anomaly_score = torch.tensor(0.)
        return anomaly_score

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, subgroups, subgroup)
            res[f'{subgroup_name}_anomaly_score'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds)
        res[f'{common_string_left(self.subgroup_names)}anomaly_score'] = result
        return res


class AUROC(Metric):
    """
    Computes the AUROC naively for each subgroup of the data individually.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)

        # Filter relevant subgroup
        subgroup_preds = preds[subgroups == subgroup]
        subgroup_targets = targets[subgroups == subgroup]

        # Compute the area under the ROC curve
        auroc = roc_auc_score(subgroup_targets, subgroup_preds)

        return torch.tensor(auroc, dtype=torch.float32)

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        auroc = roc_auc_score(targets, preds)
        return torch.tensor(auroc, dtype=torch.float32)

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_AUROC'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets)
        res[f'{common_string_left(self.subgroup_names)}AUROC'] = result
        return res


class SubgroupAUROC(Metric):
    """
    Computes the AUROC for each subgroup of the data individually.
    The TPRs are computed from each subgroup individually, while the FPRs are
    computed from the whole dataset.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)

        # Sort predictions and targets by descending prediction score
        sorted_indices = torch.argsort(preds, descending=True)
        sorted_targets = targets[sorted_indices]
        sorted_subgroups = subgroups[sorted_indices]

        # Compute the false positive rates for the whole dataset
        negatives_all = (sorted_targets == 0).sum()
        false_positives_all = torch.cumsum(sorted_targets == 0, dim=0)
        fpr = false_positives_all / (negatives_all + 1e-7)

        # Compute the true positive rates for the subgroup
        # Points on the curve where the tpr for the subgroup doesn't change
        # are kept constant
        positives_subgroup = sorted_targets[sorted_subgroups == subgroup].sum()
        true_positives_subgroup = torch.where(sorted_subgroups == subgroup, sorted_targets, 0).cumsum(dim=0)
        tpr = true_positives_subgroup / (positives_subgroup + 1e-7)

        # Insert thresholds of min_val and max_val to ensure the ROC curve starts at (0, 0) and ends at (1, 1)
        tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
        fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])

        # Compute the area under the ROC curve
        auroc = (((tpr[1:] - tpr[:-1]) / 2 + tpr[:-1]) * (fpr[1:] - fpr[:-1])).sum()

        return auroc

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_subgroupAUROC'] = result
        return res


class AveragePrecision(Metric):
    """
    Computes the Average Precision (AP) naively for each subgroup of the data individually.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)

        # Filter relevant subgroup
        subgroup_preds = preds[subgroups == subgroup]
        subgroup_targets = targets[subgroups == subgroup]

        # Compute the Average Precision
        ap = average_precision_score(subgroup_targets, subgroup_preds)

        return torch.tensor(ap, dtype=torch.float32)

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        ap = average_precision_score(targets, preds)
        return torch.tensor(ap, dtype=torch.float32)

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_AP'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets)
        res[f'{common_string_left(self.subgroup_names)}AP'] = result
        return res


class MeanPrecision(Metric):
    """
    Computes the mean precision across equally spaced thresholds for
    subgroups of the data with min and max scores are taken from the entire
    dataset.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute min and max score and thresholds for whole dataset
        min_score = preds.min()
        max_score = preds.quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute mean precision for subgroup
        subgroup_targets = targets[subgroups == subgroup, None]  # [N, 1]
        subgroup_preds = preds[subgroups == subgroup, None]  # [N, 1]
        subgroup_preds_bin = (subgroup_preds > thresholds).long()  # [N, n_thresholds]
        tp = (subgroup_preds_bin * subgroup_targets).sum(0)  # [n_thresholds]
        fp = (subgroup_preds_bin * (1 - subgroup_targets)).sum(0)  # [n_thresholds]
        precisions = tp / (tp + fp + 1e-8)  # [n_thresholds]
        return precisions.mean()

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute min and max score and thresholds
        min_score = preds.min()
        max_score = preds.quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute mean precision
        targets = targets.clone()[:, None]  # [N, 1]
        preds = preds.clone()[:, None]  # [N, 1]
        preds_bin = (preds > thresholds).long()  # [N, n_thresholds]
        tp = (preds_bin * targets).sum(0)  # [n_thresholds]
        fp = (preds_bin * (1 - targets)).sum(0)  # [n_thresholds]
        precisions = tp / (tp + fp + 1e-8)  # [n_thresholds]
        return precisions.mean()

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_meanPrecision'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets)
        res[f'{common_string_left(self.subgroup_names)}meanPrecision'] = result
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
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor,
                         subgroup: int, xfpr: float):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute FPR threshold for total dataset
        fpr, _, thresholds = roc_curve(targets, preds, pos_label=1)
        threshold_idx = np.argwhere(fpr < xfpr)
        if len(threshold_idx) == 0:
            threshold_idx = -1
        else:
            threshold_idx = threshold_idx[-1, 0]
        threshold = thresholds[threshold_idx]
        # Compute TPR for subgroup
        subgroup_targets = targets[subgroups == subgroup]  # [N_s]
        subgroup_preds = preds[subgroups == subgroup]  # [N_s]
        subgroup_preds_bin = (subgroup_preds > threshold).long()  # [N_s]
        tpr = (subgroup_preds_bin * subgroup_targets).sum() / (subgroup_targets.sum() + 1e-8)
        return tpr

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor, xfpr: float):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute FPR threshold for total dataset
        fpr, tpr, _ = roc_curve(targets, preds, pos_label=1)
        tpr_idx = np.argwhere(fpr < xfpr)
        if len(tpr_idx) == 0:
            tpr_idx = -1
        else:
            tpr_idx = tpr_idx[-1, 0]
        return torch.tensor(tpr[tpr_idx], dtype=torch.float32)

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup, self.xfpr)
            res[f'{subgroup_name}_tpr@{self.xfpr}'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets, self.xfpr)
        res[f'{common_string_left(self.subgroup_names)}tpr@{self.xfpr}'] = result
        return res


class FPR_at_TPR(Metric):
    """False positive rate at x% TPR."""
    is_differentiable: bool = False
    higher_is_better: bool = False

    def __init__(self, subgroup_names: List[str], xtpr: float = 0.95):
        super().__init__()
        assert 0 <= xtpr <= 1
        self.xtpr = xtpr
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor,
                         subgroup: int, xtpr: float):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(1.)
        # Compute TPR threshold for total dataset
        _, tpr, thresholds = roc_curve(targets, preds, pos_label=1)
        threshold = thresholds[np.argwhere(tpr > xtpr)[0, 0]]
        # Compute FPR for subgroup
        subgroup_targets = targets[subgroups == subgroup]  # [N_s]
        subgroup_preds = preds[subgroups == subgroup]  # [N_s]
        subgroup_preds_bin = (subgroup_preds > threshold).long()  # [N_s]
        fpr = (subgroup_preds_bin * (1 - subgroup_targets)).sum() / ((1 - subgroup_targets).sum() + 1e-8)
        return fpr

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor, xtpr: float):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(1.)
        fpr, tpr, _ = roc_curve(targets, preds, pos_label=1)
        fpr_idx = np.argwhere(tpr > xtpr)[0, 0]
        return torch.tensor(fpr[fpr_idx], dtype=torch.float32)

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup, self.xtpr)
            res[f'{subgroup_name}_fpr@{self.xtpr}'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets, self.xtpr)
        res[f'{common_string_left(self.subgroup_names)}fpr@{self.xtpr}'] = result
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
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Normalize preds for full dataset
        min_score = preds.min()
        max_score = preds.quantile(0.99, interpolation='lower')  # Ignore outliers
        preds_norm = (preds - min_score) / (max_score - min_score)
        # Filter relevant subgroup
        subgroup_targets = targets[subgroups == subgroup]  # [N_s]
        subgroup_preds = preds_norm[subgroups == subgroup]  # [N_s]
        # Compute cDC
        anb = (subgroup_targets * subgroup_preds).sum()  # Eq. 2
        a = subgroup_targets.sum()  # Eq. 3
        b = subgroup_preds.sum()  # Eq. 4
        c = (subgroup_targets * subgroup_preds).sum() / (subgroup_targets[subgroup_preds > 0].sum() + 1e-7)  # Eq. 6
        cDC = anb / (c * a + b + 1e-7)  # Eq. 5
        return cDC

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Normalize preds
        min_score = preds.min()
        max_score = preds.quantile(0.99, interpolation='lower')  # Ignore outliers
        preds_norm = (preds - min_score) / (max_score - min_score)
        targets = targets.clone()
        # Compute cDC
        anb = (targets * preds_norm).sum()  # Eq. 2
        a = targets.sum()  # Eq. 3
        b = preds_norm.sum()  # Eq. 4
        c = (targets * preds_norm).sum() / (targets[preds_norm > 0].sum() + 1e-7)  # Eq. 6
        cDC = anb / (c * a + b + 1e-7)  # Eq. 5
        return cDC

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_cDC'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets)
        res[f'{common_string_left(self.subgroup_names)}cDC'] = result
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
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute min and max score and thresholds for whole dataset
        min_score = preds.min()
        max_score = preds.quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute average DSC for subgroup
        subgroup_targets = targets[subgroups == subgroup, None]  # [N, 1]
        subgroup_preds = preds[subgroups == subgroup, None]  # [N, 1]
        subgroup_preds_bin = (subgroup_preds > thresholds).long()  # [N, n_thresholds]
        tp = (subgroup_preds_bin * subgroup_targets).sum(0)  # [n_thredholds]
        p = subgroup_preds_bin.sum(0)  # [n_thresholds]
        t = subgroup_targets.sum()
        DSCs = (2 * tp) / (p + t + 1e-8)  # [n_thresholds]
        return DSCs.mean()

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute min and max score and thresholds
        min_score = preds.min()
        max_score = preds.quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute average DSC
        targets = targets.clone()[:, None]  # [N, 1]
        preds = preds.clone()[:, None]  # [N, 1]
        preds_bin = (preds > thresholds).long()  # [N, n_thresholds]
        tp = (preds_bin * targets).sum(0)  # [n_thredholds]
        p = preds_bin.sum(0)  # [n_thresholds]
        t = targets.sum()
        DSCs = (2 * tp) / (p + t + 1e-8)  # [n_thresholds]
        return DSCs.mean()

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_aDSC'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets)
        res[f'{common_string_left(self.subgroup_names)}aDSC'] = result
        return res


class UpperDSC(Metric):
    """
    Computes the Dice similarity coefficient for subgroups of the data
    at the point where the DSC is maximal for the whole dataset.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute min and max score and thresholds for whole dataset
        min_score = preds.min()
        max_score = preds.quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute maximum DSC for whole dataset
        preds_bin = (preds[:, None] > thresholds).long()  # [N, n_thresholds]
        tp = (preds_bin * targets[:, None]).sum(0)  # [n_thredholds]
        p = preds_bin.sum(0)  # [n_thresholds]
        t = targets.sum()
        DSCs = (2 * tp) / (p + t + 1e-8)  # [n_thresholds]
        # Compute DSC for subgroup at maximum DSC
        max_threshold = thresholds[DSCs.argmax()]
        subgroup_preds = preds[subgroups == subgroup]  # [N]
        subgroup_targets = targets[subgroups == subgroup]  # [N]
        subgroup_preds_bin = (subgroup_preds > max_threshold).long()  # [N]
        tp_sub = (subgroup_preds_bin * subgroup_targets).sum()
        p_sub = subgroup_preds_bin.sum()
        t_sub = subgroup_targets.sum()
        DSC_sub = (2 * tp_sub) / (p_sub + t_sub + 1e-8)
        return DSC_sub

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute min and max score and thresholds
        min_score = preds.min()
        max_score = preds.quantile(0.99, interpolation='lower')  # Ignore outliers
        thresholds = torch.linspace(min_score, max_score, 1001)
        # Compute maximum DSC
        preds_bin = (preds[:, None] > thresholds).long()  # [N, n_thresholds]
        tp = (preds_bin * targets[:, None]).sum(0)  # [n_thredholds]
        p = preds_bin.sum(0)  # [n_thresholds]
        t = targets.sum()
        DSCs = (2 * tp) / (p + t + 1e-8)  # [n_thresholds]
        # Compute DSC for subgroup at maximum DSC
        max_threshold = thresholds[DSCs.argmax()]
        preds_ = preds.clone()
        targets_ = targets.clone()
        preds_bin = (preds_ > max_threshold).long()  # [N]
        tp_sub = (preds_bin * targets_).sum()
        p_sub = preds_bin.sum()
        t_sub = targets_.sum()
        DSC_sub = (2 * tp_sub) / (p_sub + t_sub + 1e-8)
        return DSC_sub

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_upperDSC'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets)
        res[f'{common_string_left(self.subgroup_names)}upperDSC'] = result
        return res


class DSC_at_EER(Metric):
    """DSC-score at equal error rate (where FPR and FNR are the same)."""
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor,
                         subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute EER for total dataset
        fpr, tpr, thresholds = roc_curve(targets, preds, pos_label=1)
        fnr = 1 - tpr
        threshold_idx = np.argmin(np.abs(fpr - fnr))
        threshold = thresholds[threshold_idx]
        # Compute F1 for subgroup
        subgroup_targets = targets[subgroups == subgroup]  # [N_s]
        subgroup_preds = preds[subgroups == subgroup]  # [N_s]
        subgroup_preds_bin = (subgroup_preds > threshold).long()  # [N_s]
        tp = (subgroup_preds_bin * subgroup_targets).sum()
        fp = (subgroup_preds_bin * (1 - subgroup_targets)).sum()
        fn = ((1 - subgroup_preds_bin) * subgroup_targets).sum()
        dsc = 2 * tp / (2 * tp + (fp + fn) + 1e-8)
        return dsc

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute EER for total dataset
        fpr, tpr, thresholds = roc_curve(targets, preds, pos_label=1)
        fnr = 1 - tpr
        threshold_idx = np.argmin(np.abs(fpr - fnr))
        threshold = thresholds[threshold_idx]
        # Compute F1 for subgroup
        targets_ = targets.clone()  # [N_s]
        preds_ = preds.clone()  # [N_s]
        preds_bin = (preds_ > threshold).long()  # [N_s]
        tp = (preds_bin * targets_).sum()
        fp = (preds_bin * (1 - targets_)).sum()
        fn = ((1 - preds_bin) * targets_).sum()
        dsc = 2 * tp / (2 * tp + (fp + fn) + 1e-8)
        return dsc

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_DSC@EER'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets)
        res[f'{common_string_left(self.subgroup_names)}DSC@EER'] = result
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


def common_string_left(strings: List[str]) -> str:
    """Returns the longest common string of all strings from the left."""
    if len(strings) == 0:
        return ''
    if len(strings) == 1:
        return strings[0]
    common = ''
    for i in range(len(strings[0])):
        if all([s.startswith(strings[0][:i + 1]) for s in strings]):
            common = strings[0][:i + 1]
        else:
            break
    return common


if __name__ == '__main__':
    subgroup_names = ['subgroup1', 'subgroup2']
    metrics = MyMetricCollection({
        'avg_anomaly_score': AvgAnomalyScore(subgroup_names),
        'AUROC': AUROC(subgroup_names),
        'subgroupAUROC': SubgroupAUROC(subgroup_names),
        'AveragePrecision': AveragePrecision(subgroup_names),
        'meanPrecision': MeanPrecision(subgroup_names),
        'DSC@EER': DSC_at_EER(subgroup_names),
        'tpr@5fpr': TPR_at_FPR(subgroup_names, xfpr=0.05),
        'fpr@5tpr': FPR_at_TPR(subgroup_names, xtpr=0.95),
        'cDC': cDC(subgroup_names),
        'aDSC': AverageDSC(subgroup_names),
        'upperDSC': UpperDSC(subgroup_names),
    })

    scores = torch.tensor([0.8, 0.6, 0.2, 0.9, 0.5, 0.7, 0.3])
    labels = torch.tensor([1, 0, 0, 1, 1, 0, 1])
    subgroups = torch.tensor([0, 1, 0, 0, 1, 1, 0])
    metrics.update(subgroups, scores, labels)
    print(metrics.compute())
