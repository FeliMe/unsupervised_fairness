from collections import defaultdict

import numpy as np
import torch

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import Tensor
from torchmetrics import Metric, MetricCollection


def build_metrics() -> MetricCollection:
    classification_metrics = MetricCollection({
        'auroc': AUROC(),
        'ap': AveragePrecision(),
        'tpr@5fpr': TPR_at_FPR(xfpr=0.05),
        'fpr@95tpr': FPR_at_TPR(xtpr=0.95),
    })
    return classification_metrics


class AUROC(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, targets: Tensor):
        assert preds.shape == targets.shape
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds).view(-1)
        targets = torch.cat(self.targets).view(-1)
        return roc_auc_score(targets, preds)


class AveragePrecision(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, targets: Tensor):
        assert preds.shape == targets.shape
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds).view(-1)
        targets = torch.cat(self.targets).view(-1)
        return average_precision_score(targets, preds)


class TPR_at_FPR(Metric):
    """True positive rate at x% FPR."""
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, xfpr: float = 0.05):
        super().__init__()
        assert 0 <= xfpr <= 1
        self.xfpr = xfpr
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, targets: Tensor):
        assert preds.shape == targets.shape
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds).view(-1)
        targets = torch.cat(self.targets).view(-1)
        fpr, tpr, _ = roc_curve(targets, preds, pos_label=1)
        idx = np.argwhere(fpr < self.xfpr)[-1, 0]
        return tpr[idx]


class FPR_at_TPR(Metric):
    """False positive rate at x% TPR."""
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, xtpr: float = 0.95):
        super().__init__()
        assert 0 <= xtpr <= 1
        self.xtpr = xtpr
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, targets: Tensor):
        assert preds.shape == targets.shape
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds).view(-1)
        targets = torch.cat(self.targets).view(-1)
        fpr, tpr, _ = roc_curve(targets, preds, pos_label=1)
        idx = np.argwhere(tpr > self.xtpr)[0, 0]
        return fpr[idx]



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
