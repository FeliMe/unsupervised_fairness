from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(512, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet(x)[:, 0]

    def loss(self, x: Tensor, y: Tensor, **kwargs) -> Dict[str, Tensor]:
        pred = self(x)
        loss = F.binary_cross_entropy_with_logits(pred, y.float())
        return {'loss': loss}

    def predict_anomaly(self, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            pred = self(x)
        anomaly_scores = torch.sigmoid(pred)
        return None, anomaly_scores  # No anomaly map for ResNet18

    def save(self, path: str):
        """Save the model weights"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load weights"""
        self.load_state_dict(torch.load(path))
