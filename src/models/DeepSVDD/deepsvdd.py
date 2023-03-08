from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GlobalAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=(2, 3))


class DeepSVDD(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.repr_dim = config.repr_dim

        # from src.models.DeepSVDD.resnet import resnet18
        # self.encoder = resnet18(pretrained=False)
        # self.encoder.fc = nn.Linear(512, self.repr_dim, bias=False)

        # Input size: 1 x 128 x 128
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 16, 16)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256, 8, 8)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=False),
            nn.ReLU(inplace=True),
            GlobalAvgPool(),
            nn.Linear(512, self.repr_dim, bias=False),
        )

        # hypersphere center c
        self.register_buffer('c', torch.randn(1, self.repr_dim) + 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if x.shape[1] == 1:
        #     x = x.repeat(1, 3, 1, 1)
        x = self.encoder(x)
        return x

    def loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute DeepSVDD loss."""
        pred = self.forward(x)
        loss = one_class_scores(pred, self.c).mean()
        return {
            'loss': loss,
            'oc_loss': loss,
        }

    def predict_anomaly(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute DeepSVDD loss."""
        with torch.no_grad():
            pred = self.forward(x)
        anomaly_scores = one_class_scores(pred, self.c)
        return None, anomaly_scores  # No anomaly map for DeepSVDD

    def save(self, path: str):
        """Save the model weights"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load weights"""
        self.load_state_dict(torch.load(path))


def one_class_scores(pred: Tensor, c: Tensor) -> Tensor:
    """Compute anomaly_score for the one-class objective."""
    return torch.sum((pred - c) ** 2, dim=1)  # (N,)


if __name__ == '__main__':
    from argparse import Namespace
    config = Namespace(
        repr_dim=128
    )
    model = DeepSVDD(config)
    model.predict_anomaly(torch.randn(2, 1, 128, 128))
