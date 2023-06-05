"""
Abnormality Detection in Chest X-ray Images Using Uncertainty Prediction Autoencoders
http://www.isee-ai.cn/~wangruixuan/files/MICCAI2020_1.pdf
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Reshape(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.size)


def build_encoder(in_channels: int, hidden_dims: List[int],
                  use_batchnorm: bool = True, dropout: float = 0.0) -> nn.Module:
    encoder = nn.Sequential()
    for i, h_dim in enumerate(hidden_dims):
        layer = nn.Sequential()

        # Convolution
        layer.add_module(f"encoder_conv_{i}",
                         nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2,
                                   padding=1, bias=False))

        # Batch normalization
        if use_batchnorm:
            layer.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm2d(h_dim))

        # LeakyReLU
        layer.add_module(f"encoder_leakyrelu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"encoder_dropout_{i}", nn.Dropout(p=dropout))

        # Add layer to encoder
        encoder.add_module(f"encoder_layer_{i}", layer)

        in_channels = h_dim

    return encoder


def build_decoder(out_channels: int, hidden_dims: List[int],
                  use_batchnorm: bool = True, dropout: float = 0.0) -> nn.Module:
    h_dims = [out_channels] + hidden_dims

    dec = nn.Sequential()
    for i in range(len(h_dims) - 1, 0, -1):
        # Add a new layer
        layer = nn.Sequential()

        # Upsample
        layer.add_module(f"decoder_upsample_{i}", nn.Upsample(scale_factor=2))

        # Convolution
        layer.add_module(f"decoder_conv_{i}",
                         nn.Conv2d(h_dims[i], h_dims[i - 1], kernel_size=3,
                                   padding=1, bias=False))

        # Batch normalization
        if use_batchnorm:
            layer.add_module(f"decoder_batchnorm_{i}",
                             nn.BatchNorm2d(h_dims[i - 1]))

        # LeakyReLU
        layer.add_module(f"decoder_leakyrelu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"decoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the decoder
        dec.add_module(f"decoder_layer_{i}", layer)

    # Final layer
    dec.add_module("decoder_conv_final",
                   nn.Conv2d(h_dims[0], out_channels, 1, bias=False))

    return dec


class UncertaintyAE(nn.Module):
    """
    A n-layer variational autoencoder
    adapted from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    """
    def __init__(self, config):
        super().__init__()

        # Unpack config
        img_size = config.img_size
        latent_dim = config.latent_dim
        hidden_dims = config.uae_hidden_dims
        use_batchnorm = config.use_batchnorm if "use_batchnorm" in config else True
        dropout = config.dropout if "dropout" in config else 0.0

        intermediate_res = img_size // 2 ** len(hidden_dims)
        intermediate_feats = intermediate_res * intermediate_res * hidden_dims[-1]

        # Build encoder
        self.encoder = build_encoder(1, hidden_dims, use_batchnorm, dropout)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(intermediate_feats, latent_dim, bias=False),
        )
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, intermediate_feats, bias=False),
            Reshape((-1, hidden_dims[-1], intermediate_res, intermediate_res)),
        )

        # Build decoder
        self.decoder = build_decoder(2, hidden_dims, use_batchnorm, dropout)

    def forward(self, inp: Tensor) -> Tensor:
        res = self.encoder(inp)
        z = self.bottleneck(res)
        decoder_inp = self.decoder_input(z)
        y = self.decoder(decoder_inp)
        mu, logvar = y.chunk(2, 1)
        return mu, logvar

    def save(self, path: str):
        """Save the model weights"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load weights"""
        self.load_state_dict(torch.load(path))

    def loss(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        mu, logvar = self.forward(x)
        rec_err = (mu - x).pow(2)
        aleatoric_map = torch.exp(-logvar) * rec_err
        rec_loss = rec_err.sqrt().mean()
        loss = (aleatoric_map + logvar).mean()
        return {'loss': loss, 'rec_loss': rec_loss}

    def predict_anomaly(self, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        # Forward pass
        mu, logvar = self.forward(x)

        # Anomaly map
        anomaly_map = torch.exp(-logvar) * (mu - x).pow(2)  # (N, C, H, W)

        # Anomaly score is the mean of the top 10 quantile of the anomaly map
        anomaly_score = anomaly_map.flatten(1).topk(
            int(0.1 * anomaly_map[0].numel()), largest=True).values.mean(1)  # (N,)

        return anomaly_map, anomaly_score
