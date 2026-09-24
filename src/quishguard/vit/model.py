"""Module C1 models: ViT autoencoder (main) and CNN autoencoder (baseline).

Idea (VT-ADL / AnoViT style): train ONLY on normal QR codes to rebuild the
input. A normal code is rebuilt well. A tampered area (sticker edge, logo,
overlay, broken grid) was never seen in training, so it is rebuilt badly.
The per-patch rebuild error is the anomaly map; the mean of the worst
patches is the anomaly score.

A narrow bottleneck (32 numbers per patch) stops the network from simply
copying the input, which would make it rebuild tampering too.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 16
GRID = 14  # 224 / 16


class ViTAutoencoder(nn.Module):
    def __init__(self, encoder: str = "vit_tiny_patch16_224", pretrained: bool = True,
                 bottleneck: int = 32, dec_depth: int = 4, dec_dim: int = 192):
        super().__init__()
        import timm
        from timm.models.vision_transformer import Block

        self.encoder = timm.create_model(encoder, pretrained=pretrained, in_chans=1, num_classes=0)
        d = self.encoder.embed_dim
        self.to_code = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, bottleneck))
        self.from_code = nn.Linear(bottleneck, dec_dim)
        self.dec_pos = nn.Parameter(torch.zeros(1, GRID * GRID, dec_dim))
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        self.decoder = nn.Sequential(*[Block(dec_dim, num_heads=3, mlp_ratio=4.0) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.to_pixels = nn.Linear(dec_dim, PATCH * PATCH)
        self.register_buffer("mean", torch.tensor(0.5))
        self.register_buffer("std", torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B x 1 x 224 x 224 in [0,1]
        tokens = self.encoder.forward_features((x - self.mean) / self.std)
        tokens = tokens[:, getattr(self.encoder, "num_prefix_tokens", 1):]      # drop CLS
        z = self.from_code(self.to_code(tokens)) + self.dec_pos
        z = self.dec_norm(self.decoder(z))
        patches = torch.sigmoid(self.to_pixels(z))                              # B x 196 x 256
        return unpatchify(patches)


class CNNAutoencoder(nn.Module):
    """Baseline with a similar bottleneck (32 channels on a 14 x 14 grid)."""

    def __init__(self, bottleneck: int = 32):
        super().__init__()
        def down(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.GELU())
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.GELU())
        self.enc = nn.Sequential(down(1, 32), down(32, 64), down(64, 128), down(128, 192),
                                 nn.Conv2d(192, bottleneck, 1))
        self.dec = nn.Sequential(nn.Conv2d(bottleneck, 192, 1), nn.GELU(), up(192, 128), up(128, 64),
                                 up(64, 32), nn.ConvTranspose2d(32, 1, 4, 2, 1))

    def forward(self, x):
        return torch.sigmoid(self.dec(self.enc(x * 2 - 1)))


def unpatchify(p: torch.Tensor) -> torch.Tensor:
    b = p.shape[0]
    p = p.reshape(b, GRID, GRID, PATCH, PATCH).permute(0, 1, 3, 2, 4)
    return p.reshape(b, 1, GRID * PATCH, GRID * PATCH)


def patch_errors(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Mean squared error per 16 x 16 patch -> B x 14 x 14."""
    e = (x - recon) ** 2
    return F.avg_pool2d(e, PATCH).squeeze(1)


def anomaly_score(pe: torch.Tensor, top_frac: float = 0.05) -> torch.Tensor:
    """Mean of the worst `top_frac` patches (≈10 of 196). Small local tampering
    such as a sticker edge still raises the score; one noisy patch does not."""
    flat = pe.flatten(1)
    k = max(1, int(round(flat.shape[1] * top_frac)))
    return flat.topk(k, dim=1).values.mean(1)


def build(arch: str, pretrained: bool = True) -> nn.Module:
    if arch == "vit":
        return ViTAutoencoder(pretrained=pretrained)
    if arch == "cnn":
        return CNNAutoencoder()
    raise ValueError(arch)
