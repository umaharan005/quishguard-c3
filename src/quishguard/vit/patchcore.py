"""Module C1, version 2: PatchCore anomaly detection on Vision Transformer features.

Why this replaces the autoencoder as the main visual model:
the reconstruction autoencoder (v1) learned to redraw ANY 16x16 patch, logos
and stickers included, because QR content is random and gives it no strong
"normal" prior (test AUROC 0.69). PatchCore (Roth et al., CVPR 2022) keeps the
same principle, learning only from normal codes, but compares features instead
of pixels:

  1. A pretrained ViT (DINO self-supervised ViT-S/16) turns each 16x16 patch
     into a feature vector, taken from middle layers so it keeps local
     structure (edges, grid alignment, grey levels) and global context
     (attention across the whole code).
  2. Features of patches from NORMAL training codes are stored in a memory
     bank, thinned with greedy coreset sampling.
  3. At test time each patch gets the distance to its nearest normal patch.
     Tampered areas (sticker edges, logo, overlay, broken grid) are far from
     anything normal. The 14x14 distance map is the heatmap; the mean of the
     worst patches is the anomaly score.

No tampered image is ever used for fitting.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 14


class PatchCoreViT(nn.Module):
    def __init__(self, backbone: str = "vit_small_patch16_224.dino", layers=(3, 6, 9),
                 proj_dim: int = 256, pretrained: bool = True):
        super().__init__()
        import timm
        try:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        except Exception:  # fallback if DINO weights are unavailable
            self.backbone = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
        self.backbone.eval().requires_grad_(False)
        self.layers = tuple(layers)
        d = self.backbone.embed_dim * len(self.layers)
        g = torch.Generator().manual_seed(0)
        # fixed random projection: keeps distances (Johnson-Lindenstrauss), 4.5x less memory
        self.register_buffer("proj", torch.randn(d, proj_dim, generator=g) / proj_dim ** 0.5)
        self.register_buffer("memory", torch.zeros(0, proj_dim))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: B x 1 x 224 x 224 in [0,1] -> B x 196 x proj_dim"""
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        bb = self.backbone
        h = bb.patch_embed(x)
        h = bb._pos_embed(h)
        h = bb.norm_pre(h) if hasattr(bb, "norm_pre") else h
        n_prefix = getattr(bb, "num_prefix_tokens", 1)
        feats = []
        for i, blk in enumerate(bb.blocks):
            h = blk(h)
            if i in self.layers:
                feats.append(h[:, n_prefix:])
            if i >= max(self.layers):
                break
        f = torch.cat(feats, dim=-1)                                   # B x 196 x (384*3)
        b = f.shape[0]
        # 3x3 neighbourhood average (PatchCore "locally aware" features)
        f = f.transpose(1, 2).reshape(b, -1, GRID, GRID)
        f = F.avg_pool2d(f, 3, stride=1, padding=1, count_include_pad=False)
        f = f.reshape(b, -1, GRID * GRID).transpose(1, 2)
        return f.float() @ self.proj

    @torch.no_grad()
    def fit(self, loader, device, max_images: int = 1500, coreset_ratio: float = 0.1, log=print):
        bank, n = [], 0
        for x in loader:
            x = x.to(device)
            bank.append(self.features(x).reshape(-1, self.proj.shape[1]).cpu())
            n += x.shape[0]
            if n >= max_images:
                break
        bank = torch.cat(bank)
        m = max(1000, int(len(bank) * coreset_ratio))
        log(f"PatchCore: {n} normal images -> {len(bank):,} patch features -> coreset {m:,}")
        self.memory = greedy_coreset(bank.to(device), m, log=log)
        return self

    @torch.no_grad()
    def patch_scores(self, x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """B x 1 x 224 x 224 -> B x 14 x 14 distance to the nearest normal patch."""
        f = self.features(x)
        b, p, d = f.shape
        q = f.reshape(-1, d)
        best = torch.empty(q.shape[0], device=q.device)
        for i in range(0, q.shape[0], chunk):
            dist = torch.cdist(q[i:i + chunk], self.memory)
            best[i:i + chunk] = dist.min(dim=1).values
        return best.reshape(b, GRID, GRID)


@torch.no_grad()
def greedy_coreset(x: torch.Tensor, m: int, log=print) -> torch.Tensor:
    """Greedy k-centre selection (as in PatchCore): picks m points that cover the bank."""
    n = x.shape[0]
    if m >= n:
        return x
    idx = [int(torch.randint(n, (1,)).item())]
    mind = torch.cdist(x, x[idx]).squeeze(1)
    for k in range(1, m):
        i = int(torch.argmax(mind))
        idx.append(i)
        mind = torch.minimum(mind, torch.cdist(x, x[i:i + 1]).squeeze(1))
        if k % 5000 == 0:
            log(f"  coreset {k:,}/{m:,}")
    return x[idx].contiguous()
