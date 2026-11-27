"""Video Swin-T + LoRA, trunk, per-criterion instance classifiers and projector."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import Swin3D_T_Weights, swin3d_t
from torchvision.models.video.swin_transformer import SwinTransformerBlock


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base, self.scaling = base.requires_grad_(False), alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    # torchvision's Swin attention reads `.weight` / `.bias` directly
    @property
    def weight(self):
        return self.base.weight + self.scaling * self.B @ self.A

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.t() @ self.B.t() * self.scaling


class Checkpointed(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


def _wrap(module, r, alpha, dropout, prefix=""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}"
        if isinstance(child, nn.Linear) and any(t in full for t in ("qkv", "proj", "mlp")):
            setattr(module, name, LoRALinear(child, r, alpha, dropout))
        else:
            _wrap(child, r, alpha, dropout, full)
            if isinstance(child, SwinTransformerBlock):
                setattr(module, name, Checkpointed(child))


class Encoder(nn.Module):
    def __init__(self, n_criteria, K, proj_dim=128, lora_r=16, lora_alpha=32,
                 lora_dropout=0.3):
        super().__init__()
        self.backbone = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        self.backbone.head = nn.Identity()
        self.backbone.requires_grad_(False)
        _wrap(self.backbone.features, lora_r, lora_alpha, lora_dropout)
        self.trunk = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256),
                                   nn.GELU(), nn.Linear(256, 256))
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, K)) for _ in range(n_criteria)])
        self.proj_base = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.proj_heads = nn.ModuleList([nn.Linear(256, proj_dim)
                                         for _ in range(n_criteria)])

    def latent(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.trunk(self.backbone(x)).float()

    def classify(self, h):
        return [f(h) for f in self.classifiers]

    def project(self, h):
        b = self.proj_base(h)
        return [F.normalize(p(b), dim=1) for p in self.proj_heads]
