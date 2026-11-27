"""Clip augmentations (Supplementary Note 5). Input/output: float [M, 3, T, H, W]
in [0, 1]; random parameters are drawn per clip and shared by its frames."""
import random

import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
LUM = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1, 1)


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def _u(x, lo=-1.0, hi=1.0, c=1):
    return torch.empty(x.size(0), c, 1, 1, 1, device=x.device).uniform_(lo, hi)


def _lum(x):
    return (x * LUM.to(x.device)).sum(1, keepdim=True)


def brightness(x, m=0.4):
    return x * (1 + m * _u(x))


def contrast(x, m=0.4):
    g = _lum(x).mean((2, 3, 4), keepdim=True)
    return (x - g) * (1 + m * _u(x)) + g


def saturation(x, m=0.4):
    y = _lum(x)
    return y + (1 + m * _u(x)) * (x - y)


def hue(x, m=0.1):
    a = _u(x) * m * 2 * torch.pi
    cos, sin = a.cos(), a.sin()
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b
    i, q = i * cos - q * sin, i * sin + q * cos
    return torch.cat([y + 0.956 * i + 0.621 * q, y - 0.272 * i - 0.647 * q,
                      y - 1.106 * i + 1.703 * q], 1)


def white_balance(x, m=0.3):
    return x * (1 + m * _u(x, c=3))


def gamma(x, lo=0.7, hi=1.5):
    return x.clamp_min(1e-6).pow(_u(x, lo, hi))


def grayscale(x, p=0.2):
    do = (torch.rand(x.size(0), device=x.device) < p).view(-1, 1, 1, 1, 1)
    return torch.where(do, _lum(x).expand_as(x), x)


def blur(x, p=0.5, lo=2.0, hi=3.5):
    sel = (torch.rand(x.size(0), device=x.device) < p).nonzero()[:, 0]
    if sel.numel() == 0:
        return x
    M, C, T, H, W = x.shape
    xb = x[sel].transpose(1, 2).reshape(-1, C, H, W)
    xb = F.interpolate(xb, scale_factor=1 / random.uniform(lo, hi),
                       mode="bilinear", align_corners=False)
    xb = F.interpolate(xb, size=(H, W), mode="bilinear", align_corners=False)
    out = x.clone()
    out[sel] = xb.reshape(-1, T, C, H, W).transpose(1, 2)
    return out


def hflip(x, p=0.5):
    do = (torch.rand(x.size(0), device=x.device) < p).view(-1, 1, 1, 1, 1)
    return torch.where(do, x.flip(-1), x)


def resized_crop(x, scale=(0.85, 1.0)):
    M, C, T, H, W = x.shape
    s = torch.empty(M, device=x.device).uniform_(*scale).sqrt()
    theta = torch.zeros(M, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = s
    theta[:, :, 2] = (1 - s)[:, None] * (torch.rand(M, 2, device=x.device) * 2 - 1)
    grid = F.affine_grid(theta, (M, C, H, W), align_corners=False)
    xt = x.transpose(1, 2).reshape(M * T, C, H, W)
    xt = F.grid_sample(xt, grid.repeat_interleave(T, 0), mode="bilinear",
                       padding_mode="reflection", align_corners=False)
    return xt.reshape(M, T, C, H, W).transpose(1, 2)


def query_view(x):
    x = resized_crop(hflip(x))
    x = hue(saturation(contrast(brightness(x, 0.3), 0.3), 0.3), 0.05)
    x = gamma(white_balance(x, 0.2).clamp(0, 1), 0.85, 1.2)
    return (x + 0.02 * torch.randn_like(x)).clamp(0, 1)


KEY_OPS = (brightness, contrast, saturation, hue, white_balance, gamma, blur)


def key_view(x, n_ops=2):
    x = hflip(x)
    for op in random.sample(KEY_OPS, n_ops):
        x = op(x)
    return grayscale(x.clamp(0, 1), p=0.1)
