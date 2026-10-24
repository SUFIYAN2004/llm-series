"""Mamba (S6)

Selective SSM: B, C and the step size depend on the input so the model chooses what to remember. Pure PyTorch parallel scan (no CUDA kernel). No MLP blocks, 16 layers.

Reference: Gu & Dao 2023
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Block, LM, lin_scan, nocast


def dt_bias_init(n):
    dt = torch.exp(torch.rand(n) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
    return dt + torch.log(-torch.expm1(-dt))


class Mamba(nn.Module):  # selective SSM (S6), parallel scan
    def __init__(s, d, E=2, N=16, conv=4):
        super().__init__(); s.Ei, s.N = E * d, N; s.dtr = math.ceil(d / 16)
        s.in_proj = nn.Linear(d, 2 * s.Ei, bias=False); s.conv = nn.Conv1d(s.Ei, s.Ei, conv, groups=s.Ei, padding=conv - 1)
        s.x_proj = nn.Linear(s.Ei, s.dtr + 2 * N, bias=False); s.dt_proj = nn.Linear(s.dtr, s.Ei)
        with torch.no_grad(): s.dt_proj.bias.copy_(dt_bias_init(s.Ei))
        s.A_log = nn.Parameter(torch.log(torch.arange(1, N + 1).float()).repeat(s.Ei, 1))
        s.D = nn.Parameter(torch.ones(s.Ei)); s.out_proj = nn.Linear(s.Ei, d, bias=False)
    def forward(s, x):
        B, T, _ = x.shape
        xi, z = s.in_proj(x).chunk(2, -1)
        xi = F.silu(s.conv(xi.transpose(1, 2))[..., :T]).transpose(1, 2)
        dt, Bm, Cm = s.x_proj(xi).split([s.dtr, s.N, s.N], -1)
        with nocast():
            dt = F.softplus(s.dt_proj(dt.float())); A = -torch.exp(s.A_log); xf = xi.float()
            a = torch.exp(dt[..., None] * A)
            b = dt[..., None] * Bm.float()[:, :, None, :] * xf[..., None]
            h = lin_scan(a, b)
            y = (h * Cm.float()[:, :, None, :]).sum(-1) + s.D * xf
        return s.out_proj(y * F.silu(z.float()))


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, Mamba(d), None) for _ in range(16)], ckpt=True)
