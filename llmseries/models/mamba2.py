"""Mamba-2 (SSD)

State Space Duality: a scalar decay per head makes the SSM equal to a masked attention-like product. Implemented in the quadratic (dual) form.

Reference: Dao & Gu 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Block, LM, RMSNorm, nocast
from .mamba import dt_bias_init


class Mamba2(nn.Module):  # SSD, quadratic (attention-like) form, scalar decay per head
    def __init__(s, d, E=2, N=32, P=32, conv=4):
        super().__init__(); s.Ei, s.N, s.P = E * d, N, P; s.H = s.Ei // P
        s.in_proj = nn.Linear(d, 2 * s.Ei + 2 * N + s.H, bias=False); cd = s.Ei + 2 * N
        s.conv = nn.Conv1d(cd, cd, conv, groups=cd, padding=conv - 1)
        s.dt_bias = nn.Parameter(dt_bias_init(s.H)); s.A_log = nn.Parameter(torch.log(torch.rand(s.H) * 15 + 1))
        s.D = nn.Parameter(torch.ones(s.H)); s.norm = RMSNorm(s.Ei); s.out = nn.Linear(s.Ei, d, bias=False)
    def forward(s, x):
        B, T, _ = x.shape; H, P, N, Ei = s.H, s.P, s.N, s.Ei
        z, xBC, dt = s.in_proj(x).split([Ei, Ei + 2 * N, H], -1)
        xBC = F.silu(s.conv(xBC.transpose(1, 2))[..., :T]).transpose(1, 2)
        xi, Bm, Cm = xBC.split([Ei, N, N], -1)
        with nocast():
            dt = F.softplus(dt.float() + s.dt_bias); A = -torch.exp(s.A_log)
            cum = torch.cumsum(dt * A, 1).transpose(1, 2)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            L = torch.exp((cum[:, :, :, None] - cum[:, :, None, :]).masked_fill(~mask, -float("inf")))
            G = Cm.float() @ Bm.float().transpose(1, 2)
            xh = xi.float().view(B, T, H, P).transpose(1, 2)
            y = (G[:, None] * L) @ (xh * dt.transpose(1, 2)[..., None]) + s.D[None, :, None, None] * xh
            y = y.transpose(1, 2).reshape(B, T, Ei)
            y = s.norm(y * F.silu(z.float()))
        return s.out(y)


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, Mamba2(d), None) for _ in range(12)], ckpt=True)
