"""Gated DeltaNet

Gated delta rule: a matrix-valued state is decayed, then corrected toward each new value (error-driven write). Implemented as a sequential loop, so it is the slowest model to train.

Reference: Schlag et al. 2021; Yang et al. 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Block, LM, RMSNorm, SwiGLU, nocast


class GatedDelta(nn.Module):  # gated delta rule, recurrent form (sequential loop)
    def __init__(s, d, H):
        super().__init__(); s.H = H
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.gates = nn.Linear(d, 2 * H)
        s.g = nn.Linear(d, d, bias=False); s.o = nn.Linear(d, d, bias=False); s.norm = RMSNorm(d // H)
    def forward(s, x):
        B, T, D = x.shape; H, hd = s.H, D // s.H
        q, k, v = s.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        bg = s.gates(x).float().transpose(1, 2)
        beta, alpha = torch.sigmoid(bg[:, :H]), torch.sigmoid(bg[:, H:] + 3.0)
        with nocast():
            q = F.normalize(F.silu(q.float()), dim=-1); k = F.normalize(F.silu(k.float()), dim=-1); v = v.float()
            S = torch.zeros(B, H, hd, hd, device=x.device); outs = []
            for t in range(T):
                kt = k[:, :, t]
                S = S * alpha[:, :, t, None, None]
                pred = (S * kt[..., None]).sum(2)
                S = S + beta[:, :, t, None, None] * kt[..., None] * (v[:, :, t] - pred)[:, :, None, :]
                outs.append((S * q[:, :, t, :, None]).sum(2))
            y = torch.stack(outs, 2)
        y = s.norm(y).transpose(1, 2).reshape(B, T, D)
        return s.o(y * F.silu(s.g(x)))


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, GatedDelta(d, H), SwiGLU(d)) for _ in range(6)], ckpt=True)
