"""xLSTM (mLSTM)

mLSTM variant of xLSTM: matrix memory, exponential input gate with stabilizer, parallel form. sLSTM blocks are not included.

Reference: Beck et al. 2024
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Block, LM, RMSNorm, SwiGLU, nocast


class MLSTM(nn.Module):  # xLSTM matrix-memory cell, parallel form with exp input gate + stabilizer
    def __init__(s, d, H):
        super().__init__(); s.H = H
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.gates = nn.Linear(d, 2 * H)
        with torch.no_grad(): s.gates.bias[H:] = 3.0
        s.g = nn.Linear(d, d, bias=False); s.o = nn.Linear(d, d, bias=False); s.norm = RMSNorm(d // H)
    def forward(s, x):
        B, T, D = x.shape; H, hd = s.H, D // s.H
        q, k, v = s.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        gp = s.gates(x).float().transpose(1, 2); ig, lf = gp[:, :H], F.logsigmoid(gp[:, H:])
        with nocast():
            cum = torch.cumsum(lf, -1); mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            logD = (cum[:, :, :, None] - cum[:, :, None, :] + ig[:, :, None, :]).masked_fill(~mask, -float("inf"))
            m = logD.max(-1, keepdim=True).values
            S = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd) * torch.exp(logD - m)
            n = torch.maximum(S.sum(-1, keepdim=True).abs(), torch.exp(-m))
            y = (S / n) @ v.float()
        y = s.norm(y).transpose(1, 2).reshape(B, T, D)
        return s.o(y * F.silu(s.g(x)))


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, MLSTM(d, H), SwiGLU(d)) for _ in range(8)])
