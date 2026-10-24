"""Griffin (RG-LRU + local attn)

Gated linear recurrence (RG-LRU) layers mixed with local sliding-window attention in a 2:1 ratio.

Reference: De et al. 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Attn, Block, LM, SwiGLU, lin_scan, nocast


class RGLRU(nn.Module):  # Griffin/Hawk recurrent block
    def __init__(s, d, conv=4):
        super().__init__()
        s.gate = nn.Linear(d, d); s.inp = nn.Linear(d, d); s.conv = nn.Conv1d(d, d, conv, groups=d, padding=conv - 1)
        s.wa = nn.Linear(d, d); s.wx = nn.Linear(d, d); s.out = nn.Linear(d, d)
        s.lam = nn.Parameter(torch.logit((torch.rand(d) * 0.099 + 0.9) ** (1 / 8)))
    def forward(s, x):
        T = x.shape[1]
        g = F.gelu(s.gate(x)); h = s.conv(s.inp(x).transpose(1, 2))[..., :T].transpose(1, 2)
        with nocast():
            h = h.float(); r = torch.sigmoid(s.wa(h)); i = torch.sigmoid(s.wx(h))
            a = torch.exp(8 * r * F.logsigmoid(s.lam))
            y = lin_scan(a, torch.sqrt(torch.clamp(1 - a * a, min=1e-6)) * (i * h))
        return s.out(y * g)


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    mix = lambda i: Attn(d, H, kvh, rope=True, window=128) if i % 3 == 2 else RGLRU(d)
    return LM(d, [Block(d, mix(i), SwiGLU(d)) for i in range(9)])
