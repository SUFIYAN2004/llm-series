"""MoE Transformer

Transformer whose FFN is a top-2 router over 8 SwiGLU experts with a load-balancing loss. 10M is TOTAL parameters; active parameters per token are lower (stored in the results JSON).

Reference: Shazeer 2017; Fedus 2021; Jiang 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Attn, Block, LM


class MoE(nn.Module):  # top-2 of 8 SwiGLU experts, Switch-style load-balance loss
    def __init__(s, d, h, E=8, k=2):
        super().__init__(); s.E, s.k = E, k; s.router = nn.Linear(d, E, bias=False)
        s.w1 = nn.Parameter(torch.randn(E, d, h) * 0.02); s.w3 = nn.Parameter(torch.randn(E, d, h) * 0.02)
        s.w2 = nn.Parameter(torch.randn(E, h, d) * 0.02); s.aux = torch.zeros(())
    def forward(s, x):
        B, T, D = x.shape; xf = x.reshape(-1, D)
        probs = F.softmax(s.router(xf).float(), -1); topv, topi = probs.topk(s.k, -1)
        topv = topv / topv.sum(-1, keepdim=True); out = torch.zeros_like(xf)
        for e in range(s.E):
            rows, slot = (topi == e).nonzero(as_tuple=True)
            if rows.numel() == 0: continue
            xe = xf[rows]; ye = (F.silu(xe @ s.w1[e]) * (xe @ s.w3[e])) @ s.w2[e]
            out.index_add_(0, rows, (ye * topv[rows, slot, None]).to(out.dtype))
        f = F.one_hot(topi[:, 0], s.E).float().mean(0); s.aux = s.E * (f * probs.mean(0)).sum()
        return out.view(B, T, D)


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, Attn(d, H, kvh, rope=True), MoE(d, d)) for _ in range(8)])
