"""GRU + Bahdanau attn

2-layer GRU plus additive attention over all previous hidden states. Simplification: attention sits on top of the GRU instead of inside the decoder loop. O(T^2) memory.

Reference: Bahdanau et al. 2014
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import V
from ..layers import init_weights


class BahdanauLM(nn.Module):  # GRU + additive (Bahdanau) attention over all previous hidden states
    def __init__(s, d, A=32):
        super().__init__(); s.emb = nn.Embedding(V, d); s.gru = nn.GRU(d, d, 2, batch_first=True)
        s.wq = nn.Linear(d, A, bias=False); s.wk = nn.Linear(d, A); s.v = nn.Linear(A, 1, bias=False)
        s.mix = nn.Linear(2 * d, d); s.head = nn.Linear(d, V, bias=False); s.apply(init_weights); s.head.weight = s.emb.weight
    def forward(s, idx):
        h = s.gru(s.emb(idx))[0]; T = h.shape[1]
        e = s.v(torch.tanh(s.wq(h)[:, :, None, :] + s.wk(h)[:, None, :, :])).squeeze(-1)
        e = e.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=h.device).tril(), -float("inf"))
        ctx = F.softmax(e.float(), -1).to(h.dtype) @ h
        return s.head(torch.tanh(s.mix(torch.cat([h, ctx], -1))))


def build(d):
    return BahdanauLM(d)
