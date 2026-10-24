"""Neural n-gram (Bengio)

Embeddings + fixed 8-token window + MLP. Bridge from counting to neural LMs; anything beyond 8 tokens is invisible.

Reference: Bengio et al. 2003
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import V
from ..layers import init_weights


class NNLM(nn.Module):  # Bengio 2003: fixed window of embeddings -> MLP
    def __init__(s, d, ctx=8):
        super().__init__(); s.ctx = ctx; s.emb = nn.Embedding(V, d); s.h = nn.Linear(ctx * d, 2 * d); s.out = nn.Linear(2 * d, V); s.apply(init_weights)
    def forward(s, idx):
        B, T = idx.shape; x = F.pad(s.emb(idx), (0, 0, s.ctx - 1, 0)).unfold(1, s.ctx, 1)
        return s.out(torch.tanh(s.h(x.reshape(B, T, -1))))


def build(d):
    return NNLM(d)
