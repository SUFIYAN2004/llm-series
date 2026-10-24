"""Vanilla RNN / LSTM / GRU

Elman RNN (tanh), 2 layers. Recurrence and BPTT; expect the weakest neural model because of vanishing gradients.

Reference: Elman 1990
"""
import torch.nn as nn
from ..config import V
from ..layers import init_weights


class RNNLM(nn.Module):
    def __init__(s, d, kind, layers=2):
        super().__init__(); s.emb = nn.Embedding(V, d)
        s.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind](d, d, layers, batch_first=True)
        s.head = nn.Linear(d, V, bias=False); s.apply(init_weights); s.head.weight = s.emb.weight
    def forward(s, idx): return s.head(s.rnn(s.emb(idx))[0])


def build(d, kind='rnn'):
    """kind in {'rnn','lstm','gru'}"""
    return RNNLM(d, kind)
