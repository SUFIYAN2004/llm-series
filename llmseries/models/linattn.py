"""Linear attention

Softmax replaced by an elu+1 feature map so attention equals an RNN with fixed-size state. Training uses the parallel masked form (still O(T^2) compute); the recurrent form is what gives O(T) inference.

Reference: Katharopoulos et al. 2020
"""
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Block, LM, SwiGLU, nocast


class LinAttn(nn.Module):  # Katharopoulos 2020, parallel (masked) form: elu+1 feature map
    def __init__(s, d, H): super().__init__(); s.H = H; s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.H, D // s.H).permute(2, 0, 3, 1, 4)
        with nocast():
            q, k, v = F.elu(q.float()) + 1, F.elu(k.float()) + 1, v.float()
            a = (q @ k.transpose(-1, -2)).tril(); y = (a @ v) / (a.sum(-1, keepdim=True) + 1e-6)
        return s.o(y.transpose(1, 2).reshape(B, T, D))


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, LinAttn(d, H), SwiGLU(d)) for _ in range(8)])
