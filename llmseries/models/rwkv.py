"""RWKV

RWKV-4 style token-shift, WKV time mixing (exponentially decaying weighted average) and channel mixing. Trains in parallel, runs as an RNN. WKV uses an exact parallel scan.

Reference: Peng et al. 2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Block, LM, lin_scan, nocast


class RWKVTime(nn.Module):  # RWKV-4 style WKV, computed with a parallel linear scan
    def __init__(s, d):
        super().__init__(); s.mu = nn.Parameter(torch.full((3, d), 0.5))
        s.k = nn.Linear(d, d, bias=False); s.v = nn.Linear(d, d, bias=False); s.r = nn.Linear(d, d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.w = nn.Parameter(torch.linspace(-5, -0.5, d)); s.u = nn.Parameter(torch.zeros(d))
    def forward(s, x):
        xs = F.pad(x, (0, 0, 1, -1)); m = s.mu
        xk, xv, xr = x * m[0] + xs * (1 - m[0]), x * m[1] + xs * (1 - m[1]), x * m[2] + xs * (1 - m[2])
        k = 15 * torch.tanh(s.k(xk).float() / 15); v = s.v(xv).float(); r = torch.sigmoid(s.r(xr))
        with nocast():
            a = torch.exp(-torch.exp(s.w.float()))[None, None].expand_as(k); ek = torch.exp(k)
            S = lin_scan(torch.cat([a, a], -1), torch.cat([ek * v, ek], -1))
            num, den = F.pad(S, (0, 0, 1, -1)).chunk(2, -1)   # state up to t-1
            eu = torch.exp(s.u + k); wkv = (num + eu * v) / (den + eu)
        return s.o(r * wkv.to(r.dtype))


class RWKVChan(nn.Module):
    def __init__(s, d):
        super().__init__(); s.mu = nn.Parameter(torch.full((2, d), 0.5))
        s.k = nn.Linear(d, 4 * d, bias=False); s.v = nn.Linear(4 * d, d, bias=False); s.r = nn.Linear(d, d, bias=False)
    def forward(s, x):
        xs = F.pad(x, (0, 0, 1, -1)); m = s.mu
        xk, xr = x * m[0] + xs * (1 - m[0]), x * m[1] + xs * (1 - m[1])
        return torch.sigmoid(s.r(xr)) * s.v(F.relu(s.k(xk)) ** 2)


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, RWKVTime(d), RWKVChan(d)) for _ in range(8)])
