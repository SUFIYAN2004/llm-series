"""Hybrid Mamba+Attn

Mamba layers with a full-attention layer every 4th layer plus MLPs. Tests whether a little attention repairs the recall weaknesses of pure SSMs.

Reference: Jamba 2024; Samba 2024
"""
from ..layers import Attn, Block, LM, SwiGLU
from .mamba import Mamba


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    mix = lambda i: Attn(d, H, kvh, rope=True) if i % 4 == 3 else Mamba(d)
    return LM(d, [Block(d, mix(i), SwiGLU(d)) for i in range(8)], ckpt=True)
