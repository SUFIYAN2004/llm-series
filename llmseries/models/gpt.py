"""GPT (decoder-only)

Decoder-only Transformer: LayerNorm, GELU MLP, learned positional embeddings, causal self-attention. The baseline everything is judged against. Learned positions cannot extrapolate past 256 tokens.

Reference: Vaswani et al. 2017; Radford et al. 2019
"""
import torch.nn as nn
from ..layers import Attn, Block, GELUMLP, LM


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, Attn(d, H), GELUMLP(d), nn.LayerNorm) for _ in range(8)],
              pos=True, norm=nn.LayerNorm)
