"""Modern Transformer (RoPE/RMS/SwiGLU/GQA)

Transformer with LLaMA-style upgrades: RoPE, RMSNorm, SwiGLU, grouped-query attention. Compare directly with GPT.

Reference: Su 2021; Zhang & Sennrich 2019; Shazeer 2020; Ainslie 2023
"""
from ..layers import Attn, Block, LM, SwiGLU


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, Attn(d, H, kvh, rope=True), SwiGLU(d)) for _ in range(8)])
