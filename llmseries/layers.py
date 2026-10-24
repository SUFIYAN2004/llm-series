"""Shared building blocks: norms, MLPs, the residual Block, the generic LM wrapper,
parallel linear scan, FFT convolution, RoPE and (multi/grouped-query, optionally windowed) attention."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import V, SEQ


def nocast(): return torch.autocast("cuda", enabled=False)


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)


class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6): super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + s.eps)).type_as(x) * s.w


class SwiGLU(nn.Module):
    def __init__(s, d, h=None):
        super().__init__(); h = h or int(8 * d / 3) // 32 * 32
        s.w13 = nn.Linear(d, 2 * h, bias=False); s.w2 = nn.Linear(h, d, bias=False)
    def forward(s, x): a, b = s.w13(x).chunk(2, -1); return s.w2(F.silu(a) * b)


class GELUMLP(nn.Module):
    def __init__(s, d): super().__init__(); s.fc = nn.Linear(d, 4 * d); s.proj = nn.Linear(4 * d, d)
    def forward(s, x): return s.proj(F.gelu(s.fc(x)))


class Block(nn.Module):
    def __init__(s, d, mixer, ffn=None, norm=RMSNorm):
        super().__init__(); s.n1 = norm(d); s.mixer = mixer
        s.n2 = norm(d) if ffn is not None else None; s.ffn = ffn
    def forward(s, x):
        x = x + s.mixer(s.n1(x))
        return x + s.ffn(s.n2(x)) if s.ffn is not None else x


class LM(nn.Module):
    def __init__(s, d, blocks, pos=False, ckpt=False, norm=RMSNorm):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos = nn.Embedding(SEQ, d) if pos else None
        s.blocks = nn.ModuleList(blocks); s.norm = norm(d); s.ckpt = ckpt
        s.head = nn.Linear(d, V, bias=False); s.apply(init_weights); s.head.weight = s.emb.weight
    def forward(s, idx):
        x = s.emb(idx)
        if s.pos is not None: x = x + s.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in s.blocks:
            x = checkpoint(b, x, use_reentrant=False) if (s.ckpt and s.training) else b(x)
        return s.head(s.norm(x))


def lin_scan(a, b):
    """h_t = a_t*h_{t-1} + b_t along dim 1 (Hillis-Steele parallel scan, O(T log T), exact)."""
    T, d = a.shape[1], 1
    while d < T:
        b = torch.cat([b[:, :d], b[:, d:] + a[:, d:] * b[:, :-d]], 1)
        a = torch.cat([a[:, :d], a[:, d:] * a[:, :-d]], 1); d *= 2
    return b


def fftconv(u, k, D):  # u (B,C,T), k (C,T), D (C,)
    T = u.shape[-1]; u = u.float()
    y = torch.fft.irfft(torch.fft.rfft(u, n=2 * T) * torch.fft.rfft(k.float(), n=2 * T), n=2 * T)[..., :T]
    return y + u * D[None, :, None]


def rope(x, base=10000.0):  # x (B,H,T,hd)
    T, hd = x.shape[2], x.shape[3]
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    ang = torch.outer(torch.arange(T, device=x.device).float(), inv)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attn(nn.Module):
    def __init__(s, d, H, kv=None, rope=False, window=None):
        super().__init__(); s.H, s.kv, s.hd, s.rope, s.window = H, kv or H, d // H, rope, window
        s.qkv = nn.Linear(d, (H + 2 * s.kv) * s.hd, bias=False); s.o = nn.Linear(d, d, bias=False)
    def forward(s, x):
        B, T, _ = x.shape; H, kv, hd = s.H, s.kv, s.hd
        q, k, v = s.qkv(x).split([H * hd, kv * hd, kv * hd], -1)
        q = q.view(B, T, H, hd).transpose(1, 2); k = k.view(B, T, kv, hd).transpose(1, 2); v = v.view(B, T, kv, hd).transpose(1, 2)
        if s.rope: q, k = rope(q), rope(k)
        if kv != H: k, v = k.repeat_interleave(H // kv, 1), v.repeat_interleave(H // kv, 1)
        if s.window:
            i = torch.arange(T, device=x.device)
            m = (i[:, None] >= i[None]) & (i[:, None] - i[None] < s.window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, -1))
