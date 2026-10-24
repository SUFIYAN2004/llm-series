"""S4D (SSM)

Diagonal state-space model run in convolution mode via FFT. Linear time-invariant: the kernel does not depend on the input.

Reference: Gu et al. 2021, 2022
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Block, LM, SwiGLU, fftconv, nocast


class S4D(nn.Module):  # diagonal SSM, conv mode via FFT
    def __init__(s, d, N=16):
        super().__init__()
        s.log_dt = nn.Parameter(torch.rand(d) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        s.log_Are = nn.Parameter(torch.log(0.5 * torch.ones(d, N)))
        s.Aim = nn.Parameter(math.pi * torch.arange(N).float().repeat(d, 1))
        s.C = nn.Parameter(torch.randn(d, N, 2) * 0.5); s.D = nn.Parameter(torch.randn(d)); s.out = nn.Linear(d, 2 * d)
    def kernel(s, T):
        dt = s.log_dt.exp(); A = -s.log_Are.exp() + 1j * s.Aim; dtA = A * dt[:, None]
        C = torch.view_as_complex(s.C.contiguous()) * (torch.exp(dtA) - 1) / A
        K = torch.exp(dtA[..., None] * torch.arange(T, device=dt.device))
        return 2 * torch.einsum("dn,dnt->dt", C, K).real
    def forward(s, x):
        T = x.shape[1]
        with nocast(): y = fftconv(x.float().transpose(1, 2), s.kernel(T), s.D)
        return F.glu(s.out(F.gelu(y.transpose(1, 2))), dim=-1)


def build(d):
    H = max(d // 32, 1)
    kvh = H // 2 if H % 2 == 0 else H
    return LM(d, [Block(d, S4D(d), SwiGLU(d)) for _ in range(8)])
