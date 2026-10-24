"""N-gram (KN 3-gram)

Count-based baseline: Kneser-Ney smoothed trigram on BPE tokens. No gradients, no embeddings; shows what pure statistics give and why sparsity motivates neural LMs.

Reference: Kneser & Ney 1995
Not a neural model: "params" are the number of stored n-gram entries.
"""
import json
import math
import os
import numpy as np

from ..config import NAMES, V


def run_ngram(tokens=20_000_000):
    os.makedirs("results", exist_ok=True)
    if os.path.exists("results/ngram.json"): return
    tr = np.load("data/train.npy")[:tokens].astype(np.int64); va = np.load("data/val.npy").astype(np.int64); V2 = V * V
    u3, c3 = np.unique(tr[:-2] * V2 + tr[1:-1] * V + tr[2:], return_counts=True)
    ab, bc = u3 // V, u3 % V2
    ab_keys, ab_i, ab_types = np.unique(ab, return_index=True, return_counts=True); ab_tot = np.add.reduceat(c3, ab_i)
    bc_keys, n_bc = np.unique(bc, return_counts=True)            # continuation counts N1+(. b c)
    b_keys, b_i, b_types = np.unique(bc_keys // V, return_index=True, return_counts=True); b_tot = np.add.reduceat(n_bc, b_i)
    uni = np.bincount(bc_keys % V, minlength=V).astype(np.float64); p_uni = (uni + 1) / (uni.sum() + V)
    disc = lambda c: (c == 1).sum() / ((c == 1).sum() + 2 * (c == 2).sum()); D3, D2 = disc(c3), disc(n_bc)
    def look(keys, q): i = np.minimum(np.searchsorted(keys, q), len(keys) - 1); return i, keys[i] == q
    a, b, c = va[:-2], va[1:-1], va[2:]; pu = p_uni[c]
    with np.errstate(all="ignore"):
        ib, fb = look(b_keys, b); ibc, fbc = look(bc_keys, b * V + c)
        p2 = np.where(fb, (np.where(fbc, np.maximum(n_bc[ibc] - D2, 0), 0.0) + D2 * b_types[ib] * pu) / b_tot[ib], pu)
        iab, fab = look(ab_keys, a * V + b); iabc, fabc = look(u3, a * V2 + b * V + c)
        p3 = np.where(fab, (np.where(fabc, np.maximum(c3[iabc] - D3, 0), 0.0) + D3 * ab_types[iab] * p2) / ab_tot[iab], p2)
    loss = float(-np.log(p3).mean())
    res = {"name": "ngram", "label": NAMES["ngram"], "params": int(len(u3) + len(bc_keys) + V), "val_loss": loss,
           "val_ppl": math.exp(loss), "val_bits_per_token": loss / math.log(2), "tok_per_s": None, "note": "params = stored n-gram entries"}
    json.dump(res, open("results/ngram.json", "w"), indent=1); print("[ngram]", res)
