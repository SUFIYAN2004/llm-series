"""Shared training loop (fp16 autocast, AdamW, cosine schedule, grad accumulation) and evaluation."""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import LR, MICRO, NAMES, SEQ, V
from .data import get_batch
from .models import fit, nparams
from .models.moe import MoE


def aux_loss(model):
    a = [m.aux for m in model.modules() if isinstance(m, MoE)]
    return sum(a) / len(a) if a else 0.0


def evaluate(model, val, T, bs, nb):
    model.eval(); g = torch.Generator().manual_seed(0); ls = []
    for _ in range(nb):
        x, y = get_batch(val, bs, T, g)
        with torch.autocast("cuda", dtype=torch.float16): lg = model(x)
        ls.append(F.cross_entropy(lg.float().view(-1, V), y.reshape(-1)).item())
    model.train(); return float(np.mean(ls))


def train(name, a):
    os.makedirs("results", exist_ok=True); os.makedirs("ckpt", exist_ok=True)
    if os.path.exists(f"results/{name}.json"): print(f"[{name}] done already, skipping"); return
    dev = "cuda"; torch.manual_seed(0)
    tr = torch.from_numpy(np.load("data/train.npy").astype(np.int16)).to(dev)
    va = torch.from_numpy(np.load("data/val.npy").astype(np.int16)).to(dev)
    model, d = fit(name, a.target); model = model.to(dev); n = nparams(model)
    print(f"[{name}] d={d} params={n/1e6:.2f}M")
    micro = MICRO.get(name, a.bs); accum = a.bs // micro; lr0 = LR.get(name, a.lr)
    dec = [p for k, p in model.named_parameters() if p.ndim >= 2 and k.split(".")[-1] in ("weight", "w1", "w2", "w3")]
    nod = [p for k, p in model.named_parameters() if not (p.ndim >= 2 and k.split(".")[-1] in ("weight", "w1", "w2", "w3"))]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.1}, {"params": nod, "weight_decay": 0.0}], lr=lr0, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda"); hist = []; eval_time = 0.0; diverged = False
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    for step in range(a.steps):
        lr = lr0 * (step + 1) / 100 if step < 100 else lr0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 100) / max(1, a.steps - 100))))
        for gp in opt.param_groups: gp["lr"] = lr
        tot = 0.0
        for _ in range(accum):
            x, y = get_batch(tr, micro, SEQ)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(x)
                loss = F.cross_entropy(lg.float().view(-1, V), y.reshape(-1))
                total = loss + 0.01 * aux_loss(model)
            scaler.scale(total / accum).backward(); tot += loss.item() / accum
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if not math.isfinite(tot): print(f"[{name}] diverged at step {step}"); diverged = True; break
        if step % 100 == 0 or step == a.steps - 1:
            ev = None
            if step % 500 == 0 or step == a.steps - 1:
                torch.cuda.synchronize(); te = time.time(); ev = evaluate(model, va, SEQ, 32, 20); eval_time += time.time() - te
            hist.append({"step": step, "train": tot, "val": ev})
            print(f"[{name}] step {step:5d} train {tot:.3f}" + (f" val {ev:.3f}" if ev else "") + f" | {time.time()-t0:.0f}s", flush=True)
    torch.cuda.synchronize(); ttrain = time.time() - t0 - eval_time
    res = {"name": name, "label": NAMES[name], "d_model": d, "params": n, "steps": step + 1, "diverged": diverged,
           "train_time_s": ttrain, "tok_per_s": (step + 1) * a.bs * SEQ / ttrain,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "history": hist}
    if name == "moe":
        ex = sum(p.numel() for k, p in model.named_parameters() if k.split(".")[-1] in ("w1", "w2", "w3") and p.ndim == 3)
        res["active_params"] = n - ex * 6 // 8
    if not diverged:
        res["val_loss"] = evaluate(model, va, SEQ, 32, 60); res["val_ppl"] = math.exp(res["val_loss"])
        res["val_bits_per_token"] = res["val_loss"] / math.log(2); res["len_eval"] = {}
        for T in (256, 512, 1024):  # trained at 256: length extrapolation
            try: res["len_eval"][T] = evaluate(model, va, T, 32 if T <= 256 else 8, 20)
            except Exception as e: res["len_eval"][T] = None; print(f"[{name}] len {T} failed: {type(e).__name__}")
        torch.save(model.state_dict(), f"ckpt/{name}.pt")
    json.dump(res, open(f"results/{name}.json", "w"), indent=1)
    print(f"[{name}] FINAL val_loss={res.get('val_loss')} ppl={res.get('val_ppl')} tok/s={res['tok_per_s']:.0f} mem={res['peak_mem_gb']:.1f}GB")
    del model, opt; torch.cuda.empty_cache()
