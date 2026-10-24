"""Aggregate results/*.json into a markdown table and a quality-vs-speed plot."""
import glob
import json
import math  # noqa: F401

from .config import ALL


def summary():
    rs = [json.load(open(f)) for f in glob.glob("results/*.json")]; rs.sort(key=lambda r: ALL.index(r["name"]))
    L = ["| # | Model | Params | Val loss | PPL | bits/tok | tok/s | Mem GB | loss@512 | loss@1024 |", "|--|--|--|--|--|--|--|--|--|--|"]
    f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    for r in rs:
        le = r.get("len_eval", {}) or {}
        L.append(f"| {ALL.index(r['name'])+1} | {r['label']} | {r['params']/1e6:.2f}M | {f(r.get('val_loss'))} | {f(r.get('val_ppl'),2)} | "
                 f"{f(r.get('val_bits_per_token'))} | {f(r.get('tok_per_s'),0)} | {f(r.get('peak_mem_gb'),1)} | {f(le.get('512'))} | {f(le.get('1024'))} |")
    open("results/summary.md", "w").write("\n".join(L)); print("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pts = [r for r in rs if r.get("tok_per_s") and r.get("val_ppl")]
        plt.figure(figsize=(9, 6))
        for r in pts: plt.scatter(r["tok_per_s"], r["val_ppl"]); plt.annotate(r["name"], (r["tok_per_s"], r["val_ppl"]))
        plt.xscale("log"); plt.xlabel("training tokens/s (T4)"); plt.ylabel("val perplexity"); plt.title("Quality vs speed @ ~10M params")
        plt.grid(alpha=.3); plt.savefig("results/quality_vs_speed.png", dpi=150, bbox_inches="tight")
    except Exception as e: print("plot skipped:", e)
