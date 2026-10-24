"""Train one or more models:  python scripts/train.py --arch gpt,mamba   |   --arch all"""
import argparse

from llmseries.config import ALL
from llmseries.data import prep_data
from llmseries.models.ngram import run_ngram
from llmseries.report import summary
from llmseries.train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="comma-separated names, or 'all'. Names: " + ",".join(ALL))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--target", type=float, default=10e6, help="parameter budget")
    a = ap.parse_args()
    prep_data()
    for n in (ALL if a.arch == "all" else a.arch.split(",")):
        run_ngram() if n == "ngram" else train(n, a)
    summary()
