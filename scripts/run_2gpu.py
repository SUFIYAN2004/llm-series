"""Kaggle 2xT4: one training process per GPU (+ n-gram on CPU). Logs in logs/gpu{0,1}.log"""
import os
import subprocess
import sys

from llmseries.data import prep_data
from llmseries.models.ngram import run_ngram
from llmseries.report import summary

GPU0 = ["nnlm", "lstm", "bahdanau", "modern", "linattn", "deltanet", "hyena", "rwkv", "xlstm"]
GPU1 = ["rnn", "gru", "gpt", "s4d", "mamba", "mamba2", "griffin", "hybrid", "moe"]

if __name__ == "__main__":
    steps = sys.argv[1] if len(sys.argv) > 1 else "4000"
    prep_data()  # once, before the workers start
    os.makedirs("logs", exist_ok=True)
    train_py = os.path.join(os.path.dirname(__file__), "train.py")
    procs = []
    for gid, names in enumerate([GPU0, GPU1]):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gid))
        procs.append(subprocess.Popen([sys.executable, train_py, "--arch", ",".join(names), "--steps", steps],
                                      env=env, stdout=open(f"logs/gpu{gid}.log", "w"), stderr=subprocess.STDOUT))
    run_ngram()
    print("n-gram done; GPU jobs running -> tail -f logs/gpu0.log logs/gpu1.log")
    for p in procs:
        p.wait()
    summary()
