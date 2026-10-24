"""Model registry. `make(name, d)` builds a model of width d; `fit` searches d to hit a parameter budget."""
import importlib

from ..config import ALL  # noqa: F401

_FILE = {"rnn": "rnn", "lstm": "rnn", "gru": "rnn"}


def make(name, d):
    mod = importlib.import_module(f".{_FILE.get(name, name)}", __package__)
    return mod.build(d, name) if name in ("rnn", "lstm", "gru") else mod.build(d)


def nparams(model):
    return sum(p.numel() for p in model.parameters())


def fit(name, target=10e6):
    """Pick the width (multiple of 32) whose parameter count is closest to `target`."""
    best = None
    for d in range(64, 1537, 32):
        n = nparams(make(name, d))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (d, n)
        if n > target:
            break
    return make(name, best[0]), best[0]
