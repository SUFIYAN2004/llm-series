"""TinyStories download, BPE tokenizer training, and batch sampling."""
import os
import numpy as np
import torch

from .config import V, EOS


def prep_data(out="data", n_train=300_000, n_val=5_000):
    if os.path.exists(f"{out}/train.npy"): return
    os.makedirs(out, exist_ok=True)
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    ds = load_dataset("roneneldan/TinyStories")
    tr = ds["train"].select(range(n_train))["text"]; va = ds["validation"].select(range(n_val))["text"]
    tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(tr[:100_000], trainers.BpeTrainer(
        vocab_size=V, special_tokens=["<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(f"{out}/tok.json")
    def enc(texts):
        parts = []
        for i in range(0, len(texts), 20_000):
            ids = []
            for e in tok.encode_batch(texts[i:i + 20_000]): ids.extend(e.ids); ids.append(EOS)
            parts.append(np.array(ids, dtype=np.uint16))
        return np.concatenate(parts)
    for n, t in [("train", tr), ("val", va)]:
        a = enc(t); np.save(f"{out}/{n}.npy", a); print(n, "tokens:", len(a))


def get_batch(data, bs, T, g=None):
    ix = torch.randint(0, len(data) - T - 1, (bs,), generator=g).to(data.device)
    idx = ix[:, None] + torch.arange(T + 1, device=data.device)
    b = data[idx].long(); return b[:, :-1], b[:, 1:]
