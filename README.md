# llm-series: 19 language-model architectures at ~10M parameters

From a Kneser-Ney n-gram to Mamba-2, RWKV, Griffin and mixture-of-experts: every model is implemented from scratch in PyTorch and trained under **one controlled protocol**, so the comparison of quality, speed and memory is fair.

The thread through the series: *how do we model sequences while paying less for memory and compute?*

## Models

| # | Model | Code | Notebook | Reference |
|---|---|---|---|---|
| 1 | N-gram (KN 3-gram) | [`ngram.py`](llmseries/models/ngram.py) | [01](notebooks/01_ngram.ipynb) | Kneser & Ney 1995 |
| 2 | Neural n-gram (Bengio) | [`nnlm.py`](llmseries/models/nnlm.py) | [02](notebooks/02_nnlm.ipynb) | Bengio et al. 2003 |
| 3 | Vanilla RNN | [`rnn.py`](llmseries/models/rnn.py) | [03](notebooks/03_rnn.ipynb) | Elman 1990 |
| 4 | LSTM | [`rnn.py`](llmseries/models/rnn.py) | [04](notebooks/04_lstm.ipynb) | Hochreiter & Schmidhuber 1997 |
| 5 | GRU | [`rnn.py`](llmseries/models/rnn.py) | [05](notebooks/05_gru.ipynb) | Cho et al. 2014 |
| 6 | GRU + Bahdanau attn | [`bahdanau.py`](llmseries/models/bahdanau.py) | [06](notebooks/06_bahdanau.ipynb) | Bahdanau et al. 2014 |
| 7 | GPT (decoder-only) | [`gpt.py`](llmseries/models/gpt.py) | [07](notebooks/07_gpt.ipynb) | Vaswani et al. 2017; Radford et al. 2019 |
| 8 | Modern Transformer (RoPE/RMS/SwiGLU/GQA) | [`modern.py`](llmseries/models/modern.py) | [08](notebooks/08_modern.ipynb) | Su 2021; Zhang & Sennrich 2019; Shazeer 2020; Ainslie 2023 |
| 9 | Linear attention | [`linattn.py`](llmseries/models/linattn.py) | [09](notebooks/09_linattn.ipynb) | Katharopoulos et al. 2020 |
| 10 | Gated DeltaNet | [`deltanet.py`](llmseries/models/deltanet.py) | [10](notebooks/10_deltanet.ipynb) | Schlag et al. 2021; Yang et al. 2024 |
| 11 | S4D (SSM) | [`s4d.py`](llmseries/models/s4d.py) | [11](notebooks/11_s4d.ipynb) | Gu et al. 2021, 2022 |
| 12 | Hyena | [`hyena.py`](llmseries/models/hyena.py) | [12](notebooks/12_hyena.ipynb) | Poli et al. 2023 |
| 13 | RWKV | [`rwkv.py`](llmseries/models/rwkv.py) | [13](notebooks/13_rwkv.ipynb) | Peng et al. 2023 |
| 14 | Mamba (S6) | [`mamba.py`](llmseries/models/mamba.py) | [14](notebooks/14_mamba.ipynb) | Gu & Dao 2023 |
| 15 | Mamba-2 (SSD) | [`mamba2.py`](llmseries/models/mamba2.py) | [15](notebooks/15_mamba2.ipynb) | Dao & Gu 2024 |
| 16 | xLSTM (mLSTM) | [`xlstm.py`](llmseries/models/xlstm.py) | [16](notebooks/16_xlstm.ipynb) | Beck et al. 2024 |
| 17 | Griffin (RG-LRU + local attn) | [`griffin.py`](llmseries/models/griffin.py) | [17](notebooks/17_griffin.ipynb) | De et al. 2024 |
| 18 | Hybrid Mamba+Attn | [`hybrid.py`](llmseries/models/hybrid.py) | [18](notebooks/18_hybrid.ipynb) | Jamba 2024; Samba 2024 |
| 19 | MoE Transformer | [`moe.py`](llmseries/models/moe.py) | [19](notebooks/19_moe.ipynb) | Shazeer 2017; Fedus 2021; Jiang 2024 |

Per-model notes and simplifications: [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md).

## Protocol (identical for all models)

| | |
|---|---|
| Data | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories): first 300k stories for training, 5k validation stories |
| Tokenizer | One shared byte-level BPE, vocab 4096 |
| Size | ~10M parameters each (width auto-searched in steps of 32, actual count reported) |
| Context | 256 tokens (also evaluated at 512 and 1024 for length extrapolation) |
| Budget | 4000 steps x 32 sequences x 256 tokens ≈ 33M tokens |
| Optimiser | AdamW (β=0.9/0.95, wd 0.1 on matrices), peak LR 2e-3 (RNN/LSTM/GRU lower), 100 warm-up steps, cosine to 10%, grad-clip 1.0, fp16 autocast |
| Metrics | validation loss / perplexity / bits per token, training tokens/s, peak GPU memory, loss at 512 and 1024 context |
| Seed | 0 |

Notes: the n-gram is not trained by gradient descent (its "params" are stored entries); the MoE model's 10M is *total* parameters, active parameters are stored in `results/moe.json`.

## Results

> Fill this table from `results/summary.md` after your runs (`python scripts/summarize.py`). Hardware: ___ (e.g. Kaggle T4).

| # | Model | Params | Val loss | PPL | Tokens/s | Mem (GB) | Loss@512 | Loss@1024 |
|---|---|---|---|---|---|---|---|---|
| 1 | N-gram (KN 3-gram) | | | | | | |
| 2 | Neural n-gram (Bengio) | | | | | | |
| 3 | Vanilla RNN | | | | | | |
| 4 | LSTM | | | | | | |
| 5 | GRU | | | | | | |
| 6 | GRU + Bahdanau attn | | | | | | |
| 7 | GPT (decoder-only) | | | | | | |
| 8 | Modern Transformer (RoPE/RMS/SwiGLU/GQA) | | | | | | |
| 9 | Linear attention | | | | | | |
| 10 | Gated DeltaNet | | | | | | |
| 11 | S4D (SSM) | | | | | | |
| 12 | Hyena | | | | | | |
| 13 | RWKV | | | | | | |
| 14 | Mamba (S6) | | | | | | |
| 15 | Mamba-2 (SSD) | | | | | | |
| 16 | xLSTM (mLSTM) | | | | | | |
| 17 | Griffin (RG-LRU + local attn) | | | | | | |
| 18 | Hybrid Mamba+Attn | | | | | | |
| 19 | MoE Transformer | | | | | | |

`results/quality_vs_speed.png` plots perplexity against training throughput.

## Quickstart

```bash
git clone https://github.com/<you>/llm-series && cd llm-series
pip install -e ".[dev]"            # PyTorch >= 2.1 must already be installed (Colab/Kaggle have it)
pytest -q                          # CPU: forward/backward + causality test for all 18 neural models

python scripts/prep_data.py        # download TinyStories + train BPE (a few minutes)
python scripts/train.py --arch gpt # one model
python scripts/train.py --arch all # all 19, sequentially
python scripts/summarize.py        # table + plot
```

Each finished model writes `results/<name>.json` (metrics + loss history) and `ckpt/<name>.pt`; models with an existing results file are skipped, so interrupted runs resume.

### Kaggle (2x T4)
Settings: Accelerator *GPU T4 x2*, Internet *On*. In a notebook:
```bash
!git clone https://github.com/<you>/llm-series && cd llm-series && pip -q install -e .
!cd llm-series && python scripts/run_2gpu.py        # one model per GPU, n-gram on CPU
```
Use *Save Version → Save & Run All* to run in the background; follow progress with `tail logs/gpu0.log`.

### Colab (1x T4)
```bash
!git clone https://github.com/<you>/llm-series && cd llm-series && pip -q install -e .
!cd llm-series && python scripts/train.py --arch all
```
Colab disconnects when idle: mount Drive and keep `results/` and `ckpt/` there, then re-run to resume.

### Notebooks
[`notebooks/`](notebooks) contains one fully self-contained notebook per model (`01`-`19`), each ending with a trained 10M model, loss curve, generated stories and an exportable zip, plus `20_compare.ipynb` to merge all results. They do not need this package; use them if you prefer a one-model-per-notebook workflow.

## Repository layout

```
llm-series/
├── llmseries/
│   ├── config.py        constants, model list, per-model overrides
│   ├── data.py          TinyStories + BPE + batching
│   ├── layers.py        norms, MLPs, Block/LM wrapper, attention, RoPE, parallel scan, FFT conv
│   ├── models/          one file per architecture (+ registry and param fitting in __init__.py)
│   ├── train.py         training loop and evaluation
│   └── report.py        results table + plot
├── scripts/             prep_data.py, train.py, run_2gpu.py, summarize.py
├── notebooks/           19 self-contained model notebooks + comparison notebook
├── tests/               shape / gradient / causality tests (run in CI on CPU)
├── docs/ARCHITECTURES.md
└── results/             metrics JSONs, summary table, plots
```

## Implementation notes and honest simplifications

- Everything is pure PyTorch, with **no custom CUDA kernels**. Mamba, RWKV and Griffin use an exact Hillis-Steele parallel scan; Gated DeltaNet is a plain sequential loop (slowest model). Speed numbers therefore reflect these implementations, not the fused kernels from the original papers.
- Linear attention and Mamba-2 are trained in their **parallel / quadratic forms**; their O(T) recurrent inference is not benchmarked here.
- GRU+Bahdanau puts attention on top of the GRU outputs rather than inside the decoder loop. xLSTM is the mLSTM variant only. There is no KV-cache or generation benchmark.
- Mamba, Hybrid, DeltaNet and Mamba-2 use gradient checkpointing and smaller micro-batches (global batch still 32).
- One seed, one dataset, one scale: differences of a few percent between models should not be over-interpreted.

## Roadmap

- [ ] Fill the results table; add loss-vs-context-length plot
- [ ] Synthetic tests: copy, induction heads, MQAR (multi-query associative recall)
- [ ] Inference benchmark: tokens/s and memory vs context length, KV-cache vs recurrent state
- [ ] 50M-parameter runs and multiple seeds
- [ ] RetNet, Hawk, sLSTM, fused-kernel versions

## References

Bengio et al. 2003 · Elman 1990 · Hochreiter & Schmidhuber 1997 · Cho et al. 2014 · Bahdanau et al. 2014 · Vaswani et al. 2017 · Su et al. 2021 (RoPE) · Katharopoulos et al. 2020 · Schlag et al. 2021 · Yang et al. 2024 (Gated DeltaNet) · Gu et al. 2021/2022 (S4/S4D) · Poli et al. 2023 (Hyena) · Peng et al. 2023 (RWKV) · Gu & Dao 2023 (Mamba) · Dao & Gu 2024 (Mamba-2) · Beck et al. 2024 (xLSTM) · De et al. 2024 (Griffin) · Shazeer et al. 2017 / Fedus et al. 2021 (MoE) · Eldan & Li 2023 (TinyStories)

## License
MIT
