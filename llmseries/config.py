"""Global constants shared by every model so all comparisons are controlled."""

V = 4096      # BPE vocabulary size
SEQ = 256     # training context length
EOS = 0       # end-of-story token id

# Training order = series order (1-based number in the README)
ALL = ['ngram',
 'nnlm',
 'rnn',
 'lstm',
 'gru',
 'bahdanau',
 'gpt',
 'modern',
 'linattn',
 'deltanet',
 's4d',
 'hyena',
 'rwkv',
 'mamba',
 'mamba2',
 'xlstm',
 'griffin',
 'hybrid',
 'moe']

NAMES = {'bahdanau': 'GRU + Bahdanau attn',
 'deltanet': 'Gated DeltaNet',
 'gpt': 'GPT (decoder-only)',
 'griffin': 'Griffin (RG-LRU + local attn)',
 'gru': 'GRU',
 'hybrid': 'Hybrid Mamba+Attn',
 'hyena': 'Hyena',
 'linattn': 'Linear attention',
 'lstm': 'LSTM',
 'mamba': 'Mamba (S6)',
 'mamba2': 'Mamba-2 (SSD)',
 'modern': 'Modern Transformer (RoPE/RMS/SwiGLU/GQA)',
 'moe': 'MoE Transformer',
 'ngram': 'N-gram (KN 3-gram)',
 'nnlm': 'Neural n-gram (Bengio)',
 'rnn': 'Vanilla RNN',
 'rwkv': 'RWKV',
 's4d': 'S4D (SSM)',
 'xlstm': 'xLSTM (mLSTM)'}

# micro-batch overrides (global batch stays 32 via gradient accumulation) for memory-hungry models
MICRO = {'mamba': 8, 'hybrid': 8, 'mamba2': 16, 'deltanet': 16, 'hyena': 16, 'bahdanau': 16}

# per-model learning-rate overrides (default 2e-3)
LR = {'rnn': 0.001, 'lstm': 0.0015, 'gru': 0.0015}
