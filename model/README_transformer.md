# `model/transformer.py` — Poker Decision Transformer

A **decoder-only transformer** (GPT-style) that reads the encoded game history
and simultaneously outputs an **action distribution** (actor/policy) and a
**state value estimate** (critic). This shared-backbone design lets PPO train
both heads with a single forward pass.

---

## Architecture Overview

```
  Input token_ids  (B, T=512)
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │  Token Embedding                             │
  │  nn.Embedding(vocab_size=1413, d_model=128)  │
  │                       +                      │
  │  SinusoidalPositionalEncoding                │
  │  PE(pos,2i)   = sin(pos / 10000^(2i/128))    │
  │  PE(pos,2i+1) = cos(pos / 10000^(2i/128))    │
  └──────────────────┬───────────────────────────┘
                     │
                     ▼  (B, T, 128)
  ┌──────────────────────────────────────────────┐  ─┐
  │  TransformerBlock                            │   │
  │  ┌─────────────────────────────────────────┐ │   │
  │  │  x = x + Attn(LayerNorm(x))             │ │   │  × n_layers
  │  │  x = x + FFN (LayerNorm(x))             │ │   │  (default 4)
  │  └─────────────────────────────────────────┘ │   │
  └──────────────────┬───────────────────────────┘  ─┘
                     │
                     ▼  Final LayerNorm
                     │
                     │  Gather the last non-PAD token's hidden state
                     │  (this position sees the full causal history)
                     │  (B, 128)
                     │
          ┌──────────┴──────────────────┐
          │                             │
          ▼                             ▼
  ┌───────────────────┐     ┌──────────────────────────┐
  │   Action Head     │     │      Value Head           │
  │                   │     │                          │
  │  Linear(128, 19)  │     │  Linear(128, 64)         │
  │                   │     │  GELU activation          │
  │  action_logits    │     │  Linear(64, 1)           │
  │  (B, 19)          │     │  state_value  (B, 1)     │
  └───────────────────┘     └──────────────────────────┘
```

---

## Causal Self-Attention (Detail)

The attention mask ensures each token can only attend to **earlier tokens**
in the sequence — future actions are invisible. This is essential because
the model must make decisions without seeing what opponents will do next.

```
  Attention mask for T=6:

  Query position →   0   1   2   3   4   5
                   ┌───┬───┬───┬───┬───┬───┐
  Key position 0   │ ✓ │   │   │   │   │   │
  Key position 1   │ ✓ │ ✓ │   │   │   │   │
  Key position 2   │ ✓ │ ✓ │ ✓ │   │   │   │
  Key position 3   │ ✓ │ ✓ │ ✓ │ ✓ │   │   │
  Key position 4   │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │   │
  Key position 5   │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │
                   └───┴───┴───┴───┴───┴───┘

  ✓ = attended   blank = masked out (-inf before softmax → 0 weight)

  Shape flow through attention:
    x            : (B, T, 128)
    Q, K, V      : (B, H=4, T, d_head=32)   via fused qkv projection
    scores       : (B, H, T, T)             Q @ K^T / sqrt(32)
    scores masked: upper triangle set to -inf
    weights      : (B, H, T, T)             softmax(scores)
    out          : (B, H, T, 32)            weights @ V
    proj         : (B, T, 128)              reshape + linear
```

---

## Pre-LayerNorm (Stability)

This implementation uses **Pre-LayerNorm** (normalize *before* the sublayer)
rather than the original transformer's Post-LayerNorm:

```
  Post-LN (original, less stable):    Pre-LN (this model, more stable):
  x = LayerNorm(x + Attn(x))          x = x + Attn(LayerNorm(x))
  x = LayerNorm(x + FFN(x))           x = x + FFN(LayerNorm(x))

  Pre-LN advantage: gradients flow through the residual path
  without passing through LayerNorm, preventing vanishing gradients
  in deep stacks. Better for training without learning rate warmup.
```

---

## Feed-Forward Block

```
  FFN(x):
    h = Linear(128 → 512)(x)      ← expand by 4×
    h = GELU(h)                   ← smooth non-linearity
    h = Dropout(0.1)(h)
    y = Linear(512 → 128)(h)      ← project back
    y = Dropout(0.1)(y)
    return y

  GELU vs ReLU:
    ReLU(x) = max(0, x)           hard zero for x < 0
    GELU(x) = x * Φ(x)           smooth, probabilistic gating
    GELU performs better in language/sequence models empirically.
```

---

## Sinusoidal Positional Encoding

Positional information is injected via fixed sin/cos encodings —
no learned position embeddings. This generalises better to sequence
lengths not seen during training.

```
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  Each dimension oscillates at a different frequency:
    dim  0: period = 2π          (changes every token)
    dim  2: period ≈ 63          (changes every ~10 tokens)
    dim 64: period ≈ 10000       (very slow)
    dim 127: period ≈ 40000      (nearly constant over 512 tokens)

  The model learns to use combinations of these as position signals.
  Registered as a buffer — not a learnable parameter.
```

---

## Value Head (Critic for PPO)

The value head estimates V(s) — the expected cumulative return from state s.
It is used in PPO to compute the **advantage** A(s,a):

```
  A(s, a) = Q(s, a) - V(s)

  Intuitively: "Was this action better or worse than average from this state?"

  A > 0 → action was better than expected → increase its probability
  A < 0 → action was worse than expected  → decrease its probability

  The value head is trained to minimise MSE(V(s), discounted_returns).
  A well-trained critic drastically reduces variance in the policy gradient,
  making training much faster.
```

---

## Default Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `d_model` | 128 | Balances capacity vs. CPU training speed |
| `n_heads` | 4 | d_head = 32; sufficient for 512-length sequences |
| `n_layers` | 4 | 4 blocks ≈ 4 levels of abstraction |
| `d_ff` | 512 | 4× d_model, standard transformer ratio |
| `dropout` | 0.1 | Light regularisation for self-play data |
| `max_len` | 512 | Covers ~3–4 full hands of history |
| `vocab_size` | 1413 | Full token vocabulary |
| `num_actions` | 19 | 3 + 8 raise buckets + 8 all-in buckets |

**Parameter count** at defaults: **~982,000 parameters** (≈1M).

---

## Scaling Guide

```
  Use Case         d_model   n_heads   n_layers   d_ff    ~Params   Device
  ───────────────  ───────   ───────   ────────   ─────   ───────   ──────
  Fast dev/test       64        2          2        256    ~250K     CPU
  Default (rec.)     128        4          4        512    ~1M       CPU
  Stronger policy    256        8          6       1024    ~12M      GPU
  Research-scale     512        8          8       2048    ~50M      GPU

  Scaling rule: doubling d_model roughly 4× the parameters.
  Larger models need proportionally more rollout data to benefit.
```

---

## Forward Pass Summary

```python
logits, values = model(token_ids)
# token_ids : (B, T)   integer sequences, 0-padded
# logits    : (B, 19)  action scores (pass through softmax for probs)
# values    : (B, 1)   state value estimates V(s)

# During training (PPO):
dist = Categorical(logits=logits)
action = dist.sample()                    # stochastic exploration
log_prob = dist.log_prob(action)          # needed for ratio computation
entropy = dist.entropy().mean()           # exploration bonus

# During evaluation (greedy):
action = logits.argmax(dim=-1)
```

---

## Usage Example

```python
from model.transformer import PokerTransformer, TransformerConfig
from model.tokenizer import PokerTokenizer
import torch

tok = PokerTokenizer(max_len=512)
cfg = TransformerConfig(
    vocab_size=tok.vocab_size,   # 1413
    num_actions=tok.num_actions, # 19
    d_model=128,
    n_heads=4,
    n_layers=4,
)
model = PokerTransformer(cfg)
print(f"Parameters: {model.count_parameters():,}")   # ~982,000

# Forward pass
token_ids = torch.randint(0, tok.vocab_size, (1, 512))
logits, values = model(token_ids)
# logits: (1, 19)   values: (1, 1)

# Sample action
action_idx = torch.distributions.Categorical(logits=logits).sample().item()
action_str = tok.decode_action(action_idx)
```
