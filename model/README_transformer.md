# `model/transformer.py` — Poker Decision Transformer

## Purpose

A **decoder-only transformer** (GPT-style) that reads the game history as a token sequence and outputs both an **action distribution** (policy head) and a **state value estimate** (critic head). This dual-output design enables PPO training without a separate critic network.

---

## Architecture

```
Input: token_ids  (B, T)   — batch of padded integer sequences
          │
          ▼
  Token Embedding  (vocab_size → d_model)
          +
  Sinusoidal Positional Encoding
          │
          ▼
  ┌───────────────────────┐
  │   TransformerBlock    │  × n_layers
  │                       │
  │  x = x + Attn(LN(x)) │  ← Pre-LayerNorm (more stable than post-LN)
  │  x = x + FFN(LN(x))  │
  └───────────────────────┘
          │
          ▼
    LayerNorm (final)
          │
    Gather last real token hidden state  (B, d_model)
          │
     ┌────┴────┐
     ▼         ▼
 Action Head  Value Head
 Linear →     Linear → GELU → Linear →
 (num_actions)         (1,)
     │              │
 action_logits   state_value
```

---

## Components

### `SinusoidalPositionalEncoding`

Fixed (non-learned) encoding using sin/cos at different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Registered as a buffer (not a parameter). Generalises to sequence lengths not seen during training.

### `MultiHeadSelfAttention`

Causal (autoregressive) attention with fused Q/K/V projection:

```
attn(i, j) = 0  if j > i   (future tokens masked)
```

Shape flow:
```
x: (B, T, d_model)
   → qkv: (B, T, 3, H, d_head)
   → q, k, v: (B, H, T, d_head)
   → scores = q @ k^T / √d_head  : (B, H, T, T)
   → masked + softmax
   → out = scores @ v             : (B, H, T, d_head)
   → reshape + proj               : (B, T, d_model)
```

### `FeedForward`

```
FFN(x) = Linear(GELU(Linear(x, d_ff)), d_model)
```

GELU activation is smoother than ReLU and performs better in transformer contexts.

### `TransformerBlock`

Pre-LayerNorm variant (more stable gradient flow):

```
x = x + Attn(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### `PokerTransformer` (main class)

```python
logits, values = model(token_ids)
# logits: (B, num_actions)  — unnormalized action scores
# values: (B, 1)            — state value estimate V(s)
```

The **last non-padded token's hidden state** is used for prediction. This position sees the full causal history and acts as the "decision state" representation.

---

## Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 128 | Embedding dimension |
| `n_heads` | 4 | Attention heads (d_model / n_heads = 32) |
| `n_layers` | 4 | Transformer blocks |
| `d_ff` | 512 | Feed-forward inner dimension (4 × d_model) |
| `dropout` | 0.1 | Applied after attention and FFN |
| `max_len` | 512 | Maximum sequence length |

**Parameter count** (default config): ~2–4M parameters.

Scale up by increasing `d_model` and `n_layers`. The 6-player game with 512-length context trains well at these defaults on CPU or a small GPU.

---

## Value Head (for PPO)

The value head outputs `V(s)` — the expected future return from state `s`. During PPO training:

- `V(s)` is the **critic baseline** used to compute advantages: `A(s,a) = R - V(s)`
- A good critic reduces variance in policy gradient estimates
- The value head shares the transformer backbone (parameter-efficient)

---

## Weight Initialisation

| Layer | Initialisation |
|-------|----------------|
| Linear layers | Xavier uniform |
| Biases | Zeros |
| Embedding | Normal(mean=0, std=0.02) |

---

## Usage Example

```python
from model.transformer import PokerTransformer, TransformerConfig
from model.tokenizer import PokerTokenizer
import torch

tok = PokerTokenizer(max_len=512)
cfg = TransformerConfig(
    vocab_size=tok.vocab_size,
    num_actions=tok.num_actions,
    d_model=128,
    n_heads=4,
    n_layers=4,
)
model = PokerTransformer(cfg)
print(f"Parameters: {model.count_parameters():,}")   # ~2-4M

# Single forward pass
token_ids = torch.randint(0, tok.vocab_size, (1, 512))
logits, values = model(token_ids)
# logits: (1, 19)
# values: (1, 1)

# Sample action
probs = torch.softmax(logits, dim=-1)
action_idx = torch.multinomial(probs, 1).item()
```

---

## Scaling Guide

| Use Case | d_model | n_heads | n_layers | d_ff | ~Params |
|----------|---------|---------|----------|------|---------|
| Fast dev/test | 64 | 2 | 2 | 256 | ~0.5M |
| Default (recommended) | 128 | 4 | 4 | 512 | ~2M |
| Stronger policy | 256 | 8 | 6 | 1024 | ~12M |
| Research-scale | 512 | 8 | 8 | 2048 | ~50M |

Larger models require GPU training and longer rollout collection to see benefits.
