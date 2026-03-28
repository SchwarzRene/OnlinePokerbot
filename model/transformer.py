"""
transformer.py — Poker Decision Transformer
============================================

A decoder-only transformer that predicts the optimal poker action
given the current game history as a token sequence.

Architecture Overview:
----------------------

    Input sequence (token ids, length T):
    ┌─────────────────────────────────────────┐
    │  [BOS] [HOLE:Ah] [HOLE:Kd] [ROUND:pre] │
    │  [P0:SB:20] [P1:BB:40] [P2:RAISE:120]  │
    │  [ROUND:flop] [BOARD:Qh] [BOARD:Jc]... │
    └─────────────────────────────────────────┘
               │
               ▼
    Token Embedding  (vocab_size → d_model)
               +
    Positional Encoding  (sinusoidal, up to max_len)
               │
               ▼
    ┌─────────────────────┐
    │  TransformerBlock   │  × n_layers
    │  ├─ MultiHeadAttn   │
    │  ├─ LayerNorm       │
    │  ├─ FeedForward     │
    │  └─ LayerNorm       │
    └─────────────────────┘
               │
               ▼
    Last token hidden state  (d_model,)
               │
        ┌──────┴──────┐
        ▼             ▼
    Action Head    Value Head
    (num_actions)  (scalar)
        │             │
    logits        V(s) estimate
                  (for PPO critic)

Design Decisions:
-----------------
  • Decoder-only (GPT-style): causal masking so each position
    only attends to past tokens. Natural fit for sequential decisions.

  • Action head: linear layer → softmax → categorical distribution.
    During training sampled stochastically; during eval argmax or
    temperature-scaled sampling.

  • Value head: linear → scalar. Used by PPO as the baseline estimator
    V(s) to reduce variance in policy gradient updates.

  • No cross-attention: we don't use encoder-decoder since the
    entire context (history) is in a single sequence.

  • Weight tying: token embedding weights are shared with the action
    head projection to reduce parameters (optional, off by default).

Hyperparameters (defaults tuned for 6-player games, 512-length context):
-------------------------------------------------------------------------
    d_model   = 128    embedding dimension
    n_heads   = 4      attention heads (d_model must be divisible by n_heads)
    n_layers  = 4      transformer blocks
    d_ff      = 512    feedforward inner dimension
    dropout   = 0.1
    max_len   = 512    maximum sequence length
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """All hyperparameters for the poker transformer."""
    vocab_size: int = 2048
    num_actions: int = 19          # 3 + 8 RAISE buckets + 8 ALL_IN buckets
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 512
    pad_id: int = 0


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic fixed sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Registered as a buffer (not a parameter) — not updated during training.
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head attention with causal (autoregressive) mask.

    Q, K, V projections are fused into a single matrix for efficiency.

    Complexity: O(T² · d_model) per layer — manageable for T ≤ 512.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, d_model)
        mask: Optional[torch.Tensor] = None,  # (1, 1, T, T) causal
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True=pad
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, n_heads, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, H, T, T)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        if key_padding_mask is not None:
            # (B, T) → (B, 1, 1, T)
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(kpm, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-Forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation.

        FFN(x) = Linear(GELU(Linear(x, d_ff)), d_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm transformer block (more stable training than post-LN).

        x = x + Attn(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal_mask, key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# PokerTransformer
# ---------------------------------------------------------------------------

class PokerTransformer(nn.Module):
    """
    Full poker decision model.

    Forward pass returns:
        action_logits : (B, num_actions)   unnormalized action probabilities
        state_values  : (B, 1)             estimated state value for PPO critic

    Usage
    -----
        cfg = TransformerConfig(vocab_size=tok.vocab_size, num_actions=tok.num_actions)
        model = PokerTransformer(cfg)
        logits, values = model(token_ids)          # token_ids: (B, T)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1)       # sample action
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        self.ln_final = nn.LayerNorm(cfg.d_model)

        # Action head
        self.action_head = nn.Linear(cfg.d_model, cfg.num_actions)

        # Value head (for PPO critic)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for linear layers; normal for embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (1=attend, 0=block). Shape: (1, 1, T, T)."""
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        return mask

    def forward(
        self,
        token_ids: torch.Tensor,              # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) 1=real, 0=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = token_ids.shape
        device = token_ids.device

        # Key-padding mask: True where padded
        if attention_mask is not None:
            key_pad = (attention_mask == 0)
        else:
            key_pad = (token_ids == self.cfg.pad_id)

        x = self.embedding(token_ids)          # (B, T, d_model)
        x = self.pos_enc(x)

        causal = self._causal_mask(T, device)

        for block in self.blocks:
            x = block(x, causal, key_pad)

        x = self.ln_final(x)

        # Use last non-padded token's hidden state as the decision state
        # (In practice: index of last real token per batch item)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1   # (B,)
        else:
            lengths = (~key_pad).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        # Gather last real token: (B, d_model)
        idx = lengths.view(B, 1, 1).expand(B, 1, x.size(-1))
        last_hidden = x.gather(1, idx).squeeze(1)

        action_logits = self.action_head(last_hidden)   # (B, num_actions)
        state_values = self.value_head(last_hidden)     # (B, 1)

        return action_logits, state_values

    def get_action_probs(self, token_ids: torch.Tensor, temperature: float = 1.0):
        """Convenience: forward → softmax. Returns probs (B, num_actions)."""
        logits, values = self.forward(token_ids)
        probs = F.softmax(logits / temperature, dim=-1)
        return probs, values

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
