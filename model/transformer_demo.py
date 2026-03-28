"""
transformer_demo.py — Tiny CPU-Optimised Demo Model
=====================================================

A drastically stripped-down model designed to run 1000 training epochs
on a regular CPU in minutes rather than hours.

Drop-in replacement for transformer.py:
    • Same forward() signature → (action_logits, state_values)
    • Same TransformerConfig dataclass (ignored fields are just unused)
    • Same count_parameters() method
    • Works with RLAgent, PPOTrainer, checkpoints — zero other changes needed

Architecture: GRU + small MLP  (no attention, no positional encoding)
----------------------------------------------------------------------

    token_ids  (B, T)
         │
         ▼  nn.Embedding(vocab, d_model=32)
         │
         ▼  nn.GRU(d_model, hidden=32, num_layers=1, batch_first=True)
         │   └─ takes the final hidden state  (B, 32)
         │
        ┌┴────────────────┐
        ▼                 ▼
    Action Head       Value Head
    Linear(32, 19)    Linear(32, 1)

Why GRU instead of Transformer for the demo?
---------------------------------------------
  Transformer attention is O(T²) per layer.  At seq_len=512 and 4 layers
  that's ~1M multiply-adds per forward pass.  A single-layer GRU is O(T)
  and highly optimised in PyTorch's C++ backend (cuDNN path on GPU,
  MKLDNN/C++ on CPU).  For the same d_model it is ~20–50× faster on CPU.

  The GRU can still learn poker strategy — it just has far less capacity.
  That's fine for a demo; the point is to verify the full training loop
  works and get a feel for reward curves before committing GPU time.

Speed comparison (approximate, 6-player game, CPU):
----------------------------------------------------
  Full model  (Transformer, 983K params, seq=512):  ~3–8 s/epoch
  Demo model  (GRU,           ~28K params, seq=64):  ~0.1–0.3 s/epoch

  → 1000 demo epochs ≈ 2–5 minutes on a typical laptop CPU.

Demo-specific config changes (set automatically by DemoConfig):
---------------------------------------------------------------
  d_model         128  →   32    (embedding + GRU hidden size)
  n_layers          4  →    1    (GRU layers)
  max_len         512  →   64    (sequence truncation in tokenizer)
  hands_per_rollout 200 →  50    (less data needed per update)
  batch_size        64  →  32

Usage
-----
    # In main.py / your script — swap one import:
    from model.transformer_demo import PokerTransformerDemo as PokerTransformer, DemoConfig as TransformerConfig

    # Everything else stays identical.

    # Or use the convenience function:
    from model.transformer_demo import build_demo_training_pair
    ppo_cfg, model_cfg, trainer = build_demo_training_pair()
    trainer.train()
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# DemoConfig
# ---------------------------------------------------------------------------

@dataclass
class DemoConfig:
    """
    Tiny hyperparameter set tuned for fast CPU training.

    Intentionally mirrors TransformerConfig field names so it is a
    drop-in replacement everywhere configs are passed around.
    """
    # Set by tokenizer at init (same as full model)
    vocab_size: int = 2048
    num_actions: int = 19

    # Tiny model dimensions
    d_model: int = 32           # was 128  — 4× smaller embedding
    n_heads: int = 1            # unused by GRU, kept for API compat
    n_layers: int = 1           # single GRU layer
    d_ff: int = 64              # unused by GRU, kept for API compat
    dropout: float = 0.0        # no dropout — model is tiny, regularisation not needed
    max_len: int = 64           # was 512  — 8× shorter sequence
    pad_id: int = 0


# ---------------------------------------------------------------------------
# PokerTransformerDemo
# ---------------------------------------------------------------------------

class PokerTransformerDemo(nn.Module):
    """
    GRU-based poker policy model.

    Identical forward interface to PokerTransformer:
        logits, values = model(token_ids)   # token_ids: (B, T)

    Parameters
    ----------
    cfg : DemoConfig (or any object with .vocab_size, .num_actions,
                      .d_model, .n_layers, .pad_id, .dropout)
    """

    def __init__(self, cfg: DemoConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding (same vocab as full model)
        self.embedding = nn.Embedding(
            cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id
        )

        # GRU: processes the token sequence left-to-right.
        # We use the final hidden state as the "decision state" —
        # equivalent to the last-token hidden state in the transformer.
        self.gru = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )

        # Action head — same output shape as full model
        self.action_head = nn.Linear(cfg.d_model, cfg.num_actions)

        # Value head — same output shape as full model
        self.value_head = nn.Linear(cfg.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        token_ids: torch.Tensor,              # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # unused, kept for API compat
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        action_logits : (B, num_actions)
        state_values  : (B, 1)
        """
        x = self.embedding(token_ids)          # (B, T, d_model)

        # GRU: output is (B, T, d_model), h_n is (n_layers, B, d_model)
        _, h_n = self.gru(x)

        # Use the last layer's final hidden state as the decision state
        last_hidden = h_n[-1]                  # (B, d_model)

        action_logits = self.action_head(last_hidden)   # (B, num_actions)
        state_values  = self.value_head(last_hidden)    # (B, 1)

        return action_logits, state_values

    def get_action_probs(self, token_ids: torch.Tensor, temperature: float = 1.0):
        """Convenience: forward → softmax. Returns (probs, values)."""
        logits, values = self.forward(token_ids)
        probs = F.softmax(logits / temperature, dim=-1)
        return probs, values

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_demo_training_pair():
    """
    Build a ready-to-run (PPOConfig, DemoConfig, PPOTrainer) triple
    tuned for fast CPU training.

    Returns
    -------
    ppo_cfg   : PPOConfig
    model_cfg : DemoConfig
    trainer   : PPOTrainer  (call trainer.train() to start)

    Example
    -------
        from model.transformer_demo import build_demo_training_pair
        ppo_cfg, model_cfg, trainer = build_demo_training_pair()
        trainer.train()   # ~2–5 minutes for 1000 epochs on CPU
    """
    from training.ppo_trainer import PPOConfig, PPOTrainer
    from model.tokenizer import PokerTokenizer

    # Tuned for CPU speed without sacrificing learning signal
    ppo_cfg = PPOConfig(
        min_players=2,
        max_players=6,          # cap at 6-max; 9-player games are rare online anyway
        total_epochs=1000,
        hands_per_rollout=50,   # was 200 — less data, faster epochs
        batch_size=32,          # was 64
        n_opt_epochs=3,         # was 4
        save_interval=100,      # checkpoint every 100 epochs
        eval_interval=50,
        eval_hands=50,
        starting_chips=2000,    # 100 BB (realistic)
        checkpoint_dir="checkpoints_demo",
        log_dir="logs_demo",
    )

    model_cfg = DemoConfig()

    # Let the tokenizer set vocab_size and num_actions (same as always)
    tok = PokerTokenizer(max_len=model_cfg.max_len)
    model_cfg.vocab_size = tok.vocab_size
    model_cfg.num_actions = tok.num_actions

    trainer = PPOTrainer(ppo_cfg, model_cfg, model_class=PokerTransformerDemo)

    return ppo_cfg, model_cfg, trainer
