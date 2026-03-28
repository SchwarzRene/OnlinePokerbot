# `training/ppo_trainer.py` — PPO Self-Play Trainer

## Purpose

Trains the `PokerTransformer` via **Proximal Policy Optimization (PPO)** with **self-play**. The agent plays against frozen copies of itself, improving iteratively.

---

## Algorithm Overview

```
for epoch in 1..total_epochs:

  ┌─ ROLLOUT PHASE ───────────────────────────────────────────┐
  │  Play hands_per_rollout hands using current policy        │
  │  Record (token_ids, action_idx, log_prob, value,          │
  │          reward, done) per RL-agent decision              │
  └───────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─ ADVANTAGE COMPUTATION ────────────────────────────────────┐
  │  GAE (Generalised Advantage Estimation):                  │
  │    δ_t = r_t + γ·V(s_{t+1}) - V(s_t)                    │
  │    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...        │
  │  Normalise advantages: A = (A - μ) / (σ + ε)             │
  └───────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─ PPO UPDATE PHASE ─────────────────────────────────────────┐
  │  for k in 1..n_opt_epochs:                                │
  │    for each mini-batch of size batch_size:                │
  │      ratio     = exp(log π_new(a|s) - log π_old(a|s))    │
  │      clip_ratio = clip(ratio, 1-ε, 1+ε)                  │
  │      L_policy  = -mean(min(ratio·A, clip_ratio·A))       │
  │      L_value   = MSE(V(s), returns)                       │
  │      L_entropy = -mean(Σ π·log π)                        │
  │      loss = L_policy + c1·L_value - c2·L_entropy         │
  │      optimizer.step()                                     │
  │    if KL_div > target_kl: break early                    │
  └───────────────────────────────────────────────────────────┘
```

---

## Self-Play Setup

```
Training agent (player 0)
    └── PokerTransformer (actively trained, explore=True)

Opponents (players 1..N-1)
    └── Frozen copies of PokerTransformer (explore=False)
         └── Updated every opponent_update_interval hands
              from training agent weights
```

This prevents the agent from exploiting a static opponent and produces more generalisable strategies (similar to AlphaGo's self-play scheme).

---

## Reward Design

```
Per-step reward:   0.0           (no intermediate reward)

Terminal reward:   (chips_after - chips_before) / big_blind

Normalisation:     divide by big_blind → reward in BB units
                   keeps reward scale stable regardless of stack size
```

**Example:** Win a 240-chip pot (blinds 10/20) → reward = +12 BB. Lose blind = −1 BB.

BB-normalised rewards have lower variance and train more stably than raw chip amounts.

---

## `Experience` — Rollout Record

```
Experience
├── token_ids   : Tensor (T,)   encoded game state
├── action_idx  : int           chosen action (0..18)
├── log_prob    : float         log π(a|s) at decision time
├── value       : float         V(s) from critic at decision time
├── reward      : float         terminal BB reward for this hand
└── done        : bool          always True (one experience per hand)
```

---

## `RolloutBuffer`

Stores all `Experience` objects from one rollout. Computes advantages and returns via GAE:

```python
adv, ret = buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)
data = buffer.to_tensors(adv, ret)
# data["token_ids"]    : (N, T)
# data["actions"]      : (N,)
# data["old_log_probs"]: (N,)
# data["advantages"]   : (N,)
# data["returns"]      : (N,)
```

---

## `PPOConfig` — All Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_players` | 6 | Players per table |
| `starting_chips` | 1000 | Chips each player starts with per session |
| `hands_per_rollout` | 200 | Hands collected per epoch |
| `clip_eps` | 0.2 | PPO clipping range |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing (0=TD, 1=MC) |
| `c1` | 0.5 | Value loss weight |
| `c2` | 0.01 | Entropy bonus weight |
| `target_kl` | 0.02 | Early-stop KL threshold |
| `n_opt_epochs` | 4 | Optimisation passes per rollout |
| `batch_size` | 64 | Mini-batch size |
| `lr` | 3e-4 | Adam learning rate |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `opponent_update_interval` | 50 | Hands between opponent weight sync |
| `total_epochs` | 500 | Training epochs |
| `save_interval` | 50 | Epochs between checkpoint saves |

---

## Training Loop Output

```
Epoch  100 | reward= 0.423 | ploss= 0.0312 | vloss= 0.8541 | ent= 2.341 | kl= 0.0089 | 14.2s
Epoch  200 | reward= 1.271 | ploss= 0.0198 | vloss= 0.6231 | ent= 2.102 | kl= 0.0143 | 13.8s
Epoch  300 | reward= 2.104 | ploss= 0.0147 | vloss= 0.4812 | ent= 1.891 | kl= 0.0112 | 14.1s
```

**Reading the metrics:**
- `reward`: mean BB/hand over rollout (higher = better)
- `ploss`: policy gradient loss (lower = policy is improving stably)
- `vloss`: critic MSE loss (lower = better value estimates)
- `ent`: policy entropy (if too low → agent is over-committing to one action)
- `kl`: estimated KL divergence from old policy (> `target_kl` → early stop)

---

## Checkpoints

Saved as `checkpoints/model_epoch_{N}.pt`:

```python
{
    "epoch": int,
    "model_state": OrderedDict,      # PokerTransformer weights
    "optimizer_state": dict,
    "metrics": dict                  # full training history
}
```

Load with:

```python
trainer.load_checkpoint("checkpoints/model_epoch_200.pt")
```

---

## Usage

```python
from training.ppo_trainer import PPOTrainer, PPOConfig
from model.transformer import TransformerConfig

cfg = PPOConfig(
    n_players=6,
    total_epochs=500,
    hands_per_rollout=200,
)
model_cfg = TransformerConfig(d_model=128, n_heads=4, n_layers=4)

trainer = PPOTrainer(cfg, model_cfg)
trainer.train()
```

Resume from checkpoint:

```python
trainer = PPOTrainer(cfg, model_cfg)
trainer.load_checkpoint("checkpoints/model_epoch_100.pt")
trainer.train()
```
