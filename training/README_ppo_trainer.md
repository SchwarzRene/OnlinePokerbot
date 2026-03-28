# `training/ppo_trainer.py` — PPO Self-Play Trainer

Trains the `PokerTransformer` using **Proximal Policy Optimization (PPO)**
with self-play. The agent plays against frozen copies of itself and
improves through iterative rollout collection and gradient updates.

---

## Algorithm: PPO (Schulman et al., 2017)

PPO is an **on-policy actor-critic** algorithm. It collects experience under
the current policy, then optimises a clipped surrogate objective that prevents
dangerously large policy updates.

### Why PPO for Poker?

```
  Poker challenges:                    PPO solution:
  ─────────────────────────────────    ─────────────────────────────────────
  Sparse rewards (only at hand end)    GAE smooths delayed credit assignment
  High variance (luck element)         Value baseline subtracts noise
  Non-stationary opponents             Self-play syncs opponents periodically
  Combinatorial action space           19-bucket discrete action head
  Partial observability                Transformer attends to full history
```

---

## Self-Play Setup

```
  Training agent  (P0)          Frozen opponent pool  (P1..P8)
  ────────────────────          ──────────────────────────────
  Uses current policy           Uses snapshot of policy from
  Stochastic (explore=True)     `opponent_update_interval` hands ago
  Gradients flow through it     Deterministic (explore=False)
  Updated every PPO epoch       Updated every 50 hands via weight copy

  Every hand: n_players = randint(2, 9)
  Opponents P1..P(n-1) are drawn from the pool of 8 frozen agents.

  Why frozen opponents?
    If both sides update simultaneously, the game becomes non-stationary
    in a way that causes training instability (similar to GAN mode collapse).
    Freezing one side gives the training agent a stable target to improve against.
```

---

## Training Loop Detail

```
  PPOTrainer.train()
  │
  ├── for epoch = 1 .. total_epochs:
  │
  │   ┌── ROLLOUT PHASE ────────────────────────────────────────────────┐
  │   │                                                                  │
  │   │   buffer.clear()                                                │
  │   │                                                                  │
  │   │   for hand = 1 .. hands_per_rollout (200):                      │
  │   │       n_players = randint(min=2, max=9)                         │
  │   │       game, players = build_game(n_players)                     │
  │   │                                                                  │
  │   │       chips_before = players[0].chips                           │
  │   │       result = game.play_hand()                                 │
  │   │       chips_after  = players[0].chips                           │
  │   │                                                                  │
  │   │       reward = (chips_after - chips_before) / starting_chips    │
  │   │       reward = clip(reward, -1.0, +1.0)                         │
  │   │                                                                  │
  │   │       # Flush ALL decisions the agent made this hand            │
  │   │       for i, exp in enumerate(rl_agent.flush_hand_buffer()):    │
  │   │           exp.reward = reward if i == last else 0.0             │
  │   │           exp.done   = (i == last)                              │
  │   │           buffer.add(exp)                                        │
  │   │       # Result: ~4-8 experiences per hand instead of 1          │
  │   │                                                                  │
  │   │       if hand_count % 50 == 0:                                  │
  │   │           frozen_model.load_state_dict(trained_model.state_dict)│
  │   │                                                                  │
  │   └──────────────────────────────────────────────────────────────────┘
  │
  │   ┌── ADVANTAGE COMPUTATION ────────────────────────────────────────┐
  │   │                                                                  │
  │   │   GAE (Generalized Advantage Estimation):                       │
  │   │                                                                  │
  │   │   delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)       │
  │   │                                                                  │
  │   │   A_t = delta_t                                                 │
  │   │         + gamma * lambda * delta_{t+1}                          │
  │   │         + (gamma * lambda)^2 * delta_{t+2}                      │
  │   │         + ...                                                    │
  │   │                                                                  │
  │   │   Intuition:                                                     │
  │   │     lambda=0 → pure TD(0), low variance, high bias              │
  │   │     lambda=1 → Monte Carlo, high variance, low bias             │
  │   │     lambda=0.95 → smooth blend (default)                        │
  │   │                                                                  │
  │   │   returns_t = A_t + V(s_t)   ← critic training target          │
  │   │   normalize: A_t = (A_t - mean(A)) / (std(A) + 1e-8)           │
  │   │                                                                  │
  │   └──────────────────────────────────────────────────────────────────┘
  │
  │   ┌── PPO UPDATE PHASE ─────────────────────────────────────────────┐
  │   │                                                                  │
  │   │   for opt_epoch = 1 .. n_opt_epochs (4):                        │
  │   │       shuffle(buffer_indices)                                    │
  │   │                                                                  │
  │   │       for mini_batch in batches(indices, batch_size=64):        │
  │   │                                                                  │
  │   │           logits, values = model(token_ids)                     │
  │   │           dist = Categorical(logits=logits)                     │
  │   │           new_log_probs = dist.log_prob(actions)                │
  │   │           entropy = dist.entropy().mean()                       │
  │   │                                                                  │
  │   │           ratio = exp(new_log_probs - old_log_probs)            │
  │   │                                                                  │
  │   │           ┌── Policy Loss (clipped surrogate objective) ─────┐  │
  │   │           │                                                   │  │
  │   │           │   unclipped = ratio * advantages                 │  │
  │   │           │   clipped   = clamp(ratio, 1-eps, 1+eps) * adv   │  │
  │   │           │   L_policy  = -mean(min(unclipped, clipped))     │  │
  │   │           │                                                   │  │
  │   │           │   eps=0.2 means ratio can't deviate by >20%:     │  │
  │   │           │   ratio in [0.8, 1.2]                            │  │
  │   │           └───────────────────────────────────────────────────┘  │
  │   │                                                                  │
  │   │           ┌── Value Loss ───────────────────────────────────┐   │
  │   │           │   L_value = MSE(values.squeeze(), returns)      │   │
  │   │           └──────────────────────────────────────────────────┘   │
  │   │                                                                  │
  │   │           ┌── Entropy Bonus ────────────────────────────────┐   │
  │   │           │   L_entropy = dist.entropy().mean()             │   │
  │   │           │   (penalising low entropy prevents premature    │   │
  │   │           │    convergence to a deterministic policy)       │   │
  │   │           └──────────────────────────────────────────────────┘   │
  │   │                                                                  │
  │   │           loss = L_policy + c1*L_value - c2*L_entropy           │
  │   │                           (c1=0.5)    (c2=0.01)                 │
  │   │                                                                  │
  │   │           optimizer.zero_grad()                                  │
  │   │           loss.backward()                                        │
  │   │           clip_grad_norm_(params, max_norm=0.5)                 │
  │   │           optimizer.step()                                       │
  │   │                                                                  │
  │   │       if mean(KL) > target_kl (0.02): break  ← early stop      │
  │   │                                                                  │
  │   └──────────────────────────────────────────────────────────────────┘
  │
  │   scheduler.step()   ← cosine LR annealing
  │
  └── (repeat until total_epochs)
```

---

## GAE Visual Intuition

```
  Hand timeline (each step = one decision the training agent makes):

  t=0              t=1                   t=N (hand ends)
  │                │                     │
  ▼                ▼                     ▼
  [act:RAISE]  [act:CALL]  ...  [act:FOLD]    reward = -1.5 BB (lost)
      │                                   │
      │◄──────────────────────────────────┘
      │  GAE propagates reward backward with decay (gamma=0.99, lambda=0.95)
      │
      │  A_0 ≈ -1.5 * (0.99*0.95)^N     (heavily discounted for early actions)
      │  A_{N-1} ≈ -1.5                  (last action gets most of the blame)

  Good sequence (won 3 BB):
    all advantages positive → those actions reinforced
  Bad sequence (lost 1.5 BB):
    all advantages negative → those actions suppressed
```

---

## Reward Design

```
  Per-step reward  = 0.0          (no intermediate reward signals)
  Terminal reward  = (chips_after - chips_before) / big_blind

  Example:
    Starting chips:  2000  (100 BB)
    Won a pot:       +200 chips  → reward = +200 / 2000 = +0.10
    Lost a hand:      -40 chips  → reward =  -40 / 2000 = -0.02
    Full stack loss: -2000 chips → reward = -2000 / 2000 = -1.0 (hard floor)

  Normalisation rationale:
    Rewards are divided by starting_chips (2000), NOT by big_blind (20).
    This keeps every reward in [-1, +1], which is critical for the value
    head. If you divide by BB instead, a full-stack pot win = +100, and
    MSE(value_pred ≈ 0, target = 100) ≈ 10,000 — the value loss explodes
    and drowns the policy gradient completely (seen as vloss > 20,000 in logs).
    After normalisation, healthy vloss should reach < 1.0 within ~50 epochs.

  Reward assignment across decisions:
    Only the final decision of a hand receives the terminal reward.
    All earlier decisions in the same hand receive reward = 0.
    GAE then propagates credit backwards through the hand with decay,
    so preflop decisions get a discounted version of the terminal signal.

  Stack management:
    Players start each session with 2000 chips (100 BB standard buy-in).
    Between hands, any player below 800 chips (40 BB) is topped up to 2000.
    This mirrors the auto top-up feature on online poker sites and ensures
    the model always trains in realistic 100 BB effective-stack situations
    rather than distorted push/fold dynamics that arise below ~20 BB.

  fold_penalty (default 0.0):
    Optional small negative reward added when the agent folds.
    Useful to discourage over-folding if the agent collapses to
    always folding, but generally left at 0 to avoid reward shaping bias.
```

---

## Hyperparameter Reference

| Parameter | Default | Effect of increasing |
|-----------|---------|----------------------|
| `starting_chips` | 2000 | Buy-in stack per player (100 BB). Also used as reward denominator. |
| `min_players` | 2 | Lower bound of table size sampling |
| `max_players` | 9 | Upper bound of table size sampling |
| `hands_per_rollout` | 200 | More data per update → stable but slower |
| `clip_eps` | 0.2 | Larger clip → bigger updates, less stable |
| `gamma` | 0.99 | Higher → rewards propagate further back |
| `gae_lambda` | 0.95 | Higher → lower bias, higher variance |
| `c1` | 0.05 | Value loss weight → better critic. Kept low so vloss never swamps ploss. |
| `c2` | 0.01 | Entropy weight → more exploration |
| `target_kl` | 0.02 | Higher → fewer early stops |
| `n_opt_epochs` | 4 | More gradient steps per rollout |
| `batch_size` | 64 | Larger → smoother gradients |
| `lr` | 3e-4 | Standard Adam rate for transformers |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `opponent_update_interval` | 50 | Hands between opponent weight sync |

---

## Experience collection

Every decision the training agent makes is recorded, not just the final one.
`RLAgent` accumulates an internal `hand_buffer` during each hand. After
`game.play_hand()` returns, the trainer calls `rl_agent.flush_hand_buffer()`
and writes all decisions to the rollout buffer:

```
  Preflop raise  → Experience(tokens, action, log_prob, value, reward=0,    done=False)
  Flop call      → Experience(tokens, action, log_prob, value, reward=0,    done=False)
  Turn check     → Experience(tokens, action, log_prob, value, reward=0,    done=False)
  River fold     → Experience(tokens, action, log_prob, value, reward=±r,   done=True)
```

GAE propagates the terminal reward backwards, so earlier decisions receive
a discounted version of the hand outcome. This gives 4–8× more gradient
signal per hand compared to storing only the final action.

---

## Training charts

Every time a checkpoint is saved, a `training_charts_{label}.png` is written
to the same directory. The chart contains four panels:

| Panel | What to look for |
|-------|------------------|
| **Mean reward** | Smoothed trend should turn positive by epoch ~100–200. Raw is noisy — ignore short dips. |
| **Policy loss + KL** | Policy loss should drift toward zero. KL should stay below `target_kl` (0.02). |
| **Value loss (log scale)** | Must drop from thousands to < 1.0. If it stays high, reward normalisation is broken. |
| **Entropy** | Should decrease slowly from ~2.8. If it collapses below 1.0 early, the model is stuck. |

Matplotlib is an optional dependency — if it is not installed, chart saving
is silently skipped and a warning is printed. Install with:
```bash
pip install matplotlib
```

---

## Usage

```python
from training.ppo_trainer import PPOTrainer, PPOConfig
from model.transformer import TransformerConfig

ppo_cfg = PPOConfig(
    min_players=2,
    max_players=9,
    total_epochs=500,
    hands_per_rollout=200,
)
model_cfg = TransformerConfig()   # defaults: d_model=128, n_layers=4

trainer = PPOTrainer(ppo_cfg, model_cfg)
trainer.train()
```

Resume from a checkpoint:
```python
trainer = PPOTrainer(ppo_cfg, model_cfg)
trainer.load_checkpoint("checkpoints/model_epoch_200.pt")
trainer.train()
```
