# Poker RL — Texas Hold'em Reinforcement Learning System

A complete system for training a transformer-based RL agent to play No-Limit Texas Hold'em
via self-play PPO. The engine layer is pure Python with zero ML dependencies; only the
model and training layers require PyTorch.

---

## Project Structure

```
poker_rl/
├── engine/                     ← Simulation layer (no ML dependencies)
│   ├── cards.py                Card, Deck, HandEvaluator
│   ├── player.py               Player state, Action, ActionType
│   ├── pot.py                  Pot manager, side-pot calculation
│   ├── game.py                 Full hand orchestration
│   └── README_*.md             Per-module documentation
│
├── model/                      ← ML layer (requires PyTorch)
│   ├── tokenizer.py            Game history → integer sequences
│   ├── transformer.py          Decoder-only transformer (policy + value heads)
│   └── README_*.md
│
├── training/
│   ├── ppo_trainer.py          PPO self-play trainer
│   └── README_ppo_trainer.md
│
├── utils/
│   ├── agents.py               RandomAgent, CallAgent, RuleBasedAgent, RLAgent
│   └── README_agents.md
│
├── tests/
│   ├── test_engine.py          Engine test suite (70 tests)
│   └── README_tests.md
│
├── main.py                     CLI entry point
└── README.md                   This file
```

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         TRAINING LOOP (PPOTrainer)                      ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │                        ROLLOUT PHASE                             │   ║
║  │                                                                  │   ║
║  │  ┌─────────────┐    GameState    ┌──────────────────────────┐   │   ║
║  │  │             │ ──────────────► │         RLAgent          │   │   ║
║  │  │  Game       │                 │  ┌────────────────────┐  │   │   ║
║  │  │  (engine)   │                 │  │  PokerTokenizer    │  │   │   ║
║  │  │             │                 │  │  history+cards     │  │   │   ║
║  │  │  2–9 seats  │                 │  │  → token_ids[512]  │  │   │   ║
║  │  │  random     │                 │  └────────┬───────────┘  │   │   ║
║  │  │  per hand   │                 │           │               │   │   ║
║  │  │             │                 │  ┌────────▼───────────┐  │   │   ║
║  │  │             │ ◄────────────── │  │  PokerTransformer  │  │   │   ║
║  │  │             │    Action        │  │  → action_logits   │  │   │   ║
║  │  └─────────────┘                 │  │  → state_value     │  │   │   ║
║  │        │                         │  └────────────────────┘  │   │   ║
║  │        │ HandResult              └──────────────────────────┘   │   ║
║  │        │ (reward = Δchips / starting_chips, clipped [-1,+1])     │   ║
║  │        ▼                                                          │   ║
║  │  ┌──────────────┐                                                │   ║
║  │  │ RolloutBuffer│  stores (token_ids, action, log_prob, value,   │   ║
║  │  │              │          reward, done) per decision step        │   ║
║  │  └──────────────┘                                                │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │                        UPDATE PHASE (PPO)                        │   ║
║  │                                                                  │   ║
║  │  Compute GAE advantages and discounted returns                   │   ║
║  │  for mini_batch in shuffle(buffer):                              │   ║
║  │      pi_new, V_new  ← model(token_ids)                          │   ║
║  │      ratio         = exp(log pi_new − log pi_old)               │   ║
║  │      L_policy      = −min(ratio·A, clip(ratio,1±ε)·A)           │   ║
║  │      L_value       = MSE(V_new, returns)                        │   ║
║  │      L_entropy     = −Σ pi·log(pi)                              │   ║
║  │      loss = L_policy + 0.5·L_value − 0.01·L_entropy            │   ║
║  │      loss.backward(); optimizer.step()                           │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow: State → Token Sequence → Action

```
  GameState
  ─────────
  street = "flop"
  pot    = 120
  board  = [Qh, Jc, 2s]
  you    = PlayerView(id=0, chips=460, hole=[Ah,Kd], bet=40)
  history= ["<HAND:7>","<ROUND:preflop>","<P0:SB:10>", ...]

        │
        │  PokerTokenizer.encode(history, hole_cards, num_players=4)
        ▼

  Token Sequence (length = max_len = 512):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ <BOS> <NUM_PLAYERS:4>                 ← table size context           │
  │ <HOLE> <CARD:Ah> <CARD:Kd>           ← your private cards           │
  │ <HAND:7>                             ← hand number                  │
  │ <ROUND:preflop>                                                      │
  │   <P0:SB:AMT1>  <P1:BB:AMT1>        ← blinds (bucket 1 = 10-19)    │
  │   <P2:RAISE:AMT3>  <P3:FOLD>        ← opponents acted               │
  │   <P0:CALL>  <P1:CALL>                                              │
  │ <ROUND:flop>                                                         │
  │   <BOARD> <CARD:Qh> <CARD:Jc> <CARD:2s>                            │
  │   <P1:CHECK>  <P2:RAISE:AMT4>                                       │
  │   [you must act here]                                                │
  │ <EOS>                                                                │
  │ <PAD> <PAD> ... <PAD>                ← zero-padded to 512           │
  └──────────────────────────────────────────────────────────────────────┘

        │
        │  PokerTransformer.forward(token_ids)
        ▼

  action_logits: (19,)   ← unnormalized scores for each possible action
  state_value:   (1,)    ← V(s) baseline for PPO

        │
        │  Categorical(softmax(logits)).sample()
        ▼

  action_idx = 7          → "<ACT:RAISE:AMT4>"

        │
        │  tokenizer.action_token_to_action_type_and_amount()
        ▼

  Action(ActionType.RAISE, amount=149, player_id=0, street="flop")
```

---

## Chip Stack & Blind Structure

The simulation uses standard online cash game conventions:

```
  Small blind            :   10 chips  =  0.5 BB
  Big blind              :   20 chips  =  1 BB
  Default starting stack : 2000 chips  = 100 BB   ← standard online buy-in
  Rebuy threshold        :  800 chips  =  40 BB   ← auto top-up threshold
```

**Buy-in vs. rebuy:** Each player is created once per session with a 100 BB
stack (`Player(..., chips=2000)`). Between hands, any player whose stack has
dropped below 40 BB is automatically topped back up to 100 BB — identical to
the "auto top-up" feature on every major online poker site (PokerStars, GGPoker,
etc.). This keeps effective stack depths realistic for the model to learn from.

## Bet Sizing / Chip Denomination Reference

The model selects from **8 amount buckets** — not exact chip counts.
When the bucket is decoded, the midpoint of that range is used as the
actual raise amount, clamped to the player's remaining chips.

```
  Bucket  Chip Range    Midpoint   BB equiv   Typical Situation
  ──────  ───────────   ────────   ─────────  ──────────────────────────────────────
    0       1 –    9        5       0.25 BB   Completing a straddle, min-bet
    1      10 –   19       14       0.7  BB   Small blind, limp, min-raise
    2      20 –   49       34       1.7  BB   Standard open-raise (1.5–2.5 BB)
    3      50 –   99       74       3.7  BB   3-bet / pot-sized continuation bet
    4     100 –  199      149       7.5  BB   Large 3-bet, flop overbet
    5     200 –  499      349      17.5  BB   Turn/river big bet, 4-bet
    6     500 –  999      749      37.5  BB   Near-stack shove (40–50 BB)
    7    1000+           1500      75+  BB    Deep shove / full stack all-in
  ──────  ───────────   ────────   ─────────  ──────────────────────────────────────

  Default starting stack : 2000 chips = 100 BB  (standard online cash game)
  Small blind            :   10 chips =   0.5 BB
  Big blind              :   20 chips =   1 BB
  Rebuy threshold        :  800 chips =  40 BB   (auto top-up, like online sites)
```

**Full action menu (19 actions total):**

```
  Index   Token                  Meaning
  ─────   ─────────────────────  ──────────────────────────────────────
    0     <ACT:FOLD>             Surrender hand, forfeit all bets
    1     <ACT:CHECK>            Pass (only legal when no bet to call)
    2     <ACT:CALL>             Match current bet exactly
    3     <ACT:RAISE:AMT0>       Raise to ~5 chips  (bucket 0)
    4     <ACT:RAISE:AMT1>       Raise to ~14 chips (bucket 1)
    5     <ACT:RAISE:AMT2>       Raise to ~34 chips (bucket 2)
    6     <ACT:RAISE:AMT3>       Raise to ~74 chips (bucket 3)
    7     <ACT:RAISE:AMT4>       Raise to ~149 chips (bucket 4)
    8     <ACT:RAISE:AMT5>       Raise to ~349 chips (bucket 5)
    9     <ACT:RAISE:AMT6>       Raise to ~749 chips (bucket 6)
   10     <ACT:RAISE:AMT7>       Raise to ~1000+ chips (bucket 7)
   11     <ACT:ALL_IN:AMT0>      All-in (short stack, ~5 chips)
   12     <ACT:ALL_IN:AMT1>      All-in (~14 chips)
   13     <ACT:ALL_IN:AMT2>      All-in (~34 chips)
   14     <ACT:ALL_IN:AMT3>      All-in (~74 chips)
   15     <ACT:ALL_IN:AMT4>      All-in (~149 chips)
   16     <ACT:ALL_IN:AMT5>      All-in (~349 chips)
   17     <ACT:ALL_IN:AMT6>      All-in (~749 chips)
   18     <ACT:ALL_IN:AMT7>      All-in (full stack, 1000+ chips)
  ─────   ─────────────────────  ──────────────────────────────────────

  The game engine silently converts RAISE to ALL_IN if the
  decoded amount >= the player's remaining chip count.
```

---

## Variable Player Count Training

The model is trained on tables of **2 to 9 players** simultaneously.
Every hand, a fresh player count is drawn uniformly at random.
The `<NUM_PLAYERS:N>` token is always the second token in every
sequence (position 1, right after `<BOS>`) so the model can
condition its strategy on the current table format:

```
  Heads-up (2p):   <BOS> <NUM_PLAYERS:2> <HOLE>...  → play very loose
  6-max:           <BOS> <NUM_PLAYERS:6> <HOLE>...  → standard ranges
  Full ring (9p):  <BOS> <NUM_PLAYERS:9> <HOLE>...  → tighten ranges
```

This prevents over-fitting to one format and produces a single
general-purpose policy for any game.

---

## PPO Training Loop

```
  for epoch = 1 .. total_epochs:
  │
  ├── COLLECT ROLLOUTS  (hands_per_rollout = 200 hands)
  │    for each hand:
  │      n_players = randint(min_players=2, max_players=9)
  │      play hand: trained agent (P0) vs N-1 frozen clones
  │      reward = clip((chips_after - chips_before) / starting_chips, -1, +1)
  │      for each decision the agent made this hand (4–8 per hand):
  │          store Experience(tokens, action, log_prob, value,
  │                           reward=reward if last else 0, done=is_last)
  │
  ├── COMPUTE GAE ADVANTAGES
  │    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
  │    A_t     = delta_t + (gamma*lambda)*delta_{t+1} + ...
  │    gamma=0.99,  lambda=0.95
  │    normalize: A_t = (A_t - mean) / (std + 1e-8)
  │
  ├── PPO UPDATE  (n_opt_epochs=4 passes, batch_size=64)
  │    for each mini-batch:
  │      ratio    = exp(log pi_new(a) - log pi_old(a))
  │      L_clip   = min(ratio * A, clamp(ratio, 0.8, 1.2) * A)
  │      L_val    = MSE(V(s), returns)
  │      L_ent    = mean entropy of pi_new
  │      loss     = -L_clip + 0.5 * L_val - 0.01 * L_ent
  │      Adam.step(); clip_grad_norm_(params, 0.5)
  │    early-stop epoch if mean KL > 0.02
  │
  └── SYNC OPPONENTS  (every 50 hands)
       frozen_model.weights ← trained_model.weights
```

---

## Transformer Architecture (Summary)

```
  token_ids  (B, T=512)
       │
       ▼  nn.Embedding(vocab=1413, d_model=128)
       │  + SinusoidalPositionalEncoding
       │
       ▼  TransformerBlock × 4
       │   ┌── Pre-LayerNorm(x)
       │   ├── MultiHeadSelfAttention (4 heads, causal mask)
       │   │   └── d_head = 32 per head
       │   └── FeedForward (Linear→GELU→Linear, d_ff=512)
       │
       ▼  Final LayerNorm
       │
       ▼  gather last non-PAD token hidden state  (B, 128)
       │
      ┌┴──────────────────────────┐
      │                           │
      ▼                           ▼
  Action Head                Value Head
  Linear(128, 19)            Linear(128, 64)
  → action_logits (B, 19)    → GELU
                              → Linear(64, 1)
                              → state_value (B, 1)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch numpy pytest

# 2. Verify engine (no ML needed)
python main.py simulate --hands 10

# 3. Run the test suite
python main.py test

# 4. Train (auto-selects 2–9 players per hand)
python main.py train --epochs 500
# → checkpoints/model_epoch_50.pt              (weights)
# → checkpoints/training_charts_epoch_50.png   (loss/reward plots)

# 4b. Fast CPU demo (~2–5 min for 1000 epochs)
python main.py train --demo --epochs 1000

# 5. Evaluate a checkpoint
python main.py eval --checkpoint checkpoints/model_epoch_200.pt --hands 500
```

---

## Performance Metrics

| Metric | Good Value | Description |
|--------|-----------|-------------|
| `BB/100` | > 5 vs rules | Big blinds won per 100 hands (eval command output) |
| `mean_reward` | > 0 | Normalised reward per hand, range [-1, +1] |
| `entropy` | > 1.5 | Higher = more exploration; collapse < 1.0 = model stuck |
| `policy_loss` | near 0 | Should trend toward zero over training |
| `value_loss` | < 1.0 | Drops from ~thousands early on; target < 1.0 after ~50 epochs |
| `kl` | < 0.02 | Update stability; > 0.02 triggers early stop per epoch |
| `buf` | 200–400 | Experiences per epoch; low buf = only last action stored (bug) |

---

## Dependencies

| Package | Min Version | Purpose |
|---------|-------------|---------|
| `torch` | 2.0 | Transformer model, PPO gradients |
| `numpy` | 1.24 | GAE computation, buffer shuffling |
| `matplotlib` | 3.5 | Training charts saved with each checkpoint (optional) |
| `pytest` | 7.4 | Test runner (optional) |

The `engine/` layer has **zero external dependencies** — runs on pure Python 3.9+.

---

## README Index

| File | README |
|------|--------|
| `engine/cards.py` | [README_cards.md](engine/README_cards.md) |
| `engine/player.py` | [README_player.md](engine/README_player.md) |
| `engine/pot.py` | [README_pot.md](engine/README_pot.md) |
| `engine/game.py` | [README_game.md](engine/README_game.md) |
| `model/tokenizer.py` | [README_tokenizer.md](model/README_tokenizer.md) |
| `model/transformer.py` | [README_transformer.md](model/README_transformer.md) |
| `training/ppo_trainer.py` | [README_ppo_trainer.md](training/README_ppo_trainer.md) |
| `utils/agents.py` | [README_agents.md](utils/README_agents.md) |
| `tests/test_engine.py` | [README_tests.md](tests/README_tests.md) |
| `main.py` | [README_main.md](README_main.md) |
