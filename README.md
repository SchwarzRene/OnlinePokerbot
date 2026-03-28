# Poker RL — Texas Hold'em Reinforcement Learning System

A complete system for simulating poker hands and training a transformer-based RL agent to play Texas Hold'em No-Limit via self-play PPO.

---

## Project Structure

```
poker_rl/
├── engine/                     ← Simulation layer (no ML dependencies)
│   ├── cards.py                Card, Deck, HandEvaluator
│   ├── player.py               Player state, Action, ActionType
│   ├── pot.py                  Pot manager, side-pot calculation
│   ├── game.py                 Full hand orchestration
│   ├── README_cards.md
│   ├── README_player.md
│   ├── README_pot.md
│   └── README_game.md
│
├── model/                      ← ML layer (requires PyTorch)
│   ├── tokenizer.py            Game history → integer sequences
│   ├── transformer.py          Decoder-only transformer (policy + value heads)
│   ├── README_tokenizer.md
│   └── README_transformer.md
│
├── training/                   ← RL training
│   ├── ppo_trainer.py          PPO self-play trainer
│   └── README_ppo_trainer.md
│
├── utils/                      ← Agent implementations
│   ├── agents.py               RandomAgent, CallAgent, RuleBasedAgent, RLAgent
│   └── README_agents.md
│
├── tests/                      ← Test suite
│   ├── test_engine.py          All edge cases
│   └── README_tests.md
│
├── main.py                     ← CLI entry point
├── README_main.md
├── requirements.txt
└── README.md                   ← This file
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
│                                                                  │
│  ┌────────────┐    GameState     ┌─────────────────────────┐    │
│  │            │ ────────────►   │       RLAgent           │    │
│  │   Game     │                 │  ┌─────────────────────┐ │    │
│  │  (engine)  │ ◄────────────   │  │PokerTokenizer       │ │    │
│  │            │    Action        │  │ history → token_ids │ │    │
│  └────────────┘                 │  └────────┬────────────┘ │    │
│        │                        │           │               │    │
│        │ HandResult             │  ┌────────▼────────────┐ │    │
│        │ (reward)               │  │PokerTransformer     │ │    │
│        ▼                        │  │ token_ids → logits  │ │    │
│  ┌────────────┐                 │  │              values │ │    │
│  │PPOTrainer  │                 │  └────────┬────────────┘ │    │
│  │            │ ◄────────────── │           │               │    │
│  │ rollout    │  (log_prob,     │  sample action_idx        │    │
│  │ buffer     │   value,        └─────────────────────────┘    │
│  │ optimizer  │   action_idx)                                   │
│  └────────────┘                                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Game → Model → Action

```
1. Game engine runs a hand:
   history = ["<HAND:5>", "<ROUND:preflop>", "<P0:SB:10>", "<P2:RAISE:60>", ...]

2. Tokenizer encodes history + hole cards:
   token_ids = [1, 847, 23, 44, 112, ...]   (length = max_len = 512)

3. Transformer forward pass:
   action_logits, state_value = model(token_ids)
   # logits: (19,)   value: (1,)

4. Sample action:
   action_idx = Categorical(softmax(logits)).sample()

5. Decode to Action:
   "<ACT:RAISE:AMT3>" → Action(ActionType.RAISE, amount=75)

6. Game engine applies action, advances hand.

7. At hand end:
   reward = (chips_after - chips_before) / big_blind
   Experience(token_ids, action_idx, log_prob, value, reward) → buffer

8. PPO optimizer uses buffer to update model.
```

---

## Transformer Input Sequence

```
<BOS>
<HOLE> <CARD:Ah> <CARD:Kd>           ← your two private cards
<HAND:42>
<ROUND:preflop>
  <P0:SB:AMT1>                       ← small blind (10 chips, bucket 1)
  <P1:BB:AMT1>                       ← big blind (20 chips, bucket 1)
  <P2:RAISE:AMT3>                    ← raise to ~75 (bucket 3)
  <P3:FOLD>
  <P4:CALL>
  <P0:FOLD>
  <P1:CALL>
<ROUND:flop>
<BOARD> <CARD:Qh> <CARD:Jc> <CARD:2s>
  <P1:CHECK>
  <P4:RAISE:AMT4>
  ...
<EOS>
<PAD> <PAD> ... <PAD>
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy pytest
```

### 2. Verify engine (no ML needed)

```bash
python main.py simulate --hands 10
```

### 3. Run all tests

```bash
python main.py test
```

### 4. Train the model

```bash
python main.py train --epochs 200
```

### 5. Evaluate a checkpoint

```bash
python main.py eval --checkpoint checkpoints/model_epoch_200.pt --hands 500
```

---

## Performance Metrics

| Metric | Description |
|--------|-------------|
| `reward` (mean BB/hand) | Average chips won per hand in big-blind units |
| `BB/100` | Standard poker metric: BBs won per 100 hands |
| `policy_loss` | Should decrease during training |
| `entropy` | Should stay > 1.5 to avoid premature convergence |
| `KL divergence` | Should stay < `target_kl` (0.02) for stable updates |

---

## Dependencies

| Package | Version | Used for |
|---------|---------|----------|
| `torch` | ≥ 2.0 | Transformer model, PPO gradients |
| `numpy` | ≥ 1.24 | GAE computation, rollout buffers |
| `pytest` | ≥ 7.4 | Test runner (optional) |

The **engine layer** (`engine/`) has **zero external dependencies** and runs on pure Python. Only `model/` and `training/` require PyTorch.

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
