# `main.py` — CLI Entry Point

## Purpose

Command-line interface for the entire poker RL system. Provides four subcommands: `train`, `eval`, `simulate`, and `test`.

---

## Commands

### `train` — Run PPO Self-Play Training

```bash
python main.py train [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--players` | 6 | Number of players at the table |
| `--epochs` | 500 | Number of training epochs |
| `--rollout-hands` | 200 | Hands collected per epoch |
| `--d-model` | 128 | Transformer embedding dimension |
| `--n-heads` | 4 | Attention heads |
| `--n-layers` | 4 | Transformer blocks |
| `--checkpoint` | None | Resume from a saved checkpoint |
| `--checkpoint-dir` | `checkpoints/` | Where to save checkpoints |

**Examples:**

```bash
# Start training from scratch
python main.py train

# Train a larger model
python main.py train --d-model 256 --n-heads 8 --n-layers 6

# Resume training
python main.py train --checkpoint checkpoints/model_epoch_200.pt
```

---

### `eval` — Evaluate a Saved Model

```bash
python main.py eval --checkpoint PATH [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to `.pt` checkpoint file |
| `--players` | 6 | Total players (1 RL + N-1 rule-based opponents) |
| `--hands` | 500 | Number of evaluation hands |
| `--verbose` | False | Print per-hand profit |

**Output:**
```
=== Evaluation Results ===
  Hands played : 500
  Total profit : +1840 chips
  BB/100 hands : +18.40
```

`BB/100` (big blinds per 100 hands) is the standard poker performance metric. A random agent scores ~0 BB/100. A winning agent should score > 5 BB/100 against rule-based opponents.

---

### `simulate` — Quick Simulation

```bash
python main.py simulate [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--players` | 6 | Players at the table |
| `--hands` | 20 | Hands to simulate |
| `--verbose` | True | Print hand summaries |
| `--show-history` | False | Print full token history per hand |

Runs `RuleBasedAgent` vs. `RuleBasedAgent` — useful for verifying the engine is working and observing the token format.

**Example output:**
```
--- Hand 1 ---
  Winners : [2]
  Winnings: {2: 60}
  Stacks  : {0: 480, 1: 470, 2: 550, 3: 500, 4: 500, 5: 500}

--- Hand 2 ---
  Winners : [0]
  Winnings: {0: 40}
  ...
```

---

### `test` — Run the Test Suite

```bash
python main.py test
```

Delegates to `pytest tests/test_engine.py -v --tb=short`. Exits with code 0 on success, 1 on failure.

---

## Typical Workflow

```
1. Verify engine works:
   python main.py simulate --hands 10

2. Run tests:
   python main.py test

3. Start training:
   python main.py train --epochs 500

4. Evaluate at checkpoints:
   python main.py eval --checkpoint checkpoints/model_epoch_200.pt --hands 1000

5. Continue training from best checkpoint:
   python main.py train --checkpoint checkpoints/model_epoch_200.pt --epochs 1000
```
