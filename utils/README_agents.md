# `utils/agents.py` — Poker Agent Implementations

## Purpose

Provides all agent implementations that conform to the game engine's agent interface: `agent(GameState) → Action`. Four agents are included, ranging from random baseline to the full RL policy.

---

## Agent Hierarchy

```
BaseAgent (ABC)
├── RandomAgent      — uniform random legal action
├── CallAgent        — always checks/calls, never raises
├── RuleBasedAgent   — heuristic strategy (hand-strength based)
└── RLAgent          — transformer model + RL policy (the learnable agent)
```

All agents receive a `GameState` snapshot and return an `Action`. None may mutate the state.

---

## `RandomAgent`

Uniformly samples from the set of legal actions. Raise amount is sampled uniformly in `[min_raise, player_chips]`.

**Used for:** opponent pool during early training, performance baseline.

```python
agent = RandomAgent(seed=42)
```

---

## `CallAgent`

Always calls or checks. Never raises, rarely folds.

**Used for:** chip-conservation testing, passive baseline.

---

## `RuleBasedAgent`

Heuristic strategy parameterised by `aggression ∈ [0, 1]`.

### Preflop Logic

```
Hand strength tier:
  Premium  (AA, KK, QQ, AK)        → raise 3× BB  (with probability ∝ aggression)
  Playable (any pair, suited conn.) → call
  Weak                              → fold if there's a bet
```

### Postflop Logic

```
Evaluate hand with HandEvaluator:
  Three-of-a-kind or better         → raise 75% pot  (if random < aggression)
  Pair                              → call
  High card / draw                  → check or fold
```

**Used for:** stronger baseline than random, pre-training opponent pool.

```python
agent = RuleBasedAgent(aggression=0.6)  # more aggressive
```

---

## `RLAgent`

The main learnable agent. Wraps `PokerTransformer` and `PokerTokenizer`.

### Decision Pipeline

```
GameState
    │
    ├─ extract hole_cards from state.you.hole_cards
    ├─ convert Card repr to strings: "A♥" → "Ah"
    │
    ▼
tokenizer.encode(history, hole_cards)
    │
    ▼  token_ids: (1, max_len)
    │
model.forward(token_ids)
    │
    ▼
action_logits (1, 19)    state_values (1, 1)
    │
    ▼
Categorical distribution
    │
    ├─ explore=True  → sample(dist)   (stochastic, used during training)
    └─ explore=False → argmax(logits) (deterministic, used during eval)
    │
    ▼
action_idx  →  decode_action()  →  action_token_to_action_type_and_amount()
    │
    ▼
Action(ActionType, amount, player_id, street)
```

### Post-Call Attributes (for PPO Trainer)

After each `__call__`, these attributes are set and consumed by `ppo_trainer.py`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `last_log_prob` | float | `log π(a\|s)` — needed for PPO ratio |
| `last_value` | float | `V(s)` from value head |
| `last_action_idx` | int | Index into action vocabulary |
| `last_token_ids` | Tensor | Encoded state, shape `(max_len,)` |

### Methods

```python
agent.set_explore(True)          # switch to stochastic mode
agent.update_model(new_model)    # hot-swap weights (for opponent sync)
```

---

## Agent Comparison

| Agent | Fold | Raise | Bluff | Uses Model | Memory |
|-------|------|-------|-------|------------|--------|
| `RandomAgent` | ~33% | ~33% | yes | No | None |
| `CallAgent` | rare | Never | No | No | None |
| `RuleBasedAgent` | situational | hand-based | No | No | None |
| `RLAgent` | learned | learned | learned | Yes | GPU |

---

## Usage Example

```python
from utils.agents import RLAgent, RandomAgent
from model.transformer import PokerTransformer, TransformerConfig
from model.tokenizer import PokerTokenizer
from engine.game import Game
from engine.player import Player

# Create RL agent
tok = PokerTokenizer()
cfg = TransformerConfig(vocab_size=tok.vocab_size, num_actions=tok.num_actions)
model = PokerTransformer(cfg)
rl_agent = RLAgent(model, tok, player_id=0, explore=True)

# Build game
players = [Player(i, f"P{i}", 1000) for i in range(6)]
agents  = {0: rl_agent, **{i: RandomAgent() for i in range(1, 6)}}
game    = Game(players, agents, small_blind=10, big_blind=20)

result = game.play_hand()
# After the hand, consume PPO info:
print(rl_agent.last_log_prob)    # log π(a|s)
print(rl_agent.last_value)       # V(s)
print(rl_agent.last_action_idx)  # chosen action index
```
