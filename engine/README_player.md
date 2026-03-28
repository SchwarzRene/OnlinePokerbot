# `engine/player.py` — Player State & Action Definitions

Defines the **Player** data class and the **Action** / **ActionType** objects.
Every decision in the simulation flows through this module.

---

## Lifecycle

```
  Player(id, name, chips)
         │
         ▼  (start of each hand)
  reset_for_hand()
         │  clears: hole_cards, bet, total_bet, is_folded, is_all_in, action_log
         ▼
  [preflop betting]
         │
         ▼  (start of flop, turn, river)
  reset_for_street()
         │  zeroes: bet  (keeps total_bet for side-pot calculation)
         ▼
  apply_action(action, call_amount)
         │  mutates: chips, bet, total_bet, is_folded, is_all_in
         ▼
  (repeat per street)
```

---

## ActionType Enum

```
  ActionType    Meaning                    Legal when
  ──────────    ─────────────────────────  ──────────────────────────────────
  FOLD          Surrender the hand         Always
  CHECK         Pass, no chips             call_amount == 0
  CALL          Match current bet          call_amount > 0
  RAISE         Increase the bet           chips > call_amount
  ALL_IN        Commit all chips           Always (overrides any bet limit)
```

---

## Action Dataclass

```
  Action
  ├── action_type : ActionType
  ├── amount      : int         chips involved (RAISE/ALL_IN); 0 for others
  ├── player_id   : int         set by game engine before logging
  └── street      : str         "preflop" | "flop" | "turn" | "river"

  Token format (for game history):
    FOLD           → "<P0:FOLD>"
    CHECK          → "<P1:CHECK>"
    CALL           → "<P2:CALL>"
    RAISE 80       → "<P3:RAISE:80>"
    ALL_IN 250     → "<P4:ALL_IN:250>"
```

---

## Player Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Seat index (0-based) |
| `name` | str | Display label |
| `chips` | int | Remaining chip stack |
| `hole_cards` | List[Card] | Private two-card hand |
| `bet` | int | Chips committed this street |
| `total_bet` | int | Cumulative chips this hand (used for side-pot sizing) |
| `is_folded` | bool | Has surrendered |
| `is_all_in` | bool | Has committed all chips |
| `action_log` | List[Action] | All actions this hand |

---

## apply_action Behaviour

```
  FOLD   → is_folded = True        (chips unchanged, exits hand)

  CHECK  → no chip movement        (only legal when call_amount == 0)

  CALL   → move = min(call_amount, self.chips)
            chips    -= move
            bet      += move
            total_bet+= move
            if chips == 0: is_all_in = True

  RAISE  → move = min(action.amount, self.chips)
            chips    -= move
            bet      += move
            total_bet+= move
            if chips == 0: is_all_in = True
            (game engine validates amount >= min_raise)

  ALL_IN → move = self.chips
            bet      += move
            total_bet+= move
            chips     = 0
            is_all_in = True
```

---

## Key Properties

| Property | Returns | Meaning |
|----------|---------|---------|
| `can_act` | bool | Not folded, not all-in, has chips, is active |
| `in_hand` | bool | Not folded (all-in players are still in hand) |

---

## Chip Conservation Invariant

At any point during or after a hand:

```
  sum(player.chips for all players)
  + sum(player.bet for all players)
  + pot.total
  == sum(starting chips for all players)

  This is verified by the test suite across 500+ hand runs.
```

---

## Usage Example

```python
from engine.player import Player, Action, ActionType

p = Player(id=0, name="Alice", chips=500)
p.reset_for_hand()

# Post small blind
p.apply_action(Action(ActionType.RAISE, 10), call_amount=10)
# p.chips=490, p.bet=10, p.total_bet=10

# Call a raise to 40 total (need 30 more)
p.apply_action(Action(ActionType.CALL), call_amount=30)
# p.chips=460, p.bet=40, p.total_bet=40

print(p.can_act)   # True
print(p.in_hand)   # True

# Go all-in on the flop
p.apply_action(Action(ActionType.ALL_IN, 460), call_amount=0)
# p.chips=0, p.bet=460, p.is_all_in=True
print(p.can_act)   # False (is_all_in)
print(p.in_hand)   # True
```
