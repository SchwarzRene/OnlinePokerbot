# `engine/player.py` — Player State & Action Definitions

## Purpose

Defines the **Player** data class and the **Action** / **ActionType** objects. Every decision in the simulation passes through this module.

---

## Lifecycle

```
Player created (id, name, chips)
        │
        ▼
reset_for_hand()       ← called at start of every new hand
        │  clears: hole_cards, bet, total_bet, is_folded, is_all_in, action_log
        ▼
reset_for_street()     ← called at start of flop / turn / river
        │  zeroes: bet  (keeps total_bet for side-pot calculation)
        ▼
apply_action(action)   ← called each time the player acts
        │  mutates: chips, bet, total_bet, is_folded, is_all_in
        ▼
(repeat per hand)
```

---

## `ActionType` Enum

| Name | Description |
|------|-------------|
| `FOLD` | Surrender the hand |
| `CHECK` | Pass without betting (only when no bet to call) |
| `CALL` | Match the current highest bet |
| `RAISE` | Increase the bet by a specified amount |
| `ALL_IN` | Commit all remaining chips |

---

## `Action` Dataclass

```
Action
├── action_type : ActionType
├── amount      : int        relevant for RAISE and ALL_IN; 0 for others
├── player_id   : int        who took this action (set by game engine)
└── street      : str        "preflop" | "flop" | "turn" | "river"
```

### Token Serialisation

Actions convert to compact token strings for transformer input:

| Action | Token |
|--------|-------|
| `FOLD` | `<FOLD>` |
| `CHECK` | `<CHECK>` |
| `CALL` | `<CALL>` |
| `RAISE 120` | `<RAISE:120>` |
| `ALL_IN 300` | `<ALL_IN:300>` |

```python
action = Action(ActionType.RAISE, 80)
action.to_token()       # "<RAISE:80>"
Action.from_token("<RAISE:80>")  # reconstructs Action
```

---

## `Player` Dataclass

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Seat index (0-based) |
| `name` | str | Display name |
| `chips` | int | Current chip stack |
| `hole_cards` | `List[Card]` | Private two-card hand |
| `bet` | int | Chips committed this street |
| `total_bet` | int | Cumulative chips committed this hand (used for side pots) |
| `is_folded` | bool | Has surrendered the hand |
| `is_all_in` | bool | Has committed all chips |
| `is_active` | bool | Is seated and participating |
| `action_log` | `List[Action]` | All actions this hand |

### Key Properties

| Property | Returns | Meaning |
|----------|---------|---------|
| `can_act` | bool | Active, not folded, not all-in, has chips |
| `in_hand` | bool | Not folded (includes all-in players) |

### `apply_action` Behaviour

```
FOLD    → is_folded = True.  No chips move.
CHECK   → No chips move.
CALL    → chips -= min(call_amount, self.chips)
           bet   += amount moved
           if chips == 0: is_all_in = True
RAISE   → chips -= min(action.amount, self.chips)
           bet   += amount moved
           if chips == 0: is_all_in = True
ALL_IN  → bet   += self.chips
           chips  = 0
           is_all_in = True
```

> **Note:** `apply_action` returns the number of chips actually moved. This value is used by the pot manager to correctly track contributions even when a player can only partially match a bet.

---

## Chip Conservation Invariant

At any point during a hand:

```
player.chips + player.bet + pot.total == starting_chips (sum over all players)
```

This invariant is enforced by the game engine tests (500+ hand runs).

---

## Usage Example

```python
from engine.player import Player, Action, ActionType

p = Player(id=0, name="Alice", chips=500)
p.reset_for_hand()

# Post small blind
p.apply_action(Action(ActionType.RAISE, 10), call_amount=10)
# p.chips == 490, p.bet == 10

# Call a raise (call_amount = how many MORE chips are needed)
p.apply_action(Action(ActionType.CALL), call_amount=40)
# p.chips == 450, p.bet == 50

print(p.can_act)   # True
print(p.in_hand)   # True
```
