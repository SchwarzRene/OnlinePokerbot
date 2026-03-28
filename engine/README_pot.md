# `engine/pot.py` — Pot Manager & Side-Pot Calculation

## Purpose

Manages chip collection during a hand and correctly distributes winnings including complex **side-pot scenarios** (multiple all-in players at different stack sizes).

---

## Concepts

### Main Pot vs. Side Pot

When a player goes all-in for less than the full bet, they can only win chips up to their own contribution from each other player. Any additional chips form a side pot that the all-in player is **not eligible** for.

```
Example: 3 players
─────────────────────────────────────────────────────
Player A: all-in for  50  → contributes  50
Player B: all-in for 100  → contributes 100
Player C: calls      100  → contributes 100

Main pot  = 50 × 3 = 150  ← A, B, C all eligible
Side pot  = 50 × 2 = 100  ← B, C eligible only
─────────────────────────────────────────────────────
```

---

## Data Model

```
Pot
├── _contributions : dict[player_id → int]   total chips each player put in
├── _side_pots     : List[SidePot]           computed by calculate_side_pots()
└── total          : int                     running chip total

SidePot
├── amount   : int
└── eligible : List[int]   player_ids who can win this sub-pot
```

---

## Side-Pot Algorithm

```
Input:
  all_in_amounts   = {player_id: total_bet}   (only all-in players)
  active_player_ids = [all non-folded player ids]

Algorithm:
  1. Sort all-in levels ascending: [30, 70]
  2. For each level:
     a. Tier size = level - previous_level
     b. Collect min(contribution, tier_size) from each player
     c. Eligible = players who contributed >= level
     d. Remove players whose all-in == level from future tiers
  3. Leftover chips above all all-in levels → final pot (non all-in players only)

Example (3 players, all-in at 30, 70, caller at 100):
  Tier 1: cap=30, take 30 from each → pot = 90, eligible = [A,B,C]
  Tier 2: cap=70, take 40 from B,C  → pot = 80, eligible = [B,C]
  Tier 3: remainder from C only     → pot = 30, eligible = [C]
```

---

## Award Logic

```python
pot.award(ranked_winners)
```

`ranked_winners` is a list of groups ordered best-hand-first. Each group is a list of player IDs that share the same hand strength (for split-pot handling).

```
ranked_winners = [[3], [0, 2], [1]]
                   ↑      ↑       ↑
                 best   tied   worst
```

For each side pot:
1. Find the first group that has at least one eligible player.
2. Split that side pot equally among those players.
3. Remainder (odd chip) goes to the first player by seat order.

---

## Chip Collection Design

**Critical design decision:** chips are collected from players **only once per street**, at the end of `_betting_round()` via `add_from_players()`.

```
During the betting round:
  player.bet tracks chips committed this street (in player object)
  pot.total does NOT change during betting

End of betting round:
  pot.add_from_players(players)   ← single collection point
    └─ for each player: pot.add(player.id, player.bet); player.bet = 0
```

This avoids double-counting. An earlier design that called `pot.add()` inside `_process_action()` AND `add_from_players()` at the end created a bug where the pot was doubled.

---

## Split Pot with Odd Chip

```
Pot = 201, tied between P0 and P1
  share    = 201 // 2 = 100
  remainder = 201 %  2 =   1
  P0 gets 100 + 1 = 101
  P1 gets 100
Total = 201 ✓
```

The odd chip always goes to the first player in the group (lowest seat index).

---

## Usage Example

```python
from engine.pot import Pot

pot = Pot()

# Collect bets (called by game engine at end of each street)
pot.add(player_id=0, amount=100)
pot.add(player_id=1, amount=50)   # P1 went all-in for 50
pot.add(player_id=2, amount=100)

# Calculate side pots
pots = pot.calculate_side_pots(
    all_in_amounts={1: 50},   # P1 total_bet was 50
    active_player_ids=[0, 1, 2]
)
# pots[0]: amount=150, eligible=[0,1,2]
# pots[1]: amount=100, eligible=[0,2]

# Award
winnings = pot.award([[1], [0], [2]])  # P1 has best hand
# P1 wins main pot: 150
# P0 wins side pot: 100
```
