# `engine/pot.py` — Pot Manager & Side-Pot Calculation

Manages chip collection during a hand and correctly distributes winnings
including **side-pot scenarios** where players go all-in for different amounts.

---

## Core Concepts

### Why Side Pots Exist

When a player goes all-in for fewer chips than the current bet, they can
only win back up to their own contribution from each other player.
Any chips beyond that form a **side pot** that the short-stack player
is not eligible for.

```
  Example: 3 players

  Player A goes all-in for  50 chips  →  contributes  50
  Player B goes all-in for 100 chips  →  contributes 100
  Player C calls            100 chips  →  contributes 100
                                          ───────────
                            Total pot      250 chips

  Main pot  = 50 × 3 = 150   ← A, B, C all eligible
  Side pot  = 50 × 2 = 100   ← B, C eligible only

  Even if A has the best hand, they can only win 150.
  B or C wins the side pot of 100.
```

---

## Data Model

```
  Pot
  ├── _contributions : dict[player_id → int]   total chips each player put in
  ├── _side_pots     : List[SidePot]           computed by calculate_side_pots()
  └── total          : int                     running chip total (never decrements
                                               during betting — set to 0 only in reset)
  SidePot
  ├── amount   : int
  └── eligible : List[int]   player_ids who can win this sub-pot
```

---

## Side-Pot Algorithm (Step by Step)

```
  Inputs:
    all_in_amounts   = {pid: total_bet}   (only all-in players)
    active_player_ids = [all non-folded players]
    _contributions    = {pid: chips_put_in}

  ─────────────────────────────────────────────────────────────────────
  Example:
    A: all-in for 30   B: all-in for 70   C: called 100 (not all-in)
    contributions = {A: 30, B: 70, C: 100}
    all_in_amounts = {A: 30, B: 70}
    active = [A, B, C]

  Step 1: Sort all-in levels ascending:  [30, 70]

  Step 2: Tier 1  (level=30, increment=30)
    Take min(contrib, 30) from each player:
      A: min(30, 30) = 30   →  remaining: 0
      B: min(70, 30) = 30   →  remaining: 40
      C: min(100,30) = 30   →  remaining: 70
    pot_amount = 90
    eligible   = [A, B, C]   (all contributed >= 30)
    SidePot(90, [A,B,C])
    Remove A from future tiers (A's all-in == 30)

  Step 3: Tier 2  (level=70, increment=40)
    Take min(contrib, 40) from remaining players [B, C]:
      B: min(40, 40) = 40   →  remaining: 0
      C: min(70, 40) = 40   →  remaining: 30
    pot_amount = 80
    eligible   = [B, C]     (contributed >= 70)
    SidePot(80, [B,C])
    Remove B from future tiers (B's all-in == 70)

  Step 4: Leftover (above all all-in levels)
    Remaining: {C: 30}
    leftover = 30
    remaining_eligible = [C]   (only non-all-in players)
    SidePot(30, [C])

  Final side pots:
    SidePot(90,  [A,B,C])   ← main pot, everyone eligible
    SidePot(80,  [B,C])     ← side pot 1
    SidePot(30,  [C])       ← side pot 2 (C's excess, C wins it back)

  Total: 90 + 80 + 30 = 200 = A(30) + B(70) + C(100) ✓
  ─────────────────────────────────────────────────────────────────────
```

---

## Award Algorithm

```
  award(ranked_winners):
    # ranked_winners: list of groups, best hand first
    # e.g. [[1], [0, 2]] = P1 has best hand; P0 and P2 tied for second

    winnings = {}

    for each side_pot:
        pot_remaining = side_pot.amount
        eligible = set(side_pot.eligible)

        for group in ranked_winners:
            contenders = [pid for pid in group if pid in eligible]
            if not contenders:
                continue                    ← this group has no eligible player

            share, remainder = divmod(pot_remaining, len(contenders))
            for pid in contenders:
                winnings[pid] += share

            if remainder:
                winnings[contenders[0]] += remainder   ← odd chip to first seat

            pot_remaining = 0
            break                           ← this side pot is fully awarded

    pot.total = 0
    return winnings
```

---

## Split Pot with Odd Chip

```
  Pot = 201, tied between P0 and P2

  share     = 201 // 2 = 100
  remainder = 201 %  2 =   1

  P0 gets 100 + 1 = 101   (first player in contenders list)
  P2 gets 100

  Total: 201 ✓   Chips are conserved.
```

---

## Chip Collection Design

```
  DURING betting round:
    player.bet tracks chips the player has committed this street
    pot.total does NOT change

  END OF EACH STREET (single collection point):
    pot.add_from_players(players)
      └─ for each player:
           pot.add(player.id, player.bet)
           player.bet = 0

  This prevents double-counting. An earlier design that called pot.add()
  both inside _process_action() AND again in add_from_players() doubled the pot.
```

---

## Usage Example

```python
from engine.pot import Pot

pot = Pot()

# End of preflop: collect bets
pot.add(player_id=0, amount=100)
pot.add(player_id=1, amount=50)    # P1 went all-in for 50
pot.add(player_id=2, amount=100)

print(pot.total)   # 250

# Calculate side pots before showdown
pots = pot.calculate_side_pots(
    all_in_amounts={1: 50},        # P1 total_bet was 50
    active_player_ids=[0, 1, 2]
)
# pots[0]: SidePot(amount=150, eligible=[0,1,2])
# pots[1]: SidePot(amount=100, eligible=[0,2])

# Award: P1 has best hand, P0 second best
winnings = pot.award([[1], [0], [2]])
# P1 wins main pot: 150
# P0 wins side pot: 100
# P2 wins nothing

print(winnings)    # {1: 150, 0: 100}
print(pot.total)   # 0  (reset after award)
```
