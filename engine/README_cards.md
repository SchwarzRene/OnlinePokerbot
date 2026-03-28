# `engine/cards.py` — Card, Deck & Hand Evaluation

Foundation layer of the entire system. Defines **Card**, **Deck**, and
**HandEvaluator** — the three primitives every other module depends on.

---

## Module Map

```
  Card(rank, suit)
        │
        ▼
     Deck
     ├── shuffle(seed?)  →  Fisher-Yates in-place randomise
     ├── deal(n)         →  remove and return n cards from top
     ├── burn()          →  discard one card (casino convention)
     └── reset()         →  restore all 52 cards
           │
           ▼
  HandEvaluator.evaluate([Card × 5..7])
           │
           ├── 5 cards: _eval5(cards)  →  HandRank directly
           └── 6-7 cards: all C(n,5) combos → max(_eval5())   (21 combos for 7 cards)
           │
           ▼
      HandRank
      ├── category     : int 0–9     (9 = Royal Flush, 0 = High Card)
      ├── tiebreakers  : tuple[int]  (descending ranks for tie-breaking)
      └── best_five    : List[Card]  (the 5 cards forming the hand)
```

---

## Hand Category Reference

```
  category   Name               Example
  ────────   ─────────────────  ────────────────────────────────
     9       Royal Flush        A K Q J T (same suit)
     8       Straight Flush     9 8 7 6 5 (same suit)
     7       Four of a Kind     A A A A 2
     6       Full House         A A A K K
     5       Flush              A J 9 7 3 (same suit)
     4       Straight           9 8 7 6 5 (mixed suits)
     3       Three of a Kind    A A A K Q
     2       Two Pair           A A K K 2
     1       One Pair           A A K Q J
     0       High Card          A J 9 7 3 (mixed suits, no structure)
  ────────   ─────────────────  ────────────────────────────────
  HandRank is a dataclass(order=True): uses (category, tiebreakers) for comparison.
  A HandRank from category 2 always beats category 1, regardless of tiebreakers.
```

---

## Card Representation

```
  Card(rank, suit)
        │
        rank: int  2–14  (2=Two, 11=Jack, 12=Queen, 13=King, 14=Ace)
        suit: int  0–3   (0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades)

  String format used in tokens: "Ah", "Kd", "2c", "Ts"
        rank chars: 2 3 4 5 6 7 8 9 T J Q K A
        suit chars: c d h s

  Card.from_str("Ah")  →  Card(rank=14, suit=2)
  repr(card)           →  "A♥"
  card.to_str()        →  "Ah"

  Cards are immutable (__slots__ + __setattr__ raises AttributeError).
  This allows safe use in sets and as dict keys.
```

---

## Deck Operations

```
  Standard 52-card deck:
  ┌──────────────────────────────────────────────────────────────────┐
  │  2c 3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac                        │
  │  2d 3d 4d 5d 6d 7d 8d 9d Td Jd Qd Kd Ad                        │
  │  2h 3h 4h 5h 6h 7h 8h 9h Th Jh Qh Kh Ah                        │
  │  2s 3s 4s 5s 6s 7s 8s 9s Ts Js Qs Ks As                        │
  └──────────────────────────────────────────────────────────────────┘

  Per-hand usage sequence:
    deck.reset()      ← restore all 52 (must be called before shuffle)
    deck.shuffle()    ← Fisher-Yates randomise in-place
    hole1 = deck.deal(2)   ← player 1's cards
    hole2 = deck.deal(2)   ← player 2's cards
    ...
    deck.burn()       ← burn one card (casino standard)
    flop = deck.deal(3)
    deck.burn()
    turn = deck.deal(1)
    deck.burn()
    river = deck.deal(1)
```

---

## 7-Card Best-Hand Selection

In Texas Hold'em, each player uses the best 5 cards from their 2 hole cards
and 5 community cards (7 total). The evaluator checks all C(7,5) = **21**
five-card combinations and returns the highest-ranked one:

```
  all_cards = hole_cards + community_cards   # 7 cards

  best = None
  for combo in combinations(all_cards, 5):   # 21 combinations
      rank = _eval5(combo)
      if best is None or rank > best:
          best = rank

  return best
```

This runs in O(21) ≈ O(1) time. No lookup tables needed.

---

## Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| A-2-3-4-5 (Wheel) | Straight, category=4, tiebreakers=(5,) — Ace plays low |
| T-J-Q-K-A unsuited | Straight, not Royal Flush |
| T-J-Q-K-A same suit | Royal Flush, category=9 |
| 7 cards containing a Royal Flush | Found among 21 combos |
| Identical hands (different suits) | HandRank.__eq__ returns True |
| Kicker differentiation | AA+K beats AA+Q: tiebreakers=(14,14,13,...) vs (14,14,12,...) |
| < 5 cards | ValueError raised |

---

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Card creation | O(1) | ~48 bytes via __slots__ |
| Deck.shuffle() | O(52) | Fisher-Yates |
| Deck.deal(n) | O(n) | List pop |
| evaluate() 5 cards | O(1) | ~15 comparisons |
| evaluate() 7 cards | O(21) | 21 × O(1) |

Throughput: ~1–5 million hand evaluations per second on modern CPU.

---

## Usage Example

```python
from engine.cards import Card, Deck, HandEvaluator

deck = Deck()
deck.shuffle(seed=42)

hole  = deck.deal(2)           # [Card, Card]  — player's hole cards
deck.burn()
flop  = deck.deal(3)           # [Card, Card, Card]
deck.burn()
turn  = deck.deal(1)
deck.burn()
river = deck.deal(1)

all_cards = hole + flop + turn + river   # 7 cards
rank = HandEvaluator.evaluate(all_cards)

print(rank.name())          # e.g. "Flush"
print(rank.category)        # e.g. 5
print(rank.tiebreakers)     # e.g. (14, 11, 9, 7, 3)
print(rank.best_five)       # e.g. [A♥, J♥, 9♥, 7♥, 3♥]

# Compare hands
rank_a = HandEvaluator.evaluate(cards_a)
rank_b = HandEvaluator.evaluate(cards_b)
if rank_a > rank_b:
    print("A wins")
elif rank_a == rank_b:
    print("Split pot")
else:
    print("B wins")
```
