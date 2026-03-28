# `engine/cards.py` — Card, Deck & Hand Evaluation

## Purpose

Foundation layer of the entire system. Defines **Card**, **Deck**, and **HandEvaluator** — the three primitives every other module depends on.

---

## Data Flow

```
Card(rank, suit)
      │
      ▼
   Deck
   ├── shuffle()  →  randomises order in-place (Fisher-Yates)
   ├── deal(n)    →  removes and returns n cards from the top
   └── burn()     →  discards one card (standard casino procedure)
         │
         ▼
HandEvaluator.evaluate([Card × 5-7])
         │
         ▼
    HandRank
    ├── category     (int 0–9)
    └── tiebreakers  (tuple of ranks, highest first)
```

---

## Classes

### `Card`

Immutable value object representing one playing card.

| Attribute | Type | Values |
|-----------|------|--------|
| `rank`    | int  | 2–14 (Ace = 14) |
| `suit`    | int  | 0=♣ 1=♦ 2=♥ 3=♠ |

- Uses `__slots__` → ~4× less memory than a plain dict.
- `__setattr__` raises `AttributeError` → fully immutable after creation.
- `Card.from_str("Ah")` parses two-character strings like `"As"`, `"Td"`, `"2c"`.

```python
c = Card(14, 2)          # Ace of hearts
c = Card.from_str("Ah")  # same thing
repr(c)                  # → "A♥"
```

### `Deck`

Standard 52-card deck backed by a Python list.

| Method | Description |
|--------|-------------|
| `shuffle(seed=None)` | Fisher-Yates in-place shuffle; optional seed for reproducibility |
| `deal(n)` | Removes and returns `n` cards from the top (index −1) |
| `burn()` | Discards one card without returning it (required before each community card deal) |
| `reset()` | Restores all 52 cards |
| `remaining()` | Returns count of undealt cards |

> **Important:** `reset()` must be called before `shuffle()` between hands. Forgetting this caused a bug where only the leftover cards from the previous hand were reshuffled.

### `HandRank`

Comparable strength record.

```
HandRank
├── category    : int         0 = High Card … 9 = Royal Flush
├── tiebreakers : tuple[int]  descending rank values for tie-breaking
└── best_five   : list[Card]  the 5 cards forming the hand
```

`HandRank` is a `@dataclass(order=True)` so `>`, `<`, `==` all work via `(category, tiebreakers)`.

| category | Name |
|----------|------|
| 9 | Royal Flush |
| 8 | Straight Flush |
| 7 | Four of a Kind |
| 6 | Full House |
| 5 | Flush |
| 4 | Straight |
| 3 | Three of a Kind |
| 2 | Two Pair |
| 1 | One Pair |
| 0 | High Card |

### `HandEvaluator`

Static class with one public method: `evaluate(cards)`.

```
evaluate([Card × 5..7])
         │
         ├─ len == 5  →  _eval5(cards)
         └─ len == 6-7 →  all C(n,5) combos → max(_eval5())

_eval5(cards)
         │
         ├─ count rank frequencies
         ├─ check flush  (all same suit)
         ├─ check straight  (5 consecutive ranks)
         │    └─ handles A-2-3-4-5 "wheel" (high = 5)
         └─ return HandRank(category, tiebreakers)
```

For 7-card input: tries all `C(7,5) = 21` five-card combos and returns the best `HandRank`. This runs in effectively O(1) time.

---

## Edge Cases Handled

| Scenario | Behaviour |
|----------|-----------|
| A-2-3-4-5 (wheel) | `category=4` (Straight), `tiebreakers=(5,)` — Ace plays low |
| T-J-Q-K-A unsuited | `category=4` (Straight), not Royal Flush |
| T-J-Q-K-A same suit | `category=9` (Royal Flush) |
| 7 cards with Royal Flush available | Correctly found among 21 combos |
| Identical hands across suits | `HandRank.__eq__` returns `True` |
| Kicker differentiates pairs | Tiebreakers include kicker ranks |

---

## Performance

| Operation | Cost |
|-----------|------|
| `Card` creation | O(1), ~48 bytes |
| `Deck.shuffle()` | O(n) = O(52) |
| `Deck.deal(n)` | O(n) |
| `HandEvaluator.evaluate` (5 cards) | O(1) |
| `HandEvaluator.evaluate` (7 cards) | O(21) ≈ O(1) |

Throughput: ~1–5 million hand evaluations/second on a modern CPU.

---

## Usage Example

```python
from engine.cards import Card, Deck, HandEvaluator

deck = Deck()
deck.shuffle(seed=42)

hole  = deck.deal(2)   # [Card, Card]  — player's private cards
deck.burn()
board = deck.deal(3)   # [Card, Card, Card]  — the flop

all_cards = hole + board
rank = HandEvaluator.evaluate(all_cards)
print(rank.name())          # e.g. "Flush"
print(rank.tiebreakers)     # e.g. (14, 11, 9, 7, 3)
```
