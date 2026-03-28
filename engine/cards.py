"""
cards.py — Card, Deck, and Hand Evaluation Engine
==================================================

Core data structures for representing playing cards and evaluating poker hands.
This module is the foundation layer for the entire simulation — every other
module depends on it.

Data Flow:
----------
    Card(rank, suit)
         │
         ▼
    Deck  ──► shuffle() ──► deal() ──► [Card, Card, ...]
         │
         ▼
    HandEvaluator.evaluate([Card...]) ──► HandRank(category, tiebreakers)

Card Representation:
--------------------
Cards are lightweight named tuples for fast copying and comparison.

    Rank:  2–14  (11=J, 12=Q, 13=K, 14=A)
    Suit:  0=♣  1=♦  2=♥  3=♠

Hand Categories (HandRank.category):
-------------------------------------
    9 = Royal Flush
    8 = Straight Flush
    7 = Four of a Kind
    6 = Full House
    5 = Flush
    4 = Straight
    3 = Three of a Kind
    2 = Two Pair
    1 = One Pair
    0 = High Card

Comparison:
-----------
    HandRank supports < / > / == via tuple comparison on (category, tiebreakers).
    This makes winner determination O(1) after evaluation.

Performance Notes:
------------------
    - Card is a __slots__ class: ~4× less memory than a plain dict.
    - Deck uses a Fisher-Yates shuffle via random.shuffle (in-place, O(n)).
    - HandEvaluator works on any 5–7 card combination and picks the best 5.
    - evaluate() on 7 cards tests C(7,5)=21 combos — fast enough for millions
      of hands per second on a modern CPU.
"""

import random
from itertools import combinations
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

RANK_NAMES = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
              9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT_NAMES = {0: "c", 1: "d", 2: "h", 3: "s"}
SUIT_SYMBOLS = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}


class Card:
    """
    Immutable playing card.

    Attributes
    ----------
    rank : int   2–14  (Ace = 14)
    suit : int   0=clubs, 1=diamonds, 2=hearts, 3=spades
    """
    __slots__ = ("rank", "suit")

    def __init__(self, rank: int, suit: int):
        if rank not in RANK_NAMES:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in SUIT_NAMES:
            raise ValueError(f"Invalid suit: {suit}")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "suit", suit)

    def __setattr__(self, *_):
        raise AttributeError("Card is immutable")

    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return self.rank * 4 + self.suit

    def __repr__(self):
        return f"{RANK_NAMES[self.rank]}{SUIT_SYMBOLS[self.suit]}"

    def __lt__(self, other):
        return (self.rank, self.suit) < (other.rank, other.suit)

    @classmethod
    def from_str(cls, s: str) -> "Card":
        """
        Parse a card from a 2-char string, e.g. "As", "Td", "2c".

        Rank chars: 2-9, T, J, Q, K, A
        Suit chars: c, d, h, s
        """
        rank_map = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
                    "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                    "Q": 12, "K": 13, "A": 14}
        suit_map = {"c": 0, "d": 1, "h": 2, "s": 3}
        s = s.strip()
        if len(s) != 2:
            raise ValueError(f"Card string must be 2 chars, got: {s!r}")
        return cls(rank_map[s[0].upper()], suit_map[s[1].lower()])


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

class Deck:
    """
    Standard 52-card deck.

    Usage
    -----
        deck = Deck()
        deck.shuffle()
        hole_cards = deck.deal(2)   # [Card, Card]
        community  = deck.deal(5)   # [Card, Card, Card, Card, Card]

    State
    -----
        _cards : list[Card]   remaining cards (top = index -1)
        _dealt : set[Card]    fast membership check for duplicates
    """

    def __init__(self):
        self._cards: List[Card] = [Card(r, s) for r in range(2, 15) for s in range(4)]
        self._dealt: set = set()

    def shuffle(self, seed: Optional[int] = None):
        """Fisher-Yates in-place shuffle. Optionally seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._cards)
        self._dealt.clear()

    def deal(self, n: int = 1) -> List[Card]:
        """Remove and return n cards from the top of the deck."""
        if n > len(self._cards):
            raise RuntimeError(f"Cannot deal {n} cards — only {len(self._cards)} remain")
        dealt = self._cards[-n:]
        self._cards = self._cards[:-n]
        self._dealt.update(dealt)
        return dealt

    def burn(self):
        """Burn one card (discard without returning)."""
        if not self._cards:
            raise RuntimeError("Deck is empty — cannot burn")
        self._cards.pop()

    def remaining(self) -> int:
        return len(self._cards)

    def reset(self):
        """Restore all 52 cards and clear dealt set."""
        self._cards = [Card(r, s) for r in range(2, 15) for s in range(4)]
        self._dealt.clear()


# ---------------------------------------------------------------------------
# HandRank
# ---------------------------------------------------------------------------

@dataclass(order=True)
class HandRank:
    """
    Comparable hand strength.

    Fields
    ------
    category    : int         0–9 (see module docstring)
    tiebreakers : tuple[int]  rank values used to break ties within category
    best_five   : list[Card]  the 5 cards that make up the winning hand

    Comparison: HandRank supports all rich comparisons via @dataclass(order=True).
    The sort key is (category, tiebreakers) — best_five is excluded.
    """
    category: int
    tiebreakers: Tuple[int, ...]
    best_five: List[Card] = field(default=None, compare=False)

    CATEGORY_NAMES = {
        9: "Royal Flush", 8: "Straight Flush", 7: "Four of a Kind",
        6: "Full House", 5: "Flush", 4: "Straight", 3: "Three of a Kind",
        2: "Two Pair", 1: "One Pair", 0: "High Card"
    }

    def name(self) -> str:
        return self.CATEGORY_NAMES[self.category]

    def __repr__(self):
        return f"HandRank({self.name()}, tiebreakers={self.tiebreakers})"


# ---------------------------------------------------------------------------
# HandEvaluator
# ---------------------------------------------------------------------------

class HandEvaluator:
    """
    Evaluate the best 5-card poker hand from any 5–7 cards.

    Algorithm
    ---------
    For 7 cards: enumerate all C(7,5)=21 five-card combos,
    evaluate each, return the maximum HandRank.

    Each 5-card eval:
      1. Count rank frequencies  → detect pairs / trips / quads
      2. Check flush             → all same suit
      3. Check straight          → 5 consecutive ranks (handles A-2-3-4-5 wheel)
      4. Combine results         → assign category + tiebreakers

    Complexity: O(21 × 5) ≈ O(1)  for 7-card input.

    Usage
    -----
        hr = HandEvaluator.evaluate(seven_cards)
        print(hr.name())           # "Flush"
        print(hr.tiebreakers)      # (14, 11, 9, 7, 3)
    """

    @staticmethod
    def evaluate(cards: List[Card]) -> HandRank:
        if len(cards) < 5:
            raise ValueError(f"Need at least 5 cards, got {len(cards)}")
        if len(cards) == 5:
            return HandEvaluator._eval5(cards)
        # For 6-7 cards, try all C(n,5) combos
        best = None
        for combo in combinations(cards, 5):
            hr = HandEvaluator._eval5(list(combo))
            if best is None or hr > best:
                best = hr
        return best

    @staticmethod
    def _eval5(cards: List[Card]) -> HandRank:
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]

        # Frequencies: {rank: count}
        freq: dict = {}
        for r in ranks:
            freq[r] = freq.get(r, 0) + 1

        # Groups sorted by (count desc, rank desc)
        groups = sorted(freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
        counts = [g[1] for g in groups]
        group_ranks = [g[0] for g in groups]

        is_flush = len(set(suits)) == 1
        is_straight, straight_high = HandEvaluator._check_straight(ranks)

        # ── Straight Flush / Royal Flush ──────────────────────────────────
        if is_flush and is_straight:
            category = 9 if straight_high == 14 else 8
            return HandRank(category, (straight_high,), cards)

        # ── Four of a Kind ────────────────────────────────────────────────
        if counts[0] == 4:
            kicker = group_ranks[1]
            return HandRank(7, (group_ranks[0], kicker), cards)

        # ── Full House ────────────────────────────────────────────────────
        if counts[0] == 3 and counts[1] == 2:
            return HandRank(6, (group_ranks[0], group_ranks[1]), cards)

        # ── Flush ─────────────────────────────────────────────────────────
        if is_flush:
            return HandRank(5, tuple(ranks), cards)

        # ── Straight ──────────────────────────────────────────────────────
        if is_straight:
            return HandRank(4, (straight_high,), cards)

        # ── Three of a Kind ───────────────────────────────────────────────
        if counts[0] == 3:
            kickers = tuple(group_ranks[1:3])
            return HandRank(3, (group_ranks[0],) + kickers, cards)

        # ── Two Pair ──────────────────────────────────────────────────────
        if counts[0] == 2 and counts[1] == 2:
            kicker = group_ranks[2]
            return HandRank(2, (group_ranks[0], group_ranks[1], kicker), cards)

        # ── One Pair ──────────────────────────────────────────────────────
        if counts[0] == 2:
            kickers = tuple(group_ranks[1:4])
            return HandRank(1, (group_ranks[0],) + kickers, cards)

        # ── High Card ─────────────────────────────────────────────────────
        return HandRank(0, tuple(ranks), cards)

    @staticmethod
    def _check_straight(sorted_ranks_desc: List[int]) -> Tuple[bool, int]:
        """Return (is_straight, high_card_rank). Handles A-low wheel."""
        r = sorted_ranks_desc
        # Normal straight
        if r[0] - r[4] == 4 and len(set(r)) == 5:
            return True, r[0]
        # Wheel: A-2-3-4-5  → ranks are [14, 5, 4, 3, 2]
        if r == [14, 5, 4, 3, 2]:
            return True, 5
        return False, 0
