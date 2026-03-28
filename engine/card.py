"""
card.py — Card and Deck primitives for the poker engine.

Represents individual playing cards as lightweight integer-encoded objects
and provides a fast, shuffleable Deck implementation.

Encoding scheme:
    card_id = rank * 4 + suit
    rank: 2–14  (2=Two, 14=Ace)
    suit: 0–3   (0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades)

    Example: Ace of Spades = 14 * 4 + 3 = 59
"""

from __future__ import annotations
import random
from typing import List

# ── Constants ────────────────────────────────────────────────────────────────

RANKS = range(2, 15)           # 2 … Ace(14)
SUITS = range(4)               # 0=C  1=D  2=H  3=S

RANK_NAMES = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}
SUIT_NAMES   = {0: "c", 1: "d", 2: "h", 3: "s"}
SUIT_SYMBOLS = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}


# ── Card ─────────────────────────────────────────────────────────────────────

class Card:
    """
    Immutable playing card backed by a single integer.

    Attributes
    ----------
    id   : int  — unique integer in [8, 59]
    rank : int  — 2–14
    suit : int  — 0–3
    """

    __slots__ = ("id",)

    def __init__(self, rank: int, suit: int) -> None:
        if rank not in RANKS:
            raise ValueError(f"Invalid rank {rank}")
        if suit not in SUITS:
            raise ValueError(f"Invalid suit {suit}")
        object.__setattr__(self, "id", rank * 4 + suit)

    @classmethod
    def from_id(cls, card_id: int) -> "Card":
        """Reconstruct a Card from its integer id."""
        rank, suit = divmod(card_id, 4)
        return cls(rank, suit)

    @classmethod
    def from_str(cls, s: str) -> "Card":
        """
        Parse a short string like 'As', 'Td', '2c'.
        Rank chars: 2-9, T, J, Q, K, A
        Suit chars: c d h s
        """
        rank_map = {v: k for k, v in RANK_NAMES.items()}
        suit_map = {v: k for k, v in SUIT_NAMES.items()}
        s = s.strip()
        if len(s) != 2:
            raise ValueError(f"Cannot parse card '{s}'")
        return cls(rank_map[s[0].upper() if s[0].isalpha() else s[0]],
                   suit_map[s[1].lower()])

    @property
    def rank(self) -> int:
        return self.id // 4

    @property
    def suit(self) -> int:
        return self.id % 4

    def __setattr__(self, *_):
        raise AttributeError("Card is immutable")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Card) and self.id == other.id

    def __hash__(self) -> int:
        return self.id

    def __repr__(self) -> str:
        return f"{RANK_NAMES[self.rank]}{SUIT_NAMES[self.suit]}"

    def __str__(self) -> str:
        return f"{RANK_NAMES[self.rank]}{SUIT_SYMBOLS[self.suit]}"

    def __lt__(self, other: "Card") -> bool:
        return self.rank < other.rank


# ── Deck ─────────────────────────────────────────────────────────────────────

class Deck:
    """
    Standard 52-card deck with fast Fisher-Yates shuffle.

    Usage
    -----
    >>> deck = Deck()
    >>> deck.shuffle()
    >>> hand = deck.deal(2)
    """

    def __init__(self) -> None:
        # Pre-build all 52 card ids; shuffle on demand
        self._ids: List[int] = [r * 4 + s for r in RANKS for s in SUITS]
        self._pos: int = 0          # next card pointer

    def shuffle(self, seed: int | None = None) -> None:
        """Fisher-Yates shuffle. Optionally seeded for reproducibility."""
        rng = random.Random(seed)
        rng.shuffle(self._ids)
        self._pos = 0

    def deal(self, n: int = 1) -> List[Card]:
        """
        Deal n cards from the top of the deck.
        Raises RuntimeError if the deck runs out.
        """
        end = self._pos + n
        if end > 52:
            raise RuntimeError