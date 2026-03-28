"""
player.py — Player State and Action Definitions
================================================

Defines the Player data class and all valid poker actions.
Each player instance tracks chip count, hole cards, betting status,
and action history for the current hand.

Data Model:
-----------

    Player
    ├── id           : int           unique seat index (0-based)
    ├── name         : str           display name / agent identifier
    ├── chips        : int           current chip stack
    ├── hole_cards   : List[Card]    private two-card hand
    ├── bet          : int           amount put in for current street
    ├── total_bet    : int           cumulative bet this hand (for side pots)
    ├── is_folded    : bool
    ├── is_all_in    : bool
    ├── is_active    : bool          False = sitting out
    └── action_log   : List[Action]  history of actions this hand

Action Types:
-------------
    FOLD    — surrender the hand
    CHECK   — pass without betting (only when no bet to call)
    CALL    — match the current highest bet
    RAISE   — increase the bet by a specified amount
    ALL_IN  — commit all remaining chips

Action Encoding (for Transformer input):
-----------------------------------------
    Each Action is serializable to a compact token string:

        "<FOLD>", "<CHECK>", "<CALL>", "<RAISE:50>", "<ALL_IN>"

    These tokens feed directly into the sequence encoder in model/tokenizer.py.

State Transitions:
------------------
    new hand
        │
        ▼
    reset_for_hand()   → clears hole_cards, bet, folded, all_in flags
        │
        ▼
    reset_for_street() → zeroes per-street bet, keeps total_bet
        │
        ▼
    apply_action(action) → updates chips, bet, flags
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.cards import Card


# ---------------------------------------------------------------------------
# ActionType
# ---------------------------------------------------------------------------

class ActionType(Enum):
    FOLD   = auto()
    CHECK  = auto()
    CALL   = auto()
    RAISE  = auto()
    ALL_IN = auto()

    def __str__(self):
        return self.name


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """
    A single player decision.

    Attributes
    ----------
    action_type : ActionType
    amount      : int   relevant for RAISE (the raise-to total) and ALL_IN;
                        0 for FOLD / CHECK / CALL (amount inferred from state)
    player_id   : int   who took this action
    street      : str   "preflop" | "flop" | "turn" | "river"
    """
    action_type: ActionType
    amount: int = 0
    player_id: int = -1
    street: str = ""

    def to_token(self) -> str:
        """
        Serialize to a token suitable for transformer input.

        Examples
        --------
            FOLD            →  "<FOLD>"
            CHECK           →  "<CHECK>"
            CALL 50         →  "<CALL>"       (amount inferred from context)
            RAISE to 120    →  "<RAISE:120>"
            ALL_IN 300      →  "<ALL_IN:300>"
        """
        if self.action_type == ActionType.RAISE:
            return f"<RAISE:{self.amount}>"
        if self.action_type == ActionType.ALL_IN:
            return f"<ALL_IN:{self.amount}>"
        return f"<{self.action_type.name}>"

    @classmethod
    def from_token(cls, token: str, player_id: int = -1, street: str = "") -> "Action":
        """Parse a token string back into an Action."""
        token = token.strip("<>")
        if ":" in token:
            name, amt = token.split(":", 1)
            return cls(ActionType[name], int(amt), player_id, street)
        return cls(ActionType[token], 0, player_id, street)

    def __repr__(self):
        if self.amount:
            return f"Action({self.action_type.name}, {self.amount})"
        return f"Action({self.action_type.name})"


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class Player:
    """
    Represents one seat at the poker table.

    Lifecycle
    ---------
        1. Created once per session with a chip stack.
        2. reset_for_hand() called at the start of every new hand.
        3. reset_for_street() called at the start of every betting round.
        4. apply_action() mutates state when the player acts.

    Properties
    ----------
        can_act     : bool  True if not folded, not all-in, has chips > 0
        effective_stack : int  min(self.chips, max_opponent_chips)
    """
    id: int
    name: str
    chips: int

    # Per-hand state (reset each hand)
    hole_cards: List["Card"] = field(default_factory=list)
    bet: int = 0           # chips put in this street
    total_bet: int = 0     # cumulative chips put in this hand
    is_folded: bool = False
    is_all_in: bool = False
    is_active: bool = True
    action_log: List[Action] = field(default_factory=list)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def reset_for_hand(self):
        """Clear hole cards and all per-hand flags."""
        self.hole_cards = []
        self.bet = 0
        self.total_bet = 0
        self.is_folded = False
        self.is_all_in = False
        self.action_log = []

    def reset_for_street(self):
        """Zero out the per-street bet (keep total_bet for side-pot calc)."""
        self.bet = 0

    # ── Action Application ─────────────────────────────────────────────────

    def apply_action(self, action: Action, call_amount: int = 0) -> int:
        """
        Mutate player state based on an action.

        Parameters
        ----------
        action      : Action
        call_amount : int   the current amount needed to call (chips to add,
                            not the absolute bet level); used for CALL.

        Returns
        -------
        int   amount of chips actually moved into the pot this action.
        """
        self.action_log.append(action)
        chips_moved = 0

        if action.action_type == ActionType.FOLD:
            self.is_folded = True

        elif action.action_type == ActionType.CHECK:
            pass  # no chips move

        elif action.action_type == ActionType.CALL:
            to_call = min(call_amount, self.chips)
            self.chips -= to_call
            self.bet += to_call
            self.total_bet += to_call
            chips_moved = to_call
            if self.chips == 0:
                self.is_all_in = True

        elif action.action_type == ActionType.RAISE:
            # action.amount = total chips player puts in this action
            to_add = min(action.amount, self.chips)
            self.chips -= to_add
            self.bet += to_add
            self.total_bet += to_add
            chips_moved = to_add
            if self.chips == 0:
                self.is_all_in = True

        elif action.action_type == ActionType.ALL_IN:
            chips_moved = self.chips
            self.bet += self.chips
            self.total_bet += self.chips
            self.chips = 0
            self.is_all_in = True

        return chips_moved

    # ── Queries ────────────────────────────────────────────────────────────

    @property
    def can_act(self) -> bool:
        return self.is_active and not self.is_folded and not self.is_all_in and self.chips > 0

    @property
    def in_hand(self) -> bool:
        """True if not yet folded (includes all-in players)."""
        return self.is_active and not self.is_folded

    def __repr__(self):
        status = "folded" if self.is_folded else ("all-in" if self.is_all_in else "active")
        return f"Player({self.name}, chips={self.chips}, {status})"
