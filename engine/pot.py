"""
pot.py — Pot Manager with Side-Pot Calculation
===============================================

Manages chip collection and distributes winnings including complex
side-pot scenarios (multiple all-in players with different stack sizes).

Concepts:
---------

    Main Pot:  All players are eligible. Capped at the smallest all-in
               player's contribution × number of contributors.

    Side Pot:  Created when a player goes all-in for less than the full bet.
               Only players who matched beyond the all-in amount are eligible.

    Example (3 players):
    ─────────────────────────────────────────────────────────
    Player A: all-in for  50  → contributes 50
    Player B: all-in for 100  → contributes 100
    Player C: calls      100  → contributes 100

    Main pot  = 50 × 3 = 150  (A, B, C eligible)
    Side pot  = 50 × 2 = 100  (B, C eligible only)
    ─────────────────────────────────────────────────────────

Data Model:
-----------
    Pot
    ├── _contributions : dict[player_id → total_chips_in_pot]
    ├── _side_pots     : List[SidePot]   computed by calculate_side_pots()
    └── total          : int             running chip total

    SidePot
    ├── amount      : int
    └── eligible    : List[int]   player_ids who can win this pot

Usage:
------
    pot = Pot()
    pot.add(player_id=0, amount=100)
    pot.add(player_id=1, amount=50)   # player 1 went all-in
    side_pots = pot.calculate_side_pots(all_in_amounts={1: 50})
    winnings = pot.award(winners_per_pot)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# SidePot
# ---------------------------------------------------------------------------

@dataclass
class SidePot:
    """
    A single pot level.

    Attributes
    ----------
    amount   : int          total chips in this sub-pot
    eligible : List[int]    player ids who can win it
    """
    amount: int
    eligible: List[int]

    def __repr__(self):
        return f"SidePot(amount={self.amount}, eligible={self.eligible})"


# ---------------------------------------------------------------------------
# Pot
# ---------------------------------------------------------------------------

class Pot:
    """
    Tracks all chips on the table and computes side pots.

    Side-Pot Algorithm:
    -------------------
    Given contributions {player_id: total_chips} and all-in levels:

    1. Sort players by total contribution (ascending).
    2. Iterate through sorted levels:
       a. Determine the "cap" = current all-in level (or max if not all-in).
       b. Each eligible player contributes min(their_total, cap) to this tier.
       c. Players whose contribution == cap are removed from future tiers.
    3. Remaining chips above all caps form the final side pot (eligible = non-folded active players).

    This correctly handles any number of all-in players at different stack sizes.
    """

    def __init__(self):
        self._contributions: Dict[int, int] = {}  # player_id → chips added
        self._side_pots: List[SidePot] = []
        self.total: int = 0

    # ── Chip Collection ────────────────────────────────────────────────────

    def add(self, player_id: int, amount: int):
        """Add chips from a player to the pot."""
        if amount < 0:
            raise ValueError(f"Cannot add negative chips: {amount}")
        self._contributions[player_id] = self._contributions.get(player_id, 0) + amount
        self.total += amount

    def add_from_players(self, players):
        """
        Convenience: collect bet amounts from a list of Player objects.
        Resets player.bet to 0 after collection.
        """
        for p in players:
            if p.bet > 0:
                self.add(p.id, p.bet)
                p.bet = 0

    # ── Side-Pot Calculation ───────────────────────────────────────────────

    def calculate_side_pots(
        self,
        all_in_amounts: Dict[int, int],   # player_id → chips they went all-in for (total_bet)
        active_player_ids: List[int],     # all non-folded players
    ) -> List[SidePot]:
        """
        Recompute self._side_pots.

        Parameters
        ----------
        all_in_amounts   : mapping of player_id → their total_bet (only all-in players)
        active_player_ids: all players still in the hand (not folded)

        Returns
        -------
        List[SidePot] — ordered from smallest to largest contribution level.
        """
        if not all_in_amounts:
            # Simple case: no all-ins, single pot
            self._side_pots = [SidePot(self.total, list(active_player_ids))]
            return self._side_pots

        contribs = dict(self._contributions)  # copy
        side_pots: List[SidePot] = []

        # All players sorted by all-in level (ascending)
        levels = sorted(set(all_in_amounts.values()))
        prev_level = 0
        remaining_eligible = list(active_player_ids)

        for level in levels:
            increment = level - prev_level
            pot_amount = 0
            for pid in list(contribs.keys()):
                take = min(contribs[pid], increment)
                contribs[pid] -= take
                pot_amount += take

            # Eligible = everyone who contributed at this level
            eligible = [pid for pid in remaining_eligible
                        if self._contributions.get(pid, 0) >= level]

            if pot_amount > 0:
                side_pots.append(SidePot(pot_amount, eligible))

            # Remove players whose all-in was exactly at this level
            remaining_eligible = [pid for pid in remaining_eligible
                                   if all_in_amounts.get(pid, float("inf")) > level]
            prev_level = level

        # Remaining chips above all all-in levels
        leftover = sum(contribs.values())
        if leftover > 0 and remaining_eligible:
            side_pots.append(SidePot(leftover, remaining_eligible))

        self._side_pots = side_pots
        return side_pots

    # ── Award ──────────────────────────────────────────────────────────────

    def award(self, ranked_winners: List[List[int]]) -> Dict[int, int]:
        """
        Distribute pots to winners.

        Parameters
        ----------
        ranked_winners : list of groups, best hand first.
            Each group is a list of player_ids with equal hand strength.
            e.g. [[3], [0, 2], [1]]  → player 3 wins, then 0 and 2 tie, etc.

        Returns
        -------
        dict mapping player_id → chips won
        """
        winnings: Dict[int, int] = {}

        for side_pot in self._side_pots:
            pot_remaining = side_pot.amount
            eligible = set(side_pot.eligible)

            for group in ranked_winners:
                contenders = [pid for pid in group if pid in eligible]
                if not contenders:
                    continue
                # Split pot among tied winners
                share, remainder = divmod(pot_remaining, len(contenders))
                for pid in contenders:
                    winnings[pid] = winnings.get(pid, 0) + share
                # Remainder goes to first player (by seat order)
                if remainder:
                    winnings[contenders[0]] = winnings.get(contenders[0], 0) + remainder
                pot_remaining = 0
                break  # this side pot is fully awarded

        self.total = 0
        return winnings

    # ── Utilities ──────────────────────────────────────────────────────────

    def reset(self):
        self._contributions.clear()
        self._side_pots.clear()
        self.total = 0

    @property
    def side_pots(self) -> List[SidePot]:
        return list(self._side_pots)

    def __repr__(self):
        return f"Pot(total={self.total}, side_pots={self._side_pots})"
