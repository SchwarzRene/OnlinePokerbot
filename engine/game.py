"""
game.py — Texas Hold'em Game Engine
=====================================

Orchestrates a complete poker hand from shuffle to showdown.
This is the central simulation loop that all training and evaluation
code interacts with.

Hand Flow:
----------

    ┌─────────────────────────────────────────────────┐
    │                  Game.play_hand()                │
    │                                                 │
    │  1. reset()          — shuffle deck, clear pot  │
    │  2. post_blinds()    — SB + BB forced bets      │
    │  3. deal_hole_cards()— 2 private cards each     │
    │                                                 │
    │  4. betting_round("preflop")                    │
    │     └─ request_action() for each active player  │
    │                                                 │
    │  5. deal_community(3) — flop                    │
    │  6. betting_round("flop")                       │
    │                                                 │
    │  7. deal_community(1) — turn                    │
    │  8. betting_round("turn")                       │
    │                                                 │
    │  9. deal_community(1) — river                   │
    │  10. betting_round("river")                     │
    │                                                 │
    │  11. showdown()      — evaluate, award chips    │
    │  12. return HandResult                          │
    └─────────────────────────────────────────────────┘

Agent Interface:
----------------
    The Game uses an "agent" callable per player:

        action = agent(game_state: GameState) → Action

    GameState is a read-only snapshot passed to the agent each time
    it must act. Agents can be:
        • RandomAgent  (baseline / opponent)
        • RLAgent      (transformer + RL policy)
        • HumanAgent   (interactive CLI)

    See utils/agents.py for implementations.

GameState Fields:
-----------------
    street          : str           current street name
    community_cards : List[Card]    face-up board cards
    pot             : int           total chips in pot
    current_bet     : int           highest bet this street (amount to call)
    call_amount     : int           chips needed to call (= current_bet - player.bet)
    min_raise       : int           minimum legal raise size
    players         : List[PlayerView]  public info about each seat
    you             : PlayerView    full info for the acting player
    history         : List[str]     token sequence of all prior actions

HandResult Fields:
------------------
    winners         : List[int]     player ids who won chips
    winnings        : Dict[int,int] player_id → chips won
    showdown_hands  : Dict[int, HandRank]  best hand per surviving player
    history         : List[str]     full token sequence of the hand
    final_stacks    : Dict[int,int] chip counts after the hand
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple

from engine.cards import Card, Deck, HandEvaluator, HandRank
from engine.player import Player, Action, ActionType
from engine.pot import Pot


# ---------------------------------------------------------------------------
# View objects (read-only snapshots for agents)
# ---------------------------------------------------------------------------

@dataclass
class PlayerView:
    """Public-facing snapshot of one player (no hidden hole cards for others)."""
    id: int
    name: str
    chips: int
    bet: int
    is_folded: bool
    is_all_in: bool
    # hole_cards only populated for the acting player themselves
    hole_cards: Optional[List[Card]] = None
    action_log_tokens: List[str] = field(default_factory=list)


@dataclass
class GameState:
    """
    Complete read-only snapshot of game state delivered to an agent.

    Passed to agent callables at every decision point.
    The agent must return an Action without modifying this object.
    """
    street: str
    community_cards: List[Card]
    pot: int
    current_bet: int         # highest total bet this street
    call_amount: int         # chips needed to call (for the acting player)
    min_raise: int
    max_raise: int           # acting player's remaining chips
    players: List[PlayerView]
    you: PlayerView          # full view of the acting player
    dealer_pos: int
    history: List[str]       # token sequence: ["<ROUND:preflop>", "<P0:RAISE:20>", ...]
    hand_number: int = 0


# ---------------------------------------------------------------------------
# HandResult
# ---------------------------------------------------------------------------

@dataclass
class HandResult:
    """Summary of a completed hand."""
    hand_number: int
    winners: List[int]                     # player ids
    winnings: Dict[int, int]              # player_id → chips won
    showdown_hands: Dict[int, HandRank]   # player_id → best hand (if went to showdown)
    history: List[str]                    # full token sequence
    final_stacks: Dict[int, int]          # player_id → chip count after hand
    pot_total: int


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class Game:
    """
    Texas Hold'em No-Limit engine.

    Parameters
    ----------
    players        : List[Player]    seats (2–9 players)
    agents         : Dict[int, Callable[[GameState], Action]]
                     maps player_id → agent function
    small_blind    : int
    big_blind      : int
    dealer_pos     : int             starting dealer button position (0-indexed)
    hand_number    : int             for logging / token sequences

    Methods
    -------
    play_hand()    → HandResult
    """

    MAX_RAISES_PER_STREET = 4   # cap on re-raises (prevents infinite loops)

    def __init__(
        self,
        players: List[Player],
        agents: Dict[int, Callable[[GameState], Action]],
        small_blind: int = 10,
        big_blind: int = 20,
        dealer_pos: int = 0,
        hand_number: int = 0,
    ):
        if len(players) < 2:
            raise ValueError("Need at least 2 players")
        self.players = players
        self.agents = agents
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_pos = dealer_pos % len(players)
        self.hand_number = hand_number

        self.deck = Deck()
        self.pot = Pot()
        self.community_cards: List[Card] = []
        self.history: List[str] = []
        self._current_bet: int = 0
        self._last_raise_size: int = big_blind

    # ── Public ─────────────────────────────────────────────────────────────

    def play_hand(self) -> HandResult:
        """
        Run one complete hand.

        Returns HandResult with winners, winnings, history, final stacks.
        """
        self._reset()
        self._post_blinds()
        self._deal_hole_cards()

        self._betting_round("preflop")

        if self._players_in_hand() > 1:
            self._deal_community(3, "flop")
            self._betting_round("flop")

        if self._players_in_hand() > 1:
            self._deal_community(1, "turn")
            self._betting_round("turn")

        if self._players_in_hand() > 1:
            self._deal_community(1, "river")
            self._betting_round("river")

        result = self._showdown()
        self._apply_winnings(result.winnings)
        return result

    # ── Setup ──────────────────────────────────────────────────────────────

    def _reset(self):
        self.deck.reset()
        self.deck.shuffle()
        self.pot.reset()
        self.community_cards = []
        self.history = [f"<HAND:{self.hand_number}>"]
        self._current_bet = 0
        self._last_raise_size = self.big_blind
        for p in self.players:
            p.reset_for_hand()

    def _post_blinds(self):
        n = len(self.players)
        # Heads-up: dealer posts SB
        if n == 2:
            sb_idx = self.dealer_pos
            bb_idx = (self.dealer_pos + 1) % n
        else:
            sb_idx = (self.dealer_pos + 1) % n
            bb_idx = (self.dealer_pos + 2) % n

        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]

        sb_action = Action(ActionType.RAISE, min(self.small_blind, sb_player.chips),
                           sb_player.id, "blinds")
        bb_action = Action(ActionType.RAISE, min(self.big_blind, bb_player.chips),
                           bb_player.id, "blinds")

        self._apply_blind(sb_player, sb_action)
        self._apply_blind(bb_player, bb_action)
        self._current_bet = bb_player.bet
        self._last_raise_size = self.big_blind

        self.history.append(f"<ROUND:preflop>")
        self.history.append(f"<P{sb_player.id}:SB:{self.small_blind}>")
        self.history.append(f"<P{bb_player.id}:BB:{self.big_blind}>")

    def _apply_blind(self, player: Player, action: Action):
        """Post a blind. Chips tracked in player.bet; collected at end of preflop betting."""
        player.apply_action(action, call_amount=action.amount)

    def _deal_hole_cards(self):
        for p in self.players:
            if p.is_active:
                p.hole_cards = self.deck.deal(2)

    def _deal_community(self, n: int, street: str):
        self.deck.burn()
        new_cards = self.deck.deal(n)
        self.community_cards.extend(new_cards)
        self.history.append(f"<ROUND:{street}>")
        cards_str = ",".join(str(c) for c in new_cards)
        self.history.append(f"<BOARD:{cards_str}>")

    # ── Betting ────────────────────────────────────────────────────────────

    def _betting_round(self, street: str):
        """
        Run one full betting round.

        Action order starts left of dealer (or left of BB preflop).
        Continues until all active players have acted and bets are matched.
        """
        n = len(self.players)

        # Reset per-street bets for postflop streets only.
        # Preflop: blind bets already posted & tracked in p.bet — zeroing them
        # would make players call the full BB again despite having already paid.
        if street != "preflop":
            for p in self.players:
                p.reset_for_street()

        if street == "preflop":
            # Preflop: action starts left of BB
            if n == 2:
                start_idx = self.dealer_pos            # SB acts first HU
            else:
                start_idx = (self.dealer_pos + 3) % n  # UTG
        else:
            # Postflop: action starts left of dealer
            if n == 2:
                start_idx = (self.dealer_pos + 1) % n
            else:
                start_idx = (self.dealer_pos + 1) % n
            self._current_bet = 0
            self._last_raise_size = self.big_blind

        # Track who needs to act
        acted = set()
        raises_this_street = 0
        order = [(start_idx + i) % n for i in range(n)]

        # Preflop: BB has option to re-raise even if everyone just called
        bb_player_id = None
        if street == "preflop" and n >= 2:
            bb_idx = (self.dealer_pos + (1 if n == 2 else 2)) % n
            bb_player_id = self.players[bb_idx].id

        while True:
            acted_this_pass = False
            for idx in order:
                player = self.players[idx]
                if not player.can_act:
                    continue
                if player.id in acted and player.bet >= self._current_bet:
                    # BB option check
                    if player.id == bb_player_id and self._current_bet == self.big_blind and len(acted) < self._active_count():
                        pass   # BB gets option
                    else:
                        continue

                action = self._request_action(player, street)
                self._process_action(player, action, street)
                acted.add(player.id)
                acted_this_pass = True

                if action.action_type == ActionType.RAISE:
                    raises_this_street += 1
                    acted = {player.id}  # everyone else must act again

                    if raises_this_street >= self.MAX_RAISES_PER_STREET:
                        # No more raises allowed — remaining players just call or fold
                        break

                if self._players_in_hand() <= 1:
                    # Collect remaining bets before exiting
                    self.pot.add_from_players(self.players)
                    return

            # Check if betting is complete
            bets_matched = all(
                p.bet == self._current_bet or p.is_all_in or p.is_folded
                for p in self.players
                if p.is_active
            )
            if bets_matched and acted_this_pass is False:
                break
            if bets_matched and all(
                p.id in acted or not p.can_act
                for p in self.players if p.is_active
            ):
                break

        # Collect bets into pot
        self.pot.add_from_players(self.players)

    def _request_action(self, player: Player, street: str) -> Action:
        """Build GameState and call the agent."""
        state = self._build_state(player, street)
        agent = self.agents.get(player.id)
        if agent is None:
            raise RuntimeError(f"No agent registered for player {player.id}")
        action = agent(state)
        return self._validate_action(action, player, state)

    def _validate_action(self, action: Action, player: Player, state: GameState) -> Action:
        """
        Enforce legal action rules.

        Illegal actions are silently converted to the closest legal one:
          • RAISE below min-raise  → adjusted to min_raise
          • RAISE with no chips    → ALL_IN
          • CHECK when there's a bet → CALL
          • CALL with 0 to call   → CHECK
          • any action with 0 chips → FOLD
        """
        at = action.action_type

        if player.chips == 0:
            return Action(ActionType.ALL_IN, 0, player.id, state.street)

        call_amount = state.call_amount

        if at == ActionType.CHECK and call_amount > 0:
            return Action(ActionType.CALL, 0, player.id, state.street)

        if at == ActionType.CALL and call_amount == 0:
            return Action(ActionType.CHECK, 0, player.id, state.street)

        if at == ActionType.RAISE:
            if action.amount >= player.chips:
                return Action(ActionType.ALL_IN, player.chips, player.id, state.street)
            if action.amount < state.min_raise:
                action = Action(ActionType.RAISE, state.min_raise, player.id, state.street)

        return action

    def _process_action(self, player: Player, action: Action, street: str):
        """Apply action, update current_bet, log token.

        NOTE: chips are NOT added to self.pot here. They are collected
        in bulk by add_from_players() at the end of each betting round.
        This avoids double-counting.
        """
        action.player_id = player.id
        action.street = street

        call_amount = max(0, self._current_bet - player.bet)
        player.apply_action(action, call_amount=call_amount)

        if action.action_type == ActionType.RAISE:
            new_total = player.bet
            raise_size = new_total - self._current_bet
            self._last_raise_size = max(raise_size, self._last_raise_size)
            self._current_bet = new_total

        elif action.action_type == ActionType.ALL_IN:
            if player.bet > self._current_bet:
                self._current_bet = player.bet

        token = f"<P{player.id}:{action.to_token().strip('<>')}>"
        self.history.append(token)

    # ── Showdown ───────────────────────────────────────────────────────────

    def _showdown(self) -> HandResult:
        """
        Determine winners and compute winnings.

        If only one player remains (everyone else folded), award pot without
        revealing hole cards. Otherwise evaluate all hands at showdown.
        """
        active = [p for p in self.players if p.in_hand]
        showdown_hands: Dict[int, HandRank] = {}

        if len(active) == 1:
            # Only one player left — wins entire pot
            winner = active[0]
            self.history.append(f"<WINNER:P{winner.id}:uncontested>")
            return HandResult(
                hand_number=self.hand_number,
                winners=[winner.id],
                winnings={winner.id: self.pot.total},
                showdown_hands={},
                history=list(self.history),
                final_stacks={p.id: p.chips for p in self.players},
                pot_total=self.pot.total,
            )

        # Evaluate all hands
        evaluations: Dict[int, HandRank] = {}
        for p in active:
            all_cards = p.hole_cards + self.community_cards
            evaluations[p.id] = HandEvaluator.evaluate(all_cards)
            showdown_hands[p.id] = evaluations[p.id]

        # Sort players by hand strength (best first)
        sorted_players = sorted(active, key=lambda p: evaluations[p.id], reverse=True)

        # Build ranked groups (handle ties)
        ranked_groups: List[List[int]] = []
        current_group = [sorted_players[0].id]
        for i in range(1, len(sorted_players)):
            if evaluations[sorted_players[i].id] == evaluations[sorted_players[i-1].id]:
                current_group.append(sorted_players[i].id)
            else:
                ranked_groups.append(current_group)
                current_group = [sorted_players[i].id]
        ranked_groups.append(current_group)

        # Calculate side pots
        all_in_amounts = {p.id: p.total_bet for p in active if p.is_all_in}
        self.pot.calculate_side_pots(all_in_amounts, [p.id for p in active])

        winnings = self.pot.award(ranked_groups)
        winners = ranked_groups[0]

        for pid, amount in winnings.items():
            name = next(p.name for p in self.players if p.id == pid)
            hand_name = showdown_hands.get(pid, None)
            hand_str = hand_name.name() if hand_name else "?"
            self.history.append(f"<WINNER:P{pid}:{hand_str}:{amount}>")

        return HandResult(
            hand_number=self.hand_number,
            winners=winners,
            winnings=winnings,
            showdown_hands=showdown_hands,
            history=list(self.history),
            final_stacks={p.id: p.chips for p in self.players},
            pot_total=self.pot.total,
        )

    def _apply_winnings(self, winnings: Dict[int, int]):
        """Add won chips back to player stacks."""
        for pid, amount in winnings.items():
            player = next(p for p in self.players if p.id == pid)
            player.chips += amount

    # ── Helpers ────────────────────────────────────────────────────────────

    def _players_in_hand(self) -> int:
        return sum(1 for p in self.players if p.in_hand and p.is_active)

    def _active_count(self) -> int:
        return sum(1 for p in self.players if p.can_act)

    def _build_state(self, acting_player: Player, street: str) -> GameState:
        """Construct a read-only GameState for the acting player's agent."""
        call_amount = max(0, self._current_bet - acting_player.bet)
        min_raise = self._current_bet + self._last_raise_size

        player_views = []
        for p in self.players:
            pv = PlayerView(
                id=p.id,
                name=p.name,
                chips=p.chips,
                bet=p.bet,
                is_folded=p.is_folded,
                is_all_in=p.is_all_in,
                hole_cards=p.hole_cards if p.id == acting_player.id else None,
                action_log_tokens=[a.to_token() for a in p.action_log],
            )
            player_views.append(pv)

        acting_view = next(pv for pv in player_views if pv.id == acting_player.id)

        return GameState(
            street=street,
            community_cards=list(self.community_cards),
            pot=self.pot.total,
            current_bet=self._current_bet,
            call_amount=call_amount,
            min_raise=min_raise,
            max_raise=acting_player.chips,
            players=player_views,
            you=acting_view,
            dealer_pos=self.dealer_pos,
            history=list(self.history),
            hand_number=self.hand_number,
        )

    def rotate_dealer(self):
        """Advance dealer button one seat clockwise."""
        self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
        self.hand_number += 1
