"""
agents.py — Poker Agent Implementations
=========================================

This module provides agent callables that conform to the interface expected
by engine/game.py:

    agent(game_state: GameState) → Action

All agents receive a GameState snapshot and return an Action.
Agents must NOT mutate the GameState.

Agent Hierarchy:
----------------

    BaseAgent (ABC)
    ├── RandomAgent       — uniform random legal action
    ├── CallAgent         — always calls (or checks), never raises (baseline)
    ├── RuleBasedAgent    — heuristic: raise strong hands, fold weak
    └── RLAgent           — transformer model + RL policy (the main model)

RandomAgent is used as the default opponent pool during training.
RLAgent wraps PokerTransformer and exposes:
  • act(state)          — sample action (exploration)
  • act_greedy(state)   — argmax (evaluation)
  • update_model(model) — hot-swap model weights mid-training

Agent-to-Token Pipeline (RLAgent):
------------------------------------

    GameState
        │
        ▼
    tokenizer.encode(history, hole_cards)
        │
        ▼
    PokerTransformer.forward(token_ids)
        │
        ▼
    action_logits  ──►  softmax  ──►  sample or argmax
        │
        ▼
    action_token  ──►  tokenizer.action_token_to_action_type_and_amount()
        │
        ▼
    Action(ActionType, amount, player_id, street)
"""

from __future__ import annotations
import random
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, List

from engine.game import GameState
from engine.player import Action, ActionType
from model.tokenizer import PokerTokenizer, bucket_midpoint, AMT_BUCKETS
from model.transformer import PokerTransformer


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base for all agents."""

    @abstractmethod
    def __call__(self, state: GameState) -> Action:
        ...

    def _legal_actions(self, state: GameState) -> List[ActionType]:
        """Return list of legal ActionTypes given the current state."""
        actions = [ActionType.FOLD]
        if state.call_amount == 0:
            actions.append(ActionType.CHECK)
        else:
            actions.append(ActionType.CALL)
        if state.you.chips > state.call_amount:  # can raise
            actions.append(ActionType.RAISE)
        actions.append(ActionType.ALL_IN)
        return actions


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------

class RandomAgent(BaseAgent):
    """
    Uniformly samples from legal actions.

    Used as:
      • Opponent pool during early RL training.
      • Baseline for performance comparison.

    Raise amounts are sampled uniformly between min_raise and player chips.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def __call__(self, state: GameState) -> Action:
        legal = self._legal_actions(state)
        chosen = random.choice(legal)
        amount = 0

        if chosen == ActionType.RAISE:
            lo = state.min_raise
            hi = state.you.chips
            amount = random.randint(lo, hi) if hi >= lo else hi

        return Action(chosen, amount, state.you.id, state.street)


# ---------------------------------------------------------------------------
# CallAgent
# ---------------------------------------------------------------------------

class CallAgent(BaseAgent):
    """
    Always calls or checks. Never raises, never folds unless forced.

    Useful as a simple passive baseline that leaks no information.
    """
    def __call__(self, state: GameState) -> Action:
        if state.call_amount == 0:
            return Action(ActionType.CHECK, 0, state.you.id, state.street)
        if state.call_amount >= state.you.chips:
            return Action(ActionType.ALL_IN, state.you.chips, state.you.id, state.street)
        return Action(ActionType.CALL, 0, state.you.id, state.street)


# ---------------------------------------------------------------------------
# RuleBasedAgent
# ---------------------------------------------------------------------------

class RuleBasedAgent(BaseAgent):
    """
    Simple heuristic agent for testing and as a stronger baseline.

    Strategy:
    ---------
    Preflop:
      • Premium hands (AA, KK, QQ, AK) → raise 3×BB
      • Playable hands (any pair, suited connectors) → call
      • Otherwise → fold (unless in BB and can check)

    Postflop:
      • If we have a made hand (pair or better) and pot odds < 30% → call/raise
      • If we're drawing (4-flush, 4-straight) → call if pot odds < 20%
      • Otherwise → check or fold

    This is NOT optimal poker — it's a useful stepping stone above random.
    """
    def __init__(self, aggression: float = 0.5):
        """aggression: 0=passive, 1=aggressive. Scales raise frequency."""
        self.aggression = aggression

    def __call__(self, state: GameState) -> Action:
        pid = state.you.id
        hole = state.you.hole_cards or []

        if state.street == "preflop":
            return self._preflop_action(state, hole)
        else:
            return self._postflop_action(state, hole)

    def _preflop_action(self, state, hole):
        if len(hole) < 2:
            return self._check_or_call(state)

        ranks = sorted([c.rank for c in hole], reverse=True)
        suited = hole[0].suit == hole[1].suit
        r1, r2 = ranks

        # Premium: raise
        is_premium = (r1 == r2 and r1 >= 10) or (r1 == 14 and r2 >= 12)
        if is_premium and random.random() < (0.8 + 0.2 * self.aggression):
            raise_to = min(3 * 20, state.you.chips)  # 3×BB
            if raise_to < state.min_raise:
                raise_to = state.min_raise
            return Action(ActionType.RAISE, raise_to, state.you.id, state.street)

        # Playable: call
        is_playable = (r1 == r2) or (suited and abs(r1 - r2) <= 2) or (r1 >= 12 and r2 >= 10)
        if is_playable:
            return self._check_or_call(state)

        # Weak: fold if there's a bet
        if state.call_amount > 0 and random.random() > self.aggression * 0.3:
            return Action(ActionType.FOLD, 0, state.you.id, state.street)
        return self._check_or_call(state)

    def _postflop_action(self, state, hole):
        # Estimate hand strength heuristically
        from engine.cards import HandEvaluator
        if len(hole) >= 2 and len(state.community_cards) >= 3:
            all_cards = hole + state.community_cards
            hr = HandEvaluator.evaluate(all_cards)
            hand_category = hr.category

            if hand_category >= 3:  # three of a kind or better
                if random.random() < self.aggression:
                    raise_to = min(int(state.pot * 0.75), state.you.chips)
                    raise_to = max(raise_to, state.min_raise)
                    return Action(ActionType.RAISE, raise_to, state.you.id, state.street)
                return self._check_or_call(state)

            if hand_category >= 1:  # pair
                return self._check_or_call(state)

        # Weak: check or fold
        if state.call_amount > 0 and random.random() > 0.3 * self.aggression:
            return Action(ActionType.FOLD, 0, state.you.id, state.street)
        return self._check_or_call(state)

    def _check_or_call(self, state):
        if state.call_amount == 0:
            return Action(ActionType.CHECK, 0, state.you.id, state.street)
        if state.call_amount >= state.you.chips:
            return Action(ActionType.ALL_IN, state.you.chips, state.you.id, state.street)
        return Action(ActionType.CALL, 0, state.you.id, state.street)


# ---------------------------------------------------------------------------
# RLAgent
# ---------------------------------------------------------------------------

class RLAgent(BaseAgent):
    """
    Transformer-based reinforcement learning agent.

    The agent converts the GameState into a token sequence, runs the
    PokerTransformer forward pass, and samples an action from the
    resulting probability distribution.

    Modes:
    ------
        explore=True  (training)  → sample from distribution (stochastic)
        explore=False (eval)      → argmax (deterministic)

    Last action info (for PPO):
    ---------------------------
        After each __call__, the following attributes are populated:
            last_log_prob  : float   log π(a|s)
            last_value     : float   V(s)  from value head
            last_action_idx: int     index into action vocabulary

    These are consumed by the PPO trainer to construct experience tuples:
        (state_tokens, action_idx, log_prob, value, reward, done)
    """

    def __init__(
        self,
        model: PokerTransformer,
        tokenizer: PokerTokenizer,
        player_id: int,
        device: str = "cpu",
        explore: bool = True,
        temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.player_id = player_id
        self.device = torch.device(device)
        self.explore = explore
        self.temperature = temperature

        # Populated after each call (used by PPO trainer)
        self.last_log_prob: float = 0.0
        self.last_value: float = 0.0
        self.last_action_idx: int = 0
        self.last_token_ids: Optional[torch.Tensor] = None

    def __call__(self, state: GameState) -> Action:
        # Encode game history
        hole_strs = [repr(c).replace("♣","c").replace("♦","d")
                     .replace("♥","h").replace("♠","s")
                     for c in (state.you.hole_cards or [])]

        token_ids = self.tokenizer.encode(state.history, hole_cards=hole_strs)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        self.model.eval() if not self.explore else self.model.train()
        with torch.no_grad() if not self.explore else torch.enable_grad():
            logits, values = self.model(token_tensor)

        probs = F.softmax(logits / self.temperature, dim=-1)   # (1, num_actions)

        if self.explore:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx).item()
        else:
            action_idx = probs.argmax(dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action_idx).item()

        # Store for PPO
        self.last_log_prob = log_prob
        self.last_value = values.squeeze().item()
        self.last_action_idx = action_idx.item()
        self.last_token_ids = token_tensor.squeeze(0)

        # Decode to Action
        action_token = self.tokenizer.decode_action(self.last_action_idx)
        action_type_str, amount = self.tokenizer.action_token_to_action_type_and_amount(
            action_token, state.call_amount, state.you.chips
        )

        # Map string to ActionType
        try:
            action_type = ActionType[action_type_str]
        except KeyError:
            action_type = ActionType.FOLD
            amount = 0

        return Action(action_type, amount, state.you.id, state.street)

    def update_model(self, new_model: PokerTransformer):
        """Replace model weights (used for target network updates)."""
        self.model.load_state_dict(new_model.state_dict())
        self.model.eval()

    def set_explore(self, explore: bool):
        self.explore = explore
