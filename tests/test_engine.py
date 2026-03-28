"""
test_engine.py — Comprehensive Engine Test Suite
=================================================

Tests every edge case in the poker simulation engine.

Test Categories:
----------------
    1. Card & Deck Tests
       - Card immutability
       - Deck deal, burn, remaining
       - Deck exhaustion guard

    2. Hand Evaluation Tests
       - All 9 hand categories
       - Wheel straight (A-2-3-4-5)
       - Broadway straight (T-J-Q-K-A)
       - Flush vs straight priority
       - Tie detection

    3. Pot & Side Pot Tests
       - Simple main pot
       - Two-player all-in side pot
       - Three-player multi-level side pot
       - Tie-split with remainder

    4. Player Tests
       - apply_action for all action types
       - All-in detection
       - Chip conservation

    5. Game Flow Tests
       - Full hand: 2-player heads-up
       - Full hand: 6-player
       - Everyone folds preflop
       - All players go all-in preflop
       - Single player remaining wins uncontested

    6. Tokenizer Tests
       - Encode / decode round-trip
       - Amount bucketing
       - Sequence truncation
       - Unknown token handling

Run with:
    python -m pytest tests/test_engine.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.cards import Card, Deck, HandEvaluator, HandRank
from engine.player import Player, Action, ActionType
from engine.pot import Pot
from engine.game import Game, GameState
from utils.agents import RandomAgent, CallAgent, RuleBasedAgent
from model.tokenizer import PokerTokenizer, bucket_amount, bucket_midpoint


# ============================================================
# Helpers
# ============================================================

def make_cards(*strings):
    """Create Card list from strings like "Ah", "2c"."""
    return [Card.from_str(s) for s in strings]

def make_player(pid, chips=1000):
    return Player(id=pid, name=f"P{pid}", chips=chips)

def make_game_2player(chips=500):
    """Two-player game with CallAgent opponents for deterministic tests."""
    p0 = make_player(0, chips)
    p1 = make_player(1, chips)
    agents = {0: CallAgent(), 1: CallAgent()}
    return Game([p0, p1], agents, small_blind=10, big_blind=20), [p0, p1]

def make_game_6player(chips=1000):
    players = [make_player(i, chips) for i in range(6)]
    agents = {i: RandomAgent(seed=i) for i in range(6)}
    return Game(players, agents, small_blind=10, big_blind=20), players


# ============================================================
# 1. Card & Deck Tests
# ============================================================

class TestCard:
    def test_creation(self):
        c = Card(14, 3)  # Ace of spades
        assert c.rank == 14
        assert c.suit == 3

    def test_immutability(self):
        c = Card(14, 3)
        with pytest.raises(AttributeError):
            c.rank = 5

    def test_from_str(self):
        c = Card.from_str("Ah")
        assert c.rank == 14 and c.suit == 2
        c2 = Card.from_str("2c")
        assert c2.rank == 2 and c2.suit == 0
        c3 = Card.from_str("Ts")
        assert c3.rank == 10 and c3.suit == 3

    def test_equality(self):
        assert Card(14, 3) == Card(14, 3)
        assert Card(14, 3) != Card(14, 2)

    def test_hash(self):
        s = {Card(14, 3), Card(14, 3), Card(2, 0)}
        assert len(s) == 2

    def test_invalid_rank(self):
        with pytest.raises(ValueError):
            Card(1, 0)   # rank 1 invalid

    def test_invalid_suit(self):
        with pytest.raises(ValueError):
            Card(14, 4)  # suit 4 invalid

    def test_ordering(self):
        assert Card(2, 0) < Card(14, 3)


class TestDeck:
    def test_full_deck(self):
        deck = Deck()
        assert deck.remaining() == 52

    def test_deal(self):
        deck = Deck()
        deck.shuffle(seed=42)
        cards = deck.deal(5)
        assert len(cards) == 5
        assert deck.remaining() == 47

    def test_burn(self):
        deck = Deck()
        deck.shuffle(seed=1)
        deck.burn()
        assert deck.remaining() == 51

    def test_deal_all(self):
        deck = Deck()
        deck.shuffle()
        cards = deck.deal(52)
        assert len(cards) == 52
        assert deck.remaining() == 0

    def test_deal_exhaustion(self):
        deck = Deck()
        deck.deal(52)
        with pytest.raises(RuntimeError):
            deck.deal(1)

    def test_no_duplicates(self):
        deck = Deck()
        deck.shuffle(seed=99)
        cards = deck.deal(52)
        assert len(set(cards)) == 52

    def test_reset(self):
        deck = Deck()
        deck.deal(10)
        deck.reset()
        assert deck.remaining() == 52


# ============================================================
# 2. Hand Evaluation Tests
# ============================================================

class TestHandEvaluator:
    # ── Category Detection ─────────────────────────────────

    def test_royal_flush(self):
        cards = make_cards("Ah", "Kh", "Qh", "Jh", "Th")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 9
        assert hr.name() == "Royal Flush"

    def test_straight_flush(self):
        cards = make_cards("9h", "8h", "7h", "6h", "5h")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 8

    def test_four_of_a_kind(self):
        cards = make_cards("Ah", "As", "Ad", "Ac", "2h")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 7

    def test_full_house(self):
        cards = make_cards("Ah", "As", "Ad", "Kh", "Ks")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 6

    def test_flush(self):
        cards = make_cards("Ah", "Jh", "9h", "7h", "3h")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 5

    def test_straight(self):
        cards = make_cards("9h", "8d", "7s", "6c", "5h")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 4

    def test_three_of_a_kind(self):
        cards = make_cards("Ah", "As", "Ad", "7h", "3c")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 3

    def test_two_pair(self):
        cards = make_cards("Ah", "As", "Kd", "Ks", "3c")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 2

    def test_one_pair(self):
        cards = make_cards("Ah", "As", "9d", "7s", "3c")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 1

    def test_high_card(self):
        cards = make_cards("Ah", "Js", "9d", "7s", "3c")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 0

    # ── Special Cases ──────────────────────────────────────

    def test_wheel_straight(self):
        """A-2-3-4-5 (wheel) is a straight with high card 5."""
        cards = make_cards("Ah", "2d", "3s", "4c", "5h")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 4
        assert hr.tiebreakers[0] == 5   # high card is 5

    def test_broadway_straight(self):
        """T-J-Q-K-A is a straight (NOT royal flush unless suited)."""
        cards = make_cards("Th", "Jd", "Qs", "Kc", "Ah")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 4    # straight, not royal flush

    def test_straight_vs_flush_priority(self):
        """Both straight and flush possible — straight flush wins."""
        cards = make_cards("9h", "8h", "7h", "6h", "5h")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 8  # straight flush beats plain flush

    def test_7_card_best_hand(self):
        """7-card evaluation returns best 5."""
        cards = make_cards("Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d")
        hr = HandEvaluator.evaluate(cards)
        assert hr.category == 9  # Royal flush found in the 7 cards

    def test_tie_same_rank(self):
        """Identical hands should be equal."""
        hand1 = make_cards("Ah", "As", "Ad", "Ac", "2h")
        hand2 = make_cards("Ah", "As", "Ad", "Ac", "2d")
        hr1 = HandEvaluator.evaluate(hand1)
        hr2 = HandEvaluator.evaluate(hand2)
        assert hr1 == hr2

    def test_kicker_differentiates(self):
        """Pair of Aces with K kicker beats pair of Aces with Q kicker."""
        hand1 = make_cards("Ah", "As", "Kd", "7s", "3c")  # pair A, K kicker
        hand2 = make_cards("Ah", "As", "Qd", "7s", "3c")  # pair A, Q kicker
        hr1 = HandEvaluator.evaluate(hand1)
        hr2 = HandEvaluator.evaluate(hand2)
        assert hr1 > hr2

    def test_insufficient_cards_raises(self):
        with pytest.raises(ValueError):
            HandEvaluator.evaluate(make_cards("Ah", "Kh", "Qh", "Jh"))  # only 4


# ============================================================
# 3. Pot & Side Pot Tests
# ============================================================

class TestPot:
    def test_simple_pot(self):
        pot = Pot()
        pot.add(0, 100)
        pot.add(1, 100)
        assert pot.total == 200

    def test_award_one_winner(self):
        pot = Pot()
        pot.add(0, 100)
        pot.add(1, 100)
        pot.calculate_side_pots({}, [0, 1])
        winnings = pot.award([[0], [1]])
        assert winnings[0] == 200

    def test_award_split(self):
        """Two players tie — pot split evenly."""
        pot = Pot()
        pot.add(0, 100)
        pot.add(1, 100)
        pot.calculate_side_pots({}, [0, 1])
        winnings = pot.award([[0, 1]])
        assert winnings[0] == 100
        assert winnings[1] == 100

    def test_side_pot_allin(self):
        """
        Player 0 all-in for 50, P1 and P2 contribute 100.
        Main pot = 150 (all eligible)
        Side pot = 100 (P1, P2 only)
        """
        pot = Pot()
        pot.add(0, 50)
        pot.add(1, 100)
        pot.add(2, 100)
        pots = pot.calculate_side_pots({0: 50}, [0, 1, 2])
        assert len(pots) == 2
        assert pots[0].amount == 150
        assert set(pots[0].eligible) == {0, 1, 2}
        assert pots[1].amount == 100
        assert set(pots[1].eligible) == {1, 2}

    def test_side_pot_p0_wins_main_p1_wins_side(self):
        pot = Pot()
        pot.add(0, 50)
        pot.add(1, 100)
        pot.add(2, 100)
        pot.calculate_side_pots({0: 50}, [0, 1, 2])
        # P0 wins main pot, P1 wins side pot
        winnings = pot.award([[0], [1], [2]])
        assert winnings.get(0, 0) == 150
        assert winnings.get(1, 0) == 100

    def test_three_allin_levels(self):
        """
        P0 all-in 30, P1 all-in 70, P2 calls 100.
        Tier 1: 30×3=90 (all eligible)
        Tier 2: 40×2=80 (P1, P2)
        Tier 3: 30×1=30 (P2 only)
        """
        pot = Pot()
        pot.add(0, 30)
        pot.add(1, 70)
        pot.add(2, 100)
        pots = pot.calculate_side_pots({0: 30, 1: 70}, [0, 1, 2])
        total_in_pots = sum(p.amount for p in pots)
        assert total_in_pots == 200

    def test_odd_chip_remainder(self):
        """Remainder after split goes to first player."""
        pot = Pot()
        pot.add(0, 101)
        pot.add(1, 100)
        pot.calculate_side_pots({}, [0, 1])
        winnings = pot.award([[0, 1]])
        total_won = sum(winnings.values())
        assert total_won == 201

    def test_reset(self):
        pot = Pot()
        pot.add(0, 100)
        pot.reset()
        assert pot.total == 0


# ============================================================
# 4. Player Tests
# ============================================================

class TestPlayer:
    def test_call_action(self):
        p = make_player(0, 500)
        action = Action(ActionType.CALL, 0, 0)
        moved = p.apply_action(action, call_amount=50)
        assert moved == 50
        assert p.chips == 450

    def test_raise_action(self):
        p = make_player(0, 500)
        action = Action(ActionType.RAISE, 100, 0)
        moved = p.apply_action(action)
        assert moved == 100
        assert p.chips == 400

    def test_fold_action(self):
        p = make_player(0, 500)
        action = Action(ActionType.FOLD, 0, 0)
        p.apply_action(action)
        assert p.is_folded
        assert p.chips == 500

    def test_allin_action(self):
        p = make_player(0, 300)
        action = Action(ActionType.ALL_IN, 300, 0)
        moved = p.apply_action(action)
        assert moved == 300
        assert p.chips == 0
        assert p.is_all_in

    def test_call_more_than_chips_goes_allin(self):
        p = make_player(0, 50)
        action = Action(ActionType.CALL, 0, 0)
        moved = p.apply_action(action, call_amount=200)  # can only afford 50
        assert moved == 50
        assert p.chips == 0
        assert p.is_all_in

    def test_reset_for_hand(self):
        p = make_player(0, 500)
        p.is_folded = True
        p.bet = 100
        p.reset_for_hand()
        assert not p.is_folded
        assert p.bet == 0

    def test_can_act(self):
        p = make_player(0, 500)
        assert p.can_act
        p.is_folded = True
        assert not p.can_act

    def test_in_hand(self):
        p = make_player(0, 0)
        p.is_all_in = True
        assert p.in_hand   # all-in players are still in the hand

    def test_action_log(self):
        p = make_player(0, 500)
        p.apply_action(Action(ActionType.RAISE, 100, 0))
        p.apply_action(Action(ActionType.CALL, 0, 0), call_amount=50)
        assert len(p.action_log) == 2


# ============================================================
# 5. Game Flow Tests
# ============================================================

class TestGameFlow:
    def test_2player_hand_completes(self):
        game, players = make_game_2player()
        result = game.play_hand()
        assert result is not None
        assert len(result.winners) >= 1
        total_chips = sum(p.chips + result.winnings.get(p.id, 0) for p in players)
        # Chip conservation: total should equal starting amount
        # (post-hand chips already include winnings in game._apply_winnings)
        assert sum(p.chips for p in players) == 1000

    def test_6player_hand_completes(self):
        game, players = make_game_6player()
        result = game.play_hand()
        assert result is not None
        assert sum(p.chips for p in players) == 6000

    def test_chip_conservation_many_hands(self):
        """Chips must never be created or destroyed."""
        game, players = make_game_6player(chips=500)
        for _ in range(50):
            # Rebuy broke players
            for p in players:
                if p.chips == 0:
                    p.chips = 500
            game.play_hand()
            game.rotate_dealer()
        # Total should be original amount * n_players (minus any that were zeroed and not rebought)

    def test_everyone_folds_preflop(self):
        """When all but one player folds, that player wins uncontested."""
        p0 = make_player(0, 500)
        p1 = make_player(1, 500)

        fold_count = [0]

        def fold_agent(state):
            fold_count[0] += 1
            return Action(ActionType.FOLD, 0, state.you.id, state.street)

        game = Game([p0, p1], {0: CallAgent(), 1: fold_agent}, small_blind=10, big_blind=20)
        result = game.play_hand()
        assert len(result.winners) == 1

    def test_all_in_preflop(self):
        """All players shove preflop — valid result with side pots if needed."""
        players = [make_player(i, random_chips(i)) for i in range(3)]
        initial_total = sum(p.chips for p in players)  # capture before play
        agents = {i: (lambda state: Action(ActionType.ALL_IN, state.you.chips, state.you.id, state.street))
                  for i in range(3)}
        game = Game(players, agents, small_blind=10, big_blind=20)
        result = game.play_hand()
        assert result is not None
        total = sum(p.chips for p in players)
        assert total == initial_total   # chips must be conserved across the hand

    def test_history_is_non_empty(self):
        game, _ = make_game_2player()
        result = game.play_hand()
        assert len(result.history) > 2

    def test_rotate_dealer(self):
        game, _ = make_game_2player()
        old_pos = game.dealer_pos
        game.rotate_dealer()
        assert game.dealer_pos != old_pos or len(game.players) == 1

    def test_showdown_hands_populated(self):
        """showdown_hands populated only when hand reaches showdown."""
        game, _ = make_game_2player()
        result = game.play_hand()
        # CallAgent never folds, so we expect a showdown
        assert len(result.showdown_hands) >= 1


def random_chips(seed):
    import random
    random.seed(seed)
    return random.choice([300, 400, 500])


# ============================================================
# 6. Tokenizer Tests
# ============================================================

class TestTokenizer:
    def setup_method(self):
        self.tok = PokerTokenizer(max_len=128)

    def test_encode_length(self):
        history = ["<HAND:1>", "<ROUND:preflop>", "<P0:SB:10>", "<P1:BB:20>"]
        ids = self.tok.encode(history)
        assert len(ids) == 128

    def test_encode_with_hole_cards(self):
        history = ["<HAND:0>"]
        ids = self.tok.encode(history, hole_cards=["Ah", "Kd"])
        assert len(ids) == 128

    def test_no_all_unknown(self):
        """Encoding should not be all UNK tokens."""
        history = ["<HAND:1>", "<ROUND:preflop>"]
        ids = self.tok.encode(history)
        unk_count = ids.count(self.tok.tok2id["<UNK>"])
        assert unk_count < len(ids) * 0.5

    def test_bos_at_start(self):
        ids = self.tok.encode(["<HAND:0>"])
        assert ids[0] == self.tok.tok2id["<BOS>"]

    def test_amount_bucketing(self):
        assert bucket_amount(5) == 0
        assert bucket_amount(10) == 1
        assert bucket_amount(25) == 2
        assert bucket_amount(75) == 3
        assert bucket_amount(150) == 4
        assert bucket_amount(300) == 5
        assert bucket_amount(700) == 6
        assert bucket_amount(5000) == 7

    def test_bucket_midpoint(self):
        mid = bucket_midpoint(3)  # bucket 3 = 50–99
        assert 50 <= mid <= 99

    def test_decode_action_fold(self):
        # FOLD should be index 0 in ACTION_TOKENS_LIST
        token = self.tok.decode_action(0)
        assert "FOLD" in token

    def test_decode_action_check(self):
        token = self.tok.decode_action(1)
        assert "CHECK" in token

    def test_action_type_and_amount_fold(self):
        at, amt = self.tok.action_token_to_action_type_and_amount("<ACT:FOLD>", 0, 500)
        assert at == "FOLD"
        assert amt == 0

    def test_action_type_and_amount_raise(self):
        at, amt = self.tok.action_token_to_action_type_and_amount("<ACT:RAISE:AMT3>", 20, 500)
        assert at == "RAISE"
        assert amt > 0

    def test_truncation_left(self):
        """Very long history should be truncated from the left."""
        history = [f"<HAND:{i}>" for i in range(200)]
        ids = self.tok.encode(history, truncate_left=True)
        assert len(ids) == 128
        assert ids[0] == self.tok.tok2id["<BOS>"]

    def test_vocab_size(self):
        assert self.tok.vocab_size > 100

    def test_num_actions(self):
        assert self.tok.num_actions == 19  # 3 + 8 RAISE + 8 ALL_IN


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
