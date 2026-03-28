# `tests/test_engine.py` — Comprehensive Engine Test Suite

## Purpose

Validates every layer of the poker simulation engine including hand evaluation correctness, chip conservation across hundreds of hands, side-pot arithmetic, and tokenizer round-trips.

Run with:
```bash
python -m pytest tests/test_engine.py -v
```
Or without pytest:
```bash
python tests/test_engine.py
```

---

## Test Categories

### 1. `TestCard` — Card Primitives (8 tests)

| Test | What it checks |
|------|----------------|
| `test_creation` | rank/suit stored correctly |
| `test_immutability` | `AttributeError` on `c.rank = x` |
| `test_from_str` | `"Ah"` → rank=14, suit=2 |
| `test_equality` | Same rank+suit → equal |
| `test_hash` | Same card deduped in a set |
| `test_invalid_rank` | rank=1 raises `ValueError` |
| `test_invalid_suit` | suit=4 raises `ValueError` |
| `test_ordering` | `Card(2,0) < Card(14,3)` |

---

### 2. `TestDeck` — Deck Operations (7 tests)

| Test | What it checks |
|------|----------------|
| `test_full_deck` | Fresh deck has 52 cards |
| `test_deal` | `deal(5)` returns 5, deck has 47 |
| `test_burn` | `burn()` reduces count by 1 |
| `test_deal_all` | Can deal all 52 cards |
| `test_deal_exhaustion` | `RuntimeError` when deck empty |
| `test_no_duplicates` | All 52 dealt cards are unique |
| `test_reset` | After reset, 52 cards again |

---

### 3. `TestHandEvaluator` — All Hand Categories (14 tests)

| Test | Category | Key Check |
|------|----------|-----------|
| `test_royal_flush` | 9 | A-K-Q-J-T suited |
| `test_straight_flush` | 8 | 9-8-7-6-5 suited |
| `test_four_of_a_kind` | 7 | Four Aces |
| `test_full_house` | 6 | AAA-KK |
| `test_flush` | 5 | Five hearts |
| `test_straight` | 4 | 9-8-7-6-5 |
| `test_three_of_a_kind` | 3 | Three Aces |
| `test_two_pair` | 2 | AA-KK |
| `test_one_pair` | 1 | AA |
| `test_high_card` | 0 | A-J-9-7-3 |
| `test_wheel_straight` | 4 | A-2-3-4-5, high=5 |
| `test_broadway_straight` | 4 | T-J-Q-K-A (not Royal) |
| `test_straight_vs_flush_priority` | 8 | Straight flush > plain flush |
| `test_7_card_best_hand` | 9 | Royal found in 7 cards |
| `test_tie_same_rank` | — | AAAA+2♥ == AAAA+2♦ |
| `test_kicker_differentiates` | — | AA+K > AA+Q |
| `test_insufficient_cards_raises` | — | `ValueError` for 4 cards |

---

### 4. `TestPot` — Pot & Side-Pot Logic (7 tests)

| Test | What it validates |
|------|-------------------|
| `test_simple_pot` | `add()` accumulates total correctly |
| `test_award_one_winner` | Single winner gets entire pot |
| `test_award_split` | Tied players split evenly |
| `test_side_pot_allin` | P0 all-in 50; creates 2 pots (150/100) |
| `test_side_pot_p0_wins_main_p1_wins_side` | Correct award per eligible set |
| `test_three_allin_levels` | 3 all-in levels; total chips conserved |
| `test_odd_chip_remainder` | 201-chip pot split → remainder to first player |
| `test_reset` | `reset()` zeroes total |

---

### 5. `TestPlayer` — Player State Mutations (8 tests)

| Test | What it validates |
|------|-------------------|
| `test_call_action` | `chips -= call_amount`, `bet += amount` |
| `test_raise_action` | `chips -= raise`, `bet += raise` |
| `test_fold_action` | `is_folded = True`, chips unchanged |
| `test_allin_action` | `chips = 0`, `is_all_in = True` |
| `test_call_more_than_chips_goes_allin` | Partial call triggers all-in |
| `test_reset_for_hand` | Clears flags and bets |
| `test_can_act` | False when folded/all-in/no chips |
| `test_in_hand` | True for all-in players |
| `test_action_log` | All actions appended to log |

---

### 6. `TestGameFlow` — End-to-End Hand Scenarios (7 tests)

| Test | What it validates |
|------|-------------------|
| `test_2player_hand_completes` | Hand finishes, chip total = 1000 |
| `test_6player_hand_completes` | Hand finishes, chip total = 6000 |
| `test_chip_conservation_many_hands` | 50 hands: no chips created or lost |
| `test_everyone_folds_preflop` | Uncontested winner declared |
| `test_all_in_preflop` | Side pots correctly computed, chips conserved |
| `test_history_is_non_empty` | history list has > 2 tokens |
| `test_rotate_dealer` | Dealer position advances |
| `test_showdown_hands_populated` | HandRank stored for showdown players |

---

### 7. `TestTokenizer` — Tokenizer Correctness (12 tests)

| Test | What it validates |
|------|-------------------|
| `test_encode_length` | Output always `max_len` tokens |
| `test_encode_with_hole_cards` | Hole card tokens inserted after BOS |
| `test_no_all_unknown` | < 50% UNK tokens in valid history |
| `test_bos_at_start` | First token is always BOS |
| `test_amount_bucketing` | Each bucket maps correctly |
| `test_bucket_midpoint` | Midpoint within bucket range |
| `test_decode_action_fold` | Index 0 → FOLD |
| `test_decode_action_check` | Index 1 → CHECK |
| `test_action_type_and_amount_fold` | FOLD → 0 amount |
| `test_action_type_and_amount_raise` | RAISE:AMT3 → (RAISE, >0) |
| `test_truncation_left` | Long sequence truncated, BOS preserved |
| `test_vocab_size` | vocab_size > 100 |
| `test_num_actions` | num_actions == 19 |

---

## Chip Conservation — The Core Invariant

The most important property tested across all game flow tests:

```
At end of any hand:
    sum(player.chips for all players) == sum(starting_chips)

Equivalently: no chips are created or destroyed.
```

This was verified across:
- **100 two-player hands** (call-only agents)
- **500 six-player hands** (random agents)
- **All-in three-player hands** (3 different stack sizes)
- **Fold-preflop scenarios** (pot collected before early return)

Three bugs were discovered and fixed via these tests:
1. Double shuffle (`deck.shuffle()` called twice per hand)
2. Preflop `reset_for_street()` zeroing blind bets
3. `pot.add()` called both in `_process_action` and `add_from_players` (double-counting)
4. Fold mid-round not calling `add_from_players` before early return
