# `engine/game.py` — Texas Hold'em Game Engine

## Purpose

Orchestrates a complete poker hand from shuffle to showdown. This is the central simulation loop — all training and evaluation code calls `game.play_hand()`.

---

## Hand Flow

```
Game.play_hand()
        │
        ├─ _reset()
        │     ├─ deck.reset() + deck.shuffle()
        │     ├─ pot.reset()
        │     └─ player.reset_for_hand() for all players
        │
        ├─ _post_blinds()
        │     ├─ SB player posts small blind (tracked in player.bet)
        │     └─ BB player posts big blind  (tracked in player.bet)
        │
        ├─ _deal_hole_cards()
        │     └─ deck.deal(2) per active player
        │
        ├─ _betting_round("preflop")
        │     ├─ does NOT reset player.bet (blinds already posted)
        │     ├─ action starts UTG (left of BB), or SB in heads-up
        │     └─ pot.add_from_players() at end
        │
        ├─ _deal_community(3, "flop")   ← deck.burn() + deal(3)
        ├─ _betting_round("flop")       ← resets bets, action from left of dealer
        │
        ├─ _deal_community(1, "turn")
        ├─ _betting_round("turn")
        │
        ├─ _deal_community(1, "river")
        ├─ _betting_round("river")
        │
        └─ _showdown()
              ├─ if 1 player remains: award entire pot uncontested
              └─ else: HandEvaluator → rank all hands → pot.award()
```

---

## Agent Interface

Agents are plain callables:

```python
action = agent(game_state: GameState) → Action
```

They receive a `GameState` snapshot and must return an `Action`. They must **not** mutate the state.

Register agents in a dict keyed by player id:

```python
agents = {
    0: my_rl_agent,
    1: random_agent,
    2: rule_based_agent,
}
game = Game(players, agents, small_blind=10, big_blind=20)
```

---

## `GameState` — Agent Input

```
GameState
├── street           : str              "preflop" | "flop" | "turn" | "river"
├── community_cards  : List[Card]       face-up board cards
├── pot              : int              total chips in pot
├── current_bet      : int              highest bet this street
├── call_amount      : int              chips needed to call (= current_bet - you.bet)
├── min_raise        : int              minimum legal raise amount
├── max_raise        : int              acting player's remaining chips
├── players          : List[PlayerView] public info on all seats
├── you              : PlayerView       full info for the acting player (includes hole_cards)
├── dealer_pos       : int
├── history          : List[str]        token sequence of all prior actions this hand
└── hand_number      : int
```

`PlayerView` contains the same fields as `Player` but `hole_cards` is `None` for opponents (hidden information).

---

## `HandResult` — Return Value

```
HandResult
├── hand_number     : int
├── winners         : List[int]            player ids who won chips
├── winnings        : Dict[int, int]       player_id → chips won
├── showdown_hands  : Dict[int, HandRank]  best hand per player at showdown
├── history         : List[str]            full token sequence
├── final_stacks    : Dict[int, int]       chip counts BEFORE winnings applied
└── pot_total       : int
```

---

## Betting Round Logic

```
_betting_round(street):
  1. Preflop: do NOT reset player.bet (blinds already posted and tracked)
     Postflop: reset player.bet = 0 for all; reset current_bet = 0

  2. Determine action order:
     Preflop:  UTG = dealer+3 (or SB in heads-up)
     Postflop: left of dealer (dealer+1)

  3. Loop until all active players have acted AND bets are matched:
     a. Skip players who: folded | all-in | already acted with matched bet
     b. Call agent(GameState) → Action
     c. Validate action (fix illegal moves silently)
     d. Apply action (update player.bet, chips)
     e. If RAISE: reset acted set (everyone must re-act)
     f. If only 1 player in hand: collect bets → early return

  4. pot.add_from_players(players)   ← single chip collection
```

### Action Validation

Illegal actions are silently corrected:

| Illegal | Fixed To |
|---------|----------|
| `CHECK` when there's a bet | `CALL` |
| `CALL` with nothing to call | `CHECK` |
| `RAISE` below min-raise | `RAISE` adjusted to min_raise |
| `RAISE` more than stack | `ALL_IN` |
| Any action with 0 chips | `ALL_IN` |

---

## History Token Format

Each hand produces a list of string tokens stored in `result.history`:

```
["<HAND:5>", "<ROUND:preflop>",
 "<P0:SB:10>", "<P1:BB:20>",
 "<P2:RAISE:60>", "<P0:FOLD>", "<P1:CALL>",
 "<ROUND:flop>", "<BOARD:Qh,Jc,2s>",
 "<P1:CHECK>", "<P2:BET:80>", "<P1:CALL>",
 ...
 "<WINNER:P2:Flush:340>"]
```

This sequence is the primary input to the tokenizer and transformer model.

---

## Multi-Hand Sessions

```python
game = Game(players, agents, small_blind=10, big_blind=20)

for hand_num in range(1000):
    # Rebuy players who busted
    for p in players:
        if p.chips == 0:
            p.chips = starting_chips

    result = game.play_hand()
    game.rotate_dealer()     # advances dealer_pos, increments hand_number
```

`rotate_dealer()` advances the button one seat clockwise and increments `hand_number` for logging.

---

## Known Design Decisions

| Decision | Reason |
|----------|--------|
| Preflop does NOT call `reset_for_street()` | Blinds are posted into `player.bet` before the round; resetting would force players to call the full blind a second time |
| `pot.add()` only called via `add_from_players()` | Prevents double-counting that occurred when both `_process_action()` and `add_from_players()` added chips |
| `deck.reset()` before each hand | Without this, only the ~40 remaining cards from the previous hand were reshuffled, causing the deck to run out during community card dealing |
| Early return collects bets first | When all fold mid-round, `add_from_players()` is called before `return` to ensure the pot contains the committed chips |
