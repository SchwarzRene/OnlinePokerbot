# `engine/game.py` — Texas Hold'em Game Engine

Orchestrates a complete hand of No-Limit Texas Hold'em from shuffle to showdown.
All training, evaluation, and simulation code calls `game.play_hand()`.

---

## Hand Flow

```
  play_hand()
  │
  ├── _reset()
  │     ├── deck.reset()           restore all 52 cards
  │     ├── deck.shuffle()         Fisher-Yates randomise
  │     ├── pot.reset()            clear chips and side pots
  │     └── player.reset_for_hand() clear hole cards, bets, fold flags
  │
  ├── _post_blinds()
  │     ├── SB player posts 10 chips  (stored in player.bet)
  │     └── BB player posts 20 chips  (stored in player.bet)
  │     Blinds are posted into player.bet — NOT collected into pot yet.
  │     The pot collects all bets at the END of each betting round.
  │
  ├── _deal_hole_cards()
  │     └── deck.deal(2) per active player  (dealt in seat order)
  │
  ├── _betting_round("preflop")
  │     Action starts UTG (seat left of BB), or SB in heads-up.
  │     Bets are NOT reset — blinds count as opening bets.
  │     └── pot.add_from_players() at the end
  │
  ├── _deal_community(3, "flop")
  │     └── deck.burn(), deck.deal(3)
  │
  ├── _betting_round("flop")
  │     Bets reset to 0. Action starts left of dealer.
  │     └── pot.add_from_players()
  │
  ├── _deal_community(1, "turn")
  ├── _betting_round("turn")
  │
  ├── _deal_community(1, "river")
  ├── _betting_round("river")
  │
  └── _showdown()
        ├── 1 player left → entire pot awarded uncontested
        └── 2+ players   → HandEvaluator ranks all hands → pot.award()
```

---

## Betting Round Logic

```
  _betting_round(street):

  1. If postflop: reset player.bet = 0 for all; current_bet = 0
     If preflop:  skip reset — blinds are already in player.bet

  2. Determine first actor:
     Preflop:  UTG = dealer_pos + 3  (mod n_players)
               Heads-up: SB acts first preflop (special rule)
     Postflop: first active player left of dealer

  3. Action loop:

     acted = set()
     while True:
         skip if: player.folded | player.all_in | already_acted_with_matched_bet

         state = build_GameState(player)
         action = agent(state)               ← agent provides Action
         action = _validate(action, state)   ← fix illegal moves
         _process_action(player, action)     ← update player state

         if action is RAISE:
             acted.clear()                  ← everyone must re-act

         acted.add(player.id)

         if only 1 player in hand:
             break                          ← hand ends early

         if all active non-all-in players have acted with matched bets:
             break

  4. pot.add_from_players(players)    ← collect all player.bet into pot
```

---

## Action Validation (Silent Correction)

Illegal actions are silently fixed rather than raising errors.
This ensures agents always produce valid game states.

```
  Illegal action              │  Fixed to
  ───────────────────────────────────────────────────────
  CHECK when there's a bet    │  CALL
  CALL  with nothing to call  │  CHECK
  RAISE below min_raise       │  RAISE to min_raise
  RAISE above player chips    │  ALL_IN
  Any action with 0 chips     │  ALL_IN
  Negative raise amount       │  FOLD
```

---

## GameState — Agent Input

Every time an agent must act, it receives a `GameState` snapshot:

```
  GameState
  ├── street           : str              "preflop" | "flop" | "turn" | "river"
  ├── community_cards  : List[Card]       face-up board (0 on preflop, 3/4/5 later)
  ├── pot              : int              total chips in pot
  ├── current_bet      : int             highest bet this street
  ├── call_amount      : int             extra chips needed to call
  │                                      = current_bet - you.bet
  ├── min_raise        : int             minimum legal raise size
  ├── max_raise        : int             acting player's remaining chips
  ├── players          : List[PlayerView] public info on all seats
  ├── you              : PlayerView       your full info (includes hole_cards)
  ├── dealer_pos       : int
  ├── history          : List[str]        token sequence of all actions so far
  └── hand_number      : int
```

`PlayerView` for opponents has `hole_cards = None` (hidden information).

---

## HandResult — Return Value

```
  HandResult
  ├── hand_number     : int
  ├── winners         : List[int]            player_ids who won chips
  ├── winnings        : Dict[int, int]       player_id → chips won
  ├── showdown_hands  : Dict[int, HandRank]  best 5-card hand per player
  ├── history         : List[str]            full token list for this hand
  ├── final_stacks    : Dict[int, int]       chip counts AFTER winnings applied
  └── pot_total       : int                  total chips that were in the pot
```

---

## History Token Format

Every hand produces a `List[str]` of action tokens stored in `result.history`.
This is the primary input to the tokenizer.

```
  ["<HAND:5>",
   "<ROUND:preflop>",
   "<P0:SB:10>",        ← SB posted 10
   "<P1:BB:20>",        ← BB posted 20
   "<P2:RAISE:60>",     ← UTG raises to 60
   "<P0:FOLD>",
   "<P1:CALL>",
   "<ROUND:flop>",
   "<BOARD:Qh,Jc,2s>",
   "<P1:CHECK>",
   "<P2:BET:80>",
   "<P1:CALL>",
   "<ROUND:turn>",
   "<BOARD:Qh,Jc,2s,Kd>",
   "<P1:CHECK>",
   "<P2:CHECK>",
   "<ROUND:river>",
   "<BOARD:Qh,Jc,2s,Kd,7h>",
   "<P1:RAISE:120>",
   "<P2:FOLD>",
   "<WINNER:P1:uncontested:200>"]
```

---

## Multi-Hand Sessions

```python
game = Game(players, agents, small_blind=10, big_blind=20)

for hand_num in range(1000):
    # Rebuy busted players
    for p in players:
        if p.chips < big_blind * 5:
            p.chips = starting_chips

    result = game.play_hand()
    game.rotate_dealer()   # advance button, increment hand_number
```

---

## Known Design Decisions

| Decision | Reason |
|----------|--------|
| Preflop does NOT call `reset_for_street()` | Blinds are posted into `player.bet` before the round; resetting would force players to re-pay the blind |
| `pot.add()` only called via `add_from_players()` | Single collection point prevents double-counting |
| `deck.reset()` before each hand | Without this, only leftover cards from the previous hand were reshuffled, causing deck exhaustion |
| Early fold returns collect bets first | `add_from_players()` called before `return` so pot has the committed chips |
| Silent action correction | Agents need not handle every edge case; the engine never crashes on bad agent output |
