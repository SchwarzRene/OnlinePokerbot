# `model/tokenizer.py` — Poker History Tokenizer

Converts raw game history strings (produced by the game engine) into **integer
sequences** for the transformer, and defines the **action output vocabulary**
the model's decision head predicts over.

---

## Pipeline Position

```
  Game.history  (List[str])
       │  e.g. ["<HAND:5>", "<ROUND:preflop>", "<P0:SB:10>", "<P2:RAISE:60>"]
       │
       ▼  PokerTokenizer.encode(history, hole_cards, num_players)
       │
       │  outputs [1, 4, 847, 23, 44, 112, 0, 0, ...]  (padded to max_len=512)
       │           ^  ^    ^
       │           │  │    └── history tokens
       │           │  └─────── <NUM_PLAYERS:N> context token
       │           └────────── <BOS>
       │
       ▼  PokerTransformer.forward(token_ids)
       │
       ▼  action_logits  →  Categorical(softmax)  →  action_idx
       │
       ▼  PokerTokenizer.decode_action(action_idx)
       │  e.g. "<ACT:RAISE:AMT3>"
       │
       ▼  action_token_to_action_type_and_amount()
       │
       ▼  Action(ActionType.RAISE, amount=74)
```

---

## Vocabulary Structure

Total vocabulary size: **1,413 tokens**

```
  Group             Count    Example tokens
  ────────────────  ───────  ──────────────────────────────────────
  Special                4   <PAD>  <BOS>  <EOS>  <UNK>
  Table size             8   <NUM_PLAYERS:2> … <NUM_PLAYERS:9>
  Hand numbers        1000   <HAND:0> … <HAND:999>
  Streets                4   <ROUND:preflop/flop/turn/river>
  Board separator        1   <BOARD>
  Hole separator         1   <HOLE>
  Card tokens           52   <CARD:2c> … <CARD:As>
  Player actions       279   <P0:FOLD>, <P3:RAISE:AMT4>, …
    (9 seats × [3 amount-free + 4×8 amount-bucketed + 1 WINNER])
  Action targets        19   <ACT:FOLD>, <ACT:RAISE:AMT3>, …
  ────────────────  ───────  ──────────────────────────────────────
  TOTAL              1,413
```

**Special token IDs:**

| Token | ID |
|-------|----|
| `<PAD>` | 0 |
| `<BOS>` | 1 |
| `<EOS>` | 2 |
| `<UNK>` | 3 |
| `<NUM_PLAYERS:2>` | 4 |
| … | … |
| `<NUM_PLAYERS:9>` | 11 |

---

## Table-Size Context Token

Every encoded sequence begins with `<BOS>` followed immediately by a
`<NUM_PLAYERS:N>` token (N = 2..9). This tells the model how many players
are seated at the current table before it sees a single action:

```
  Sequence start:
  ┌─────┬──────────────────┬──────────┬──────────────┬─────────┐
  │ BOS │ <NUM_PLAYERS:6>  │  <HOLE>  │  <CARD:Ah>   │  ...    │
  └─────┴──────────────────┴──────────┴──────────────┴─────────┘
    [1]        [6]            [HOLE_ID]  [CARD_ID]

  Why this matters:
    2 players → play very wide, positional edge huge
    6 players → standard 6-max strategy
    9 players → only play tight premium hands
```

---

## Amount Bucketing

Chip amounts in raise/blind tokens are mapped to one of 8 buckets.
This caps vocabulary growth while preserving bet-sizing information.

```
  Bucket   Range         Midpoint   Real-world sizing (at 50 BB stack)
  ──────   ──────────    ────────   ────────────────────────────────────
    0       1 –    9        5       Tiny (< 0.5 BB)
    1      10 –   19       14       Min-bet / small blind  (0.5–1 BB)
    2      20 –   49       34       Open raise             (1–2.5 BB)
    3      50 –   99       74       3-bet / pot bet        (2.5–5 BB)
    4     100 –  199      149       Large 3-bet / c-bet    (5–10 BB)
    5     200 –  499      349       Re-raise / big bet     (10–25 BB)
    6     500 –  999      749       Near-shove             (25–50 BB)
    7    1000+           1000       Full-stack jam         (50 BB+)
  ──────   ──────────    ────────   ────────────────────────────────────

  Encoding:  "<P0:RAISE:120>"  →  bucket_amount(120) = 4  →  "<P0:RAISE:AMT4>"
  Decoding:  "<ACT:RAISE:AMT4>"  →  bucket_midpoint(4) = 149 chips
             then clamp to player's remaining stack
```

---

## Full Encoded Sequence Layout

```
  Position   Token                    Notes
  ─────────  ───────────────────────  ────────────────────────────────
  0          <BOS>                    always first
  1          <NUM_PLAYERS:N>          table size (2–9), always second
  2          <HOLE>                   hole card separator
  3          <CARD:Ah>                your first private card
  4          <CARD:Kd>                your second private card
  5          <HAND:7>                 hand number (mod 1000)
  6          <ROUND:preflop>          street marker
  7          <P0:SB:AMT1>             small blind posted (10 chips → bucket 1)
  8          <P1:BB:AMT1>             big blind posted   (20 chips → bucket 1)
  9          <P2:RAISE:AMT3>          UTG raises
  10         <P3:FOLD>                fold
  11         <P0:CALL>                call
  12         <P1:CALL>                call
  13         <ROUND:flop>             street marker
  14         <BOARD>                  board card separator
  15         <CARD:Qh>                flop card 1
  16         <CARD:Jc>                flop card 2
  17         <CARD:2s>                flop card 3
  18         <P1:CHECK>               opponent checks
  19         <P2:RAISE:AMT4>          opponent raises
  20         <EOS>                    end of current history
  21–511     <PAD>                    zero-padding to max_len=512
  ─────────  ───────────────────────  ────────────────────────────────

  Left-truncation: if history exceeds 512 tokens, oldest tokens are
  dropped. BOS and NUM_PLAYERS are always re-inserted at positions 0–1.
```

---

## Action Output Vocabulary (19 actions)

```
  Index   Token                   Decoded amount    Category
  ─────   ─────────────────────   ──────────────    ────────────
    0     <ACT:FOLD>              0                 fold
    1     <ACT:CHECK>             0                 check
    2     <ACT:CALL>              0 (auto)          call
    3     <ACT:RAISE:AMT0>        ~5 chips          raise
    4     <ACT:RAISE:AMT1>        ~14 chips         raise
    5     <ACT:RAISE:AMT2>        ~34 chips         raise
    6     <ACT:RAISE:AMT3>        ~74 chips         raise
    7     <ACT:RAISE:AMT4>        ~149 chips        raise
    8     <ACT:RAISE:AMT5>        ~349 chips        raise
    9     <ACT:RAISE:AMT6>        ~749 chips        raise
   10     <ACT:RAISE:AMT7>        ~1000+ chips      raise
   11     <ACT:ALL_IN:AMT0>       player stack      all-in
   12     <ACT:ALL_IN:AMT1>       player stack      all-in
   13     <ACT:ALL_IN:AMT2>       player stack      all-in
   14     <ACT:ALL_IN:AMT3>       player stack      all-in
   15     <ACT:ALL_IN:AMT4>       player stack      all-in
   16     <ACT:ALL_IN:AMT5>       player stack      all-in
   17     <ACT:ALL_IN:AMT6>       player stack      all-in
   18     <ACT:ALL_IN:AMT7>       player stack      all-in
  ─────   ─────────────────────   ──────────────    ────────────

  ALL_IN indices 11–18 have different bucket labels but identical
  real-world effect: commit all remaining chips. The bucket label
  carries implied stack-depth information the model can learn from.
```

---

## Encoding Algorithm

```
  encode(history, hole_cards, num_players):

  ids = [BOS_ID]

  if num_players:
      ids += [tok2id["<NUM_PLAYERS:{n}>"]]   # table context

  if hole_cards:
      ids += [HOLE_ID]
      for card in hole_cards:
          ids += [tok2id["<CARD:{card}>"]]

  for raw_token in history:
      ids += _encode_token(raw_token)         # bucketing happens here

  ids += [EOS_ID]

  if len(ids) > max_len:
      ids = [BOS_ID, NUM_PLAYERS_ID] + ids[-(max_len - 2):]   # left-truncate

  ids += [PAD_ID] * (max_len - len(ids))
  return ids   # always length max_len
```

---

## Usage Example

```python
from model.tokenizer import PokerTokenizer

tok = PokerTokenizer(max_len=512)
print(f"Vocab size:  {tok.vocab_size}")   # 1413
print(f"Num actions: {tok.num_actions}")  # 19

# Encode a game history with table context
history = ["<HAND:3>", "<ROUND:preflop>", "<P0:SB:10>", "<P1:BB:20>"]
ids = tok.encode(history, hole_cards=["Ah", "Kd"], num_players=6)
# ids: list[int] of length 512

# Decode model output
action_token = tok.decode_action(action_idx=7)         # "<ACT:RAISE:AMT4>"
action_type, amount = tok.action_token_to_action_type_and_amount(
    action_token, call_amount=20, player_chips=480
)
# action_type = "RAISE",  amount = 149
```
