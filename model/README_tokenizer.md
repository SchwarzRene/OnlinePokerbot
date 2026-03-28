# `model/tokenizer.py` — Poker History Tokenizer

## Purpose

Converts raw game history token strings (produced by the game engine) into **integer sequences** suitable for input to the transformer model. Also defines the **action output vocabulary** used by the model's decision head.

---

## Pipeline Position

```
Game.history (List[str])
        │
        │  e.g. ["<HAND:5>", "<ROUND:preflop>", "<P0:SB:10>", "<P2:RAISE:60>"]
        ▼
PokerTokenizer.encode(history, hole_cards)
        │
        │  outputs [1, 847, 23, 44, 112, 0, 0, 0, ...]  (padded to max_len)
        ▼
PokerTransformer.forward(token_ids)
        │
        ▼
action_logits  →  softmax  →  action_idx
        │
PokerTokenizer.decode_action(action_idx)
        │
        │  e.g. "<ACT:RAISE:AMT3>"
        ▼
action_token_to_action_type_and_amount()
        │
        ▼
Action(ActionType.RAISE, amount=75)
```

---

## Vocabulary Structure

Total vocabulary size: **~1,400 tokens**

| Group | Tokens | Example |
|-------|--------|---------|
| Special | 4 | `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` |
| Hand numbers | 1000 | `<HAND:0>` … `<HAND:999>` |
| Streets | 4 | `<ROUND:preflop>` … `<ROUND:river>` |
| Board separator | 1 | `<BOARD>` |
| Player actions | ~9 × 288 | `<P0:RAISE:AMT3>`, `<P2:FOLD>`, … |
| Card tokens | 53 | `<HOLE>`, `<CARD:Ah>`, `<CARD:2c>`, … |
| Action targets | 19 | `<ACT:FOLD>`, `<ACT:RAISE:AMT0>`, … |

---

## Amount Bucketing

Chip amounts in tokens like `<RAISE:120>` are **bucketed** into 8 ranges to keep the vocabulary size manageable while preserving bet-sizing information:

| Bucket | Range | Midpoint (for decoding) |
|--------|-------|------------------------|
| 0 | 1 – 9 | 5 |
| 1 | 10 – 19 | 14 |
| 2 | 20 – 49 | 34 |
| 3 | 50 – 99 | 74 |
| 4 | 100 – 199 | 149 |
| 5 | 200 – 499 | 349 |
| 6 | 500 – 999 | 749 |
| 7 | 1000+ | 1000 |

```
Raw token:  "<P0:RAISE:120>"
Bucketed:   "<P0:RAISE:AMT4>"   (bucket 4 = 100–199)
```

When the model outputs `<ACT:RAISE:AMT4>`, the decoded amount is `bucket_midpoint(4) = 149`, clipped to the player's actual stack.

---

## Encoding Details

```
encode(history, hole_cards=["Ah","Kd"])

Output sequence:
┌────────────────────────────────────────────────────────┐
│ <BOS>  <HOLE>  <CARD:Ah>  <CARD:Kd>                   │
│ <HAND:5>  <ROUND:preflop>                              │
│ <P0:SB:AMT1>  <P1:BB:AMT1>                            │
│ <P2:RAISE:AMT3>  <P0:FOLD>  <P1:CALL>                 │
│ <ROUND:flop>  <BOARD>  <CARD:Qh>  <CARD:Jc>  <CARD:2s>│
│ ...                                                    │
│ <EOS>  <PAD>  <PAD>  ...  <PAD>                       │
└────────────────────────────────────────────────────────┘
         ←────────────── max_len = 512 ─────────────────→
```

**Truncation:** If the sequence exceeds `max_len`, oldest tokens are dropped (left-truncation). The most recent context is preserved. `<BOS>` is always re-inserted at position 0.

---

## Action Output Vocabulary

The model's action head outputs a probability distribution over **19 actions**:

| Index | Token | Meaning |
|-------|-------|---------|
| 0 | `<ACT:FOLD>` | Fold |
| 1 | `<ACT:CHECK>` | Check |
| 2 | `<ACT:CALL>` | Call |
| 3–10 | `<ACT:RAISE:AMT0>` … `<ACT:RAISE:AMT7>` | Raise (8 size buckets) |
| 11–18 | `<ACT:ALL_IN:AMT0>` … `<ACT:ALL_IN:AMT7>` | All-in (8 size buckets) |

---

## Special Token IDs

| Token | ID |
|-------|----|
| `<PAD>` | 0 |
| `<BOS>` | 1 |
| `<EOS>` | 2 |
| `<UNK>` | 3 |

---

## Usage Example

```python
from model.tokenizer import PokerTokenizer

tok = PokerTokenizer(max_len=512)
print(f"Vocab size:  {tok.vocab_size}")   # ~1400
print(f"Num actions: {tok.num_actions}")  # 19

# Encode a game history
history = ["<HAND:3>", "<ROUND:preflop>", "<P0:SB:10>", "<P1:BB:20>"]
ids = tok.encode(history, hole_cards=["Ah", "Kd"])
# ids: list[int] of length 512, padded with 0s

# Decode a model output
action_token = tok.decode_action(action_idx=5)   # e.g. "<ACT:RAISE:AMT2>"
action_type, amount = tok.action_token_to_action_type_and_amount(
    action_token,
    call_amount=20,
    player_chips=480
)
# action_type = "RAISE", amount = 34  (midpoint of bucket 2)
```
