"""
tokenizer.py — Poker History Tokenizer
========================================

Converts raw game history token strings into integer sequences
suitable for input to the transformer model.

Token Vocabulary:
-----------------

    Special tokens:
        <PAD>       0   padding to fixed sequence length
        <BOS>       1   beginning of sequence
        <EOS>       2   end of sequence
        <UNK>       3   unknown token

    Structural tokens:
        <HAND:N>    hand number N
        <ROUND:X>   X ∈ {preflop, flop, turn, river}
        <BOARD:...> community card string

    Player action tokens:
        <P{id}:SB:N>         small blind posted
        <P{id}:BB:N>         big blind posted
        <P{id}:FOLD>         fold
        <P{id}:CHECK>        check
        <P{id}:CALL>         call
        <P{id}:RAISE:N>      raise to N
        <P{id}:ALL_IN:N>     all-in for N
        <WINNER:P{id}:...>   hand result

    Card tokens (hole cards + board):
        Each card gets its own token: "2c", "Th", "As", etc.
        52 card tokens + suits = 52 distinct tokens

    Amount tokens:
        Bet amounts are bucketed into ranges to limit vocab size:
        [1-9], [10-19], [20-49], [50-99], [100-199], [200-499],
        [500-999], [1000+]
        Represented as "AMT:bucket_id"

Token Sequence Format (example):
---------------------------------

    Input (what model sees):
    ─────────────────────────────────────────────────────
    <BOS> <HAND:5> <ROUND:preflop>
    <P0:SB:10> <P1:BB:20>
    <HOLE:Ah><HOLE:Kd>          ← your hole cards
    <P2:RAISE:60> <P0:FOLD> <P1:CALL>
    <ROUND:flop> <BOARD:Qh,Jc,2s>
    <P1:CHECK> <P2:BET:80>
    [... you must act ...]
    ─────────────────────────────────────────────────────

    Target (what model predicts):
        Action token index: FOLD | CHECK | CALL | RAISE:bucket | ALL_IN

Encoding Details:
-----------------
    Amounts in tokens like <RAISE:120> are bucketed:
        bucket 0:   1  –   9
        bucket 1:  10  –  19
        bucket 2:  20  –  49
        bucket 3:  50  –  99
        bucket 4: 100  – 199
        bucket 5: 200  – 499
        bucket 6: 500  – 999
        bucket 7: 1000+

    This keeps the vocabulary manageable (~300 tokens total)
    while preserving relative bet-sizing signal.
"""

from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Amount bucketing
# ---------------------------------------------------------------------------

AMT_BUCKETS = [(1, 9), (10, 19), (20, 49), (50, 99),
               (100, 199), (200, 499), (500, 999), (1000, 10_000_000)]

def bucket_amount(amount: int) -> int:
    """Return bucket index (0–7) for a chip amount."""
    for i, (lo, hi) in enumerate(AMT_BUCKETS):
        if lo <= amount <= hi:
            return i
    return len(AMT_BUCKETS) - 1

def bucket_midpoint(bucket: int) -> int:
    """Return representative value for a bucket (for display / decoding)."""
    lo, hi = AMT_BUCKETS[bucket]
    return (lo + hi) // 2 if hi < 10_000_000 else lo


# ---------------------------------------------------------------------------
# Vocabulary Builder
# ---------------------------------------------------------------------------

def _build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build the complete token ↔ id mapping."""
    tokens = [
        "<PAD>", "<BOS>", "<EOS>", "<UNK>",
    ]

    # Structural
    for i in range(1000):  # hand numbers 0-999
        tokens.append(f"<HAND:{i}>")
    for street in ("preflop", "flop", "turn", "river"):
        tokens.append(f"<ROUND:{street}>")
    tokens.append("<BOARD>")   # board prefix (cards follow as card tokens)

    # Player ids (up to 9 seats)
    for pid in range(9):
        # Amount-independent actions — added once per player, not per bucket
        tokens.append(f"<P{pid}:FOLD>")
        tokens.append(f"<P{pid}:CHECK>")
        tokens.append(f"<P{pid}:CALL>")
        for amt_b in range(len(AMT_BUCKETS)):
            tokens.append(f"<P{pid}:SB:AMT{amt_b}>")
            tokens.append(f"<P{pid}:BB:AMT{amt_b}>")
            tokens.append(f"<P{pid}:RAISE:AMT{amt_b}>")
            tokens.append(f"<P{pid}:ALL_IN:AMT{amt_b}>")
        tokens.append(f"<P{pid}:WINNER>")

    # Card tokens  e.g. "2c", "Th", "Ah", "As"
    RANK_CHARS = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
    SUIT_CHARS = ["c","d","h","s"]
    tokens.append("<HOLE>")  # hole card separator
    for r in RANK_CHARS:
        for s in SUIT_CHARS:
            tokens.append(f"<CARD:{r}{s}>")

    # Action targets (output vocabulary for the model head)
    ACTION_TOKENS = [f"<ACT:FOLD>", "<ACT:CHECK>", "<ACT:CALL>"]
    for b in range(len(AMT_BUCKETS)):
        ACTION_TOKENS.append(f"<ACT:RAISE:AMT{b}>")
        ACTION_TOKENS.append(f"<ACT:ALL_IN:AMT{b}>")
    tokens.extend(ACTION_TOKENS)

    tok2id = {t: i for i, t in enumerate(tokens)}
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok


TOK2ID, ID2TOK = _build_vocab()
VOCAB_SIZE = len(TOK2ID)
PAD_ID = TOK2ID["<PAD>"]
BOS_ID = TOK2ID["<BOS>"]
EOS_ID = TOK2ID["<EOS>"]
UNK_ID = TOK2ID["<UNK>"]

# Action output head indices
ACTION_TOKENS_LIST = (
    ["<ACT:FOLD>", "<ACT:CHECK>", "<ACT:CALL>"]
    + [f"<ACT:RAISE:AMT{b}>" for b in range(len(AMT_BUCKETS))]
    + [f"<ACT:ALL_IN:AMT{b}>" for b in range(len(AMT_BUCKETS))]
)
NUM_ACTIONS = len(ACTION_TOKENS_LIST)
ACTION2IDX = {t: i for i, t in enumerate(ACTION_TOKENS_LIST)}
IDX2ACTION = {i: t for t, i in ACTION2IDX.items()}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class PokerTokenizer:
    """
    Converts game history strings → token id sequences and back.

    Usage
    -----
        tok = PokerTokenizer(max_len=512)
        ids = tok.encode(history_tokens, hole_cards=["Ah", "Kd"])
        # ids is a list[int] of length max_len, padded with PAD_ID

        action_idx = 5
        action_str = tok.decode_action(action_idx)   # "<ACT:RAISE:AMT2>"
    """

    def __init__(self, max_len: int = 512):
        self.max_len = max_len
        self.tok2id = TOK2ID
        self.id2tok = ID2TOK

    def encode(
        self,
        history: List[str],
        hole_cards: Optional[List[str]] = None,
        truncate_left: bool = True,
    ) -> List[int]:
        """
        Encode a game history into a padded integer sequence.

        Parameters
        ----------
        history      : raw token strings from game.history
        hole_cards   : e.g. ["Ah", "Kd"] — inserted after BOS if provided
        truncate_left: if too long, drop oldest tokens (keep most recent context)

        Returns
        -------
        List[int] of length self.max_len
        """
        ids = [BOS_ID]

        # Insert hole cards
        if hole_cards:
            ids.append(self.tok2id.get("<HOLE>", UNK_ID))
            for card_str in hole_cards:
                card_token = f"<CARD:{card_str}>"
                ids.append(self.tok2id.get(card_token, UNK_ID))

        # Encode history
        for raw_token in history:
            encoded = self._encode_token(raw_token)
            ids.extend(encoded)

        ids.append(EOS_ID)

        # Truncate / pad
        if len(ids) > self.max_len:
            if truncate_left:
                ids = [BOS_ID] + ids[-(self.max_len - 1):]
            else:
                ids = ids[:self.max_len]

        padding = [PAD_ID] * (self.max_len - len(ids))
        return ids + padding

    def _encode_token(self, raw: str) -> List[int]:
        """
        Map a raw game token to one or more vocab ids.

        Handles amount bucketing for RAISE / ALL_IN / blind tokens.
        """
        raw = raw.strip()

        # HAND token: <HAND:N>
        m = re.match(r"<HAND:(\d+)>", raw)
        if m:
            n = int(m.group(1)) % 1000
            return [self.tok2id.get(f"<HAND:{n}>", UNK_ID)]

        # ROUND token
        m = re.match(r"<ROUND:(\w+)>", raw)
        if m:
            return [self.tok2id.get(f"<ROUND:{m.group(1)}>", UNK_ID)]

        # BOARD token: <BOARD:Qh,Jc,2s>
        m = re.match(r"<BOARD:(.+)>", raw)
        if m:
            ids = [self.tok2id.get("<BOARD>", UNK_ID)]
            for card_str in m.group(1).split(","):
                card_token = f"<CARD:{card_str.strip()}>"
                ids.append(self.tok2id.get(card_token, UNK_ID))
            return ids

        # Player action tokens: <P{id}:ACTION> or <P{id}:ACTION:amount>
        m = re.match(r"<P(\d+):(\w+)(?::(\d+))?>", raw)
        if m:
            pid = int(m.group(1))
            action = m.group(2)
            amount = int(m.group(3)) if m.group(3) else None

            if action in ("FOLD", "CHECK", "CALL"):
                return [self.tok2id.get(f"<P{pid}:{action}>", UNK_ID)]
            elif action in ("RAISE", "ALL_IN", "SB", "BB"):
                amt_b = bucket_amount(amount) if amount else 0
                return [self.tok2id.get(f"<P{pid}:{action}:AMT{amt_b}>", UNK_ID)]

        # WINNER token
        m = re.match(r"<WINNER:P(\d+):.+>", raw)
        if m:
            pid = int(m.group(1))
            return [self.tok2id.get(f"<P{pid}:WINNER>", UNK_ID)]

        return [UNK_ID]

    def decode_action(self, action_idx: int, acting_player_chips: int = 500) -> str:
        """
        Convert an action index (model output) to a human-readable action string.

        Parameters
        ----------
        action_idx           : int  output of the model's action head
        acting_player_chips  : int  used to resolve ALL_IN amount

        Returns
        -------
        str like "<ACT:RAISE:AMT3>" or "<ACT:FOLD>"
        """
        if action_idx < 0 or action_idx >= NUM_ACTIONS:
            return "<ACT:FOLD>"
        return IDX2ACTION[action_idx]

    def action_token_to_action_type_and_amount(
        self, action_token: str, call_amount: int, player_chips: int
    ) -> Tuple[str, int]:
        """
        Convert model output token to (action_type_str, amount).

        Example:
            "<ACT:RAISE:AMT3>" → ("RAISE", 75)   (midpoint of bucket 3)
            "<ACT:FOLD>"       → ("FOLD",   0)
        """
        token = action_token.replace("<ACT:", "").rstrip(">")
        if ":" in token:
            action_type, amt_str = token.split(":", 1)
            bucket = int(amt_str.replace("AMT", ""))
            amount = min(bucket_midpoint(bucket), player_chips)
        else:
            action_type = token
            amount = 0
        return action_type, amount

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS
