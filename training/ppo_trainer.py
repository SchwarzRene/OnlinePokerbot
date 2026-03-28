"""
ppo_trainer.py — Proximal Policy Optimization Trainer
=======================================================

Implements PPO (Schulman et al., 2017) to train the PokerTransformer
via self-play reinforcement learning.

Algorithm Overview:
-------------------

    PPO is an on-policy actor-critic algorithm.
    The poker transformer serves as both actor (policy) and critic (value).

    Training Loop:
    ──────────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        1. COLLECT ROLLOUTS
           ─ Play N hands using current policy
           ─ Record (token_ids, action_idx, log_prob, value, reward, done)
             per decision step
           ─ Compute discounted returns and GAE advantages

        2. OPTIMIZE  (K mini-batch epochs over collected data)
           ─ For each mini-batch:
               π_new(a|s)  ← model forward
               ratio       = exp(log π_new - log π_old)
               clip_ratio  = clip(ratio, 1-ε, 1+ε)
               policy_loss = -mean(min(ratio × A, clip_ratio × A))
               value_loss  = MSE(V(s), returns)
               entropy_bonus = -mean(Σ π log π)   (encourages exploration)
               loss = policy_loss + c1×value_loss - c2×entropy_bonus
               optimizer.step()
           ─ Early stop if KL divergence > target_kl
    ──────────────────────────────────────────────────────────────

Key Hyperparameters:
--------------------
    clip_eps     = 0.2      PPO clip ratio
    gamma        = 0.99     discount factor
    gae_lambda   = 0.95     GAE smoothing
    c1           = 0.5      value loss coefficient
    c2           = 0.01     entropy bonus coefficient
    lr           = 3e-4     Adam learning rate
    n_epochs     = 4        optimization passes per rollout batch
    batch_size   = 64       mini-batch size

GAE (Generalized Advantage Estimation):
----------------------------------------
    Advantage A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    δ_t = r_t + γV(s_{t+1}) - V(s_t)

    GAE trades off bias (λ→0 = TD) vs. variance (λ→1 = MC).

Reward Design:
--------------
    Per-step reward = 0.0    (no intermediate reward)
    Terminal reward = (chips_won - chips_invested) / big_blind
                      normalized to BB units to keep reward scale stable

    Optional: small negative reward for folding equity (encourages
    defending in +EV spots). Controlled by fold_penalty coefficient.

Self-Play Setup:
----------------
    The trained agent plays against N-1 copies of itself (frozen opponent).
    Every `opponent_update_interval` hands, opponent weights are synced
    from the training agent.

    This prevents exploiting a static opponent and produces more robust
    strategies similar to AlphaGo's self-play scheme.
"""

from __future__ import annotations
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from engine.cards import Card
from engine.player import Player, ActionType
from engine.game import Game, HandResult
from engine.pot import Pot
from model.transformer import PokerTransformer, TransformerConfig
from model.tokenizer import PokerTokenizer
from utils.agents import RLAgent, RandomAgent, CallAgent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """All training hyperparameters."""
    # Environment
    n_players: int = 6
    starting_chips: int = 1000
    small_blind: int = 10
    big_blind: int = 20
    hands_per_rollout: int = 200    # hands collected before each update
    max_steps_per_hand: int = 50    # safety cap

    # PPO
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    c1: float = 0.5                 # value loss weight
    c2: float = 0.01                # entropy bonus weight
    target_kl: float = 0.02        # early stop KL threshold
    n_opt_epochs: int = 4
    batch_size: int = 64

    # Optimizer
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    # Self-play
    n_opponents: int = 5            # frozen opponent copies
    opponent_update_interval: int = 50  # hands between opponent weight sync

    # Reward
    fold_penalty: float = 0.0       # small negative for unnecessary folds
    reward_scale: float = 1.0       # multiply all rewards by this

    # Training loop
    total_epochs: int = 500
    save_interval: int = 50
    eval_interval: int = 25
    eval_hands: int = 100

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


# ---------------------------------------------------------------------------
# Experience Buffer
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """One decision step."""
    token_ids: torch.Tensor     # (T,)
    action_idx: int
    log_prob: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    """Stores experiences from one rollout, computes advantages."""

    def __init__(self):
        self.experiences: List[Experience] = []

    def add(self, exp: Experience):
        self.experiences.append(exp)

    def clear(self):
        self.experiences = []

    def __len__(self):
        return len(self.experiences)

    def compute_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        last_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and discounted returns.

        Returns
        -------
        advantages : (N,)
        returns    : (N,)   (advantages + values, used as critic target)
        """
        n = len(self.experiences)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        values = np.array([e.value for e in self.experiences] + [last_value], dtype=np.float32)
        rewards = np.array([e.reward for e in self.experiences], dtype=np.float32)
        dones = np.array([e.done for e in self.experiences], dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return torch.tensor(advantages), torch.tensor(returns)

    def to_tensors(
        self, advantages: torch.Tensor, returns: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Pack buffer into tensor dict for mini-batch training."""
        token_ids = torch.stack([e.token_ids for e in self.experiences])
        actions = torch.tensor([e.action_idx for e in self.experiences], dtype=torch.long)
        old_log_probs = torch.tensor([e.log_prob for e in self.experiences])
        old_values = torch.tensor([e.value for e in self.experiences])

        return {
            "token_ids": token_ids,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "advantages": advantages,
            "returns": returns,
        }


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Trains the PokerTransformer via PPO + self-play.

    Usage
    -----
        cfg = PPOConfig()
        model_cfg = TransformerConfig(vocab_size=..., num_actions=...)
        trainer = PPOTrainer(cfg, model_cfg)
        trainer.train()
    """

    def __init__(self, cfg: PPOConfig, model_cfg: TransformerConfig):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPOTrainer] Device: {self.device}")

        # Tokenizer
        self.tokenizer = PokerTokenizer(max_len=model_cfg.max_len)
        model_cfg.vocab_size = self.tokenizer.vocab_size
        model_cfg.num_actions = self.tokenizer.num_actions

        # Models
        self.model = PokerTransformer(model_cfg).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.total_epochs
        )

        # RL Agent (training)
        self.rl_agent = RLAgent(
            self.model, self.tokenizer, player_id=0,
            device=str(self.device), explore=True
        )

        # Opponent agents (frozen copies + random)
        self.opponents: List = self._build_opponents()

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Logging
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        # Hand counter for opponent sync
        self._hand_count = 0

    # ── Build Environment ──────────────────────────────────────────────────

    def _build_opponents(self):
        """Create frozen RL agent copies + random fallback."""
        opponents = []
        frozen_model = PokerTransformer(self.model_cfg).to(self.device)
        frozen_model.load_state_dict(self.model.state_dict())
        for i in range(1, self.cfg.n_players):
            agent = RLAgent(
                frozen_model, self.tokenizer, player_id=i,
                device=str(self.device), explore=False
            )
            opponents.append(agent)
        return opponents

    def _build_game(self) -> Tuple[Game, List[Player]]:
        """Create a fresh game with randomly re-stacked players."""
        players = [
            Player(id=i, name=f"P{i}", chips=self.cfg.starting_chips)
            for i in range(self.cfg.n_players)
        ]
        agents = {0: self.rl_agent}
        for i, opp in enumerate(self.opponents, start=1):
            opp.player_id = i
            agents[i] = opp

        game = Game(
            players=players,
            agents=agents,
            small_blind=self.cfg.small_blind,
            big_blind=self.cfg.big_blind,
        )
        return game, players

    # ── Rollout Collection ─────────────────────────────────────────────────

    def collect_rollouts(self) -> Dict[str, float]:
        """
        Play `hands_per_rollout` hands and fill self.buffer.

        Returns dict of rollout statistics.
        """
        self.buffer.clear()
        self.model.train()

        total_reward = 0.0
        total_hands = 0
        game, players = self._build_game()

        for _ in range(self.cfg.hands_per_rollout):
            # Re-stack if anyone is broke
            for p in players:
                if p.chips < self.cfg.big_blind * 5:
                    p.chips = self.cfg.starting_chips

            chips_before = players[0].chips
            result = game.play_hand()
            chips_after = players[0].chips

            # Terminal reward for training player (player 0)
            reward = (chips_after - chips_before) / self.cfg.big_blind
            reward *= self.cfg.reward_scale
            total_reward += reward
            total_hands += 1

            # Record experience for last RL agent action this hand
            if self.rl_agent.last_token_ids is not None:
                exp = Experience(
                    token_ids=self.rl_agent.last_token_ids.cpu(),
                    action_idx=self.rl_agent.last_action_idx,
                    log_prob=self.rl_agent.last_log_prob,
                    value=self.rl_agent.last_value,
                    reward=float(reward),
                    done=True,
                )
                self.buffer.add(exp)

            self._hand_count += 1

            # Sync opponents periodically
            if self._hand_count % self.cfg.opponent_update_interval == 0:
                self._sync_opponents()

            game.rotate_dealer()

        stats = {
            "mean_reward": total_reward / max(total_hands, 1),
            "total_reward": total_reward,
            "buffer_size": len(self.buffer),
        }
        return stats

    # ── PPO Update ─────────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        Run PPO optimization on the collected buffer.

        Returns dict of training metrics.
        """
        if len(self.buffer) == 0:
            return {}

        adv, ret = self.buffer.compute_advantages(self.cfg.gamma, self.cfg.gae_lambda)
        data = self.buffer.to_tensors(adv, ret)

        N = len(self.buffer)
        indices = np.arange(N)

        policy_losses, value_losses, entropies, kls = [], [], [], []

        for _ in range(self.cfg.n_opt_epochs):
            np.random.shuffle(indices)

            for start in range(0, N, self.cfg.batch_size):
                batch_idx = indices[start:start + self.cfg.batch_size]
                batch = {k: v[batch_idx].to(self.device) for k, v in data.items()}

                token_ids = batch["token_ids"]                     # (B, T)
                actions = batch["actions"]                         # (B,)
                old_log_probs = batch["old_log_probs"]            # (B,)
                advantages = batch["advantages"]                   # (B,)
                returns = batch["returns"]                         # (B,)

                logits, values = self.model(token_ids)
                dist = torch.distributions.Categorical(logits=logits)

                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # PPO ratio and clip
                ratio = (new_log_probs - old_log_probs).exp()
                clip_ratio = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()

                # Value loss (clipped)
                values = values.squeeze(-1)
                value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = (policy_loss
                        + self.cfg.c1 * value_loss
                        - self.cfg.c2 * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                # KL divergence estimate
                kl = (old_log_probs - new_log_probs).mean().item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                kls.append(kl)

            # Early stop on KL
            if np.mean(kls) > self.cfg.target_kl:
                break

        self.scheduler.step()

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "kl": float(np.mean(kls)),
        }

    # ── Training Loop ──────────────────────────────────────────────────────

    def train(self):
        """Main training loop."""
        print(f"[PPO] Starting training: {self.cfg.total_epochs} epochs")
        print(f"[PPO] Model params: {self.model.count_parameters():,}")

        for epoch in range(1, self.cfg.total_epochs + 1):
            t0 = time.time()

            rollout_stats = self.collect_rollouts()
            update_stats = self.update()

            elapsed = time.time() - t0

            # Log
            all_stats = {**rollout_stats, **update_stats, "epoch": epoch}
            for k, v in all_stats.items():
                self.metrics[k].append(v)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"reward={rollout_stats.get('mean_reward', 0):.3f} | "
                    f"ploss={update_stats.get('policy_loss', 0):.4f} | "
                    f"vloss={update_stats.get('value_loss', 0):.4f} | "
                    f"ent={update_stats.get('entropy', 0):.4f} | "
                    f"kl={update_stats.get('kl', 0):.4f} | "
                    f"{elapsed:.1f}s"
                )

            if epoch % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch)

        print("[PPO] Training complete.")

    # ── Utilities ──────────────────────────────────────────────────────────

    def _sync_opponents(self):
        """Copy current model weights to all frozen opponent agents."""
        state_dict = self.model.state_dict()
        for opp in self.opponents:
            opp.model.load_state_dict(state_dict)

    def save_checkpoint(self, epoch: int):
        path = os.path.join(self.cfg.checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": dict(self.metrics),
        }, path)
        print(f"[PPO] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.metrics.update(ckpt.get("metrics", {}))
        print(f"[PPO] Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt["epoch"]
