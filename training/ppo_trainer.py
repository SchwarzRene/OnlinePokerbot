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
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in headless/training environments
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from engine.cards import Card
from engine.player import Player, ActionType
from engine.game import Game, HandResult
from engine.pot import Pot
from model.transformer import PokerTransformer, TransformerConfig
# model_class can be overridden to use PokerTransformerDemo or any compatible model
from model.tokenizer import PokerTokenizer
from utils.agents import RLAgent, RandomAgent, CallAgent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """All training hyperparameters."""
    # Environment
    min_players: int = 2            # minimum players per hand (heads-up)
    max_players: int = 9            # maximum players per hand (full ring)
    starting_chips: int = 2000      # 100 BB — standard online cash game buy-in
    small_blind: int = 10
    big_blind: int = 20
    hands_per_rollout: int = 200    # hands collected before each update
    max_steps_per_hand: int = 50    # safety cap

    # PPO
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    c1: float = 0.05               # value loss weight — kept small so it never drowns policy gradient
    c2: float = 0.01                # entropy bonus weight
    target_kl: float = 0.02        # early stop KL threshold
    n_opt_epochs: int = 4
    batch_size: int = 64

    # Optimizer
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    # Self-play
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


# Experience is defined in utils.agents to avoid circular imports.
# Import it here so the rest of this file works unchanged.
from utils.agents import Experience


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

    def __init__(self, cfg: PPOConfig, model_cfg: TransformerConfig, model_class=None):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self._model_class = model_class or PokerTransformer  # swap in demo model here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPOTrainer] Device: {self.device}")

        # Tokenizer
        self.tokenizer = PokerTokenizer(max_len=model_cfg.max_len)
        model_cfg.vocab_size = self.tokenizer.vocab_size
        model_cfg.num_actions = self.tokenizer.num_actions

        # Models
        self.model = self._model_class(model_cfg).to(self.device)
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
        """Create max_players-1 frozen RL agent copies (pool for variable table sizes)."""
        opponents = []
        frozen_model = self._model_class(self.model_cfg).to(self.device)
        frozen_model.load_state_dict(self.model.state_dict())
        for i in range(1, self.cfg.max_players):   # up to 8 opponents for a 9-seat table
            agent = RLAgent(
                frozen_model, self.tokenizer, player_id=i,
                device=str(self.device), explore=False
            )
            opponents.append(agent)
        return opponents

    def _build_game(self) -> Tuple[Game, List[Player]]:
        """Create a fresh game with a random number of players (min_players..max_players)."""
        n_players = random.randint(self.cfg.min_players, self.cfg.max_players)
        players = [
            Player(id=i, name=f"P{i}", chips=self.cfg.starting_chips)
            for i in range(n_players)
        ]
        agents = {0: self.rl_agent}
        for i in range(1, n_players):
            opp = self.opponents[i - 1]   # reuse from pre-built pool
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
                if p.chips < self.cfg.big_blind * 40:  # rebuy when below 40BB (standard short-stack threshold)
                    p.chips = self.cfg.starting_chips

            chips_before = players[0].chips
            result = game.play_hand()
            chips_after = players[0].chips

            # Terminal reward for training player (player 0).
            # Normalise to [-1, +1] by dividing by the full starting stack in BB.
            # A win of +100 BB (entire stack) → +1.0; losing the BB → -0.01.
            # This keeps the value head's targets in a small range and prevents
            # MSE from exploding to tens of thousands (the vloss bug).
            stack_in_bb = self.cfg.starting_chips / self.cfg.big_blind  # = 100
            reward = (chips_after - chips_before) / (self.cfg.big_blind * stack_in_bb)
            reward = float(np.clip(reward, -1.0, 1.0))  # hard clip for safety
            reward *= self.cfg.reward_scale
            total_reward += reward
            total_hands += 1

            # Record experience for last RL agent action this hand
            # Flush every decision the agent made this hand (not just the last).
            # The terminal reward is assigned to the final action; earlier actions
            # get reward=0 (no intermediate signal) so GAE propagates credit back.
            hand_exps = self.rl_agent.flush_hand_buffer()
            for i, exp in enumerate(hand_exps):
                is_last = (i == len(hand_exps) - 1)
                exp.reward = float(reward) if is_last else 0.0
                exp.done   = is_last
                self.buffer.add(exp)

            self._hand_count += 1

            # Sync opponents periodically
            if self._hand_count % self.cfg.opponent_update_interval == 0:
                self._sync_opponents()

            # Rebuild game with a fresh random player count each hand
            game, players = self._build_game()

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
                # Clipped value loss — prevents large critic updates from
                # destabilising the policy (mirrors the policy clipping idea).
                v_clipped = batch["old_values"] + (values - batch["old_values"]).clamp(
                    -self.cfg.clip_eps, self.cfg.clip_eps
                )
                value_loss = torch.max(
                    F.mse_loss(values, returns),
                    F.mse_loss(v_clipped, returns),
                ).mean()

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
        print(f"[PPO] Checkpoint dir: {self.cfg.checkpoint_dir}")
        print(f"[PPO] Save interval: every {self.cfg.save_interval} epochs (+ final)")
        print("-" * 80)

        training_start = time.time()

        for epoch in range(1, self.cfg.total_epochs + 1):
            t0 = time.time()

            rollout_stats = self.collect_rollouts()
            update_stats = self.update()

            elapsed = time.time() - t0
            total_elapsed = time.time() - training_start

            # Log
            all_stats = {**rollout_stats, **update_stats, "epoch": epoch, "elapsed": elapsed}
            for k, v in all_stats.items():
                self.metrics[k].append(v)

            # Log every epoch (was every 10 — unhelpful for short runs)
            lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, "get_last_lr") else self.cfg.lr
            print(
                f"[Epoch {epoch:4d}/{self.cfg.total_epochs}] "
                f"reward={rollout_stats.get('mean_reward', 0):+.3f} | "
                f"buf={rollout_stats.get('buffer_size', 0):4d} | "
                f"ploss={update_stats.get('policy_loss', 0):.4f} | "
                f"vloss={update_stats.get('value_loss', 0):.4f} | "
                f"ent={update_stats.get('entropy', 0):.4f} | "
                f"kl={update_stats.get('kl', 0):.4f} | "
                f"lr={lr:.2e} | "
                f"{elapsed:.1f}s (total {total_elapsed:.0f}s)"
            )

            # Save on interval OR always save when total_epochs is very small
            if epoch % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch)

        # Always save the final model, even if the last epoch wasn't a save_interval multiple
        print("-" * 80)
        self.save_checkpoint(epoch, label="final")
        total_time = time.time() - training_start
        print(f"[PPO] Training complete in {total_time:.1f}s ({total_time/60:.1f} min).")

    # ── Utilities ──────────────────────────────────────────────────────────

    def _sync_opponents(self):
        """Copy current model weights to all frozen opponent agents."""
        state_dict = self.model.state_dict()
        for opp in self.opponents:
            opp.model.load_state_dict(state_dict)

    def save_checkpoint(self, epoch: int, label: str = None):
        filename = f"model_{label}.pt" if label else f"model_epoch_{epoch}.pt"
        path = os.path.join(self.cfg.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": dict(self.metrics),
            "ppo_config": self.cfg.__dict__,
        }, path)
        print(f"[PPO] Saved checkpoint -> {path}")
        self.save_charts(epoch, label)

    def save_charts(self, epoch: int, label: str = None):
        """
        Save a 2×2 grid of training charts alongside the checkpoint.

        Plots:
          • Mean reward (smoothed 20-epoch rolling average + raw)
          • Policy loss + KL divergence
          • Value loss  (log scale — starts very high, should trend to < 1)
          • Entropy     (should decrease slowly; collapse < 1.0 is a warning)

        Output: checkpoints/training_charts_{label or epoch}.png
        Silently skipped if matplotlib is not installed.
        """
        if not _MATPLOTLIB_AVAILABLE:
            return
        if len(self.metrics.get("mean_reward", [])) < 2:
            return  # not enough data yet

        m = self.metrics
        epochs = list(range(1, len(m["mean_reward"]) + 1))

        def smooth(vals, window=20):
            """Simple trailing moving average."""
            out = []
            for i in range(len(vals)):
                start = max(0, i - window + 1)
                out.append(sum(vals[start:i + 1]) / (i - start + 1))
            return out

        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor("#1a1a2e")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        panel_bg   = "#16213e"
        col_raw    = "#4a9eda"   # muted blue  — raw / secondary series
        col_smooth = "#e8a838"   # amber       — smoothed / primary series
        col_loss   = "#e05c5c"   # coral red   — loss curves
        col_kl     = "#7ecba1"   # teal        — KL
        col_ent    = "#b07fd4"   # purple      — entropy
        txt_color  = "#c8ccd4"
        grid_color = "#2a2a4a"

        def _style_ax(ax, title):
            ax.set_facecolor(panel_bg)
            ax.set_title(title, color=txt_color, fontsize=11, pad=8)
            ax.tick_params(colors=txt_color, labelsize=8)
            ax.xaxis.label.set_color(txt_color)
            ax.yaxis.label.set_color(txt_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(grid_color)
            ax.grid(color=grid_color, linewidth=0.5, alpha=0.7)
            ax.set_xlabel("Epoch", fontsize=9)

        # ── Panel 1: Reward ───────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        reward = m["mean_reward"]
        ax1.plot(epochs, reward, color=col_raw, alpha=0.35, linewidth=0.8, label="Raw")
        ax1.plot(epochs, smooth(reward), color=col_smooth, linewidth=1.8, label="Smoothed (20ep)")
        ax1.axhline(0, color="#888", linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.set_ylabel("Reward (normalised)", fontsize=9)
        ax1.legend(fontsize=8, facecolor=panel_bg, labelcolor=txt_color, framealpha=0.8)
        _style_ax(ax1, "Mean reward per epoch")

        # ── Panel 2: Policy loss + KL ─────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ploss = m.get("policy_loss", [])
        kl    = m.get("kl", [])
        if ploss:
            ax2.plot(epochs[:len(ploss)], ploss, color=col_loss, linewidth=1.2, label="Policy loss")
        ax2.set_ylabel("Policy loss", fontsize=9, color=col_loss)
        ax2.tick_params(axis="y", colors=col_loss)
        if kl:
            ax2b = ax2.twinx()
            ax2b.plot(epochs[:len(kl)], kl, color=col_kl, linewidth=1.0,
                      linestyle="--", alpha=0.8, label="KL")
            ax2b.set_ylabel("KL divergence", fontsize=9, color=col_kl)
            ax2b.tick_params(axis="y", colors=col_kl)
            ax2b.tick_params(colors=txt_color, labelsize=8)
            for spine in ax2b.spines.values():
                spine.set_edgecolor(grid_color)
            # Combined legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2b.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2,
                       fontsize=8, facecolor=panel_bg, labelcolor=txt_color, framealpha=0.8)
        _style_ax(ax2, "Policy loss & KL divergence")

        # ── Panel 3: Value loss (log scale) ───────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        vloss = m.get("value_loss", [])
        if vloss:
            ax3.semilogy(epochs[:len(vloss)], [max(v, 1e-6) for v in vloss],
                         color=col_loss, linewidth=1.2, label="Value loss")
            ax3.semilogy(epochs[:len(vloss)], smooth([max(v, 1e-6) for v in vloss]),
                         color=col_smooth, linewidth=1.8, linestyle="--", label="Smoothed")
            ax3.axhline(1.0, color="#888", linewidth=0.8, linestyle=":", alpha=0.6,
                        label="Target < 1.0")
        ax3.set_ylabel("Value loss (log scale)", fontsize=9)
        ax3.legend(fontsize=8, facecolor=panel_bg, labelcolor=txt_color, framealpha=0.8)
        _style_ax(ax3, "Value loss  (target: < 1.0)")

        # ── Panel 4: Entropy ──────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        entropy = m.get("entropy", [])
        if entropy:
            ax4.plot(epochs[:len(entropy)], entropy, color=col_ent, linewidth=1.2)
            ax4.plot(epochs[:len(entropy)], smooth(entropy), color=col_smooth,
                     linewidth=1.8, linestyle="--", label="Smoothed")
            ax4.axhline(1.0, color=col_loss, linewidth=0.8, linestyle=":",
                        alpha=0.7, label="Collapse warning (< 1.0)")
        ax4.set_ylabel("Entropy", fontsize=9)
        ax4.legend(fontsize=8, facecolor=panel_bg, labelcolor=txt_color, framealpha=0.8)
        _style_ax(ax4, "Policy entropy  (should stay > 1.5)")

        # ── Title + save ──────────────────────────────────────────────────
        tag = label if label else f"epoch_{epoch}"
        fig.suptitle(
            f"Training progress — {tag}  ({epoch} epochs)",
            color=txt_color, fontsize=13, y=0.98
        )

        chart_name = f"training_charts_{tag}.png"
        chart_path = os.path.join(self.cfg.checkpoint_dir, chart_name)
        fig.savefig(chart_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[PPO] Saved training charts -> {chart_path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.metrics.update(ckpt.get("metrics", {}))
        print(f"[PPO] Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt["epoch"]
