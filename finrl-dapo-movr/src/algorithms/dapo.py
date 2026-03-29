"""
src/algorithms/dapo.py — DAPO Algorithm
=======================================
Adapted from FinRL-DAPO-SR base repo (https://github.com/Ruijian-Zha/FinRL-DAPO-SR).
Corresponds to: dapo_algorithm.py in the original repo.
Paper: arXiv:2505.06408, Algorithm 1.

Key DAPO innovations over standard GRPO:
  1. Asymmetric epsilon clipping (epsilon_low != epsilon_high)
  2. Dynamic sampling — filters groups where all rewards are identical
  3. Token-level policy gradient (as opposed to outcome-level)

Inputs: gymnasium environment, config dict
Outputs: trained policy + value network weights
"""

import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional


class ActorCritic(nn.Module):
    """Shared MLP backbone with separate policy and value heads."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        mean = torch.tanh(self.policy_mean(features))
        std = torch.exp(self.policy_log_std.clamp(-20, 2))
        value = self.value_head(features).squeeze(-1)
        return mean, std, value

    def get_action(self, obs):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy


class DAPOAlgorithm:
    """
    DAPO: Decoupled-Clip Asymmetric Policy Optimisation.

    Key differences from PPO/GRPO:
    - epsilon_low != epsilon_high (asymmetric clipping)
    - dynamic_sampling=True filters uninformative gradient groups
    """

    def __init__(
        self,
        env,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.28,
        dynamic_sampling: bool = True,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        target_kl: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        symmetric_clip: bool = False,      # set True for vanilla GRPO behaviour
    ):
        self.env = env
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.dynamic_sampling = dynamic_sampling
        self.lr = learning_rate
        self.gamma = gamma
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.device = device
        self.symmetric_clip = symmetric_clip

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.ac = ActorCritic(obs_dim, action_dim).to(device)
        self.pi_optimizer = optim.Adam(
            list(self.ac.shared.parameters())
            + list(self.ac.policy_mean.parameters())
            + [self.ac.policy_log_std],
            lr=self.lr,
        )
        self.v_optimizer = optim.Adam(
            list(self.ac.shared.parameters())
            + list(self.ac.value_head.parameters()),
            lr=self.lr,
        )

    def collect_rollout(self, steps: int):
        """Collect trajectory data from the environment."""
        obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []

        obs, _ = self.env.reset()
        for _ in range(steps):
            obs_t = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.ac.get_action(obs_t)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated

            obs_list.append(obs)
            act_list.append(action.cpu().numpy())
            logp_list.append(log_prob.cpu().item())
            rew_list.append(reward)
            val_list.append(value.cpu().item())
            done_list.append(done)

            obs = next_obs if not done else self.env.reset()[0]

        return {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(act_list, dtype=np.float32),
            "log_probs": np.array(logp_list, dtype=np.float32),
            "rewards": np.array(rew_list, dtype=np.float32),
            "values": np.array(val_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=bool),
        }

    def compute_returns(self, rewards, values, dones):
        """GAE-lambda returns computation."""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_val = 0.0
                last_gae = 0.0
            else:
                next_val = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * 0.95 * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
        return returns, advantages

    def _apply_dynamic_sampling(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        DAPO dynamic sampling: zero out gradients for groups where all
        advantages are equal (uninformative updates).
        """
        if not self.dynamic_sampling:
            return advantages
        # Simple heuristic: mask samples where |advantage| < eps
        mask = (advantages.abs() > 1e-6).float()
        return advantages * mask

    def update(self, batch):
        """Perform DAPO policy and value updates."""
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch["log_probs"]).to(self.device)
        returns, advantages = self.compute_returns(
            batch["rewards"], batch["values"], batch["dones"]
        )
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Dynamic sampling (DAPO innovation)
        advantages = self._apply_dynamic_sampling(advantages)

        # -- Policy update --
        for _ in range(self.train_pi_iters):
            log_probs, values, entropy = self.ac.evaluate(obs, actions)
            ratio = torch.exp(log_probs - old_log_probs)

            # Asymmetric clipping (DAPO innovation)
            eps_lo = self.epsilon_low
            eps_hi = self.epsilon_high if not self.symmetric_clip else self.epsilon_low

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi) * advantages
            pi_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.pi_optimizer.step()

            # Early stopping on KL divergence
            with torch.no_grad():
                kl = (old_log_probs - log_probs).mean().item()
            if kl > 1.5 * self.target_kl:
                break

        # -- Value update --
        for _ in range(self.train_v_iters):
            _, values, _ = self.ac.evaluate(obs, actions)
            v_loss = ((values - returns) ** 2).mean()
            self.v_optimizer.zero_grad()
            v_loss.backward()
            self.v_optimizer.step()

        return {"pi_loss": pi_loss.item(), "v_loss": v_loss.item(), "kl": kl}

    def train(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> list:
        """
        Full training loop.

        Returns
        -------
        list of dicts with per-epoch metrics
        """
        training_log = []
        for epoch in range(total_epochs):
            batch = self.collect_rollout(steps_per_epoch)
            metrics = self.update(batch)
            ep_return = batch["rewards"].sum()
            metrics["epoch"] = epoch
            metrics["ep_return"] = float(ep_return)
            training_log.append(metrics)

            if verbose and epoch % 10 == 0:
                print(
                    f"[Epoch {epoch:03d}/{total_epochs}] "
                    f"Return: {ep_return:.4f} | "
                    f"PI Loss: {metrics['pi_loss']:.4f} | "
                    f"KL: {metrics['kl']:.5f}"
                )

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.ac.state_dict(), checkpoint_dir / "actor_critic.pt")
            print(f"Checkpoint saved to {checkpoint_dir}")

        return training_log

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        self.ac.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        print(f"Loaded checkpoint from {checkpoint_path}")

    def get_action_deterministic(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action for evaluation (uses policy mean)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, _, _ = self.ac(obs_t)
        return mean.squeeze(0).cpu().numpy()
