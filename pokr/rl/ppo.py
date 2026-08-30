"""PPO over the hand-episodes RLStrategy records.

The reward structure here is unusual and simplifies things: every intra-hand
reward is 0 and the whole signal arrives at the terminal step as the hand's
net bb. With gamma=1, lam=1 the GAE recursion collapses to (reward - V(s_t))
for every step; lam < 1 trades a little bias for variance, which is worth
having when a single stack-off is 100x a typical pot.

Rewards are divided by a CONSTANT `reward_scale` before use. Measured over 2k
hands the per-hand reward has std ~274bb and tails past +/-3000bb (engine
stacks inflate over a session: bench's _rebuy tops up busted players but never
caps winners, so late hands are played 200-300bb deep). Unscaled, the value
head cannot fit targets spanning two orders of magnitude.

A constant divisor is deliberate. Per-hand normalization -- dividing by the
hero's stack, say -- would be better conditioned but changes the objective:
maximizing E[bb/stack] over-weights short-stacked hands, which is not the game
we want to play. A constant is an exact linear rescale, so the argmax is
untouched. Rewards are also not clipped: big pots are exactly the decisions
that matter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .agent import Episode, RolloutBuffer
from .net import PolicyValueNet


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 1024
    gamma: float = 1.0        # no discounting: a hand is one short episode
    lam: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 100.0
    normalize_advantage: bool = True
    target_kl: float | None = 0.02   # early-stop the epoch loop; None disables


@dataclass
class Batch:
    obs: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    logps: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return self.actions.shape[0]


def episode_advantages(ep: Episode, gamma: float, lam: float,
                       reward_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) for one terminal-reward episode -> (advantages, returns)."""
    t = len(ep.actions)
    values = ep.values.astype(np.float64)
    rewards = np.zeros(t)
    rewards[-1] = ep.reward / reward_scale
    adv = np.zeros(t)
    running = 0.0
    for i in range(t - 1, -1, -1):
        next_value = values[i + 1] if i + 1 < t else 0.0   # terminal
        delta = rewards[i] + gamma * next_value - values[i]
        running = delta + gamma * lam * running
        adv[i] = running
    return adv.astype(np.float32), (adv + values).astype(np.float32)


def make_batch(buffer: RolloutBuffer, cfg: PPOConfig) -> Batch:
    """Flatten recorded episodes into one padded-free training batch."""
    if not buffer.episodes:
        raise ValueError("empty rollout buffer")
    advs, rets = [], []
    for ep in buffer.episodes:
        a, r = episode_advantages(ep, cfg.gamma, cfg.lam, cfg.reward_scale)
        advs.append(a)
        rets.append(r)
    return Batch(
        obs=torch.from_numpy(np.concatenate([e.obs for e in buffer.episodes])),
        masks=torch.from_numpy(np.concatenate([e.masks for e in buffer.episodes])),
        actions=torch.from_numpy(np.concatenate([e.actions for e in buffer.episodes])),
        logps=torch.from_numpy(np.concatenate([e.logps for e in buffer.episodes])),
        advantages=torch.from_numpy(np.concatenate(advs)),
        returns=torch.from_numpy(np.concatenate(rets)),
    )


class PPOTrainer:
    def __init__(self, net: PolicyValueNet, cfg: PPOConfig | None = None,
                 generator: torch.Generator | None = None) -> None:
        self.net = net
        self.cfg = cfg or PPOConfig()
        self.opt = torch.optim.Adam(net.parameters(), lr=self.cfg.lr)
        self.generator = generator

    def update(self, buffer: RolloutBuffer) -> dict:
        cfg = self.cfg
        batch = make_batch(buffer, cfg)
        n = len(batch)
        adv = batch.advantages
        if cfg.normalize_advantage and n > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        updates = 0
        stopped_early = False
        for _ in range(cfg.epochs):
            order = torch.randperm(n, generator=self.generator)
            for start in range(0, n, cfg.minibatch):
                idx = order[start:start + cfg.minibatch]
                dist, value = self.net.distribution(batch.obs[idx], batch.masks[idx])
                logp = dist.log_prob(batch.actions[idx])
                ratio = torch.exp(logp - batch.logps[idx])
                a = adv[idx]
                pg = -torch.min(ratio * a,
                                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vf = 0.5 * (value - batch.returns[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = float(((ratio - 1) - (logp - batch.logps[idx])).mean())
                    stats["policy_loss"] += float(pg)
                    stats["value_loss"] += float(vf)
                    stats["entropy"] += float(ent)
                    stats["approx_kl"] += kl
                    stats["clip_frac"] += float(
                        ((ratio - 1).abs() > cfg.clip).float().mean())
                updates += 1
            if cfg.target_kl is not None and updates and \
                    stats["approx_kl"] / updates > cfg.target_kl:
                stopped_early = True
                break

        for k in stats:
            stats[k] /= max(updates, 1)
        stats["steps"] = n
        stats["episodes"] = len(buffer.episodes)
        stats["mean_reward_bb"] = float(np.mean([e.reward for e in buffer.episodes]))
        stats["early_stop"] = stopped_early
        return stats
