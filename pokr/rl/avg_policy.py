"""AvgPolicyNet: the average strategy Pi(s) of NFSP, and its supervised fit.

Design note 2: Q is the behaviour, Pi is the output — "Evaluating the Q-net
is the classic implementation bug". This module owns the output side only:
one MLP trunk, one masked policy head, and cross-entropy training on the
(s, a, mask) records of the reservoir (memory.py). Ladder B's loop is
`Pi <- supervised fit on reservoir of BR data`, and that fit is this file.

Deliberately NOT in PolicyValueNet (design note 5): PPO depends on that
class and its checkpoint contract, and Pi has no value head — a subclass
would inherit a head it never trains. The masked-logit convention is the
same as net.py (mask True = legal, illegal slots filled with -1e8 before
softmax), so a record set can flow through either learner unchanged.

Why cross-entropy IS the fictitious average: the CE minimiser of a fixed
dataset of (s, a) samples is the empirical action distribution at each s.
A reservoir sample of past best responses is, by construction, a uniform
draw from all past behaviour — so fitting Pi on it converges Pi to the
average of past behaviours, the exact object the convergence proof needs.
This is asserted in the tests: frequencies in, frequencies out.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .encode import NUM_ACTIONS, OBS_DIM

_MASK_FILL = -1e8  # same value as net.py: finite, so no NaN, and e^-1e8
# underflows to exactly 0.0 in float32 — illegal slots get zero probability
# AND zero gradient, so the head's illegal logits cannot drift.


class AvgPolicyNet(nn.Module):
    """Policy-only MLP with masked action probabilities."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.pi = nn.Linear(prev, num_actions)
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = tuple(hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw, unmasked logits [B, num_actions]."""
        return self.pi(self.trunk(obs))

    def masked_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).masked_fill(~mask, _MASK_FILL)

    @torch.no_grad()
    def probs(self, obs: np.ndarray | torch.Tensor,
              mask: np.ndarray | torch.Tensor) -> np.ndarray:
        """[B, num_actions] legal-only probabilities from numpy inputs."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        if mask_t.ndim == 1:
            mask_t = mask_t.unsqueeze(0)
        return torch.softmax(self.masked_logits(obs_t, mask_t), dim=-1).numpy()

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray,
            generator: torch.Generator | None = None) -> tuple[int, float]:
        """Single-state step: sample Pi, return (action, log-prob).

        Sampled, never argmax: design note 3.4 — the argmax of an approximate
        equilibrium is a pure strategy and is maximally exploitable. This
        net has no greedy path on purpose; there is no flag to set wrong.
        """
        probs = self.probs(obs, mask)[0]
        action = int(torch.multinomial(torch.as_tensor(probs).unsqueeze(0),
                                       1, generator=generator).item())
        return action, float(np.log(probs[action])) if probs[action] > 0 else -float("inf")

    def config(self) -> dict:
        return {"obs_dim": self.obs_dim, "num_actions": self.num_actions,
                "hidden": self.hidden}


_RESERVED = ("net_config", "state_dict")


def save(net: AvgPolicyNet, path: str, **extra) -> None:
    """Same contract as net.save: architecture under \"net_config\", and a
    caller key colliding with the reserved ones is an error, not a surprise."""
    clash = [k for k in extra if k in _RESERVED]
    if clash:
        raise ValueError(f"reserved checkpoint keys: {clash}")
    torch.save({"net_config": net.config(), "state_dict": net.state_dict(), **extra}, path)


def load(path: str) -> tuple[AvgPolicyNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = AvgPolicyNet(**ckpt["net_config"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


# -- the supervised fit ------------------------------------------------------


def sl_loss(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Mean -log p(action) over rows of already-MASKED logits.

    Illegal slots hold _MASK_FILL, which log_softmax maps to probability
    exactly 0 in float32 and to zero gradient at the gather — but an
    *illegal* target action would score -log(0) = inf. Callers must feed
    only actions legal at their state; fit_avg_policy checks once, up
    front, rather than trusting it.
    """
    return -torch.log_softmax(logits, dim=-1).gather(
        1, actions.unsqueeze(1)).squeeze(1).mean()


def fit_avg_policy(
    net: AvgPolicyNet,
    obs: np.ndarray,
    masks: np.ndarray | list[np.ndarray],
    actions: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    generator: torch.Generator | None = None,
) -> list[float]:
    """Minibatch Adam fit of Pi on a reservoir sample. Returns per-epoch mean
    loss (so a training curve is a value, not a log scrape).

    masks may be a [N, A] array or a ragged list of per-row bool arrays.
    Shuffling uses torch.randperm under `generator`, so a seeded run
    reproduces line by line — same discipline as the benchmark seeds.
    """
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
    masks_t = (torch.as_tensor(np.asarray(masks, dtype=bool))
               if not isinstance(masks, list) else torch.as_tensor(np.stack(masks)))
    actions_t = torch.as_tensor(np.asarray(actions, dtype=np.int64))
    n = obs_t.shape[0]
    if n == 0:
        raise ValueError("empty training set")
    if not masks_t[torch.arange(n), actions_t].all():
        bad = torch.nonzero(~masks_t[torch.arange(n), actions_t]).flatten()
        raise ValueError(
            f"{len(bad)} illegal target actions (rows {bad[:5].tolist()}): "
            "M_SL must hold only actions taken while following the behaviour")

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = sl_loss(net.masked_logits(obs_t[idx], masks_t[idx]),
                           actions_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        losses.append(total / n)
    return losses
