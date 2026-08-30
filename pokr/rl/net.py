"""Policy/value network for the RL agent: one MLP trunk, two heads.

Small on purpose (2x256, CPU): the observation already carries Monte Carlo
equity and opponent-model features, so the net is not being asked to learn
hand strength from raw cards.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .encode import NUM_ACTIONS, OBS_DIM

_MASK_FILL = -1e8  # finite, so softmax over an all-masked row can't produce NaN


class PolicyValueNet(nn.Module):
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
        self.v = nn.Linear(prev, 1)
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = tuple(hidden)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs [B, OBS_DIM], mask [B, NUM_ACTIONS] bool -> (masked logits, value [B])."""
        h = self.trunk(obs)
        logits = self.pi(h).masked_fill(~mask, _MASK_FILL)
        return logits, self.v(h).squeeze(-1)

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor):
        logits, value = self.forward(obs, mask)
        return torch.distributions.Categorical(logits=logits), value

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        generator: torch.Generator | None = None,
        greedy: bool = False,
    ) -> tuple[int, float, float]:
        """Single-state step. Returns (action index, log-prob, value estimate)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        logits, value = self.forward(obs_t, mask_t)
        if greedy:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1, generator=generator).item())
        logp = float(torch.log_softmax(logits, dim=-1)[0, action].item())
        return action, logp, float(value.item())

    def config(self) -> dict:
        return {"obs_dim": self.obs_dim, "num_actions": self.num_actions,
                "hidden": self.hidden}


_RESERVED = ("net_config", "state_dict")


def save(net: PolicyValueNet, path: str, **extra) -> None:
    """Write weights plus arbitrary training metadata.

    The architecture lives under "net_config" rather than "config" because
    callers pass their own `config=vars(args)`; sharing the key silently
    replaced the architecture with CLI arguments and made every checkpoint
    unloadable, so a collision is now an error rather than a surprise.
    """
    clash = [k for k in extra if k in _RESERVED]
    if clash:
        raise ValueError(f"reserved checkpoint keys: {clash}")
    torch.save({"net_config": net.config(), "state_dict": net.state_dict(), **extra}, path)


def load(path: str) -> tuple[PolicyValueNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(**ckpt["net_config"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
