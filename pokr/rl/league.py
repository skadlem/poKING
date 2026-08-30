"""A league of frozen past selves to train against.

Training only against a fixed pool of scripted bots plus the heuristic teaches
the agent to beat those five opponents, and nothing stops it drifting into a
style that a slightly different opponent punishes. The measured symptom was
non-transitivity: the retrained agent beat its predecessor's record against the
heuristic while LOSING to that predecessor head-to-head (-288 bb/100 over 20k
hands). Seating frozen snapshots of itself in the opponent pool is the standard
fix -- the agent has to stay good against what it used to be, not just against
what it currently faces.

This is not self-play. Live self-play in an imperfect-information game cycles
rather than converging (the reason NFSP and Deep CFR exist); the opponents here
are FROZEN, so each iteration still faces a stationary environment.

Snapshots are deep copies. Holding a reference to the live network would make
the whole league track the agent as it trains, which is exactly live self-play
wearing a disguise and would look like it was working.
"""
from __future__ import annotations

import copy
import random
from collections import deque
from typing import Callable

from ..strategy import Strategy
from .agent import RLStrategy
from .net import PolicyValueNet


class League:
    """Frozen snapshots of the agent, sampled uniformly.

    max_size caps memory (each snapshot is ~440KB at the default architecture).
    Old snapshots are the valuable ones for breaking cycles, so pick a snapshot
    interval that keeps the whole run rather than relying on the cap: at
    --league-every 25 over 600 iterations the deque never fills.
    """

    def __init__(self, max_size: int = 30) -> None:
        self.max_size = max_size
        self._nets: deque[PolicyValueNet] = deque(maxlen=max_size)
        self.snapshots_taken = 0

    def snapshot(self, net: PolicyValueNet) -> None:
        frozen = copy.deepcopy(net)
        frozen.eval()
        frozen.requires_grad_(False)
        self._nets.append(frozen)
        self.snapshots_taken += 1

    def __len__(self) -> int:
        return len(self._nets)

    def state(self) -> list[dict]:
        """Snapshot weights for checkpointing. Without this, --resume restarts
        with an empty league and silently discards the single change that most
        improved the agent."""
        return [{k: v.clone() for k, v in n.state_dict().items()} for n in self._nets]

    def restore(self, states: list[dict], config: dict) -> None:
        self._nets.clear()
        for state in states:
            net = PolicyValueNet(**config)
            net.load_state_dict(state)
            net.eval()
            net.requires_grad_(False)
            self._nets.append(net)
        self.snapshots_taken = max(self.snapshots_taken, len(self._nets))

    def sample(self, rng: random.Random) -> PolicyValueNet | None:
        return rng.choice(list(self._nets)) if self._nets else None

    def opponent_factory(
        self,
        rng: random.Random,
        mc_iters: int = 0,
        mc_fast: bool = False,
        num_players: int = 6,
    ) -> Callable[[random.Random], Strategy] | None:
        """A factory seating one randomly chosen past self, or None if empty.

        The snapshot is chosen once here rather than per hand, so a session
        faces one consistent opponent (its opponent models mean something).
        Frozen selves play stochastically, not greedily: a deterministic
        opponent is trivially exploitable and would teach the agent a
        counter-strategy that beats nothing else.
        """
        net = self.sample(rng)
        if net is None:
            return None

        def factory(r: random.Random) -> Strategy:
            return RLStrategy(net=net, rng=r, num_players=num_players,
                              mc_iters=mc_iters, mc_fast=mc_fast,
                              greedy=False, record=False)

        factory.__name__ = "league"
        return factory
