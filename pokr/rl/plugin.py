"""Connector plugin for the trained PPO agent.

Deliberately torch-free at import time: pokr.connector imports this module at
startup, and torch (like rlcard in rlcard_adapter.py) must stay an optional
dependency. The checkpoint and the network are loaded on the first decision.
"""
from __future__ import annotations

import os
import random

from ..connector import register_plugin
from ..strategy import Action, BaseStrategy

DEFAULT_CKPT = "models/rl/ppo_final.pt"


class TrainedRLStrategy(BaseStrategy):
    """Greedy play from a PPO checkpoint trained by train_rl.py.

    Point it at a checkpoint with the POKR_RL_CKPT env var; the default is
    models/rl/ppo_final.pt.
    """

    def __init__(self, ckpt_path: str | None = None, rng: random.Random | None = None,
                 num_players: int = 6, mc_iters: int = 30, mc_fast: bool = True,
                 greedy: bool = True) -> None:
        self.ckpt_path = ckpt_path or os.environ.get("POKR_RL_CKPT", DEFAULT_CKPT)
        self.rng = rng or random.Random()
        self.num_players = num_players
        self.mc_iters = mc_iters
        self.mc_fast = mc_fast
        self.greedy = greedy
        self._inner = None

    def _load(self):
        if self._inner is None:
            from .agent import RLStrategy
            from .net import load
            net, ckpt = load(self.ckpt_path)
            # the checkpoint's own training config wins: the equity feature must
            # be fed the same way it was during training or the obs is a lie
            cfg = ckpt.get("config") or {}
            self._inner = RLStrategy(
                net=net, rng=self.rng, num_players=self.num_players,
                mc_iters=cfg.get("mc_iters", self.mc_iters),
                mc_fast=cfg.get("fast", self.mc_fast),
                greedy=self.greedy, record=False)
        return self._inner

    def decide(self, state, player_id: int) -> Action:
        return self._load().decide(state, player_id)

    def on_hand_end(self, result, my_seat: int) -> None:
        if self._inner is not None:
            self._inner.on_hand_end(result, my_seat)


def rl_factory() -> TrainedRLStrategy:
    return TrainedRLStrategy()


register_plugin("rl", rl_factory)


class TrainedNFSPStrategy(BaseStrategy):
    """Sampled deployment of an NFSP Pi checkpoint (roadmap step 2).

    Same lazy-torch shape as TrainedRLStrategy. The decisive difference is
    what is NOT here: there is no greedy parameter at all. Design note 3.4 —
    the argmax of an approximate equilibrium is a pure strategy and is
    maximally exploitable — so this plugin structurally cannot deploy Pi
    wrong. NFSPStrategy's own act path samples; nothing wires a flag to
    bypass it.

    Checkpoint: POKR_NFSP_CKPT, default models/nfsp/nfsp_final.pt (which
    does not exist until step 9 trains one; constructing is lazy, the first
    decision loads).
    """

    DEFAULT_CKPT = "models/nfsp/nfsp_final.pt"

    def __init__(self, ckpt_path: str | None = None,
                 rng: random.Random | None = None,
                 num_players: int = 2) -> None:
        self.ckpt_path = ckpt_path or os.environ.get("POKR_NFSP_CKPT",
                                                     self.DEFAULT_CKPT)
        self.rng = rng or random.Random()
        self.num_players = num_players
        self._inner = None

    def _load(self):
        if self._inner is None:
            from .avg_policy import load
            from .nfsp import NFSPConfig, NFSPStrategy
            net, _ckpt = load(self.ckpt_path)
            # heads-up only: NFSP's guarantee is 2p0s (design note 3.5) and
            # the checkpoint's observation is meaningless seated at 6 max
            self._inner = NFSPStrategy(
                net=net, config=NFSPConfig(fit_every=0), rng=self.rng,
                num_players=self.num_players, record=False)
        return self._inner

    def decide(self, state, player_id: int) -> Action:
        return self._load().decide(state, player_id)

    def on_hand_end(self, result, my_seat: int) -> None:
        if self._inner is not None:
            self._inner.on_hand_end(result, my_seat)


def nfsp_factory() -> TrainedNFSPStrategy:
    return TrainedNFSPStrategy()


register_plugin("nfsp", nfsp_factory)
