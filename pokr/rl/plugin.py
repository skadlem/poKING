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
