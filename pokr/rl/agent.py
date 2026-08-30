"""RLStrategy: the torch agent as an engine Strategy, recording itself.

PokerGame is callback-driven -- it calls decide() -- while an RL loop wants to
be the caller. Rather than invert control with a gym wrapper or threads, the
agent plays as an ordinary Strategy and buffers what it saw: decide() appends
a step, on_hand_end() stamps the hand's net result on the whole trajectory as
the terminal reward. One hand is one episode, so `bench.play_session` doubles
as the rollout collector.

Reward is result.winnings[seat] / big_blind -- byte-for-byte the per-hand bb
that bench.run_matchup reports, so the training objective and the benchmark
metric are the same number.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..bot import PokerBot
from ..cards import monte_carlo_equity
from ..models import ModelManager
from ..strategy import Action, BaseStrategy
from .encode import NUM_ACTIONS, OBS_DIM, action_mask, decode, encode_obs
from .net import PolicyValueNet


@dataclass
class Episode:
    """One hand from the agent's seat. reward is terminal (bb won this hand)."""
    obs: np.ndarray            # [T, OBS_DIM] float32
    masks: np.ndarray          # [T, NUM_ACTIONS] bool
    actions: np.ndarray        # [T] int64
    logps: np.ndarray          # [T] float32
    values: np.ndarray         # [T] float32
    reward: float


@dataclass
class _Step:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    logp: float
    value: float


@dataclass
class RolloutBuffer:
    episodes: list[Episode] = field(default_factory=list)

    def clear(self) -> None:
        self.episodes.clear()

    @property
    def num_steps(self) -> int:
        return sum(len(e.actions) for e in self.episodes)


class RLStrategy(BaseStrategy):
    """Torch policy playing inside the pokr engine.

    net=None plays uniformly at random over the legal mask, which is what the
    day-0 legality gate exercises: no torch weights involved, but the exact
    encode/decode path the trained agent will use.
    """

    def __init__(
        self,
        net: PolicyValueNet | None = None,
        rng: random.Random | None = None,
        num_players: int = 6,
        mc_iters: int = 0,
        mc_fast: bool = False,
        model_opponents: bool = True,
        greedy: bool = False,
        record: bool = False,
        buffer: RolloutBuffer | None = None,
    ) -> None:
        self.net = net
        self.rng = rng or random.Random()
        self.num_players = num_players
        self.model_opponents = model_opponents
        self.models = ModelManager(num_players) if model_opponents else None
        self.mc_iters = mc_iters
        self.mc_fast = mc_fast
        self.greedy = greedy
        self.record = record
        self.buffer = buffer if buffer is not None else RolloutBuffer()
        self._steps: list[_Step] = []
        self._equity_cache: dict[tuple, float] = {}

    # -- Strategy ---------------------------------------------------------

    def decide(self, state, player_id: int) -> Action:
        obs = self.observe(state, player_id)
        mask = action_mask(state, player_id)
        if self.net is None:
            idx = self._random_action(mask)
            logp = -float(np.log(mask.sum()))
            value = 0.0
        else:
            idx, logp, value = self.net.act(obs, mask, greedy=self.greedy)
        if self.record:
            self._steps.append(_Step(obs, mask, idx, logp, value))
        return decode(state, player_id, idx)

    def on_hand_end(self, result, my_seat: int) -> None:
        if self.models is not None:
            self.models.observe(result, my_seat)
        if self.record and self._steps:
            reward = result.winnings[my_seat] / result.big_blind
            self.buffer.episodes.append(self._pack(reward))
        self._steps.clear()
        self._equity_cache.clear()

    def reset_models(self) -> None:
        """Drop opponent stats. Call whenever the lineup changes: the models are
        keyed by seat, so a new opponent in seat 1 inherits the old one's read
        (bench.run_benchmark uses a fresh bot per matchup for the same reason)."""
        if self.model_opponents:
            self.models = ModelManager(self.num_players)

    def clone(self, **overrides) -> "RLStrategy":
        """A twin sharing this agent's net -- for evaluation seats (greedy, not
        recording) or self-play opponents, without copying weights."""
        kwargs = dict(net=self.net, rng=self.rng, num_players=self.num_players,
                      mc_iters=self.mc_iters, mc_fast=self.mc_fast,
                      model_opponents=self.model_opponents, greedy=self.greedy,
                      record=False)
        kwargs.update(overrides)
        return RLStrategy(**kwargs)

    # -- internals --------------------------------------------------------

    def observe(self, state, player_id: int) -> np.ndarray:
        """The observation this agent would encode for `state` (also the seam
        tests and diagnostics use to inspect features without acting)."""
        opponents = [q.id for q in state.players if q.id != player_id and not q.folded]
        summary = None
        if self.models is not None and opponents:
            target = PokerBot._target_opponent(state, player_id, opponents)
            summary = self.models.summary(target)
        return encode_obs(state, player_id, self._equity(state, player_id, len(opponents)),
                          summary)

    def _equity(self, state, player_id: int, num_opponents: int) -> float | None:
        if self.mc_iters <= 0 or num_opponents <= 0:
            return None
        me = state.players[player_id]
        key = (tuple(sorted((c.rank, c.suit) for c in me.hole)),
               tuple(sorted((c.rank, c.suit) for c in state.community)),
               num_opponents)
        hit = self._equity_cache.get(key)
        if hit is None:
            equity_fn = monte_carlo_equity
            if self.mc_fast:
                from .._fastcards import monte_carlo_equity_fast
                equity_fn = monte_carlo_equity_fast
            hit = equity_fn(me.hole, state.community, num_opponents,
                            self.mc_iters, self.rng)
            self._equity_cache[key] = hit
        return hit

    def _random_action(self, mask: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(mask).tolist()))

    def _pack(self, reward: float) -> Episode:
        return Episode(
            obs=np.stack([s.obs for s in self._steps]).astype(np.float32),
            masks=np.stack([s.mask for s in self._steps]),
            actions=np.array([s.action for s in self._steps], dtype=np.int64),
            logps=np.array([s.logp for s in self._steps], dtype=np.float32),
            values=np.array([s.value for s in self._steps], dtype=np.float32),
            reward=float(reward),
        )
