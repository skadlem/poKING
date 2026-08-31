"""NFSPStrategy: the average policy Pi as an engine Strategy (roadmap step 7).

Deployment contract first (design note 3.4): this agent samples. There is no
greedy path, no argmax behind a flag — the argmax of an approximate
equilibrium is a pure strategy and is maximally exploitable, so the rope is
not in this API. `AvgPolicyNet.act` already enforces it; this class never
bypasses it.

What Pi needs from an observation is strictly less than what PPO trained on:
it must be a true function of the information state, nothing more. That is
exactly what step 1 + the encoding work guarantee — deterministic equity,
betting history as the last block, opponent model available (step 3's
`model_opponents=False` is permitted by the proof, not required). This class
holds no new observation logic on purpose: it composes an RLStrategy
(net=None) purely as the observe/decode seam, so there is ONE encode path in
the repo and the history/layout contract can never drift between agents.
The RLStrategy never acts here — its net is None and record is False; this
class does the acting and the recording.

Two behaviour modes, because the roadmap's ladder decision (design note 4:
"B first") sits between step 7's original ladder-A wording and what steps
8-9 will actually run:

- behaviour="ppo" (Ladder B, the default): every hand is played by Pi and
  every hand's steps land in the reservoir as br_mode=False. Fitting Pi on
  its own past is a self-averaging placeholder — it does NOT have the
  fictitious-play convergence guarantee — but it is a correct, measurable
  agent from the first round of rollouts, and the outer loop (train_nfsp.py,
  step 9) upgrades it by calling `record_episode` on the PPO best-response
  hands harvested from exploit.py's oracle (br_mode=True). `fit` prefers BR
  rows whenever any exist: design note 2's "M_SL holds *only* actions taken
  while following the behaviour" is enforced at fit time, not at add time,
  because both kinds legitimately share the buffer.

- behaviour="epsilon" (Ladder A shape, dormant): the per-hand sigma coin
  flip from section 2 — with probability eta, play eps-greedy over a
  tabular Q head (`q_table`, a dict info-state-key -> per-action values);
  those hands are marked br_mode=True and Pi is never asked. The Q learner
  itself does not exist (that is ladder A's DQN); with q_table=None the
  behaviour degrades to uniform-random, which is correct but weak. The
  coin flip, the marking, and the fit preference are all tested here so
  that ladder A is a learner drop-in, not a rewrite.

Recording is per hand (an episode), rows are per step: M_SL's unit is the
(s, mask, a, br_mode) tuple. Rows carry the history-bearing observation of
step 5917f88 and the deterministic equity of step 44238bf, so one info
state means one row across workers, across hands, across seeds.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from ..strategy import Action, BaseStrategy
from .agent import Episode, RLStrategy, _equity_key
from .avg_policy import AvgPolicyNet, fit_avg_policy
from .encode import action_mask, decode
from .memory import ReservoirBuffer


@dataclass
class NFSPConfig:
    capacity: int = 100_000            # M_SL rows (steps, not hands)
    fit_every: int = 500               # hands between supervised fits (0 = manual)
    epochs: int = 10
    batch_size: int = 256              # also the minimum rows before an auto-fit
    lr: float = 1e-3
    behaviour: str = "ppo"             # "ppo" (ladder B) | "epsilon" (ladder A)
    eta: float = 0.1                   # ladder A: P(play the BR this hand)
    eps_start: float = 0.08            # ladder A: eps schedule, paper's start
    eps_min: float = 0.0
    hidden: tuple[int, ...] = (256, 256)


def _state_key(state, player_id: int) -> tuple:
    """Ladder A's tabular-Q identity for a decision point.

    Immutable values only — never the raw Action objects (dataclasses
    without __hash__; a mutable key is how a table silently forks). The
    betting history is what makes this an information state rather than a
    chip snapshot (the 5917f88 argument, in table form). Contains strings,
    which hash per-process under spawn — fine here: the Q table lives in
    the trainer process only (rollout workers receive nets as state_dicts,
    never this table), and the repo's fork context keeps the seed common
    even if that ever changes."""
    me = state.players[player_id]
    base = _equity_key(me.hole, state.community, 0)
    actions = tuple((seat, street, act.action_type.name)
                    for seat, street, act in state.action_history)
    return base + (actions, player_id, state.street)


def select_fit_rows(rows: list) -> list:
    """The design note 2 contract, as a pure function: Pi is fitted on the
    BEHAVIOUR's rows (br_mode=True). Pi's own rows (br_mode=False) are what
    the fit produces — including them re-learns the current policy and
    stalls the fictitious average toward itself — so they serve ONLY as the
    documented bootstrap when no behaviour rows exist yet (ladder B before
    the first oracle round; ladder A with no Q head)."""
    br = [r for r in rows if r[3]]
    return br if br else list(rows)


class NFSPStrategy(BaseStrategy):
    """Samples Pi inside the engine; collects (obs, mask, action, br_mode)
    rows into the reservoir; refits Pi by cross-entropy every fit_every."""

    def __init__(self, net: AvgPolicyNet | None = None,
                 config: NFSPConfig | None = None,
                 rng: random.Random | None = None,
                 num_players: int = 2,
                 mc_iters: int = 30, mc_fast: bool = True,
                 model_opponents: bool = False,
                 record: bool = True,
                 q_table: dict | None = None,
                 buffer: ReservoirBuffer | None = None) -> None:
        cfg = config or NFSPConfig()
        if cfg.behaviour not in ("ppo", "epsilon"):
            raise ValueError(f"unknown behaviour {cfg.behaviour!r}; "
                             "use 'ppo' (ladder B) or 'epsilon' (ladder A)")
        self.config = cfg
        self.rng = rng or random.Random()
        self.q_table = q_table                      # ladder A's BR; None -> uniform
        self.net = net if net is not None else AvgPolicyNet(hidden=cfg.hidden)
        self.buffer = buffer if buffer is not None else \
            ReservoirBuffer(cfg.capacity, self.rng)
        self.record = record
        self._steps: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._hand_is_br: bool | None = None    # None = flip on first decision
        self._flips = 0
        self._hands_since_fit = 0
        self.last_fit_loss: float | None = None
        # the observation/decode seam: an RLStrategy with net=None, so its
        # decide() can never be what plays — observe() is all we take from it
        self._view = RLStrategy(net=None, rng=self.rng, num_players=num_players,
                                mc_iters=mc_iters, mc_fast=mc_fast,
                                model_opponents=model_opponents,
                                greedy=False, record=False, history=True)

    # -- deployment --------------------------------------------------------

    def decide(self, state, player_id: int) -> Action:
        obs = self._view.observe(state, player_id)
        mask = action_mask(state, player_id)
        if self._hand_is_br is None:               # first decision of a hand:
            self._hand_is_br = self._play_br_this_hand()   # flip sigma ONCE
            self._flips += 1                       # per hand (design note 2)
        if self._hand_is_br:
            idx = self._epsilon_greedy(state, player_id, mask)
        else:
            idx, _logp = self.net.act(obs, mask)     # sampled; never argmax
        if self.record:
            self._steps.append((obs, mask, idx))
        return decode(state, player_id, idx)

    def on_hand_end(self, result, my_seat: int) -> None:
        if self.record and self._steps:
            flag = bool(self._hand_is_br)
            for obs, mask, idx in self._steps:
                self.buffer.add((obs, mask, idx, flag))
        self._steps.clear()
        self._hand_is_br = None
        self._view._equity_cache.clear()             # hand boundary, as in PPO
        self._hands_since_fit += 1
        if (self.config.fit_every
                and self._hands_since_fit >= self.config.fit_every
                and len(self.buffer) >= self.config.batch_size):
            self.fit()
            self._hands_since_fit = 0

    def reset_models(self) -> None:
        self._view.reset_models()

    def clone_for_evaluation(self) -> "NFSPStrategy":
        """A non-recording twin sharing net, buffer and view state — for
        eval seats and opponents, exactly RLStrategy.clone's role in PPO.

        fit_every is forced to 0: a twin that reaches the fit threshold
        during an evaluation session would mutate the shared net mid-
        scoring, and the eval numbers would describe an agent that no
        longer exists. Evaluation must be a pure read of the checkpoint."""
        from dataclasses import replace
        twin = NFSPStrategy.__new__(NFSPStrategy)
        twin.config = replace(self.config, fit_every=0)
        twin.rng = self.rng
        twin.q_table = self.q_table
        twin.net = self.net
        twin.buffer = self.buffer
        twin.record = False
        twin._steps = []
        twin._hand_is_br = None
        twin._flips = self._flips
        twin._hands_since_fit = self._hands_since_fit
        twin.last_fit_loss = self.last_fit_loss
        twin._view = self._view            # shared: models advance together
        return twin

    # -- the learners' seams ------------------------------------------------

    def record_episode(self, ep: Episode, br_mode: bool = True) -> None:
        """Ladder B: the outer loop harvested a PPO best response's hand;
        its steps are behaviour rows M_SL exists to hold. The reservoir's
        unit is the step, so one call fans the episode out."""
        for t in range(len(ep.actions)):
            self.buffer.add((ep.obs[t], ep.masks[t], int(ep.actions[t]), br_mode))

    def fit(self, **overrides) -> float:
        """Supervised fit of Pi on the reservoir; returns the final epoch loss.

        Row selection is the design note 2 contract: BR rows are the
        behaviour; fitting on Pi's own rows (br_mode=False) merely re-learns
        the current policy and stalls the fictitious average, so they are
        used ONLY when no BR rows exist yet — the documented ladder-B
        bootstrap (and the epsilon-mode degenerate case), never mixed in.
        """
        cfg = {**vars(self.config), **overrides}
        fit_rows = select_fit_rows(self.buffer.contents())
        if not fit_rows:
            raise ValueError("cannot fit an empty reservoir")
        obs = np.stack([r[0] for r in fit_rows])
        masks = np.stack([r[1] for r in fit_rows])
        acts = np.array([r[2] for r in fit_rows])
        import torch
        losses = fit_avg_policy(self.net, obs, masks, acts,
                                epochs=cfg["epochs"],
                                batch_size=cfg["batch_size"], lr=cfg["lr"],
                                generator=torch.Generator().manual_seed(
                                    self.rng.randrange(1 << 30)))
        self.last_fit_loss = losses[-1]
        return losses[-1]

    # -- internals -----------------------------------------------------------

    def _play_br_this_hand(self) -> bool:
        """The per-hand sigma coin flip (design note 2). Ladder B never
        flips: its BR data arrives via record_episode from the oracle."""
        if self.config.behaviour != "epsilon":
            return False
        if self.q_table is None:
            return True      # epsilon mode without a Q: the behaviour IS
                             # uniform-random (still a behaviour -> still BR)
        return self.rng.random() < self.config.eta

    def _epsilon(self) -> float:
        """eps ∝ 1/sqrt(rows seen), the paper's schedule, floored at
        eps_min."""
        t = max(self.buffer.seen, 1)
        return max(self.config.eps_min, self.config.eps_start / math.sqrt(t))

    def _epsilon_greedy(self, state, player_id: int,
                        mask: np.ndarray | None = None) -> int:
        if mask is None:
            mask = action_mask(state, player_id)
        legal = np.flatnonzero(mask).tolist()
        if self.q_table is None:
            return int(self.rng.choice(legal))
        row = self.q_table.get(_state_key(state, player_id))
        if row is None or self.rng.random() < self._epsilon():
            return int(self.rng.choice(legal))
        vals = np.where(mask, np.asarray(row, dtype=np.float64)[:len(mask)],
                        -np.inf)
        best = np.flatnonzero(vals == vals.max()).tolist()
        return int(self.rng.choice(best))
