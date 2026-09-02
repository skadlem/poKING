"""FSP on Kuhn: the gate (design note 6) an equilibrium algorithm passes
before it sees NLHE. torch-free at import; the neural variant lazy-imports
torch inside its function, same contract as plugin.py.

Kuhn's exploitability (`kuhn.exploitability`) is EXACT — a maximum over 64
pure strategies, nothing to converge — so "the average reached < 0.05" here
is a correctness fact about the implementation, not a noisy reading. Two
variants, deliberately testing different things:

- `tabular_cfr`: ground truth. Vanilla CFR with full exact traversal and
  linear averaging. If THIS one failed, the harness itself would be broken.

- `neural_avg_cfr`: the load-bearing one for NFSP. Regrets stay tabular
  (they are not what the gate is about), but the AVERAGE STRATEGY becomes
  `AvgPolicyNet` fitted by cross-entropy on a `WeightedReservoir` of
  behaviour rows — exactly the pipeline steps 4+5 built and steps 7+9 will
  run on NLHE: sampler -> weighted inclusion -> CE recovers the behaviour
  average. On Kuhn every info set is one observation, so a good fit
  reproduces the reservoir's empirical action frequencies; the gap between
  the neural policy and the tabular average IS the sampler's and the
  fitter's error, measured against exact exploitability. An off-by-one in
  the reservoir cannot hide.

Weighted reservoir sampling (Efraimidis & Spirakis A-Res) lives here
because it is the linear-averaging half of the gate; `memory.py` stays the
pure uniform/floored pair the design note specified.
"""
from __future__ import annotations

import heapq
import math
import random

from .kuhn import (
    BET,
    DEALS,
    INFO_SETS,
    INFO_SETS_BY_PLAYER,
    PASS,
    info_set,
    is_terminal,
    player_to_act,
    terminal_payoff,
)

GATE_EXPLOITABILITY = 0.05   # design note 6: the number that must come down


class WeightedReservoir:
    """A-Res: reservoir sampling where item i with weight w_i is included
    with (asymptotically) probability proportional to w_i; uniform weights
    reduce it to Algorithm R — asserted in the tests.

    Linear weighting over iterations is what makes the reservoir the
    *linearly*-weighted average of behaviours, the standard CFR averaging
    that converges faster than a plain uniform reservoir, which leaves the
    near-random early iterations in at full weight.

    Keys live in log space: the A-Res key is U^(1/w), monotone in log(U)/w,
    which stays sane for tiny weights (key -> -inf, never enters). A zero
    weight is skipped outright rather than drawn. The heap keeps the weakest
    item on top, so an add is O(log capacity), not a re-sort.

    NOTE on the fixed-size variant: proportional inclusion is asymptotic,
    so the stream must overflow the capacity for the weighting to bite —
    a reservoir that never fills keeps every row once and reproduces row
    COUNTS, not weights. `neural_avg_cfr` sizes the default capacity to
    overflow; tests check both regimes and say which they are in.
    """

    def __init__(self, capacity: int, rng: random.Random | None = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.rng = rng or random.Random()
        self._heap: list[tuple[float, int, object]] = []
        self._seq = 0
        self._seen = 0
        # sum of weights ever added; compared against capacity by callers
        # that must not overfill (A-Res inclusion is only proportional once
        # the stream overflows the capacity — see class docstring).
        self.weight_sum = 0.0

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def seen(self) -> int:
        return self._seen

    def add(self, item, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError(f"negative weight {weight}")
        self._seen += 1
        self.weight_sum += weight
        if weight == 0:
            return                                   # zero weight never enters
        u = self.rng.random()
        while u == 0.0:                              # log(0) = -inf key
            u = self.rng.random()
        key = math.log(u) / weight
        entry = (key, self._seq, item)
        self._seq += 1
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
        elif key > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def contents(self) -> list:
        return [item for _, _, item in self._heap]


# -- regret matching and the exact tabular reference ------------------------


def regret_match(regrets: list[float]) -> tuple[float, float]:
    """sigma(a) ∝ max(R(a), 0); uniform when no positive regret."""
    pos = [r if r > 0.0 else 0.0 for r in regrets]
    s = pos[PASS] + pos[BET]
    if s <= 0.0:
        return (0.5, 0.5)
    return (pos[PASS] / s, pos[BET] / s)


def _strategy_tables(regrets: dict) -> dict:
    return {p: {key: regret_match(regrets[p][key])
                for key in INFO_SETS_BY_PLAYER[p]} for p in (0, 1)}


def _cfr_walk(history: str, cards: tuple[int, int], regrets: dict,
              strategy: dict, cf_reach: tuple[float, float]) -> float:
    """One exact traversal: accumulate counterfactual regrets, return the
    expected payoff to player 0 from this history.

    cf_reach[p] is player p's counterfactual reach: deal probability times
    the OPPONENT's action probabilities along the path (never the acting
    player's own), which is what the regret update must be weighted by.
    """
    if is_terminal(history):
        return terminal_payoff(history, cards)
    p = player_to_act(history)
    key = info_set(history, cards)
    sigma = strategy[p][key]
    other = 1 - p
    child = {}
    for a in (PASS, BET):
        nxt = list(cf_reach)
        nxt[other] *= sigma[a]        # opponent's sigma folds into their reach
        child[a] = _cfr_walk(history + "pb"[a], cards, regrets, strategy,
                             (nxt[0], nxt[1]))
    node_ev = sigma[PASS] * child[PASS] + sigma[BET] * child[BET]
    sign = 1.0 if p == 0 else -1.0    # utilities are stated to player 0
    r = regrets[p][key]
    r[PASS] += cf_reach[p] * sign * (child[PASS] - node_ev)
    r[BET] += cf_reach[p] * sign * (child[BET] - node_ev)
    return node_ev


def _accumulate(avg_sum: dict, strategy: dict, weight: float) -> None:
    for p in (0, 1):
        for key, sigma in strategy[p].items():
            avg_sum[key][PASS] += weight * sigma[PASS]
            avg_sum[key][BET] += weight * sigma[BET]


def _normalize(avg_sum: dict) -> dict:
    return {key: (s[0] / (s[0] + s[1]), s[1] / (s[0] + s[1]))
            for key, s in avg_sum.items()}


def tabular_cfr(iters: int) -> dict:
    """Vanilla CFR on the full tree with LINEAR averaging (weight t at
    iteration t). Returns the average strategy as a kuhn.Strategy table.

    The traversal doubles as the harness's self-check: the average's EV
    must converge to kuhn.GAME_VALUE (asserted in the gate tests).
    """
    regrets = {p: {key: [0.0, 0.0] for key in keys}
               for p, keys in enumerate(INFO_SETS_BY_PLAYER)}
    avg_sum = {key: [0.0, 0.0] for key in INFO_SETS}
    deal_w = 1.0 / len(DEALS)
    for t in range(1, iters + 1):
        strategy = _strategy_tables(regrets)
        for cards in DEALS:
            _cfr_walk("", cards, regrets, strategy, (deal_w, deal_w))
        _accumulate(avg_sum, strategy, t)
    return _normalize(avg_sum)


# -- the neural path: reservoir-sampled, CE-fitted average ------------------


def _onehot_encoder():
    """Kuhn state = info-set key; one-hot over INFO_SETS, mask all-legal.
    Both actions are always legal in Kuhn, so the mask is [1, 1] — the gate
    does not exercise masking; avg_policy's own tests do."""
    idx = {key: i for i, key in enumerate(INFO_SETS)}
    dim = len(INFO_SETS)

    def row(key: str, action: int):
        import numpy as np
        obs = np.zeros(dim, dtype=np.float32)
        obs[idx[key]] = 1.0
        mask = np.ones(2, dtype=bool)
        return (obs, mask, action)

    return row, dim


def neural_avg_cfr(iters: int, *, capacity: int = 4000, epochs: int = 60,
                   batch: int = 256, lr: float = 5e-3, fit_every: int = 200,
                   seed: int = 0):
    """CFR whose average strategy is an AvgPolicyNet trained by masked CE
    on a weighted reservoir of (obs, mask, action) rows harvested from the
    behaviour at every iteration.

    The harvest adds one row per (t, key, a) labelled a with weight
    t * sigma_t(a), so a uniform draw from an overflowing reservoir
    reproduces the action frequencies of the linearly-weighted average
    behaviour — the same object tabular_cfr accumulates in closed form.
    Fitting CE on that sample is precisely the step 4+5 pipeline, and the
    capacity default (4000 << ~20 rows/iter * iters) keeps the stream
    overflowing so the weighting actually bites.

    Returns (net, table_avg, stats): the net is the deliverable, the table
    is the control it must match, stats carries the last fit's loss.
    """
    import numpy as np
    import torch

    from .avg_policy import AvgPolicyNet, fit_avg_policy

    rng = random.Random(seed)
    regrets = {p: {key: [0.0, 0.0] for key in keys}
               for p, keys in enumerate(INFO_SETS_BY_PLAYER)}
    avg_sum = {key: [0.0, 0.0] for key in INFO_SETS}
    reservoir = WeightedReservoir(capacity, rng)
    deal_w = 1.0 / len(DEALS)
    row, dim = _onehot_encoder()
    net = AvgPolicyNet(obs_dim=dim, num_actions=2, hidden=(32, 32))

    last_loss = float("nan")
    for t in range(1, iters + 1):
        strategy = _strategy_tables(regrets)
        for cards in DEALS:
            _cfr_walk("", cards, regrets, strategy, (deal_w, deal_w))
        _accumulate(avg_sum, strategy, t)
        for p in (0, 1):
            for key, sigma in strategy[p].items():
                for a in (PASS, BET):
                    if sigma[a] > 0.0:
                        reservoir.add(row(key, a), weight=t * sigma[a])
        if t % fit_every == 0:
            sample = reservoir.contents()
            obs = np.stack([r[0] for r in sample])
            masks = np.stack([r[1] for r in sample])
            acts = np.array([r[2] for r in sample])
            losses = fit_avg_policy(net, obs, masks, acts, epochs=epochs,
                                    batch_size=batch, lr=lr,
                                    generator=torch.Generator().manual_seed(
                                        rng.randrange(1 << 30)))
            last_loss = losses[-1]

    return net, _normalize(avg_sum), {
        "final_loss": last_loss, "kept": len(reservoir),
        "seen": reservoir.seen}


def net_strategy(net) -> dict:
    """Read a kuhn.Strategy table out of an AvgPolicyNet over the one-hot
    info-set encoding — the deployment view of the averaged policy."""
    import numpy as np

    keys = list(INFO_SETS)
    obs = np.eye(len(keys), dtype=np.float32)
    masks = np.ones((len(keys), 2), dtype=bool)
    probs = net.probs(obs, masks)
    return {key: (float(probs[i, PASS]), float(probs[i, BET]))
            for i, key in enumerate(keys)}
