"""The gate (design note 6): an equilibrium algorithm must converge on a
game with a KNOWN answer before it is ever pointed at NLHE.

Kuhn exploitability is exact (max over 64 pure strategies per player), so
every assertion here is about the implementation, not the metric:

- tabular_cfr: ground-truth CFR. If this fails, the harness is broken.
- neural_avg_cfr: the step 4+5 pipeline — WeightedReservoir harvest ->
  CE fit of AvgPolicyNet — measured against the same exact metric. Its
  net must (a) clear the 0.05 bar, (b) match the tabular average it is a
  sampled version of, and (c) not be fooling itself: the fit's CE loss
  must sit at the entropy of the averaged behaviour, the floor no policy
  can beat (measured: loss 0.2641 vs floor 0.2634). That check is what
  distinguishes 'converged' from 'underfit and got lucky on the metric'.

Statistical bars were calibrated on the real sampler before being trusted
(same discipline as test_memory's chi-square): the uniform-weight null
runs chi2 ~79 ±11 on 99 dof over 40 seeds; the weighted-overflow share
runs 0.760 ±0.034 against the 0.75 expectation for weights 1:3.
"""
import math
import random

import pytest

from pokr.rl.fsp import (
    GATE_EXPLOITABILITY,
    WeightedReservoir,
    neural_avg_cfr,
    net_strategy,
    regret_match,
    tabular_cfr,
)
from pokr.rl.kuhn import (
    BET,
    GAME_VALUE,
    INFO_SETS,
    expected_value,
    exploitability,
    nash,
    uniform,
)


# -- regret matching ---------------------------------------------------------


def test_regret_match_normalises_positive_part_only():
    assert regret_match([0.0, 0.0]) == (0.5, 0.5)          # no regrets: uniform
    assert regret_match([-5.0, -1.0]) == (0.5, 0.5)        # all negative: uniform
    assert regret_match([-2.0, 6.0]) == pytest.approx((0.0, 1.0))
    assert regret_match([2.0, 6.0]) == pytest.approx((0.25, 0.75))
    assert regret_match([1.0, -1.0]) == pytest.approx((1.0, 0.0))


# -- WeightedReservoir --------------------------------------------------------


def test_uniform_weights_reduce_to_algorithm_r_inclusion():
    """Chi-square over stream positions, same shape as test_memory's gate:
    with every weight 1 the weighted sampler must BE Algorithm R
    (P(kept) = cap/n for all positions, including the first cap).
    Bar: empirical null over 40 seeds is 79 ±11 (max 100); 200 is >10
    sigma of a sound sampler and far under what a broken reduction scores."""
    cap, n, trials = 20, 100, 3000
    hits = [0] * n
    rng = random.Random(0)
    for _ in range(trials):
        res = WeightedReservoir(cap, random.Random(rng.randrange(1 << 40)))
        for i in range(n):
            res.add(i, 1.0)
        for i in res.contents():
            hits[i] += 1
    assert sum(hits) == cap * trials
    e = cap * trials / n
    stat = sum((h - e) ** 2 / e for h in hits)
    assert stat < 200, f"chi2={stat:.0f} on 99 dof, calibrated null 79±11"


def test_weights_bite_only_once_the_stream_overflows():
    """Documented asymmetry, asserted both ways so nobody 'fixes' it by
    accident: a reservoir that never fills keeps everything (weights
    inert); overflowing, kept share follows the weights (1:3 over equal
    arrival counts -> 0.75; calibrated: 0.760 ±0.034)."""
    cap, trials = 10, 2000
    keeps = 0
    rng = random.Random(1)
    for _ in range(trials):
        res = WeightedReservoir(cap, random.Random(rng.randrange(1 << 40)))
        res.add("light", 1.0)
        res.add("heavy", 100.0)
        keeps += "light" in res.contents()
    assert keeps == trials                                   # underfilled: no decay

    shares = []
    rng = random.Random(2)
    for _ in range(200):
        res = WeightedReservoir(cap, random.Random(rng.randrange(1 << 40)))
        for i in range(1000):
            res.add("L" if i % 2 == 0 else "H", 1.0 if i % 2 == 0 else 3.0)
        c = res.contents()
        shares.append(c.count("H") / len(c))
    m = sum(shares) / len(shares)
    sd = math.sqrt(sum((x - m) ** 2 for x in shares) / (len(shares) - 1))
    tol = 5 * sd / math.sqrt(len(shares))
    assert abs(m - 0.75) <= tol + 0.02, f"heavy share {m:.3f} vs 0.75±{tol:.3f}"


def test_zero_weight_never_enters_and_length_respects_capacity():
    rng = random.Random(3)
    res = WeightedReservoir(4, rng)
    for _ in range(50):
        res.add("ghost", 0.0)
    assert res.contents() == [] and len(res) == 0 and res.seen == 50
    for i in range(10):
        res.add(i, 1.0)
    assert len(res) == 4
    with pytest.raises(ValueError):
        res.add("neg", -1.0)
    with pytest.raises(ValueError):
        WeightedReservoir(0)


# -- the tabular reference ----------------------------------------------------


def test_tabular_cfr_converges_and_the_harness_agrees_with_itself():
    """The traversal is validated twice at once: exploitability falls, and
    the average strategy's EV approaches the analytic game value. A
    reach-weighting bug moves one of those two."""
    assert exploitability(uniform()) > 0.4          # the baseline to beat
    early = tabular_cfr(50)
    late = tabular_cfr(400)
    assert exploitability(early) < exploitability(uniform()) / 2
    assert exploitability(late) < GATE_EXPLOITABILITY
    assert abs(expected_value(late) - GAME_VALUE) < 0.005
    assert exploitability(nash()) < 1e-12           # the metric itself: exact


# -- the neural gate (the thing this roadmap step exists for) ------------------


@pytest.fixture(scope="module")
def gate_run():
    """1000 iterations: the size calibrated before the bars below were
    trusted (net expl 0.0096, table 0.0132, fit loss 0.2641)."""
    pytest.importorskip("torch")
    return neural_avg_cfr(1000, seed=0)


def test_neural_gate_clears_the_bar(gate_run):
    net, table, stats = gate_run
    assert exploitability(table) < GATE_EXPLOITABILITY
    assert exploitability(net_strategy(net)) < GATE_EXPLOITABILITY
    assert stats["seen"] > 2 * stats["kept"], "the weighting must be in the " \
        "overflowing regime, or the sampler silently degenerates to counts"


def test_neural_average_matches_the_tabular_average(gate_run):
    """Same regret dynamics, same linear weights: the net's table must sit
    on the closed-form average at every info set within sampler + fitter
    slack. 0.05 per slot is not the gate — the GATE is exploitability; this
    is the diagnostic that says WHERE a failure lives."""
    net, table, _ = gate_run
    ns = net_strategy(net)
    drift = {k: abs(ns[k][BET] - table[k][BET]) for k in INFO_SETS}
    worst = max(drift.values())
    assert worst < 0.05, f"worst per-set drift {worst:.3f}: {drift}"


def test_fit_loss_sits_on_the_entropy_floor(gate_run):
    """Distinguish 'the pipeline converged' from 'the fitter underfit and
    the reservoir was generous': CE loss cannot go below the per-state
    entropy of the behaviour it is fitting, and ending far above it means
    the fit is unfinished. Floor computed from the returned table, not
    hardcoded; measured gap at the calibrated run: 0.0007 nats."""
    net, table, stats = gate_run

    def entropy(p):
        return -sum(x * math.log(x) for x in p if x > 0.0)

    floor = sum(entropy(table[k]) for k in INFO_SETS) / len(INFO_SETS)
    assert stats["final_loss"] > floor - 0.01, \
        "loss BELOW the behaviour's entropy means the sampler collapsed " \
        "(fewer distinct rows than believed: an off-by-one made it memorise)"
    assert stats["final_loss"] < floor + 0.05, \
        f"loss {stats['final_loss']:.4f} vs floor {floor:.4f}: underfit"


def test_the_gate_actually_gates():
    """Negative control: the same harness on a policy that did NOT run the
    pipeline must fail the bar loudly, or the bar measures nothing."""
    assert exploitability(uniform()) > GATE_EXPLOITABILITY * 5
