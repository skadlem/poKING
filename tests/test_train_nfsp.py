"""train_nfsp.py smoke: the ladder-B loop must actually run — harvest fills
M_SL with behaviour rows, fit consumes them, checkpoints land.

The gate is cheap (1 round, 8 PPO iters, 40 hands): correctness of the
wiring, not strength of the agent. Strength is step 10's exploit probe and
gets measured on a real run, never asserted here.

Campaign #1 (uniform reservoir, no burn-in, global-iteration weights)
produced a Pi whose exploitability measured WORSE than random play would
be and whose fit loss sat next to the coin-flip floor. These tests pin the
three properties that were absent and are the whole point of the redesign:
burn-in discards the oracle's random openings, weights are per-ROUND and
linear, and the reservoir is actually OVERFLOWED by weight mass — A-Res
without overflow silently reduces to keeping everything.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import train_nfsp
from pokr.opponents import RandomBot
from pokr.rl.avg_policy import load as load_pi
from pokr.rl.exploit import best_response
from pokr.rl.ppo import PPOConfig


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    # capacity 300 vs ~700 items/round GUARANTEES the A-Res overflow the
    # weighting needs (item count, not weight mass — the class docstring's
    # documented trap); burn_in 0.75 of 8 iters keeps only iters 6-7.
    ckpt = tmp_path_factory.mktemp("nfsp")
    stats = train_nfsp.train(
        rounds=2, iters_per_round=8, hands_per_iter=40,
        capacity=300, burn_in=0.75,
        fit_epochs=3, fit_batch=32, lr=3e-3, max_fit_rows=10_000,
        hidden=(32, 32), ckpt_dir=str(ckpt), seed=11, quiet=True,
        br_cfg=PPOConfig(minibatch=32))
    return stats, ckpt


def test_two_rounds_produce_round_stats_and_a_checkpoint(smoke_run):
    stats, ckpt = smoke_run
    assert len(stats) == 2
    for st in stats:
        assert st.rows > 0
        assert np.isfinite(st.fit_loss), f"CE loss not finite: {st.fit_loss}"
    assert (ckpt / "pi_last.pt").exists()
    assert stats[1].seen > stats[0].seen, "weight mass must accumulate"


def test_burn_in_discards_and_round_weights_accumulate(smoke_run):
    """Weight mass (seen) must exceed row count — if every row had weight
    1.0 the burn-in/round-weight wiring silently vanished. Round 2 rows
    carry weight 2, so mass is strictly more than items even after
    burn-in discarded 6 of every 8 iterations."""
    stats, _ckpt = smoke_run
    assert stats[1].seen > stats[1].rows


def test_reservoir_is_in_the_overflowing_regime(smoke_run):
    """A-Res proportional inclusion only bites once the ITEM stream
    overflows capacity; an underfilled reservoir keeps everything and
    weights are inert. Capacity 300 vs ~700 items/round must saturate:
    rows == capacity exactly, or the smoke run measured the wrong regime.
    The real run (250k cap, ~150k items/round x 30) overflows 18x."""
    stats, _ckpt = smoke_run
    assert stats[1].rows == 300, (
        f"reservoir held {stats[1].rows}: not saturated, weights inert")


def test_burn_in_validation():
    with pytest.raises(ValueError, match="burn_in"):
        train_nfsp.train(rounds=1, burn_in=1.0)


def test_checkpoint_roundtrips_through_the_plugin_path(smoke_run):
    """The exact load path TrainedNFSPStrategy._load takes: avg_policy.load
    rebuilds the net from net_config alone."""
    _stats, ckpt = smoke_run
    net, meta = load_pi(str(ckpt / "pi_last.pt"))
    assert meta["round"] == 1
    assert net.obs_dim == 176 and net.hidden == (32, 32)
    obs = np.zeros(net.obs_dim, dtype=np.float32)
    mask = np.zeros(9, dtype=bool)
    mask[[0, 2]] = True
    a, _ = net.act(obs, mask)
    assert a in (0, 2)


def test_probe_with_eval_hands_zero_is_loud_not_zero():
    """The NFSP round loop passes eval_hands=0; the probe must answer nan,
    never 0.0 — the retraction this repo already lived through came from a
    clamped small number reading as good news."""
    rep = best_response(lambda rng: RandomBot(), "rnd", iterations=1,
                        hands_per_iter=10, seed=3, eval_hands=0)
    assert math.isnan(rep.bb_per_100)
    assert not rep.converged, "a nan eval must never read as converged"
    assert "PROBE FAILED" in rep.format()


def test_default_probe_path_untouched():
    """Regression guard for every number ever measured with best_response:
    no harvest, default eval — the report must be a normal resolved probe."""
    rep = best_response(lambda rng: RandomBot(), "rnd", iterations=1,
                        hands_per_iter=50, seed=3, eval_hands=100)
    assert not math.isnan(rep.bb_per_100)
    assert rep.eval_hands == 100 and len(rep.curve) == 1


def test_harvest_hook_receives_the_iteration_index():
    seen_its = []
    best_response(
        lambda rng: RandomBot(), "rnd", iterations=3, hands_per_iter=20,
        seed=4, eval_hands=0, cfg=PPOConfig(minibatch=32),
        harvest=lambda eps, it: seen_its.append(it))
    assert seen_its == [0, 1, 2]
