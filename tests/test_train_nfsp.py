"""train_nfsp.py smoke: the ladder-B loop must actually run — harvest fills
M_SL with behaviour rows, fit consumes them, checkpoints land.

The gate is cheap (1 round, 2 PPO iters, 40 hands): correctness of the
wiring, not strength of the agent. Strength is step 10's exploit probe and
gets measured on a real run, never asserted here.
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
    ckpt = tmp_path_factory.mktemp("nfsp")
    stats = train_nfsp.train(
        rounds=1, iters_per_round=2, hands_per_iter=40,
        capacity=10_000, fit_epochs=3, fit_batch=32, lr=3e-3,
        hidden=(32, 32), ckpt_dir=str(ckpt), seed=11, quiet=True,
        br_cfg=PPOConfig(minibatch=32))
    return stats, ckpt


def test_one_round_produces_round_stats_and_a_checkpoint(smoke_run):
    stats, ckpt = smoke_run
    assert len(stats) == 1
    st = stats[0]
    assert st.rows > 0 and st.seen == st.rows, "harvest must fill M_SL"
    assert np.isfinite(st.fit_loss), f"CE loss not finite: {st.fit_loss}"
    assert (ckpt / "pi_last.pt").exists()


def test_checkpoint_roundtrips_through_the_plugin_path(smoke_run):
    """The exact load path TrainedNFSPStrategy._load takes: avg_policy.load
    rebuilds the net from net_config alone."""
    _stats, ckpt = smoke_run
    net, meta = load_pi(str(ckpt / "pi_last.pt"))
    assert meta["round"] == 0
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
