"""Correctness gate for the multiprocess rollout collector (pokr/rl/rollout.py).

Kept fast on purpose (mc_iters=0, small hand counts, no-net random-policy
opponents) so the fork pool this exercises stays a couple hundred ms per test
rather than seconds -- see the module docstring in rollout.py for why fork
was chosen and what each worker must reseed.
"""
from __future__ import annotations

import numpy as np
import pytest

from pokr.rl.encode import NUM_ACTIONS, OBS_DIM
from pokr.rl.rollout import collect_parallel

OPPONENTS = ["cs", "tag", "maniac", "random", "cs"]  # 5 names for a 6-max table


def _assert_well_formed(episodes):
    assert len(episodes) > 0
    for e in episodes:
        t = len(e.actions)
        assert t > 0
        assert e.obs.shape == (t, OBS_DIM) and e.obs.dtype == np.float32
        assert e.masks.shape == (t, NUM_ACTIONS) and e.masks.dtype == np.bool_
        assert e.logps.shape == (t,) and e.values.shape == (t,)
        # every recorded action must have been legal under its own mask
        assert e.masks[np.arange(t), e.actions].all()
        assert np.isfinite(e.obs).all()


def test_workers_1_returns_well_formed_episodes():
    episodes, per_hand_bb = collect_parallel(
        net=None, opponent_names=OPPONENTS, num_hands=100, seed=1, workers=1)
    _assert_well_formed(episodes)
    assert len(per_hand_bb) == 100
    assert len(episodes) <= 100  # one episode per hand seat-0 actually acted in


def test_workers_2_collects_roughly_the_requested_hands():
    episodes, per_hand_bb = collect_parallel(
        net=None, opponent_names=OPPONENTS, num_hands=200, seed=2, workers=2)
    _assert_well_formed(episodes)
    assert len(per_hand_bb) == 200


def test_same_seed_and_worker_count_is_reproducible():
    a_eps, a_bb = collect_parallel(
        net=None, opponent_names=OPPONENTS, num_hands=150, seed=42, workers=2)
    b_eps, b_bb = collect_parallel(
        net=None, opponent_names=OPPONENTS, num_hands=150, seed=42, workers=2)
    assert a_bb == pytest.approx(b_bb)
    assert len(a_eps) == len(b_eps)
    for ea, eb in zip(a_eps, b_eps):
        assert ea.reward == pytest.approx(eb.reward)
        assert np.array_equal(ea.actions, eb.actions)
        assert np.array_equal(ea.obs, eb.obs)


def test_different_worker_counts_need_not_match():
    """Documented, expected divergence: workers changes how hands are split
    and in what order they're dealt, so the collected set differs -- this
    test just pins that workers=1 and workers=2 are NOT required to agree,
    as a guard against someone "fixing" that later under a mistaken belief
    the outputs must be worker-count-invariant."""
    eps1, bb1 = collect_parallel(
        net=None, opponent_names=OPPONENTS, num_hands=100, seed=9, workers=1)
    eps2, bb2 = collect_parallel(
        net=None, opponent_names=OPPONENTS, num_hands=100, seed=9, workers=2)
    _assert_well_formed(eps1)
    _assert_well_formed(eps2)
    assert len(bb1) == len(bb2) == 100


def test_workers_1_is_single_process_and_matches_direct_play_session():
    """workers=1 must be the same code path as calling play_session directly
    (no multiprocessing involved), so it should reproduce bit-for-bit what
    the existing single-process training loop already gets."""
    import random

    from pokr.bench import calling_station_factory, maniac_factory, random_factory, tight_aggressive_factory
    from pokr.rl.agent import RLStrategy
    from pokr.bench import play_session

    factories = [calling_station_factory, tight_aggressive_factory, maniac_factory,
                random_factory, calling_station_factory]
    agent = RLStrategy(net=None, rng=random.Random(5), num_players=6, record=True)
    direct_bb, _ = play_session(agent, factories, 80, 5, num_seats=6)

    eps, bb = collect_parallel(net=None, opponent_names=OPPONENTS, num_hands=80,
                               seed=5, workers=1)
    assert bb == pytest.approx(direct_bb)
    assert len(eps) == len(agent.buffer.episodes)
    for a, b in zip(eps, agent.buffer.episodes):
        assert a.reward == pytest.approx(b.reward)
        assert np.array_equal(a.actions, b.actions)


@pytest.mark.parametrize("num_hands,workers", [(7, 3), (10, 4), (1, 5)])
def test_remainder_is_not_dropped(num_hands, workers):
    """An odd hand count split across workers must still sum to num_hands --
    no hands silently lost to integer-division truncation."""
    _episodes, per_hand_bb = collect_parallel(
        net=None, opponent_names=OPPONENTS, num_hands=num_hands, seed=3, workers=workers)
    assert len(per_hand_bb) == num_hands


def test_unknown_opponent_name_raises_before_any_worker_spawns():
    with pytest.raises(ValueError, match="unknown opponent name"):
        collect_parallel(net=None, opponent_names=["cs", "tag", "bogus", "random", "cs"],
                         num_hands=10, seed=1, workers=3)


def test_wrong_number_of_opponent_names_raises():
    with pytest.raises(ValueError, match="opponent names"):
        collect_parallel(net=None, opponent_names=["cs", "tag"], num_hands=10, seed=1,
                         workers=1, num_seats=6)


def test_torch_net_round_trips_through_workers():
    """A real PolicyValueNet must survive the state_dict/config round trip
    into a forked worker and produce well-formed, legal-under-mask episodes."""
    torch = pytest.importorskip("torch")
    from pokr.rl.net import PolicyValueNet
    torch.manual_seed(0)
    net = PolicyValueNet(hidden=(16, 16))
    episodes, per_hand_bb = collect_parallel(
        net=net, opponent_names=OPPONENTS, num_hands=40, seed=11, workers=2)
    _assert_well_formed(episodes)
    assert len(per_hand_bb) == 40


def test_invalid_workers_and_num_hands_raise():
    with pytest.raises(ValueError):
        collect_parallel(net=None, opponent_names=OPPONENTS, num_hands=10, seed=1, workers=0)
    with pytest.raises(ValueError):
        collect_parallel(net=None, opponent_names=OPPONENTS, num_hands=0, seed=1, workers=1)
