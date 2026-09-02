"""The league's one load-bearing property: snapshots must be frozen copies.

A league holding references to the live network would silently degrade into
live self-play -- and would still run, still log, and still look like it was
working -- so the isolation is asserted directly.
"""
import random

import pytest

torch = pytest.importorskip("torch")

from pokr.rl.agent import RLStrategy  # noqa: E402
from pokr.rl.encode import OBS_DIM  # noqa: E402
from pokr.rl.league import League  # noqa: E402
from pokr.rl.net import PolicyValueNet  # noqa: E402


def logits(net, obs_dim):
    with torch.no_grad():
        mask = torch.ones(1, net.num_actions, dtype=torch.bool)
        return net.distribution(torch.ones(1, obs_dim), mask)[0].probs.clone()


def test_snapshot_is_isolated_from_later_training():
    torch.manual_seed(0)
    live = PolicyValueNet(obs_dim=16, hidden=(8,))
    league = League()
    league.snapshot(live)
    frozen = league.sample(random.Random(0))
    before = logits(frozen, 16)

    with torch.no_grad():                      # simulate an optimizer step
        for p in live.parameters():
            p.add_(torch.randn_like(p))

    assert not torch.allclose(logits(live, 16), before), "live net should have moved"
    assert torch.allclose(logits(frozen, 16), before), "snapshot tracked the live net"


def test_snapshot_requires_no_grad():
    league = League()
    league.snapshot(PolicyValueNet(obs_dim=16, hidden=(8,)))
    frozen = league.sample(random.Random(0))
    assert not any(p.requires_grad for p in frozen.parameters())
    assert not frozen.training


def test_empty_league_yields_no_opponent():
    league = League()
    assert len(league) == 0
    assert league.sample(random.Random(0)) is None
    assert league.opponent_factory(random.Random(0)) is None


def test_max_size_evicts_oldest():
    league = League(max_size=3)
    for _ in range(5):
        league.snapshot(PolicyValueNet(obs_dim=16, hidden=(8,)))
    assert len(league) == 3
    assert league.snapshots_taken == 5


def test_opponent_factory_builds_a_frozen_non_recording_agent():
    # a real obs_dim, unlike the stubs above: this test actually seats an
    # RLStrategy, which reads its observation layout off the net's obs_dim
    league = League()
    league.snapshot(PolicyValueNet(obs_dim=OBS_DIM, hidden=(8,)))
    factory = league.opponent_factory(random.Random(0), num_players=6)
    agent = factory(random.Random(1))
    assert isinstance(agent, RLStrategy)
    assert agent.record is False, "a league opponent must not collect training data"
    assert agent.greedy is False, "a deterministic opponent is trivially exploitable"


def test_factory_pins_one_snapshot_for_the_whole_session():
    """Chosen once at factory build, not per hand, so a session's opponent
    models describe a single consistent opponent."""
    league = League()
    for _ in range(4):
        league.snapshot(PolicyValueNet(obs_dim=OBS_DIM, hidden=(8,)))
    factory = league.opponent_factory(random.Random(3))
    nets = {id(factory(random.Random(i)).net) for i in range(5)}
    assert len(nets) == 1


# -- checkpoint persistence -----------------------------------------------


def test_state_restore_round_trip_preserves_behaviour():
    """--resume used to restart with an empty league, silently discarding the
    change that most improved the agent."""
    torch.manual_seed(0)
    config = {"obs_dim": 16, "num_actions": 9, "hidden": (8,)}
    league = League()
    for _ in range(3):
        league.snapshot(PolicyValueNet(**config))
    before = [logits(n, 16) for n in [league._nets[i] for i in range(3)]]

    revived = League()
    revived.restore(league.state(), config)

    assert len(revived) == 3
    after = [logits(revived._nets[i], 16) for i in range(3)]
    for a, b in zip(before, after):
        assert torch.allclose(a, b)
    assert not any(p.requires_grad for n in revived._nets for p in n.parameters())


def test_state_is_detached_from_the_live_snapshots():
    league = League()
    net = PolicyValueNet(obs_dim=16, hidden=(8,))
    league.snapshot(net)
    saved = league.state()
    with torch.no_grad():
        for p in league._nets[0].parameters():
            p.add_(1.0)
    revived = League()
    revived.restore(saved, {"obs_dim": 16, "num_actions": 9, "hidden": (8,)})
    assert not torch.allclose(logits(revived._nets[0], 16), logits(league._nets[0], 16))


def test_restore_into_a_populated_league_replaces_it():
    config = {"obs_dim": 16, "num_actions": 9, "hidden": (8,)}
    league = League()
    for _ in range(4):
        league.snapshot(PolicyValueNet(**config))
    other = League()
    other.snapshot(PolicyValueNet(**config))
    league.restore(other.state(), config)
    assert len(league) == 1
