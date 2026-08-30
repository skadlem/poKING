"""CLI-level guards for train_rl.py. The training loop itself is exercised by
running it; these cover the argument parsing that silently mistrains if wrong.
"""
import pytest

torch = pytest.importorskip("torch")

import train_rl  # noqa: E402


@pytest.mark.parametrize("spec,expected", [
    ("6", [6]),
    ("2,6", [2, 6]),
    (" 2 , 6 , 9 ", [2, 6, 9]),
    ("2,", [2]),
])
def test_parse_seats(spec, expected):
    assert train_rl.parse_seats(spec) == expected


@pytest.mark.parametrize("spec", ["", "1", "0,6", "-2", "6,1", "abc"])
def test_parse_seats_rejects_impossible_tables(spec):
    with pytest.raises(ValueError):
        train_rl.parse_seats(spec)


def test_bad_seats_spec_exits_nonzero(capsys):
    assert train_rl.main(["--seats", "1", "--iters", "1"]) == 2
    assert "error" in capsys.readouterr().out


def test_zero_opponent_mc_iters_is_rejected(capsys):
    """PokerBot divides by its iteration count, so 0 is a ZeroDivisionError the
    engine's sandbox would swallow into a silent fold on every decision."""
    assert train_rl.main(["--opp-mc-iters", "0", "--iters", "1"]) == 2
    assert "opp-mc-iters" in capsys.readouterr().out


def test_unknown_pool_entry_is_rejected(capsys):
    assert train_rl.main(["--pool", "cs,nope", "--iters", "1"]) == 2
    assert "unknown pool" in capsys.readouterr().out


def test_opponents_train_at_pokerbots_own_strength():
    """Training against a weakened heuristic (opp mc_iters=10) produced an agent
    that drew with the weak version and lost 225 bb/100 to the real one. The
    default must track PokerBot's own, not drift back down."""
    import inspect
    from pokr.bot import PokerBot
    pokerbot_default = inspect.signature(PokerBot.__init__).parameters["mc_iters"].default
    assert train_rl.build_parser().get_default("opp_mc_iters") == pokerbot_default


def test_seats_default_includes_heads_up_and_ring():
    assert train_rl.parse_seats(train_rl.build_parser().get_default("seats")) == [2, 6]


# -- league wiring --------------------------------------------------------


def test_league_is_in_the_default_pool():
    assert "league" in train_rl.build_parser().get_default("pool")
    assert train_rl.build_parser().get_default("league_every") > 0


def test_league_in_pool_without_snapshots_is_rejected(capsys):
    assert train_rl.main(["--pool", "cs,league", "--league-every", "0",
                          "--iters", "1"]) == 2
    assert "league" in capsys.readouterr().out


def test_build_pool_falls_back_when_the_league_is_empty():
    """Before the first snapshot a league seat must still produce a real
    opponent, not None -- play_session would fail on a None factory."""
    import random
    from pokr.rl.league import League
    factories = train_rl.build_pool(["league", "league"], 10, False,
                                    League(), random.Random(0))
    assert len(factories) == 2
    assert all(callable(f) for f in factories)
    assert all(f(random.Random(1)) is not None for f in factories)


def test_build_pool_seats_a_frozen_self_once_populated():
    import random
    from pokr.rl.agent import RLStrategy
    from pokr.rl.league import League
    from pokr.rl.net import PolicyValueNet
    league = League()
    league.snapshot(PolicyValueNet())
    factory, = train_rl.build_pool(["league"], 10, False, league,
                                   random.Random(0), agent_mc_iters=30)
    opponent = factory(random.Random(1))
    assert isinstance(opponent, RLStrategy)
    # a frozen self must see observations encoded the way it was trained
    assert opponent.mc_iters == 30
    assert not opponent.record


def test_unknown_pool_entry_still_rejected_with_league_present(capsys):
    assert train_rl.main(["--pool", "league,bogus", "--iters", "1"]) == 2
    assert "unknown pool" in capsys.readouterr().out
