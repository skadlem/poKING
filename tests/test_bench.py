import random

from pokr.bench import (
    MatchupReport,
    calling_station_factory,
    leak_hunter_factory,
    random_factory,
    run_benchmark,
    run_matchup,
)
from pokr.bot import PokerBot
from pokr.opponents import CallingStation


def test_deterministic_reports():
    r1 = run_matchup(PokerBot(random.Random(1)), calling_station_factory, 200, seed=42)
    r2 = run_matchup(PokerBot(random.Random(1)), calling_station_factory, 200, seed=42)
    assert r1 == r2


def test_selfplay_is_symmetric():
    r = run_matchup(PokerBot(random.Random(3)), lambda rng: PokerBot(rng), 400, seed=7)
    # no rake: seat 0's total should be within 4 std of 0 over 400 hands
    std_total = 20 * (r.var_bb_per_100 ** 0.5) if r.var_bb_per_100 > 0 else 0.0
    assert abs(r.total_bb) <= 4 * std_total


def test_report_math():
    rep = MatchupReport("x", hands=100, total_bb=50.0, bb_per_100=50.0, win_rate=0.5, var_bb_per_100=1.0)
    assert rep.bb_per_100 == 50.0
    assert rep.hands == 100


def test_run_benchmark_includes_self_and_leak_hunter():
    reports = run_benchmark(PokerBot(random.Random(5)), num_hands=50, seed=1)
    names = [r.name for r in reports]
    assert any("self" in n.lower() for n in names)
    assert any("leak" in n.lower() for n in names)


def test_seat_rotation_dealer_cycles():
    # structural check: with num_seats=6, dealer position h % 6 covers all seats
    from pokr.engine import PokerGame
    seen = set()
    bot = PokerBot(random.Random(7))
    for h in range(6):
        g = PokerGame([bot] + [CallingStation() for _ in range(5)], [200] * 6,
                      rng=random.Random(h), initial_dealer=h % 6)
        seen.add(g.initial_dealer)
    assert seen == {0, 1, 2, 3, 4, 5}
