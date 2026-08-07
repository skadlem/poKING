import random

from pokr.bench import (
    MatchupReport,
    bot_own_stats,
    calling_station_factory,
    format_hand,
    leak_hunter_factory,
    maniac_factory,
    play_session,
    random_factory,
    run_benchmark,
    run_matchup,
    tight_aggressive_factory,
)
from pokr.bot import PokerBot
from pokr.opponents import CallingStation


def test_deterministic_reports():
    r1 = run_matchup(PokerBot(random.Random(1), mc_iters=10), calling_station_factory, 100, seed=42)
    r2 = run_matchup(PokerBot(random.Random(1), mc_iters=10), calling_station_factory, 100, seed=42)
    assert r1 == r2


def test_selfplay_is_symmetric():
    r = run_matchup(PokerBot(random.Random(3), mc_iters=10), lambda rng: PokerBot(rng, mc_iters=10), 150, seed=7)
    # no rake: seat 0's total should be within 4 std of 0 over 150 hands
    std_total = 20 * (r.var_bb_per_100 ** 0.5) if r.var_bb_per_100 > 0 else 0.0
    assert abs(r.total_bb) <= 4 * std_total


def test_report_math():
    rep = MatchupReport("x", hands=100, total_bb=50.0, bb_per_100=50.0, win_rate=0.5, var_bb_per_100=1.0)
    assert rep.bb_per_100 == 50.0
    assert rep.hands == 100


def test_run_benchmark_includes_self_and_leak_hunter():
    reports = run_benchmark(PokerBot(random.Random(5), mc_iters=10), num_hands=30, seed=1)
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


def test_play_session_mixed_lineup():
    # Game benchmark: bot at seat 0 vs a mixed lineup; session is deterministic
    # and reports the bot's own play stats.
    factories = [calling_station_factory, tight_aggressive_factory,
                 maniac_factory, random_factory, leak_hunter_factory]
    bb1, results1 = play_session(PokerBot(random.Random(1), mc_iters=10),
                                 factories, 20, seed=9)
    bb2, results2 = play_session(PokerBot(random.Random(1), mc_iters=10),
                                 factories, 20, seed=9)
    assert bb1 == bb2
    assert len(results1) == 20
    stats = bot_own_stats(results1, seat=0)
    assert stats["hands"] == 20
    assert 0.0 <= stats["vpip"] <= 1.0
    assert 0.0 <= stats["aggression_freq"] <= 1.0


def test_format_hand_uses_real_dealer():
    _, results = play_session(PokerBot(random.Random(1), mc_iters=10),
                              [calling_station_factory] * 5, 5, seed=3)
    text = format_hand(results[2], ["you"] + ["cs"] * 5,
                       hand_label="Hand #2 of 5")
    assert "Hand #2 of 5" in text
    assert "dealer" in text
