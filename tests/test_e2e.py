import random

from pokr.bench import run_benchmark
from pokr.bot import PokerBot


def test_smoke_benchmark_runs():
    # Smoke check of run_benchmark wiring; mc_iters=10 keeps it fast while
    # exercising the same code path (equity precision is covered elsewhere).
    reports = run_benchmark(PokerBot(random.Random(0), mc_iters=10), num_hands=30, seed=0)
    assert len(reports) >= 6  # 4 canned + self_play + leak_hunter
    for r in reports:
        assert r.hands == 30
        assert 0.0 <= r.win_rate <= 1.0
