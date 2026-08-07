import random

from pokr.bench import run_benchmark
from pokr.bot import PokerBot


def test_smoke_benchmark_runs():
    reports = run_benchmark(PokerBot(random.Random(0)), num_hands=30, seed=0)
    assert len(reports) >= 6  # 4 canned + self_play + leak_hunter
    for r in reports:
        assert r.hands == 30
        assert 0.0 <= r.win_rate <= 1.0
