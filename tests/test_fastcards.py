import random

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from pokr._fastcards import _code, eval_n, monte_carlo_equity_fast  # noqa: E402
from pokr.cards import all_cards, card_from_str, evaluate_hand  # noqa: E402


def hs(*strs):
    return [card_from_str(s) for s in " ".join(strs).split()]


def test_eval_n_matches_evaluate_hand_ordering():
    rng = random.Random(42)
    deck = all_cards()
    for _ in range(3000):
        size = rng.choice([5, 6, 7])
        pick = rng.sample(deck, size * 2)
        a, b = pick[:size], pick[size:]
        ta, tb = evaluate_hand(a), evaluate_hand(b)
        fa = eval_n(np.array([_code(c) for c in a], np.int64))
        fb = eval_n(np.array([_code(c) for c in b], np.int64))
        assert (ta > tb) == (fa > fb), (a, b)
        assert (ta == tb) == (fa == fb), (a, b)


def test_fast_equity_deterministic_given_seed():
    a = monte_carlo_equity_fast(hs("As Ah"), [], 1, 200, random.Random(3))
    b = monte_carlo_equity_fast(hs("As Ah"), [], 1, 200, random.Random(3))
    assert a == b


def test_fast_equity_ranges():
    rng = random.Random(1)
    assert 0.75 <= monte_carlo_equity_fast(hs("As Ah"), [], 1, 2000, rng) <= 0.95
    assert 0.20 <= monte_carlo_equity_fast(hs("7h 2d"), [], 1, 2000, rng) <= 0.50
    eq = monte_carlo_equity_fast(hs("As Ah"), hs("Ks Kd 2c"), 2, 500, random.Random(4))
    assert 0.0 <= eq <= 1.0


def test_policy_mc_fast_plays_legally():
    from pokr.bench import calling_station_factory, run_matchup
    from pokr.bot import PokerBot

    r = run_matchup(PokerBot(random.Random(1), mc_iters=10, mc_fast=True),
                    calling_station_factory, 30, seed=42)
    assert r.hands == 30
