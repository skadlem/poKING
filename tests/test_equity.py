import random

from pokr.cards import card_from_str, monte_carlo_equity


def hs(*strs):
    return [card_from_str(s) for s in " ".join(strs).split()]


def test_aces_equity_high():
    rng = random.Random(1)
    eq = monte_carlo_equity(hs("As Ah"), [], 1, 2000, rng)
    assert 0.75 <= eq <= 0.95


def test_72_equity_low():
    rng = random.Random(2)
    eq = monte_carlo_equity(hs("7h 2d"), [], 1, 2000, rng)
    assert 0.20 <= eq <= 0.50


def test_deterministic_given_seed():
    a = monte_carlo_equity(hs("As Ah"), [], 1, 500, random.Random(3))
    b = monte_carlo_equity(hs("As Ah"), [], 1, 500, random.Random(3))
    assert a == b


def test_partial_board():
    rng = random.Random(4)
    eq = monte_carlo_equity(hs("As Ah"), hs("Ks Kd 2c"), 2, 500, rng)
    assert 0.0 <= eq <= 1.0
