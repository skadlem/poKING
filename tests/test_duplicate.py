"""Duplicate-deck evaluator: the pairing must be exact, and the estimator must
report an honest variance reduction rather than a flattering one.

The load-bearing claim is that both runs of a deck see identical cards no
matter how differently the heroes play. That is asserted directly here, since
every number the module produces depends on it.
"""
import random

import numpy as np
import pytest

from pokr.bench import (
    calling_station_factory,
    random_factory,
    tight_aggressive_factory,
)
from pokr.bot import PokerBot
from pokr.duplicate import DuplicateReport, _reseed, main, run_duplicate
from pokr.opponents import CallingStation


def heuristic(rng):
    return PokerBot(rng, mc_iters=8, mc_fast=False)


# -- the pairing is exact -------------------------------------------------


def test_both_runs_of_a_deck_see_identical_cards():
    """Hole cards are dealt before any action and board cards come off the same
    deck in a fixed order, so divergent play cannot change the cards. Board
    length may differ (one run may end preflop); the shorter must be a prefix."""
    checked = 0

    def on_hand(h, r1, r2):
        nonlocal checked
        assert r1.hole == r2.hole, f"deck {h}: hole cards diverged"
        short, long = sorted((r1.community, r2.community), key=len)
        assert long[:len(short)] == short, f"deck {h}: board diverged"
        checked += 1

    run_duplicate(heuristic, calling_station_factory,
                  [tight_aggressive_factory, random_factory],
                  num_hands=120, seed=5, on_hand=on_hand)
    assert checked == 120


def test_identical_deterministic_heroes_score_exactly_zero():
    """CallingStation is deterministic, so both runs of a deck are bit-identical
    and the measured gap must be exactly 0 -- not merely small."""
    report = run_duplicate(lambda rng: CallingStation(), lambda rng: CallingStation(),
                           [tight_aggressive_factory], num_hands=150, seed=11)
    assert np.array_equal(report.per_deck_a, report.per_deck_b)
    assert report.diff_bb_per_100 == 0.0
    assert report.se_diff == 0.0
    assert not report.resolved


def test_heads_up_is_zero_sum():
    report = run_duplicate(heuristic, tight_aggressive_factory, [],
                           num_hands=200, seed=3)
    assert report.bb_per_100_a == pytest.approx(-report.bb_per_100_b, abs=1e-9)
    assert report.se_a == pytest.approx(report.se_b, rel=1e-9)


def test_table_size_follows_the_opponent_list():
    seen = []

    def probe(rng):
        bot = CallingStation()
        seen.append(bot)
        return bot

    def on_hand(h, r1, r2):
        assert len(r1.starting_stacks) == 4 and len(r2.starting_stacks) == 4

    run_duplicate(probe, probe, [calling_station_factory, random_factory],
                  num_hands=5, seed=1, on_hand=on_hand)
    # two heroes x two seats: each hero must get its own instance per seat, or a
    # stateful bot would see one seat change identity between runs
    assert len(seen) == 4
    assert len({id(x) for x in seen}) == 4


def test_stacks_reset_every_deck():
    """Unlike bench.run_matchup, each deck starts at the buy-in -- that is what
    makes decks independent and keeps depth at the intended 100bb."""
    def on_hand(h, r1, r2):
        assert set(r1.starting_stacks) == {200}
        assert set(r2.starting_stacks) == {200}

    run_duplicate(heuristic, calling_station_factory, [random_factory],
                  num_hands=40, seed=2, buy_in=200, on_hand=on_hand)


# -- common random numbers ------------------------------------------------


def test_reseed_reaches_a_shared_rng_object():
    """PokerBot hands its Random to Policy, so reseeding must happen in place;
    rebinding bot.rng would leave the policy on the old stream."""
    bot = PokerBot(random.Random(1), mc_iters=4)
    assert bot.policy.rng is bot.rng
    _reseed([bot], 12345)
    first = bot.policy.rng.random()
    _reseed([bot], 12345)
    assert bot.policy.rng.random() == first


def test_reseed_ignores_strategies_without_an_rng():
    class NoRng:
        pass
    _reseed([NoRng(), CallingStation()], 7)   # must not raise


# -- the statistics -------------------------------------------------------


def test_report_arithmetic():
    a = np.array([2.0, -1.0, 3.0, 0.0])
    b = np.array([1.0, 0.0, 1.0, 2.0])
    from pokr.duplicate import _report
    r = _report(a, b, np.repeat(a, 2), np.repeat(b, 2), "A", "B")
    assert r.decks == 4 and r.hands_per_hero == 8
    assert r.bb_per_100_a == pytest.approx(100.0)   # mean 1.0 bb -> 100 bb/100
    assert r.bb_per_100_b == pytest.approx(100.0)
    assert r.diff_bb_per_100 == pytest.approx(0.0)


def test_resolved_tracks_the_two_se_convention():
    from pokr.duplicate import _report
    a = np.array([5.0, 5.0, 5.0, 5.0])       # zero variance, clear separation
    b = np.zeros(4)
    r = _report(a, b, np.repeat(a, 2), np.repeat(b, 2), "A", "B")
    assert r.resolved
    noisy = _report(np.array([9.0, -9.0, 9.0, -9.0]), b,
                    np.repeat(a, 2), np.repeat(b, 2), "A", "B")
    assert not noisy.resolved


def test_variance_reduction_is_reported_honestly():
    """Duplicate averaging buys ~1.1x in NLHE, not the 5-10x it buys in bridge
    (see the module docstring). Guard the range so a future change that claims
    a large reduction gets looked at rather than believed."""
    report = run_duplicate(heuristic, tight_aggressive_factory, [],
                           num_hands=400, seed=3)
    assert 0.9 < report.variance_reduction < 2.0, report.variance_reduction
    assert report.unpaired_se_a > 0 and report.se_a > 0


def test_format_names_the_verdict():
    from pokr.duplicate import _report
    a, b = np.full(6, 4.0), np.zeros(6)
    text = _report(a, b, np.repeat(a, 2), np.repeat(b, 2), "Alpha", "Beta").format()
    assert "Alpha" in text and "Beta" in text
    assert "resolved" in text and "UNRESOLVED" not in text


# -- CLI ------------------------------------------------------------------


def test_cli_rejects_an_unknown_bot(capsys):
    assert main(["--a", "nope", "--b", "cs", "--lineup", "", "--hands", "2"]) == 2
    assert "unknown bot" in capsys.readouterr().out


def test_cli_runs_heads_up(capsys):
    assert main(["--a", "cs", "--b", "tag", "--lineup", "", "--hands", "20",
                 "--mc-iters", "5", "--fast"]) == 0
    out = capsys.readouterr().out
    assert "duplicate:" in out and "bb/100" in out and "heads-up" in out
