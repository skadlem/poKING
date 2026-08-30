"""Tests for pokr.allin_ev: all-in EV adjustment.

Fixtures mostly build HandResult directly (no real game needed to exercise
the pure post-processing logic), plus one integration test that runs a real
hand through PokerGame to prove the action-log reconstruction matches what
the engine actually does.
"""
from __future__ import annotations

import random

import pytest

import pokr.allin_ev as allin_ev
from pokr.allin_ev import allin_adjusted_winnings, allin_street, completion_plan
from pokr.cards import Deck, card_from_str
from pokr.engine import HandResult, PokerGame
from pokr.strategy import Action, BaseStrategy


class Scripted(BaseStrategy):
    """Plays actions in order; repeats the last one forever."""

    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0

    def decide(self, state, pid):
        if self._i < len(self._actions):
            a = self._actions[self._i]
            self._i += 1
            return a
        return self._actions[-1]


def C(*names: str):
    return [card_from_str(n) for n in names]


def make_result(*, starting_stacks, hole, community, actions, big_blind=2, dealer=0,
                 ending_stacks=None, hand_number=1):
    """Build a HandResult by hand, filling in the fields allin_ev doesn't read
    (ending_stacks/winnings) with plausible placeholders unless given.
    """
    ending = ending_stacks or list(starting_stacks)
    return HandResult(
        hand_number=hand_number,
        starting_stacks=starting_stacks,
        ending_stacks=ending,
        hole=hole,
        community=community,
        actions=actions,
        winnings=[e - s for e, s in zip(ending, starting_stacks)],
        big_blind=big_blind,
        dealer=dealer,
    )


# -- no adjustment needed ---------------------------------------------------

def test_no_adjustment_everyone_folded_to_one():
    result = make_result(
        starting_stacks=[100, 100],
        hole=[C("As", "Ah"), C("Kd", "Kc")],
        community=[],
        actions=[(1, "preflop", Action.fold("x"))],
    )
    assert allin_street(result) is None
    assert allin_adjusted_winnings(result) is None


def test_no_adjustment_betting_reaches_river():
    # Board is complete and there's still a decision on the river itself —
    # nothing left to average away even if someone later goes all-in there.
    result = make_result(
        starting_stacks=[100, 100],
        hole=[C("As", "Ah"), C("Kd", "Kc")],
        community=C("2c", "7d", "9h", "Tc", "3d"),
        actions=[(0, "river", Action.bet(10, "x")), (1, "river", Action.call(10, "y"))],
    )
    assert allin_street(result) is None
    assert allin_adjusted_winnings(result) is None


def test_no_adjustment_showdown_nobody_all_in():
    result = make_result(
        starting_stacks=[100, 100],
        hole=[C("As", "Ah"), C("Kd", "Kc")],
        community=C("2c", "7d", "9h", "Tc", "3d"),
        actions=[(0, "river", Action.check("x")), (1, "river", Action.check("y"))],
    )
    assert allin_street(result) is None
    assert allin_adjusted_winnings(result) is None


def test_no_adjustment_betting_ceased_early_but_nobody_all_in():
    # Directly exercises detection rule #3: last action is before the river,
    # 2+ players live, but nobody's total commitment matches their starting
    # stack. Can't happen from a real PokerGame hand (the engine keeps
    # dealing streets whenever 2+ active players remain), but the detector
    # must not assume that invariant when handed a HandResult directly.
    result = make_result(
        starting_stacks=[1000, 1000],
        hole=[C("As", "Ah"), C("Kd", "Kc")],
        community=C("2c", "7d", "9h"),
        actions=[(0, "flop", Action.check("x")), (1, "flop", Action.check("y"))],
    )
    assert allin_street(result) is None
    assert allin_adjusted_winnings(result) is None


# -- river-only (44 outs), hand-checked -------------------------------------

def _river_only_result():
    # Hero already has nines-full-of-fives (999-55) after the turn; villain
    # holds Kc/Qc. Verified by brute force (see task notes): of the 44
    # possible river cards, exactly one (5s) flips the result — it completes
    # quad fives on the board, decided by kicker (villain's K beats hero's
    # 9). Every other river leaves hero's boat (trip rank 9) ahead of any
    # trip-fives-based boat villain could make, since full houses compare by
    # trip rank first.
    return make_result(
        starting_stacks=[100, 100],
        hole=[C("9h", "9d"), C("Kc", "Qc")],
        community=C("5c", "5d", "5h", "9s", "2c"),  # 5th card (2c) is the
        # *actual* dealt river — irrelevant, since it's past the known
        # prefix and gets replaced by enumeration.
        actions=[
            (0, "preflop", Action.call(1, "sb completes")),
            (1, "preflop", Action.check("bb")),
            (1, "flop", Action.check("x")),
            (0, "flop", Action.check("x")),
            (1, "turn", Action.check("x")),
            (0, "turn", Action.bet(98, "shove")),
            (1, "turn", Action.call(98, "call")),
        ],
        ending_stacks=[200, 0],
    )


def test_river_only_exact_hand_checked():
    result = _river_only_result()
    assert allin_street(result) == "turn"

    plan = completion_plan(result)
    assert plan.method == "exact"
    assert plan.cards_needed == 1
    assert plan.combo_count == 44

    adjusted = allin_adjusted_winnings(result)
    assert adjusted is not None
    expected_hero = (43 * 100 + 1 * (-100)) / 44
    assert adjusted[0] == pytest.approx(expected_hero, abs=1e-9)
    assert adjusted[1] == pytest.approx(-expected_hero, abs=1e-9)


# -- dominated all-in: EV strictly between the two extremes -----------------

def test_dominated_allin_ev_strictly_between_extremes():
    # AA vs KK, all-in on the flop (need turn+river only -> 990 combos,
    # enumerated exactly). AA is a big favorite but KK isn't drawing dead.
    result = make_result(
        starting_stacks=[100, 100],
        hole=[C("Ac", "Ad"), C("Kc", "Kd")],
        community=C("2s", "7h", "9d", "Tc", "3d"),
        actions=[
            (0, "preflop", Action.call(1, "sb completes")),
            (1, "preflop", Action.check("bb")),
            (1, "flop", Action.check("x")),
            (0, "flop", Action.bet(98, "shove")),
            (1, "flop", Action.call(98, "call")),
        ],
        ending_stacks=[200, 0],
    )
    assert allin_street(result) == "flop"
    plan = completion_plan(result)
    assert plan.method == "exact"
    assert plan.combo_count == 990

    adjusted = allin_adjusted_winnings(result)
    assert adjusted is not None
    assert -100 < adjusted[0] < 100
    assert -100 < adjusted[1] < 100
    assert adjusted[0] > 0  # AA is the favorite
    assert adjusted[0] == pytest.approx(-adjusted[1], abs=1e-9)


# -- zero-sum -----------------------------------------------------------

def test_adjusted_winnings_are_zero_sum():
    result = _river_only_result()
    adjusted = allin_adjusted_winnings(result)
    assert adjusted is not None
    assert sum(adjusted) == pytest.approx(0.0, abs=1e-6)


# -- drawing dead: adjusted equals realized exactly --------------------------

def test_drawing_dead_matches_realized_exactly():
    # Hero already holds a complete royal flush (board Tc,Jc,Qc,Kc + hole
    # Ac) after the turn. No river card can ever let villain catch up, so
    # the adjustment should leave the realized result untouched.
    result = make_result(
        starting_stacks=[100, 100],
        hole=[C("Ac", "2d"), C("As", "Ad")],
        community=C("Tc", "Jc", "Qc", "Kc", "2h"),
        actions=[
            (0, "preflop", Action.call(1, "sb completes")),
            (1, "preflop", Action.check("bb")),
            (1, "flop", Action.check("x")),
            (0, "flop", Action.check("x")),
            (1, "turn", Action.check("x")),
            (0, "turn", Action.bet(98, "shove")),
            (1, "turn", Action.call(98, "call")),
        ],
        ending_stacks=[200, 0],
    )
    assert allin_street(result) == "turn"
    adjusted = allin_adjusted_winnings(result)
    assert adjusted == [pytest.approx(100.0), pytest.approx(-100.0)]
    assert adjusted == pytest.approx(result.winnings, abs=1e-9)


# -- side pots: short stack can't win more than the main pot ----------------

def test_side_pots_short_stack_capped_at_main_pot():
    # 3-handed. Seat 0 is the short stack (30 total) and holds quad nines
    # made on the flop -- unbeatable by seats 1/2's unpaired hole cards
    # (verified by brute force), so it always wins the main pot and never
    # contends for the side pot. Seats 1/2 (100 each) fight over the side
    # pot on equity (Ac/Kd vs Qc/Jd).
    # dealer=0, n=3: sb=seat1, bb=seat2, so seat0 starts the preflop street
    # uncommitted. Everyone limps to 2, then goes all-in on the flop: seat0
    # (28 left) shoves first, seat1 raises all-in to 98, seat2 calls all-in.
    result = make_result(
        starting_stacks=[30, 100, 100],
        hole=[C("9c", "9d"), C("Ac", "Kd"), C("Qc", "Jd")],
        community=C("9h", "9s", "2c", "5d", "8h"),
        actions=[
            (0, "preflop", Action.call(2, "limp")),
            (1, "preflop", Action.call(1, "sb completes")),
            (2, "preflop", Action.check("bb")),
            (0, "flop", Action.bet(28, "shove")),
            (1, "flop", Action.raise_to(98, "raise")),
            (2, "flop", Action.call(98, "call")),
        ],
        ending_stacks=[90, 140, 0],
        dealer=0,
        big_blind=2,
    )
    assert allin_street(result) == "flop"
    plan = completion_plan(result)
    assert plan.method == "exact"

    adjusted = allin_adjusted_winnings(result)
    assert adjusted is not None
    # Main pot = 30 * 3 = 90; seat 0 always wins it and never touches the
    # side pot, so its net EV is exactly (main pot - its own commitment).
    assert adjusted[0] == pytest.approx(60.0, abs=1e-6)
    assert adjusted[1] == pytest.approx(4.263565891472865, abs=1e-6)
    assert adjusted[2] == pytest.approx(-64.26356589147287, abs=1e-6)
    assert sum(adjusted) == pytest.approx(0.0, abs=1e-6)
    # The short stack's total payout never exceeds the main pot (90).
    committed0 = result.starting_stacks[0]  # fully all-in
    assert committed0 + adjusted[0] == pytest.approx(90.0, abs=1e-6)


# -- exact vs sampled agreement ----------------------------------------------

def test_exact_vs_sampled_agreement(monkeypatch):
    result = _river_only_result()
    exact = allin_adjusted_winnings(result)
    assert completion_plan(result).method == "exact"

    # Force the sampling path on a case small enough to enumerate exactly,
    # so we can compare the two methods against each other directly.
    monkeypatch.setattr(allin_ev, "_EXACT_ENUMERATION_LIMIT", 0)
    plan = completion_plan(result)
    assert plan.method == "sample"

    sampled = allin_adjusted_winnings(result, iterations=20_000, rng=random.Random(42))
    assert sampled is not None
    for e, s in zip(exact, sampled):
        assert s == pytest.approx(e, abs=3.0)  # a few percent of a 100-chip pot


# -- integration: a real hand played through PokerGame ----------------------

def test_integration_real_preflop_shove():
    deck = Deck(random.Random(1), C("As", "Ah", "Kd", "Kc", "2c", "2d", "9h", "3h", "4h"),
                shuffle=False)
    strategies = [Scripted([Action.raise_to(100, "shove")]), Scripted([Action.call(98, "call")])]
    game = PokerGame(strategies, [100, 100], rng=random.Random(1), initial_dealer=0, deck=deck)
    result = game.play_hand()

    # Sanity: matches the hand-computed integration values from development.
    assert result.actions[0][1] == "preflop"
    assert result.winnings == [100, -100]

    assert allin_street(result) == "preflop"
    plan = completion_plan(result)
    assert plan.street == "preflop"
    assert plan.method == "sample"  # C(46,5) far exceeds the exact-enumeration cutoff

    adjusted = allin_adjusted_winnings(result, iterations=5_000, rng=random.Random(7))
    assert adjusted is not None
    assert sum(adjusted) == pytest.approx(0.0, abs=1e-6)
    # AA is a big favorite over KK preflop; EV should be well clear of zero
    # but nowhere near the realized +-100 extremes.
    assert adjusted[0] > 40
    assert adjusted[0] < 100
