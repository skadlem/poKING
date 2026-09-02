"""The Kuhn harness has to be right before it can certify anything else.

pokr/rl/kuhn.py exists to gate an equilibrium algorithm (NFSP, see
docs/design/nfsp.md section 6) on a game with a known answer. A gate that is
itself wrong is worse than no gate, so these tests pin the two facts everything
downstream leans on -- the Nash family has exploitability exactly zero, and the
game value is -1/18 -- plus the invariants that hold for every strategy.
"""
import random

import pytest

from pokr.rl.kuhn import (
    BET,
    CARDS,
    DEALS,
    GAME_VALUE,
    INFO_SETS,
    INFO_SETS_BY_PLAYER,
    PASS,
    average_of,
    best_response_value,
    expected_value,
    exploitability,
    info_set,
    is_terminal,
    nash,
    play,
    player_to_act,
    sampling_policy,
    terminal_payoff,
    uniform,
)

ALPHAS = [0.0, 0.05, 1 / 6, 0.3, 1 / 3]


def random_strategy(rng):
    table = {}
    for key in INFO_SETS:
        p = rng.random()
        table[key] = (p, 1 - p)
    return table


# -- the game itself ------------------------------------------------------


def test_deals_are_the_six_ordered_pairs_of_distinct_cards():
    assert len(DEALS) == 6
    assert len(set(DEALS)) == 6
    assert all(a != b and a in CARDS and b in CARDS for a, b in DEALS)


def test_info_sets_partition_by_player_and_are_disjoint():
    p0, p1 = INFO_SETS_BY_PLAYER
    assert len(p0) == len(p1) == 6
    assert not set(p0) & set(p1)
    assert set(INFO_SETS) == set(p0) | set(p1)


@pytest.mark.parametrize("history,player", [
    ("", 0), ("p", 1), ("b", 1), ("pb", 0)])
def test_player_to_act_alternates(history, player):
    assert not is_terminal(history)
    assert player_to_act(history) == player


def test_info_set_shows_only_the_acting_players_own_card():
    # player 0 holds the king, player 1 the jack
    assert info_set("", (2, 0)) == "2"
    assert info_set("b", (2, 0)) == "0b"


@pytest.mark.parametrize("history", ["pp", "pbp", "pbb", "bp", "bb"])
def test_terminal_payoffs_are_zero_sum_and_bounded(history):
    for deal in DEALS:
        u0 = terminal_payoff(history, deal)
        assert abs(u0) in (1.0, 2.0)


def test_folding_loses_the_ante_regardless_of_cards():
    """A fold pays 1 whoever held what -- the hand never reaches showdown."""
    assert {terminal_payoff("pbp", d) for d in DEALS} == {-1.0}
    assert {terminal_payoff("bp", d) for d in DEALS} == {1.0}


def test_showdown_pays_the_high_card():
    assert terminal_payoff("pp", (2, 0)) == 1.0
    assert terminal_payoff("pp", (0, 2)) == -1.0
    assert terminal_payoff("bb", (2, 0)) == 2.0
    assert terminal_payoff("bb", (0, 2)) == -2.0


def test_non_terminal_history_is_a_loud_error():
    with pytest.raises(ValueError, match="not a terminal history"):
        terminal_payoff("p", (0, 1))


# -- the reference equilibrium --------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_nash_family_is_unexploitable(alpha):
    assert exploitability(nash(alpha)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_nash_family_all_share_the_game_value(alpha):
    assert expected_value(nash(alpha)) == pytest.approx(GAME_VALUE, abs=1e-12)
    assert GAME_VALUE == pytest.approx(-1 / 18)


@pytest.mark.parametrize("alpha", [-0.01, 0.4, 1.0])
def test_alpha_outside_the_family_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        nash(alpha)


def test_uniform_play_is_exploitable():
    """11/24 antes per hand. Pinned as a regression anchor: any change to the
    tree or the payoffs moves this number."""
    assert exploitability(uniform()) == pytest.approx(11 / 24)


# -- exploitability invariants --------------------------------------------


def test_exploitability_is_never_negative():
    rng = random.Random(7)
    for _ in range(50):
        assert exploitability(random_strategy(rng)) >= -1e-12


def test_best_response_never_does_worse_than_the_strategy_it_replaces():
    rng = random.Random(11)
    for _ in range(20):
        sigma = random_strategy(rng)
        ev0 = expected_value(sigma)
        assert best_response_value(sigma, 0) >= ev0 - 1e-12
        assert best_response_value(sigma, 1) >= -ev0 - 1e-12


def test_best_response_to_a_player_who_always_folds_is_to_always_bet():
    """Hand-computable: bet every hand, villain folds every hand, +1 an ante."""
    sigma = dict(uniform())
    for card in CARDS:
        sigma[f"{card}b"] = (1.0, 0.0)   # folds to any bet
        sigma[f"{card}p"] = (1.0, 0.0)   # never bets itself
    assert best_response_value(sigma, 0) == pytest.approx(1.0)


# -- episode generation ---------------------------------------------------


def test_play_visits_only_real_info_sets_in_turn_order():
    rng = random.Random(3)
    policy = sampling_policy(uniform(), rng)
    for _ in range(200):
        steps, _ = play(policy, policy, rng)
        assert 2 <= len(steps) <= 3
        for player, key, action in steps:
            assert key in INFO_SETS_BY_PLAYER[player]
            assert action in (PASS, BET)


def test_play_payoffs_are_zero_sum():
    rng = random.Random(5)
    policy = sampling_policy(uniform(), rng)
    for _ in range(200):
        _, (u0, u1) = play(policy, policy, rng)
        assert u0 + u1 == 0.0


def test_forced_deal_makes_a_hand_deterministic():
    always_bet = lambda key, player: BET      # noqa: E731
    steps, payoffs = play(always_bet, always_bet, random.Random(0), cards=(2, 0))
    assert [s[1] for s in steps] == ["2", "0b"]
    assert payoffs == (2.0, -2.0)             # king wins a doubled pot


def test_sampled_play_converges_to_the_analytic_value():
    """The tree walk and the episode generator must describe the same game --
    if they drift, a learner trains on one and is scored on the other, which is
    the depth-mismatch bug this repo already paid for once."""
    rng = random.Random(17)
    sigma = nash(1 / 3)
    policy = sampling_policy(sigma, rng)
    n = 40_000
    total = sum(play(policy, policy, rng)[1][0] for _ in range(n))
    # per-hand payoff is bounded by 2, so the standard error is under 2/sqrt(n)
    assert total / n == pytest.approx(expected_value(sigma), abs=4 / n ** 0.5)


def test_a_policy_returning_a_bad_action_is_a_loud_error():
    with pytest.raises(ValueError, match="expected 0 or 1"):
        play(lambda key, player: 2, lambda key, player: 0, random.Random(0))


# -- averaging ------------------------------------------------------------


def test_average_of_two_strategies_is_their_midpoint():
    a = {key: (1.0, 0.0) for key in INFO_SETS}
    b = {key: (0.0, 1.0) for key in INFO_SETS}
    assert average_of([a, b]) == uniform()


def test_averaging_nothing_is_an_error():
    with pytest.raises(ValueError, match="no strategies"):
        average_of([])
