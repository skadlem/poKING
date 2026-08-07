import random

from pokr.bankroll import SimpleBankrollManager
from pokr.bot import PokerBot
from pokr.risk import BankrollManager


def test_implements_protocol():
    # Structural conformance with the risk.py Protocol seam.
    assert isinstance(SimpleBankrollManager(), BankrollManager)


def test_session_budget_scales_with_bankroll():
    m = SimpleBankrollManager(fraction_risked=0.02)
    assert m.session_budget(10_000) == 200.0
    assert m.session_budget(50_000) == 1000.0


def test_session_budget_clamps():
    m = SimpleBankrollManager(fraction_risked=0.02, min_budget=20.0, max_budget=500.0)
    assert m.session_budget(100) == 20.0        # floor
    assert m.session_budget(1_000_000) == 500.0  # ceiling
    assert m.session_budget(50_000) == 500.0     # 1000 -> capped to 500


def test_should_stop_loss():
    m = SimpleBankrollManager(stop_loss_budgets=1.0, stop_win_budgets=2.0)
    budget = m.session_budget(10_000)  # 200
    assert m.should_stop(10_000, 9_800, budget)   # -200 exactly = boundary, stop (<=)
    assert m.should_stop(10_000, 9_799, budget)
    assert m.should_stop(10_000, 9_500, budget)


def test_should_stop_win():
    m = SimpleBankrollManager(stop_loss_budgets=1.0, stop_win_budgets=2.0)
    budget = m.session_budget(10_000)  # 200
    assert not m.should_stop(10_000, 10_199, budget)
    assert m.should_stop(10_000, 10_400, budget)   # +400 = +2 budgets
    assert m.should_stop(10_000, 12_000, budget)


def test_no_stop_within_bands():
    m = SimpleBankrollManager(stop_loss_budgets=1.0, stop_win_budgets=2.0)
    budget = m.session_budget(10_000)
    assert not m.should_stop(10_000, 10_100, budget)
    assert not m.should_stop(10_000, 9_900, budget)


def test_begin_session_wires_budget_into_policy():
    m = SimpleBankrollManager(fraction_risked=0.02)
    bot = PokerBot(random.Random(1), bankroll_manager=m)
    assert bot.policy.risk_cfg.session_budget == 200.0  # default unchanged
    bot.begin_session(25_000)
    assert bot.policy.risk_cfg.session_budget == 500.0
    bot.begin_session(5_000)
    assert bot.policy.risk_cfg.session_budget == 100.0


def test_begin_session_without_manager_is_noop():
    bot = PokerBot(random.Random(1))
    assert bot.policy.risk_cfg.session_budget == 200.0
    bot.begin_session(25_000)
    assert bot.policy.risk_cfg.session_budget == 200.0
