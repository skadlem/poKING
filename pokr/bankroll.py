"""Cross-session bankroll management (handoff step 4).

Implements the `BankrollManager` Protocol from pokr/risk.py. A manager turns a
total bankroll into a per-session budget (which Policy's Kelly sizing consumes
via RiskConfig.session_budget) and decides when a session should stop
(stop-loss / stop-win). Pure logic, no I/O: the caller (e.g. a poker client)
queries `should_stop` between hands and calls `begin_session` on the bot at
session start.

Buy-in variation (choosing a table with a buy-in that fits the session budget)
is intentionally left to the caller: it needs table information this module
does not have.
"""
from __future__ import annotations

from dataclasses import dataclass

from .risk import BankrollManager


@dataclass
class SimpleBankrollManager:
    """Risk a fixed fraction of the total bankroll per session.

    session_budget = clamp(bankroll * fraction_risked, min_budget, max_budget).

    Stop rules are expressed in session budgets: stop when the session has
    lost at least `stop_loss_budgets` budgets or won at least
    `stop_win_budgets` budgets (measured against the bankroll at session
    start). Defaults: risk 2% per session, stop at -1 budget (down 2% of
    bankroll) or +2 budgets (up 4% of bankroll).
    """

    fraction_risked: float = 0.02
    min_budget: float = 20.0
    max_budget: float = 2000.0
    stop_loss_budgets: float = 1.0
    stop_win_budgets: float = 2.0

    def session_budget(self, bankroll: float) -> float:
        """Session budget in chips for a given total bankroll."""
        raw = bankroll * self.fraction_risked
        return max(self.min_budget, min(self.max_budget, raw))

    def should_stop(self, bankroll_start: float, bankroll_now: float,
                    session_budget: float) -> bool:
        """True when the session should stop (stop-loss or stop-win hit)."""
        delta = bankroll_now - bankroll_start
        if delta <= -self.stop_loss_budgets * session_budget:
            return True
        if delta >= self.stop_win_budgets * session_budget:
            return True
        return False
