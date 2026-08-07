from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class BankrollManager(Protocol):
    """Seam for cross-session bankroll management (future work).

    A real implementation would scale the session budget by total bankroll,
    add stop-loss/stop-win, and vary buy-in size.
    """

    def session_budget(self, bankroll: float) -> float: ...


@dataclass
class RiskConfig:
    kelly_fraction: float = 1.0          # fraction of full Kelly to use
    max_bet_fraction_of_stack: float = 0.35
    max_bet_as_pot_fraction: float = 1.0
    session_budget: float = 200.0        # in chips; the BankrollManager seam feeds this


def kelly_fraction(win_prob: float, pot_odds: float) -> float:
    """Full-Kelly fraction f* = (b*p - q) / b, clamped to [0, 1].

    b = pot odds (reward/risk), p = P(win), q = 1 - p.
    """
    b = max(pot_odds, 1e-9)
    q = 1.0 - win_prob
    f = (b * win_prob - q) / b
    return min(max(f, 0.0), 1.0)


def risk_adjusted_bet_size(
    win_prob: float,
    pot: int,
    stack: int,
    bankroll: float,
    cfg: RiskConfig,
) -> int:
    """Bet size in chips: Kelly-sized off bankroll, capped by stack and pot."""
    b = pot / max(pot, 1)
    f = kelly_fraction(win_prob, b) * cfg.kelly_fraction
    by_bankroll = f * bankroll
    by_stack = cfg.max_bet_fraction_of_stack * stack
    by_pot = cfg.max_bet_as_pot_fraction * pot
    return max(int(min(by_bankroll, by_stack, by_pot)), 0)
