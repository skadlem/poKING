from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .botdetect import DetectionResult
from .cards import monte_carlo_equity
from .engine import GameState
from .models import OpponentSummary
from .risk import RiskConfig, risk_adjusted_bet_size
from .strategy import Action, ActionType

_VALUE_EQUITY = 0.6
_BLUFF_EQUITY_MAX = 0.35
_BET_FRACTIONS = (0.33, 0.66, 1.0)
_MIRROR_BET_FRACTIONS = (0.66, 1.0, 1.5)
_TEMPERATURE = 3.0
_MIRROR_THRESHOLD = 0.6
_BOT_THRESHOLD = 0.6


@dataclass
class _Candidate:
    action: Action
    ev: float


class Policy:
    """Combines hand EV, opponent model, bot detection, and risk into one action."""

    def __init__(
        self,
        rng: random.Random,
        risk_cfg: RiskConfig | None = None,
        mc_iters: int = 150,
    ) -> None:
        self.rng = rng
        self.risk_cfg = risk_cfg or RiskConfig()
        self.mc_iters = mc_iters

    def decide(
        self,
        state: GameState,
        player_id: int,
        summary: OpponentSummary | None,
        detection: DetectionResult | None,
    ) -> Action:
        p = state.players[player_id]
        to_call = state.current_bet - p.street_committed
        pot = state.pot
        stack = p.stack
        opp_count = max(sum(1 for q in state.players if q.id != player_id and not q.folded), 1)
        equity = monte_carlo_equity(p.hole, state.community, opp_count, self.mc_iters, self.rng)

        fold_freq = summary.fold_rate_postflop if summary else 0.3
        mirror = detection is not None and detection.p_mirror >= _MIRROR_THRESHOLD
        if detection is not None and detection.p_is_bot >= _BOT_THRESHOLD:
            fold_freq = max(fold_freq, 0.5)  # predictable opponents fold more to bets

        cands: list[_Candidate] = []
        for la in state.legal_actions:
            t = la.action_type
            if t == ActionType.FOLD:
                cands.append(_Candidate(Action.fold("ev 0"), 0.0))
            elif t == ActionType.CHECK:
                cands.append(_Candidate(Action.check("check"), equity * pot))
            elif t == ActionType.CALL:
                ev = equity * pot - (1 - equity) * to_call
                cands.append(_Candidate(Action.call(la.min_amount, "call"), ev))
            elif t in (ActionType.BET, ActionType.RAISE):
                fractions = _MIRROR_BET_FRACTIONS if mirror else _BET_FRACTIONS
                for frac in fractions:
                    bet = int(pot * frac)
                    amt = max(la.min_amount, bet) if bet > 0 else la.min_amount
                    amt = min(amt, la.max_amount)
                    incremental = amt - p.street_committed if t == ActionType.RAISE else amt
                    ev = fold_freq * pot + (1 - fold_freq) * (
                        equity * (pot + incremental) - (1 - equity) * incremental
                    )
                    reason = "value" if equity >= _VALUE_EQUITY else (
                        "bluff" if equity <= _BLUFF_EQUITY_MAX else "semi"
                    )
                    action = Action.raise_to(amt, reason) if t == ActionType.RAISE else Action.bet(amt, reason)
                    cands.append(_Candidate(action, ev))

        action = self._choose(cands)

        if action.action_type in (ActionType.BET, ActionType.RAISE):
            # Value bets are Kelly-capped off the bankroll; bluffs (EV-positive only
            # through fold equity) are capped by stack and pot fractions so the
            # dynamic risk core never zeroes out the bluff arm of the strategy.
            if equity >= _VALUE_EQUITY:
                cap = risk_adjusted_bet_size(equity, pot, stack, self.risk_cfg.session_budget, self.risk_cfg)
            else:
                cap = min(int(self.risk_cfg.max_bet_fraction_of_stack * stack),
                          int(self.risk_cfg.max_bet_as_pot_fraction * pot))
            if action.amount > cap:
                la = [x for x in state.legal_actions if x.action_type == action.action_type]
                if la and la[0].min_amount <= cap <= la[0].max_amount:
                    amt = int(cap)
                    action = Action.raise_to(amt, action.reason) if action.action_type == ActionType.RAISE \
                        else Action.bet(amt, action.reason)
                else:
                    # capping makes the action illegal; fall back to check/call
                    call_la = [x for x in state.legal_actions if x.action_type == ActionType.CALL]
                    if call_la:
                        action = Action.call(call_la[0].min_amount, "risk cap fallback call")
                    else:
                        action = Action.check("risk cap fallback check")
        return action

    def _choose(self, cands: list[_Candidate]) -> Action:
        if not cands:
            return Action.check("no legal actions")
        evs = [max(c.ev, 0.0) for c in cands]
        total = sum(evs)
        if total <= 0:
            return max(cands, key=lambda c: c.ev).action
        probs = [math.exp((e / total) * _TEMPERATURE) for e in evs]
        s = sum(probs)
        r = self.rng.random() * s
        acc = 0.0
        for c, pr in zip(cands, probs):
            acc += pr
            if r <= acc:
                return c.action
        return cands[-1].action
