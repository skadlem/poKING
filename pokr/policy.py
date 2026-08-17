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
# Range-aware equity discount: an opponent's betting range is stronger than a
# random hand. Tight (low VPIP) + aggressive (high postflop aggression)
# opponents bet a much stronger range, so facing their bets, equity is
# discounted by up to _RANGE_DISCOUNT_MAX. Loose opponents' bets are close to
# random; no discount.
_RANGE_VPIP_FULL = 0.3       # vpip at or below this counts as fully tight
_RANGE_AGGR_LO = 0.15        # aggression at or below this counts as not aggressive
_RANGE_AGGR_SPAN = 0.4       # aggression span over _RANGE_AGGR_LO to reach full aggression
_RANGE_DISCOUNT_MAX = 0.3
_RANGE_FOLD_STRENGTH = 0.25  # at/above this range strength, a -EV call folds (no bluff-raise rescue)
_RANGE_CBET_FOLD_MAX = 0.4   # opponents who rarely fold to c-bets call down with strong hands
_OOP_CALL_EQUITY = 0.42      # min preflop equity to call a raise from SB/BB (OOP realization)
# ponytail: fixed equity line; position-aware ranges if the blinds still bleed


@dataclass
class _Candidate:
    action: Action
    ev: float


def _is_blind(state: GameState, player_id: int) -> bool:
    """SB/BB seats of the current hand (out of position preflop)."""
    n = len(state.players)
    if n == 2:
        sb, bb = state.dealer, (state.dealer + 1) % 2
    else:
        sb = (state.dealer + 1) % n
        bb = (state.dealer + 2) % n
    return player_id in (sb, bb)


class Policy:
    """Combines hand EV, opponent model, bot detection, and risk into one action."""

    def __init__(
        self,
        rng: random.Random,
        risk_cfg: RiskConfig | None = None,
        mc_iters: int = 150,
        mc_fast: bool = False,
    ) -> None:
        self.rng = rng
        self.risk_cfg = risk_cfg or RiskConfig()
        self.mc_iters = mc_iters
        self.mc_fast = mc_fast
        # Opt-in numba equity (~10-120x faster; different RNG draw stream, so
        # results differ from the pure path seed-for-seed). Default stays pure
        # python so tuned benchmark numbers remain comparable and numba stays
        # lazily imported.
        if mc_fast:
            from ._fastcards import monte_carlo_equity_fast
            self._equity = monte_carlo_equity_fast
        else:
            self._equity = monte_carlo_equity

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
        equity = self._equity(p.hole, state.community, opp_count, self.mc_iters, self.rng)

        # Range-aware equity: the opponent's betting/raising range is stronger
        # than a random hand, so discount equity when facing a bet or raise.
        # Postflop, tight (low VPIP) + aggressive (high postflop aggression)
        # opponents bet a much stronger range. Preflop, an opener who raises a
        # large share of their voluntarily played hands (high pfr/vpip) is
        # representing the same strength. Loose opponents' bets are close to
        # random; no discount. (Measured leak vs the TAG archetype: the bot
        # called/raised into a ~5% premium open range as if it were random.)
        strong_range = False
        preflop_strength = 0.0
        if to_call > 0 and summary is not None and summary.hands_observed >= 5:
            tight = max(0.0, 1.0 - summary.vpip / _RANGE_VPIP_FULL)
            if state.street != "preflop":
                aggressive = max(0.0, min(1.0, (summary.aggression_freq - _RANGE_AGGR_LO) / _RANGE_AGGR_SPAN))
                strength = min(1.0, tight * aggressive)
                # Also treat a tight opponent who rarely folds to c-bets as
                # betting a strong range: they call down with made hands, so our
                # marginal holdings are dominated. Combined with low VPIP, this
                # is the tight-aggressive "win small, lose big" profile.
                if summary.fold_to_cbet > 0 and summary.fold_to_cbet < _RANGE_CBET_FOLD_MAX:
                    strength = max(strength, tight * 0.5)
            else:
                # Preflop, an opener who raises a large share of their
                # voluntarily played hands (high pfr/vpip) represents strength.
                pfr_ratio = summary.pfr / summary.vpip if summary.vpip > 0 else 0.0
                aggressive = max(0.0, min(1.0, (pfr_ratio - _RANGE_AGGR_LO) / _RANGE_AGGR_SPAN))
                preflop_strength = strength = min(1.0, tight * aggressive)
            equity *= 1.0 - _RANGE_DISCOUNT_MAX * strength
            # Facing a strong betting range (tight + aggressive) with a -EV call,
            # don't rescue the hand with a bluff-raise: fold. Raising a
            # tight-aggressive bettor's strong range is -EV (the "win small, lose
            # big" leak measured vs the TAG archetype).
            strong_range = strength >= _RANGE_FOLD_STRENGTH
            if strong_range:
                call_ev = equity * pot - (1.0 - equity) * to_call
                if call_ev <= 0:
                    return Action.fold("fold vs strong range")

        fold_freq = summary.fold_rate_postflop if summary else 0.3
        mirror = detection is not None and detection.p_mirror >= _MIRROR_THRESHOLD
        if detection is not None and detection.p_is_bot >= _BOT_THRESHOLD:
            fold_freq = max(fold_freq, 0.5)  # predictable opponents fold more to bets
        if to_call > 0 and state.street == "preflop" and summary is not None \
                and preflop_strength > 0:
            # Facing a preflop raise from a TIGHT opener (the only case where
            # their raise range is meaningfully stronger than random), their
            # fold-to-reraise is near zero: scale fold equity down so
            # bluff-reraises into tight opens die (the TAG leak). Loose openers
            # keep their fold equity, so steals survive and mirror-like
            # opponents (whose tightness is ~0) are unaffected - keeping the
            # action-profile mirror signal intact.
            fold_freq *= (1.0 - preflop_strength) ** 2

        cands: list[_Candidate] = []
        for la in state.legal_actions:
            t = la.action_type
            if t == ActionType.FOLD:
                cands.append(_Candidate(Action.fold("ev 0"), 0.0))
            elif t == ActionType.CHECK:
                cands.append(_Candidate(Action.check("check"), equity * pot))
            elif t == ActionType.CALL:
                ev = equity * pot - (1 - equity) * to_call
                # OOP blind defense (measured 6-max self-play leak): from SB/BB
                # facing a preflop raise, raw pot odds justify calling almost
                # any two cards, but postflop realization out of position is
                # poor — the BB was the worst seat in self-play, bleeding ~10x
                # the blind cost. Fold marginal hands instead of paying to see
                # flops OOP. Strong hands still defend. Heads-up is excluded:
                # measured regression vs PyPokerEngine's HonestPlayer there
                # (SB defense vs wide ranges is a different game).
                if (len(state.players) > 2 and state.street == "preflop"
                        and to_call > 0 and _is_blind(state, player_id)
                        and equity < _OOP_CALL_EQUITY):
                    cands.append(_Candidate(Action.fold("oop fold marginal"), 0.0))
                else:
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
            # cap is a chip SIZE (incremental risk); for RAISE the amount is the
            # raise-to, so compare/cap the incremental part.
            incremental = action.amount - p.street_committed
            if incremental > cap:
                capped_to = p.street_committed + int(cap)
                la = [x for x in state.legal_actions if x.action_type == action.action_type]
                if la and la[0].min_amount <= capped_to <= la[0].max_amount:
                    action = Action.raise_to(capped_to, action.reason) if action.action_type == ActionType.RAISE \
                        else Action.bet(capped_to, action.reason)
                else:
                    # Capping makes the action illegal. A fold-equity-driven
                    # raise (bluff) that can't be legally sized must not
                    # degrade into a call with garbage: call only when the call
                    # itself is +EV (semi-bluff with equity), else fold. This
                    # also starves the preflop re-raise escalation (mirror
                    # self-play coin flips) of capped-bluff calls that grow
                    # pots until bigger re-raises become legal.
                    call_la = [x for x in state.legal_actions if x.action_type == ActionType.CALL]
                    if call_la:
                        call_ev = equity * pot - (1.0 - equity) * to_call
                        # Same OOP rule in the fallback: a capped bluff raise
                        # from the blinds with a marginal hand must not sneak
                        # back into the call it was just folded for.
                        oop_marginal = (len(state.players) > 2
                                        and state.street == "preflop" and to_call > 0
                                        and _is_blind(state, player_id)
                                        and equity < _OOP_CALL_EQUITY)
                        if oop_marginal:
                            action = Action.fold("oop fallback fold")
                        elif equity >= _VALUE_EQUITY or call_ev > 0:
                            action = Action.call(call_la[0].min_amount, "risk cap fallback call")
                        else:
                            action = Action.fold("risk cap fallback fold")
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
