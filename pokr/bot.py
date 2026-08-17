from __future__ import annotations

import random

from .botdetect import BotDetector, DetectionResult
from .models import ModelManager
from .policy import Policy
from .risk import BankrollManager, RiskConfig
from .strategy import Action, ActionType, BaseStrategy


class PokerBot(BaseStrategy):
    """Our bot: opponent models + bot detection + risk + policy.

    Known leaks (for mirror exploitation): this bot over-folds to large river
    bets and under-bluffs rivers, so against a mirror it sizes river bets
    bigger. The policy's mirror mode already shifts to larger bet fractions.
    """

    MIRROR_THRESHOLD = 0.6

    def __init__(
        self,
        rng: random.Random | None = None,
        risk_cfg: RiskConfig | None = None,
        num_players: int = 6,
        mc_iters: int = 150,
        bankroll_manager: BankrollManager | None = None,
        mc_fast: bool = False,
    ) -> None:
        self.rng = rng or random.Random()
        self.models = ModelManager(num_players)
        self.detector = BotDetector()
        self.policy = Policy(self.rng, risk_cfg, mc_iters, mc_fast)
        self.num_players = num_players
        self.mirror_mode = False
        self.bankroll_manager = bankroll_manager

    def begin_session(self, bankroll: float) -> None:
        """Feed the session budget from the bankroll manager (no-op without one).

        The budget feeds Kelly sizing through policy.risk_cfg.session_budget.
        """
        if self.bankroll_manager is not None:
            self.policy.risk_cfg.session_budget = self.bankroll_manager.session_budget(bankroll)

    def decide(self, state, player_id):
        opponents = [q.id for q in state.players if q.id != player_id and not q.folded]
        if not opponents:
            return Action.check("no opponents")
        opp = self._target_opponent(state, player_id, opponents)
        summary = self.models.summary(opp)
        detection = self.detector.detect(opp, summary)
        if detection.p_mirror >= self.MIRROR_THRESHOLD:
            self.mirror_mode = True
        return self.policy.decide(state, player_id, summary, detection)

    @staticmethod
    def _target_opponent(state, player_id, opponents):
        """Whose model/detection to read: the last bettor/raiser on the current
        street (their range defines the bet we are facing), else the last
        aggressor anywhere in the hand (a preflop raiser still defines the pot
        when a later street is unopened), else the first live opponent.
        Reading opponents[0] instead made the bot apply one seat's stats to
        whoever happened to act, which mis-targets every multiway hand."""
        opp_set = set(opponents)

        def last_aggressor(street=None):
            for pid, s, a in reversed(state.action_history):
                if street is not None and s != street:
                    continue
                if pid != player_id and pid in opp_set and \
                        a.action_type in (ActionType.BET, ActionType.RAISE):
                    return pid
            return None

        return last_aggressor(state.street) or last_aggressor() or opponents[0]

    def on_hand_end(self, result, my_seat):
        self.models.observe(result, my_seat)
        self.detector.observe(result, my_seat)
