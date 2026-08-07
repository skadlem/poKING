from __future__ import annotations

import random

from .botdetect import BotDetector, DetectionResult
from .bankroll import BankrollManager
from .models import ModelManager
from .policy import Policy
from .risk import RiskConfig
from .strategy import Action, BaseStrategy


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
    ) -> None:
        self.rng = rng or random.Random()
        self.models = ModelManager(num_players)
        self.detector = BotDetector()
        self.policy = Policy(self.rng, risk_cfg, mc_iters)
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
        opp = opponents[0]
        summary = self.models.summary(opp)
        detection = self.detector.detect(opp, summary)
        if detection.p_mirror >= self.MIRROR_THRESHOLD:
            self.mirror_mode = True
        return self.policy.decide(state, player_id, summary, detection)

    def on_hand_end(self, result, my_seat):
        self.models.observe(result, my_seat)
        self.detector.observe(result, my_seat)
