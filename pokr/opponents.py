from __future__ import annotations

import random

from .cards import evaluate_hand
from .models import OpponentModel
from .strategy import Action, ActionType, BaseStrategy


class CallingStation(BaseStrategy):
    """Never folds to a callable bet; checks when there is no bet."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def decide(self, state, player_id):
        p = state.players[player_id]
        to_call = state.current_bet - p.street_committed
        if to_call > 0:
            return Action.call(min(to_call, p.stack), "calling station calls")
        return Action.check("calling station checks")


class RandomBot(BaseStrategy):
    """Uniform random over legal actions."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def decide(self, state, player_id):
        la = self.rng.choice(state.legal_actions)
        if la.action_type in (ActionType.BET, ActionType.RAISE):
            amt = self.rng.randint(la.min_amount, la.max_amount)
            cls = Action.bet if la.action_type == ActionType.BET else Action.raise_to
            return cls(amt, "random")
        if la.action_type == ActionType.CALL:
            return Action.call(la.min_amount, "random")
        if la.action_type == ActionType.FOLD:
            return Action.fold("random")
        return Action.check("random")


def _is_premium(hole) -> bool:
    a, b = hole
    hi, lo = sorted((a.rank, b.rank), reverse=True)
    if a.rank == b.rank:
        return a.rank >= 10  # TT+
    if hi == 14 and lo == 13:
        return True  # AK
    if hi == 14 and lo == 12:
        return True  # AQ
    if hi == 14 and lo == 11 and a.suit == b.suit:
        return True  # AJs
    if hi == 13 and lo == 12 and a.suit == b.suit:
        return True  # KQs
    return False


class TightAggressive(BaseStrategy):
    """Raises 3BB preflop with premium hands; plays made hands postflop."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def decide(self, state, player_id):
        p = state.players[player_id]
        to_call = state.current_bet - p.street_committed
        if state.street == "preflop":
            if _is_premium(p.hole):
                target = max(3 * state.min_raise, state.current_bet + state.min_raise) \
                    if to_call > 0 else 3 * state.min_raise
                la = [x for x in state.legal_actions if x.action_type == ActionType.RAISE]
                if la:
                    amt = min(max(target, la[0].min_amount), la[0].max_amount)
                    return Action.raise_to(amt, "tag premium")
                if la := [x for x in state.legal_actions if x.action_type == ActionType.CALL]:
                    return Action.call(la[0].min_amount, "tag premium call")
                return Action.check("tag premium check")
            if to_call > 0:
                return Action.fold("tag folds junk")
            return Action.check("tag checks junk")
        # postflop
        score = evaluate_hand(p.hole + state.community)
        cat = score[0]
        if to_call > 0:
            if cat >= 1:  # at least one pair
                return Action.call(min(to_call, p.stack), "tag calls made hand")
            return Action.fold("tag folds weak")
        if cat >= 2:  # two pair or better: bet 2/3 pot
            bet = max(2, int(state.pot * 2 / 3))
            la = [x for x in state.legal_actions if x.action_type == ActionType.BET]
            if la:
                amt = min(max(bet, la[0].min_amount), la[0].max_amount)
                return Action.bet(amt, "tag value bet")
        return Action.check("tag checks")


class Maniac(BaseStrategy):
    """Aggressive: raises ~60%, calls ~30%, folds ~10%; bets pot when checked to."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def decide(self, state, player_id):
        p = state.players[player_id]
        to_call = state.current_bet - p.street_committed
        r = self.rng.random()
        if to_call > 0:
            raise_la = [x for x in state.legal_actions if x.action_type == ActionType.RAISE]
            call_la = [x for x in state.legal_actions if x.action_type == ActionType.CALL]
            if r < 0.6 and raise_la:
                amt = self.rng.randint(raise_la[0].min_amount, raise_la[0].max_amount)
                return Action.raise_to(amt, "maniac raise")
            if r < 0.9 and call_la:
                return Action.call(call_la[0].min_amount, "maniac call")
            return Action.fold("maniac fold")
        bet_la = [x for x in state.legal_actions if x.action_type == ActionType.BET]
        if r < 0.8 and bet_la:
            amt = min(max(state.pot, bet_la[0].min_amount), bet_la[0].max_amount)
            return Action.bet(amt, "maniac bet")
        return Action.check("maniac check")


class LeakHunter(BaseStrategy):
    """Exploitability proxy: models the bot's action frequencies and counter-adjusts."""

    def __init__(self, rng: random.Random | None = None, target_seat: int = 0) -> None:
        self.rng = rng or random.Random()
        self.target_seat = target_seat
        self.model = OpponentModel()

    def on_hand_end(self, result, my_seat):
        self.model.update(result, my_seat, self.target_seat)

    def decide(self, state, player_id):
        p = state.players[player_id]
        to_call = state.current_bet - p.street_committed
        s = self.model.summary()
        if s.hands_observed < 5:
            if to_call > 0:
                return Action.call(min(to_call, p.stack), "hunter default call")
            return Action.check("hunter default check")

        bluff_mode = s.fold_rate_postflop > 0.6
        tight_mode = s.aggression_freq > 0.6
        call_wide = s.vpip > 0.5

        if state.street != "preflop":
            score = evaluate_hand(p.hole + state.community)
            cat = score[0]
        else:
            cat = 2 if _is_premium(p.hole) else 0

        if to_call > 0:
            if tight_mode and cat < 1:
                return Action.fold("hunter folds to aggressive target")
            if call_wide and cat >= 1:
                return Action.call(min(to_call, p.stack), "hunter calls wide vs loose target")
            if cat >= 2 or not tight_mode:
                return Action.call(min(to_call, p.stack), "hunter call")
            return Action.fold("hunter fold")
        # checked to
        bet_la = [x for x in state.legal_actions if x.action_type == ActionType.BET]
        if bluff_mode and bet_la and (cat < 2 or self.rng.random() < 0.5):
            amt = min(max(int(state.pot * 0.5), bet_la[0].min_amount), bet_la[0].max_amount)
            return Action.bet(amt, "hunter bluff vs folder")
        if cat >= 2 and bet_la:
            amt = min(max(int(state.pot * 0.66), bet_la[0].min_amount), bet_la[0].max_amount)
            return Action.bet(amt, "hunter value bet")
        return Action.check("hunter check")
