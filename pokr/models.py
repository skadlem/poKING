from __future__ import annotations

from dataclasses import dataclass

from .strategy import ActionType


@dataclass
class OpponentSummary:
    hands_observed: int
    vpip: float
    pfr: float
    aggression_freq: float
    fold_to_cbet: float
    fold_rate_postflop: float
    raise_sizes_exact: list[float]


class OpponentModel:
    """Per-opponent statistics, updated from HandResult after each hand."""

    def __init__(self) -> None:
        self.hands = 0
        self.vpip_n = 0
        self.pfr_n = 0
        self.postflop_aggr = 0
        self.postflop_calls = 0
        self.postflop_actions = 0
        self.postflop_folds = 0
        self.cbet_faced = 0
        self.cbet_fold = 0
        self.raise_sizes: list[float] = []

    def update(self, result, observer_id: int, target_id: int) -> None:
        if target_id == observer_id:
            return
        self.hands += 1
        preflop = [a for (t, s, a) in result.actions if t == target_id and s == "preflop"]
        voluntary = any(a.action_type in (ActionType.CALL, ActionType.BET, ActionType.RAISE) for a in preflop)
        if voluntary:
            self.vpip_n += 1
        raised = any(a.action_type in (ActionType.BET, ActionType.RAISE) for a in preflop)
        if raised:
            self.pfr_n += 1
            for a in preflop:
                if a.action_type == ActionType.RAISE:
                    self.raise_sizes.append(a.amount / result.big_blind)
                    break
        postflop = [a for (t, s, a) in result.actions
                    if t == target_id and s in ("flop", "turn", "river")]
        self.postflop_actions += len(postflop)
        self.postflop_folds += sum(1 for a in postflop if a.action_type == ActionType.FOLD)
        self.postflop_aggr += sum(1 for a in postflop if a.action_type in (ActionType.BET, ActionType.RAISE))
        self.postflop_calls += sum(1 for a in postflop if a.action_type == ActionType.CALL)
        # fold-to-cbet: raised preflop, saw flop, faced a flop bet before first flop action
        flop_actions = [(t, s, a) for (t, s, a) in result.actions if s == "flop"]
        target_idx = next((i for i, (t, s, a) in enumerate(flop_actions) if t == target_id), None)
        if raised and len(result.community) >= 3 and target_idx is not None:
            faced = any(a.action_type in (ActionType.BET, ActionType.RAISE)
                        for _, _, a in flop_actions[:target_idx])
            if faced:
                self.cbet_faced += 1
                if flop_actions[target_idx][2].action_type == ActionType.FOLD:
                    self.cbet_fold += 1

    def summary(self) -> OpponentSummary:
        denom = max(self.hands, 1)
        agg_denom = max(self.postflop_aggr + self.postflop_calls, 1)
        return OpponentSummary(
            hands_observed=self.hands,
            vpip=self.vpip_n / denom,
            pfr=self.pfr_n / denom,
            aggression_freq=self.postflop_aggr / agg_denom,
            fold_to_cbet=(self.cbet_fold / self.cbet_faced) if self.cbet_faced else 0.0,
            fold_rate_postflop=(self.postflop_folds / self.postflop_actions) if self.postflop_actions else 0.0,
            raise_sizes_exact=list(self.raise_sizes),
        )


class ModelManager:
    def __init__(self, num_players: int) -> None:
        self._models = {i: OpponentModel() for i in range(num_players)}

    def observe(self, result, observer_id: int) -> None:
        for tid in self._models:
            if tid != observer_id:
                self._models[tid].update(result, observer_id, tid)

    def summary(self, target_id: int) -> OpponentSummary:
        return self._models[target_id].summary()
