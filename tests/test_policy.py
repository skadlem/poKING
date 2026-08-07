import random

from pokr.botdetect import DetectionResult
from pokr.engine import GameState, LegalAction, PlayerView
from pokr.models import OpponentSummary
from pokr.policy import Policy
from pokr.risk import RiskConfig
from pokr.strategy import Action, ActionType


def hs(*strs):
    from pokr.cards import card_from_str
    return [card_from_str(s) for s in " ".join(strs).split()]


def make_state(hole, community, pot=100, to_call=0, stack=200, legal=None):
    ps = [PlayerView(0, stack, hole=hole),
          PlayerView(1, stack, hole=hs("Ks Kd"))]
    if legal is None:
        legal = []
        if to_call > 0:
            legal += [LegalAction(ActionType.FOLD),
                      LegalAction(ActionType.CALL, to_call, to_call),
                      LegalAction(ActionType.RAISE, to_call + 4, stack)]
        else:
            legal += [LegalAction(ActionType.CHECK),
                      LegalAction(ActionType.BET, 2, stack)]
    return GameState(ps, community, pot, to_call, 2, "flop", 0, 0, legal)


def _is_legal(a, state):
    if a.action_type == ActionType.FOLD:
        return a.amount == 0
    if a.action_type == ActionType.CHECK:
        return a.amount == 0
    la = [x for x in state.legal_actions if x.action_type == a.action_type]
    if not la:
        return False
    if a.action_type == ActionType.CALL:
        return a.amount == la[0].min_amount
    return la[0].min_amount <= a.amount <= la[0].max_amount


def test_ev_positive_calls(monkeypatch):
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.9)
    p = Policy(random.Random(1))
    state = make_state(hs("As Ah"), hs("Ks Kd 2c"), pot=100, to_call=10)
    act = p.decide(state, 0, None, None)
    assert act.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.BET)
    assert act.reason != ""


def test_ev_negative_folds(monkeypatch):
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.02)
    p = Policy(random.Random(1))
    state = make_state(hs("7h 2d"), hs("Ks Kd 2c"), pot=100, to_call=80)
    act = p.decide(state, 0, None, None)
    assert act.action_type == ActionType.FOLD


def test_randomization_over_repeated_states(monkeypatch):
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.85)
    p = Policy(random.Random(5))
    state = make_state(hs("As Ah"), hs("Ks Kd 2c"), pot=100, to_call=0)
    acts = {p.decide(state, 0, None, None).action_type for _ in range(100)}
    assert len(acts) >= 2


def test_bluffs_sometimes(monkeypatch):
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.2)
    p = Policy(random.Random(9))
    state = make_state(hs("7h 2d"), hs("2c 3c 4c"), pot=100, to_call=0)
    acts = [p.decide(state, 0, None, None).action_type for _ in range(200)]
    assert ActionType.BET in acts
    assert ActionType.CHECK in acts


def test_bluff_frequency_rises_with_fold_freq(monkeypatch):
    def run(fold):
        p = Policy(random.Random(3))
        s = make_state(hs("7h 2d"), hs("2c 3c 4c"), pot=100, to_call=0)
        ssum = OpponentSummary(50, 0.3, 0.1, 0.5, fold, fold, {}, [3.0] * 50, {})
        monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.2)
        return sum(1 for _ in range(300)
                   if p.decide(s, 0, ssum, None).action_type in (ActionType.BET, ActionType.RAISE))
    assert run(0.8) > run(0.1)


def test_mirror_mode_increases_bet_size(monkeypatch):
    def mean_bet(mirror):
        p = Policy(random.Random(11))
        s = make_state(hs("As Ah"), hs("Ks Kd 2c"), pot=100, to_call=0)
        det = DetectionResult(0.5, mirror, 50)
        monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.85)
        bets = []
        for _ in range(400):
            a = p.decide(s, 0, None, det)
            if a.action_type == ActionType.BET:
                bets.append(a.amount)
        return sum(bets) / max(len(bets), 1)
    assert mean_bet(0.9) > mean_bet(0.1)


def test_risk_cap_respected(monkeypatch):
    cfg = RiskConfig(max_bet_fraction_of_stack=0.2)
    p = Policy(random.Random(2), cfg)
    s = make_state(hs("As Ah"), hs("Ks Kd 2c"), pot=1000, to_call=0, stack=200)
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.95)
    for _ in range(100):
        a = p.decide(s, 0, None, None)
        if a.action_type == ActionType.BET:
            assert a.amount <= 40


def test_actions_always_legal(monkeypatch):
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.3)
    p = Policy(random.Random(4))
    for _ in range(200):
        state = make_state(hs("7h 2d"), hs("2c 3c 4c"), pot=100, to_call=0)
        a = p.decide(state, 0, None, None)
        assert _is_legal(a, state)


def test_tight_aggressive_folds_more_to_bets_than_loose(monkeypatch):
    # Same marginal hand facing a large bet. A tight+aggressive opponent's betting
    # range is far stronger than a random hand, so the bot must fold more often
    # than against a loose opponent (whose bets are closer to random).
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.45)

    def fold_count(summary):
        p = Policy(random.Random(9))
        s = make_state(hs("As Kh"), hs("2c 3c 4c"), pot=100, to_call=80, stack=400)
        return sum(1 for _ in range(300)
                   if p.decide(s, 0, summary, None).action_type == ActionType.FOLD)

    tight = OpponentSummary(50, 0.10, 0.08, 0.60, 0.5, 0.5, {}, [6.0] * 50, {})
    loose = OpponentSummary(50, 0.60, 0.30, 0.30, 0.5, 0.5, {}, [3.0] * 50, {})
    assert fold_count(tight) > fold_count(loose) + 30



def test_value_raise_not_degraded_by_cap(monkeypatch):
    # Facing a bet, with a cap above the incremental raise, the value raise must be
    # preserved (not degraded to call by comparing raise-to against the cap).
    from pokr.risk import RiskConfig
    cfg = RiskConfig(kelly_fraction=1.0, max_bet_fraction_of_stack=1.0,
                     max_bet_as_pot_fraction=10.0, session_budget=100000.0)
    p = Policy(random.Random(6), cfg)
    s = make_state(hs("As Ah"), hs("Ks Kd 2c"), pot=100, to_call=20, stack=200)
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.9)
    seen_raise = False
    for _ in range(300):
        a = p.decide(s, 0, None, None)
        if a.action_type == ActionType.RAISE:
            seen_raise = True
            assert a.amount > 20  # raise-to above the call amount
    assert seen_raise


def test_raise_cap_uses_incremental_amount(monkeypatch):
    # Regression for the cap-unit bug: with street_committed=90, a raise-to of 104 is
    # only 14 chips of incremental risk. The old code compared the raise-to (104)
    # against the cap and degraded the value raise to a call.
    from pokr.risk import RiskConfig
    cfg = RiskConfig(kelly_fraction=1.0, max_bet_fraction_of_stack=0.2,
                     max_bet_as_pot_fraction=10.0, session_budget=100000.0)
    p = Policy(random.Random(6), cfg)
    ps = [PlayerView(0, 200, hole=hs("As Ah"), street_committed=90),
          PlayerView(1, 200, hole=hs("Ks Kd"))]
    legal = [LegalAction(ActionType.FOLD),
             LegalAction(ActionType.CALL, 10, 10),
             LegalAction(ActionType.RAISE, 104, 200)]
    s = GameState(ps, hs("Ks Kd 2c"), pot=200, current_bet=100, min_raise=4,
                  street="flop", dealer=0, current_player=0, legal_actions=legal)
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.9)
    seen_raise = False
    for _ in range(200):
        a = p.decide(s, 0, None, None)
        if a.action_type == ActionType.RAISE:
            seen_raise = True
            assert a.amount >= 104  # legal raise-to, never a degraded call
            assert a.amount <= 200
    assert seen_raise
