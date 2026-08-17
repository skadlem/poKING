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
        ssum = OpponentSummary(50, 0.3, 0.1, 0.5, fold, fold, 1.0)
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


def _preflop_state(hole, to_call=4, pot=9, street_committed=2, stack=200):
    """BB facing a 3BB (6-chip) open: pot 9, call 4, min raise-to 8."""
    ps = [PlayerView(0, stack, hole=hole, street_committed=street_committed),
          PlayerView(1, stack, hole=hs("Ks Kd"), street_committed=6)]
    legal = [LegalAction(ActionType.FOLD),
             LegalAction(ActionType.CALL, to_call, to_call),
             LegalAction(ActionType.RAISE, 8, stack)]
    return GameState(ps, [], pot, current_bet=6, min_raise=2, street="preflop",
                     dealer=0, current_player=0, legal_actions=legal)


def _preflop_3bet_state(hole, to_call=18, pot=45, street_committed=12, stack=200):
    """Hero raised to 12, faces a 3-bet to 30: pot 45, call 18, min raise-to 34.
    Pot is big enough that the risk cap does not clip the raise, so the test
    exercises the policy (not the cap)."""
    ps = [PlayerView(0, stack, hole=hole, street_committed=street_committed),
          PlayerView(1, stack, hole=hs("Ks Kd"), street_committed=30)]
    legal = [LegalAction(ActionType.FOLD),
             LegalAction(ActionType.CALL, to_call, to_call),
             LegalAction(ActionType.RAISE, 34, stack)]
    return GameState(ps, [], pot, current_bet=30, min_raise=4, street="preflop",
                     dealer=0, current_player=0, legal_actions=legal)


def test_no_bluff_reraises_into_tight_opener(monkeypatch):
    # A tight opener (pfr == vpip: every voluntarily played hand is a raise)
    # almost never folds to a reraise. The bot must not 4-bet-bluff into a
    # tight 3-bet with garbage; fold or call only. Regression for the TAG leak
    # (raising into a ~5% premium open range).
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.25)
    p = Policy(random.Random(9))
    s = _preflop_3bet_state(hs("7h 2d"))
    tag = OpponentSummary(200, 0.06, 0.06, 0.5, 0.5, 0.7, 1.0)
    for _ in range(100):
        a = p.decide(s, 0, tag, None)
        assert a.action_type != ActionType.RAISE


def test_preflop_call_discounted_vs_tight_raiser(monkeypatch):
    # Facing a tight raiser's open with a marginal hand, the call must be
    # discounted (their range is far stronger than a random hand) and folded.
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.33)
    p = Policy(random.Random(9))
    s = _preflop_state(hs("9h 8d"))
    tag = OpponentSummary(200, 0.06, 0.06, 0.5, 0.5, 0.7, 1.0)
    for _ in range(100):
        a = p.decide(s, 0, tag, None)
        assert a.action_type == ActionType.FOLD


def test_preflop_reraise_still_allowed_vs_wide_folder(monkeypatch):
    # A wide opener who folds a lot still gets 4-bet (the steal arm must
    # survive the fold-equity scaling).
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.25)
    p = Policy(random.Random(9))
    s = _preflop_3bet_state(hs("7h 2d"))
    rand = OpponentSummary(200, 0.5, 0.15, 0.3, 0.5, 0.3, 1.0)
    acts = {p.decide(s, 0, rand, None).action_type for _ in range(200)}
    assert ActionType.RAISE in acts


def _big_pot_state(hole, pot=200, to_call=80, stack=200):
    """Deep-pot flop where the bluff risk cap (0.66xpot) sits below the legal
    raise-to, so a desired raise gets capped into an illegal amount and the
    fallback path runs."""
    ps = [PlayerView(0, stack, hole=hole, street_committed=0),
          PlayerView(1, stack, hole=hs("Ks Kd"), street_committed=80)]
    legal = [LegalAction(ActionType.FOLD),
             LegalAction(ActionType.CALL, to_call, to_call),
             LegalAction(ActionType.RAISE, to_call + 4, stack)]
    return GameState(ps, hs("2c 3c 4c"), pot=pot, current_bet=to_call,
                     min_raise=4, street="flop", dealer=0, current_player=0,
                     legal_actions=legal)


def test_capped_bluff_raise_falls_back_to_fold_not_call(monkeypatch):
    # A fold-equity-driven raise (bluff, garbage equity) that the risk cap clips
    # below the legal raise-to must not degrade into a -EV call with garbage:
    # the cap fallback must fold, never call.
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.2)
    p = Policy(random.Random(3))
    s = _big_pot_state(hs("7h 2d"))
    seen_fallback_fold = False
    for _ in range(300):
        a = p.decide(s, 0, None, None)
        assert a.reason != "risk cap fallback call"
        if a.reason == "risk cap fallback fold":
            seen_fallback_fold = True
    assert seen_fallback_fold


def test_capped_semibluff_raise_still_calls_when_call_is_ev_positive(monkeypatch):
    # A semi-bluff with enough equity that the call itself is +EV keeps the
    # call fallback when the raise is capped below legality.
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.5)
    p = Policy(random.Random(3))
    s = _big_pot_state(hs("7h 6h"))
    seen_call = False
    for _ in range(200):
        a = p.decide(s, 0, None, None)
        if a.action_type == ActionType.CALL:
            seen_call = True
    assert seen_call


def test_capped_value_raise_falls_back_to_call(monkeypatch):
    # A value raise capped below legality keeps the call fallback (call is +EV
    # with 0.9 equity).
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.9)
    p = Policy(random.Random(3))
    s = _big_pot_state(hs("As Ah"))
    seen_call = False
    for _ in range(200):
        a = p.decide(s, 0, None, None)
        if a.action_type == ActionType.CALL:
            seen_call = True
    assert seen_call


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

    tight = OpponentSummary(50, 0.10, 0.08, 0.60, 0.5, 0.5, 1.0)
    loose = OpponentSummary(50, 0.60, 0.30, 0.30, 0.5, 0.5, 1.0)
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


def _blind_state(hole, seat, dealer, committed=2, to_call=4, pot=9, stack=200, n=6):
    """6-max preflop: hero at `seat` (BB committed 2) faces a 3BB (6-chip) open:
    pot 9, call 4, min raise-to 8. Other seats committed 6 (the opener)."""
    ps = [PlayerView(i, stack, hole=(hole if i == seat else hs("Ks Kd")),
                     street_committed=(committed if i == seat else 6))
          for i in range(n)]
    legal = [LegalAction(ActionType.FOLD),
             LegalAction(ActionType.CALL, to_call, to_call),
             LegalAction(ActionType.RAISE, 8, stack)]
    return GameState(ps, [], pot, current_bet=6, min_raise=2, street="preflop",
                     dealer=dealer, current_player=seat, legal_actions=legal)


def test_oop_blind_folds_marginal_preflop_vs_raise(monkeypatch):
    # BB (dealer 3 -> BB 5) facing an open with a marginal hand: raw pot odds
    # justify the call, but OOP realization is poor, so the call must die
    # (fold or raise, never call) — regression for the self-play blind leak.
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.35)
    p = Policy(random.Random(9))
    s = _blind_state(hs("9h 8d"), seat=5, dealer=3)
    for _ in range(200):
        assert p.decide(s, 5, None, None).action_type != ActionType.CALL


def test_oop_blind_marginal_fold_survives_cap_fallback(monkeypatch):
    # With a folder-heavy opponent the bluff raise wins the softmax, gets
    # risk-capped below the legal raise-to, and must fall back to fold — not
    # sneak back into a marginal OOP call.
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.35)
    p = Policy(random.Random(9))
    s = _blind_state(hs("9h 8d"), seat=5, dealer=3)
    folder = OpponentSummary(200, 0.5, 0.2, 0.3, 0.3, 0.8, 1.0)
    for _ in range(200):
        assert p.decide(s, 5, folder, None).action_type != ActionType.CALL


def test_oop_blind_calls_strong_hand(monkeypatch):
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.55)
    p = Policy(random.Random(9))
    s = _blind_state(hs("As Kd"), seat=5, dealer=3)
    seen = {p.decide(s, 5, None, None).action_type for _ in range(200)}
    assert ActionType.CALL in seen


def test_in_position_marginal_call_still_allowed(monkeypatch):
    # Same marginal hand in position (seat 1, not a blind): the OOP fold rule
    # must not apply, the +EV call survives.
    monkeypatch.setattr("pokr.policy.monte_carlo_equity", lambda *a, **k: 0.35)
    p = Policy(random.Random(9))
    s = _blind_state(hs("9h 8d"), seat=1, dealer=3)
    seen = {p.decide(s, 1, None, None).action_type for _ in range(200)}
    assert ActionType.CALL in seen
