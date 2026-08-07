import random

from pokr.cards import card_from_str
from pokr.engine import GameState, HandResult, LegalAction, PlayerView
from pokr.opponents import LeakHunter
from pokr.strategy import Action, ActionType


def hs(*strs):
    return [card_from_str(s) for s in " ".join(strs).split()]


def make_result(actions, n=6):
    return HandResult(1, [200] * n, [200] * n, [[] for _ in range(n)], [],
                      actions, [0] * n, 2)


def test_hunter_observes_target():
    h = LeakHunter(random.Random(1), target_seat=0)
    h.on_hand_end(make_result([(0, "preflop", Action.raise_to(6, "r"))]), my_seat=1)
    assert h.model.summary().hands_observed == 1


def test_hunter_bluffs_more_when_target_folds_postflop():
    h = LeakHunter(random.Random(2), target_seat=0)
    for _ in range(40):
        h.on_hand_end(make_result([(0, "flop", Action.fold("f"))]), my_seat=1)
    ps = [PlayerView(0, 200, hole=hs("Ks Kd")),
          PlayerView(1, 200, hole=hs("7h 2d"))]
    state = GameState(ps, hs("2c 3c 4c"), pot=100, current_bet=0, min_raise=2,
                      street="flop", dealer=0, current_player=1,
                      legal_actions=[LegalAction(ActionType.CHECK),
                                     LegalAction(ActionType.BET, 2, 200)])
    bets = sum(1 for _ in range(40) if h.decide(state, 1).action_type == ActionType.BET)
    assert bets >= 20  # target folds a lot -> hunter bluffs more than half the time


def test_hunter_defaults_to_call_when_facing_bet():
    h = LeakHunter(random.Random(3), target_seat=0)
    ps = [PlayerView(0, 200, hole=hs("Ks Kd")),
          PlayerView(1, 200, hole=hs("7h 2d"))]
    state = GameState(ps, hs("2c 3c 4c"), pot=100, current_bet=20, min_raise=2,
                      street="flop", dealer=0, current_player=1,
                      legal_actions=[LegalAction(ActionType.FOLD),
                                     LegalAction(ActionType.CALL, 20, 20),
                                     LegalAction(ActionType.RAISE, 24, 200)])
    a = h.decide(state, 1)
    assert a.action_type == ActionType.CALL
