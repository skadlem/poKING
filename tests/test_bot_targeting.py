import random

from pokr.bot import PokerBot
from pokr.engine import GameState, LegalAction, PlayerView
from pokr.strategy import Action, ActionType


def _state(history, current_bet, to_call):
    ps = [PlayerView(i, 200, hole=[]) for i in range(4)]
    from pokr.cards import card_from_str
    ps[0].hole = [card_from_str("As"), card_from_str("Ah")]
    legal = []
    if to_call > 0:
        legal = [LegalAction(ActionType.FOLD),
                 LegalAction(ActionType.CALL, to_call, to_call),
                 LegalAction(ActionType.RAISE, current_bet + 2, 200)]
    else:
        legal = [LegalAction(ActionType.CHECK), LegalAction(ActionType.BET, 2, 200)]
    return GameState(ps, [], pot=20, current_bet=current_bet, min_raise=2,
                     street="preflop", dealer=0, current_player=0,
                     legal_actions=legal, action_history=history)


def _spy_target(bot, state):
    seen = []
    orig = bot.models.summary

    def spy(tid):
        seen.append(tid)
        return orig(tid)

    bot.models.summary = spy
    bot.decide(state, 0)
    return seen


def test_bot_targets_last_aggressor_not_lowest_seat():
    # Seat 1 called, seat 3 raised, seat 1 called back: the model that matters
    # is seat 3's (the aggressor), not seat 1's (lowest live opponent).
    bot = PokerBot(random.Random(5), mc_iters=5)
    history = [
        (1, "preflop", Action.call(2, "call")),
        (3, "preflop", Action.raise_to(6, "raise")),
        (1, "preflop", Action.call(4, "call")),
    ]
    state = _state(history, current_bet=6, to_call=4)
    assert _spy_target(bot, state) == [3]


def test_bot_targets_prior_street_aggressor_when_unopened():
    # No action yet this street: fall back to the last aggressor anywhere in
    # the history (the preflop raiser defines the pot's range).
    bot = PokerBot(random.Random(5), mc_iters=5)
    history = [
        (2, "preflop", Action.raise_to(6, "raise")),
        (1, "preflop", Action.call(4, "call")),
    ]
    state = _state(history, current_bet=0, to_call=0)
    state.street = "flop"
    state.legal_actions = [LegalAction(ActionType.CHECK),
                           LegalAction(ActionType.BET, 2, 200)]
    assert _spy_target(bot, state) == [2]


def test_bot_targets_first_live_opponent_without_aggressor():
    bot = PokerBot(random.Random(5), mc_iters=5)
    history = [(1, "preflop", Action.call(2, "call")),
               (3, "preflop", Action.check("check"))]
    state = _state(history, current_bet=0, to_call=0)
    state.street = "flop"
    state.legal_actions = [LegalAction(ActionType.CHECK),
                           LegalAction(ActionType.BET, 2, 200)]
    assert _spy_target(bot, state) == [1]
