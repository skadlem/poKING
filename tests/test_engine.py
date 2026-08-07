import random

import pytest

from pokr.cards import Deck, card_from_str
from pokr.engine import IllegalAction, PokerGame
from pokr.strategy import Action, ActionType, BaseStrategy


class Scripted(BaseStrategy):
    """Plays actions in order; repeats the last one forever."""

    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0

    def decide(self, state, pid):
        if self._i < len(self._actions):
            a = self._actions[self._i]
            self._i += 1
            return a
        return self._actions[-1]


class Recorder(BaseStrategy):
    def __init__(self):
        self.calls = []

    def decide(self, state, pid):
        self.calls.append(pid)
        # NOTE (task 6 fix): UTG faces the BB blind, so a check would be
        # IllegalAction. Fold is the legal action that still proves who acts first.
        return Action.fold("r")


def make_game(strategies, deck_cards=None, dealer=0, stacks=None):
    deck = Deck(random.Random(1), [card_from_str(s) for s in deck_cards], shuffle=False) if deck_cards else None
    return PokerGame(strategies, stacks or [200] * len(strategies),
                     rng=random.Random(1), initial_dealer=dealer, deck=deck)


def test_check_when_facing_bet_raises():
    g = make_game([Scripted([Action.check("x")]), Scripted([Action.check("x")])],
                  stacks=[200, 200], dealer=0)
    with pytest.raises(IllegalAction):
        g.play_hand()


def test_fold_when_no_bet_raises():
    # 3 players; everyone checks preflop (SB calls, BB checks), then UTG folds postflop with no bet
    g = make_game(
        [Scripted([Action.call(1, "c"), Action.fold("f")]),      # UTG seat 0: call 1 (SB)
         Scripted([Action.check("x"), Action.check("x")]),       # BB seat 1
         Scripted([Action.fold("f")])],                          # seat 2 folds preflop
        stacks=[200, 200, 200], dealer=2,
    )
    with pytest.raises(IllegalAction):
        g.play_hand()


def test_raise_below_min_raises():
    # 6 players, dealer 0: SB=1, BB=2, preflop first actor = 3 (UTG)
    # UTG raises to 3, but BB=2 and min raise-to is 4
    g = make_game(
        [Scripted([Action.fold("f")]),        # 0 button
         Scripted([Action.fold("f")]),        # 1 SB
         Scripted([Action.check("x")]),       # 2 BB
         Scripted([Action.raise_to(3, "r")]), # 3 UTG -> illegal
         Scripted([Action.fold("f")]),        # 4
         Scripted([Action.fold("f")])],       # 5
        dealer=0,
    )
    with pytest.raises(IllegalAction):
        g.play_hand()


def test_first_preflop_actor_is_utg():
    r = Recorder()
    g = make_game(
        [Scripted([Action.fold("f")]), Scripted([Action.fold("f")]),
         Scripted([Action.check("x")]), r,
         Scripted([Action.fold("f")]), Scripted([Action.fold("f")])],
        dealer=0,
    )
    g.play_hand()
    assert r.calls == [3]


def test_everyone_folds_to_bb_wins_blinds():
    g = make_game(
        [Scripted([Action.fold("f")]),        # 0 button
         Scripted([Action.fold("f")]),        # 1 SB
         Scripted([Action.check("x")]),       # 2 BB wins
         Scripted([Action.fold("f")]),        # 3
         Scripted([Action.fold("f")]),        # 4
         Scripted([Action.fold("f")])],       # 5
        dealer=0,
    )
    r = g.play_hand()
    # BB nets +1: wins pot of 3 having posted the 2-chip blind.
    assert r.winnings == [0, -1, 1, 0, 0, 0]


def test_showdown_award_hole_cards():
    # HU, dealer 0 -> SB=0 (AA), BB=1 (KK). SB shoves, BB calls. Board runs out.
    # Board 2c 3d 4h 5s 8c is ragged (no straight/flush): AA > KK decides.
    g = make_game(
        [Scripted([Action.raise_to(200, "shove")]), Scripted([Action.call(198, "call")])],
        deck_cards=["As", "Ah", "Ks", "Kh", "2c", "3d", "4h", "5s", "8c"],
        stacks=[200, 200], dealer=0,
    )
    r = g.play_hand()
    assert r.winnings == [200, -200]
    assert len(r.community) == 5


def test_winnings_sum_to_zero():
    # HU, dealer=0: SB=0, BB=1, first=0. SB calls the blind, BB checks, then check down.
    g = make_game(
        [Scripted([Action.call(1, "c"), Action.check("x"), Action.check("x"), Action.check("x")]),
         Scripted([Action.check("x"), Action.check("x"), Action.check("x")])],
        stacks=[200, 200], dealer=0,
    )
    r = g.play_hand()
    assert sum(r.winnings) == 0
    assert sum(r.ending_stacks) == sum(r.starting_stacks)


def test_community_dealt_progressively():
    # 2 players, both all-in preflop -> full board dealt
    g = make_game(
        [Scripted([Action.raise_to(200, "s")]), Scripted([Action.call(198, "c")])],
        deck_cards=["As", "Ah", "Ks", "Kh", "2c", "3c", "4c", "5c", "6c"],
        stacks=[200, 200], dealer=0,
    )
    r = g.play_hand()
    assert len(r.community) == 5


def test_first_actor_responds_to_bet_after_wrap():
    # Regression test for the engine wrap bug: 6 players, dealer=0 (SB=1, BB=2, first preflop=3).
    # Preflop: 3,4,5,0 fold; SB (1) calls 1; BB (2) checks -> heads-up 1 vs 2.
    # Flop: first_actor=1 (SB) checks, BB bets 20, wrap over folded 3,4,5,0 back to 1.
    # Correct poker: seat 1 must respond to the bet (here: fold).
    g = make_game(
        [Scripted([Action.fold("f")]),                        # 0 button folds preflop
         Scripted([Action.call(1, "c"), Action.check("x"), Action.fold("f")]),  # 1 SB
         Scripted([Action.check("x"), Action.bet(20, "b")]),  # 2 BB
         Scripted([Action.fold("f")]),                        # 3 folds preflop
         Scripted([Action.fold("f")]),                        # 4 folds preflop
         Scripted([Action.fold("f")])],                       # 5 folds preflop
        stacks=[200] * 6, dealer=0,
    )
    r = g.play_hand()
    seat1 = [a for (t, s, a) in r.actions if t == 1 and s == "flop"]
    assert [a.action_type for a in seat1] == [ActionType.CHECK, ActionType.FOLD]
    assert r.winnings == [0, -2, 2, 0, 0, 0]
