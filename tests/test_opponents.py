import random
from collections import Counter

from pokr.cards import Deck, card_from_str
from pokr.engine import PokerGame
from pokr.opponents import CallingStation, Maniac, RandomBot, TightAggressive
from pokr.strategy import Action, ActionType, BaseStrategy


class Scripted(BaseStrategy):
    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0

    def decide(self, state, pid):
        if self._i < len(self._actions):
            a = self._actions[self._i]
            self._i += 1
            return a
        return self._actions[-1]


def make_game(strategies, deck_cards=None, dealer=0, stacks=None):
    deck = Deck(random.Random(1), [card_from_str(s) for s in deck_cards], shuffle=False) if deck_cards else None
    return PokerGame(strategies, stacks or [200] * len(strategies),
                     rng=random.Random(1), initial_dealer=dealer, deck=deck)


def test_calling_station_calls_bet():
    # dealer=0, n=3: SB=1 (CS), BB=2, first=0. Seat0 raises to 10, CS calls, seat2 folds.
    cs = CallingStation()
    g = make_game(
        [Scripted([Action.raise_to(10, "r"), Action.check("x"), Action.check("x"), Action.check("x")]),
         cs,
         Scripted([Action.fold("f")])],
        dealer=0,
    )
    r = g.play_hand()
    cs_actions = [a for (t, s, a) in r.actions if t == 1]
    assert cs_actions[0].action_type == ActionType.CALL
    assert cs_actions[0].amount == 9  # to_call = 10 - SB 1


def test_calling_station_never_folds():
    rng = random.Random(5)
    lineup = [Maniac(random.Random(50 + i)) if i % 2 == 0 else CallingStation() for i in range(6)]
    g = make_game(lineup, stacks=[200] * 6)
    for _ in range(20):
        r = g.play_hand()
        for (t, s, a) in r.actions:
            if isinstance(lineup[t], CallingStation):
                assert a.action_type != ActionType.FOLD


def test_tag_raises_premium_preflop():
    # n=3, dealer=2: SB=0 (TAG), BB=1, first=2. Seat2 folds, TAG (SB) raises 3BB with AKs,
    # BB calls. TAG's first action must be a 6-chip raise.
    tag = TightAggressive(random.Random(1))
    g = make_game(
        [tag,
         Scripted([Action.call(4, "c"), Action.check("x"), Action.check("x"),
                   Action.check("x"), Action.check("x")]),
         Scripted([Action.fold("f")])],
        deck_cards=["As", "Ks", "2c", "3c", "4c", "5c", "6c", "7c", "8c", "9c", "Jd"],
        dealer=2,
    )
    r = g.play_hand()
    tag_actions = [a for (t, s, a) in r.actions if t == 0]
    assert tag_actions[0].action_type == ActionType.RAISE
    assert tag_actions[0].amount == 6  # 3BB


def test_tag_checks_junk_when_unraised():
    # n=3, dealer=1: SB=2, BB=0 (TAG), first=1. Seat1 folds, SB calls, TAG checks junk as BB.
    tag = TightAggressive(random.Random(2))
    g = make_game(
        [tag,
         Scripted([Action.fold("f")]),
         Scripted([Action.call(1, "c"), Action.check("x"), Action.check("x"),
                   Action.check("x"), Action.check("x")])],
        deck_cards=["7h", "2d", "3d", "6h", "9s", "Qc", "Ks", "3c", "8d", "Jh", "Qh"],
        dealer=1,
    )
    r = g.play_hand()
    tag_actions = [a for (t, s, a) in r.actions if t == 0]
    assert tag_actions[0].action_type == ActionType.CHECK


def test_maniac_raises_most_of_the_time():
    class Flex(BaseStrategy):
        """Calls when facing a bet, checks otherwise."""

        def decide(self, state, pid):
            p = state.players[pid]
            to_call = state.current_bet - p.street_committed
            if to_call > 0:
                return Action.call(min(to_call, p.stack), "flex call")
            return Action.check("flex check")

    rng = random.Random(7)
    maniac = Maniac(rng)
    lineup = [maniac] + [Flex() for _ in range(5)]
    g = make_game(lineup, stacks=[200] * 6)
    counts = Counter()
    for _ in range(30):
        r = g.play_hand()
        for (t, s, a) in r.actions:
            if t == 0:
                counts[a.action_type] += 1
    assert counts[ActionType.RAISE] + counts[ActionType.BET] > counts[ActionType.CALL] + counts[ActionType.FOLD]


def test_random_bot_returns_legal_actions():
    lineup = [RandomBot(random.Random(1)) for _ in range(6)]
    g = make_game(lineup, stacks=[200] * 6)
    for _ in range(50):
        g.play_hand()  # engine would raise IllegalAction on any bad action
