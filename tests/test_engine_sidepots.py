import random

from pokr.cards import Deck, card_from_str
from pokr.engine import PokerGame
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


def make_game(strategies, deck_cards, stacks, dealer=0):
    deck = Deck(random.Random(1), [card_from_str(s) for s in deck_cards], shuffle=False)
    return PokerGame(strategies, stacks, rng=random.Random(1), initial_dealer=dealer, deck=deck)


def test_side_pot_multiway():
    # dealer=0, n=3: SB=1, BB=2, first preflop=0.
    # A(0) all-in 50, B(1) calls, C(2) raises to 150, B calls. B,C check down.
    # Hands: A=AA (best), B=KK, C=QQ. Board: 2c3c4c5c6c.
    g = make_game(
        [Scripted([Action.raise_to(50, "a"), Action.check("x"), Action.check("x"), Action.check("x")]),
         Scripted([Action.call(49, "b"), Action.call(100, "b"), Action.check("x"),
                   Action.check("x"), Action.check("x"), Action.check("x")]),
         Scripted([Action.raise_to(150, "c"), Action.check("x"), Action.check("x"), Action.check("x")])],
        deck_cards=["As", "Ah", "Ks", "Kh", "Qs", "Qh", "2c", "3d", "4h", "5s", "9c"],
        stacks=[50, 200, 200], dealer=0,
    )
    r = g.play_hand()
    # A wins main (150) -> +100; B wins side (200) -> +50; C loses all -> -150
    assert r.winnings == [100, 50, -150]


def test_split_pot_with_odd_chip():
    # dealer=0, n=3: SB=1, BB=2, first=0. A(0) all-in 51, B(1) all-in 50 (SB 1 + call 50),
    # C(2) all-in 49 (BB 2 + call 49). A,B tie with AA; C has KK. Pot = 153.
    # Split 153 -> 76 each + 1 odd chip to first winner after button (seat 1 = B).
    g = make_game(
        [Scripted([Action.raise_to(51, "a")]),
         Scripted([Action.call(50, "b")]),
         Scripted([Action.call(49, "c")])],
        deck_cards=["As", "Ah", "Ad", "Ac", "Ks", "Kh", "2c", "3d", "4h", "5s", "9c"],
        stacks=[51, 51, 51], dealer=0,
    )
    r = g.play_hand()
    assert r.winnings == [25, 26, -51]


def test_all_in_runout_deals_full_board():
    # 6 players, dealer=0: SB=1, BB=2, first preflop=3. Seats 3,4,5 fold, seat0 shoves,
    # seat1 (SB) calls all-in, seat2 (BB) folds. Board runs out for the showdown.
    g = make_game(
        [Scripted([Action.raise_to(200, "s")]),
         Scripted([Action.call(199, "c")]),
         Scripted([Action.fold("f")]),
         Scripted([Action.fold("f")]),
         Scripted([Action.fold("f")]),
         Scripted([Action.fold("f")])],
        deck_cards=["As", "Ah", "Ks", "Kh", "2d", "3d", "4d", "5d", "6d",
                    "7d", "8d", "9d", "2c", "3c", "4c", "5c", "6c"],
        stacks=[200, 200, 200, 200, 200, 200], dealer=0,
    )
    r = g.play_hand()
    assert len(r.community) == 5


def test_conservation_stress_random_bots():
    class LocalRandomBot(BaseStrategy):
        def __init__(self, rng):
            self.rng = rng

        def decide(self, state, pid):
            la = self.rng.choice(state.legal_actions)
            if la.action_type in (ActionType.BET, ActionType.RAISE):
                amt = self.rng.randint(la.min_amount, la.max_amount)
                cls = Action.bet if la.action_type == ActionType.BET else Action.raise_to
                return cls(amt, "rand")
            if la.action_type == ActionType.CALL:
                return Action.call(la.min_amount, "rand")
            if la.action_type == ActionType.FOLD:
                return Action.fold("rand")
            return Action.check("rand")

    lineup = [LocalRandomBot(random.Random(100 + i)) for i in range(6)]
    stacks = [200] * 6
    for _ in range(200):
        g = PokerGame(lineup, stacks, rng=random.Random(9), initial_dealer=0)
        r = g.play_hand()
        assert sum(r.winnings) == 0
        assert sum(r.ending_stacks) == sum(r.starting_stacks)
        assert all(s >= 0 for s in r.ending_stacks)
        stacks = r.ending_stacks



def test_zero_stack_player_folded_at_deal():
    # dealer=0, HU: SB=0 has 0 chips (folded at deal), BB=1 posts 2, wins uncontested.
    # Net: BB gets his own blind back (winnings [0, 0]); seat 0 never wins a pot it
    # didn't contribute to, and chips are conserved.
    g = make_game(
        [Scripted([Action.check("x")]), Scripted([Action.check("x")])],
        deck_cards=["As", "Ah", "Ks", "Kh"],
        stacks=[0, 200], dealer=0,
    )
    r = g.play_hand()
    assert r.winnings == [0, 0]
    assert sum(r.ending_stacks) == sum(r.starting_stacks)
