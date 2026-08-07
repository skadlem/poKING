import random

from pokr.bot import PokerBot
from pokr.cards import Deck, card_from_str
from pokr.engine import PokerGame
from pokr.opponents import CallingStation
from pokr.strategy import Action, ActionType, BaseStrategy


def test_bot_learns_opponent():
    bot = PokerBot(random.Random(1))
    lineup = [bot] + [CallingStation() for _ in range(5)]
    for _ in range(10):
        g = PokerGame(lineup, [200] * 6, rng=random.Random(2), initial_dealer=0)
        g.play_hand()
    assert bot.models.summary(1).hands_observed == 10


def test_bot_decides_legally():
    bot = PokerBot(random.Random(3))
    lineup = [bot] + [CallingStation() for _ in range(5)]
    for _ in range(20):
        g = PokerGame(lineup, [200] * 6, rng=random.Random(4), initial_dealer=0)
        g.play_hand()  # engine raises IllegalAction on any bad action


def test_mirror_mode_triggered_by_detection(monkeypatch):
    from pokr.botdetect import DetectionResult
    bot = PokerBot(random.Random(5))
    # monkeypatch detection to report a mirror
    class FakeDetector:
        def observe(self, result, my_seat):
            pass

        def detect(self, opponent_id, summary):
            return DetectionResult(0.5, 0.99, 100)

    bot.detector = FakeDetector()
    # build a minimal game state
    from pokr.engine import GameState, LegalAction, PlayerView
    ps = [PlayerView(0, 200, hole=[card_from_str("As"), card_from_str("Ah")]),
          PlayerView(1, 200, hole=[card_from_str("Ks"), card_from_str("Kh")])]
    state = GameState(ps, [card_from_str("2c"), card_from_str("3c"), card_from_str("4c")],
                      pot=100, current_bet=0, min_raise=2, street="flop",
                      dealer=0, current_player=0,
                      legal_actions=[LegalAction(ActionType.CHECK), LegalAction(ActionType.BET, 2, 200)])
    bot.decide(state, 0)
    assert bot.mirror_mode is True


def test_two_identical_bots_drift_to_mirror():
    b0 = PokerBot(random.Random(1))
    b1 = PokerBot(random.Random(1))
    lineup = [b0] + [b1] + [CallingStation() for _ in range(4)]
    for h in range(200):
        # rotate the dealer so both bots experience every position equally
        g = PokerGame(lineup, [200] * 6, rng=random.Random(100 + h),
                      initial_dealer=h % 6)
        g.play_hand()
    d = b0.detector.detect(1, b0.models.summary(1))
    assert d.p_mirror > 0.25
