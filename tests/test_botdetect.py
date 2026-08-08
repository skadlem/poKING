import math

from pokr.botdetect import BotDetector, _hellinger
from pokr.engine import HandResult
from pokr.models import OpponentSummary, _is_round
from pokr.strategy import Action


def summary(vpip=0.25, pfr=0.12, aggr=0.4, fold=0.4, round_frac=1.0, hands=50, sizes=None):
    if sizes is not None:
        round_frac = sum(1.0 for x in sizes if _is_round(x)) / len(sizes)
    return OpponentSummary(hands, vpip, pfr, aggr, fold, 0.3,
                           round_frac)


def test_robot_like_scores_high():
    s = summary(hands=50, sizes=[3.0] * 50)
    assert BotDetector().detect(1, s).p_is_bot > 0.6


def test_humanlike_scores_low():
    s = summary(hands=50, round_frac=0.1, vpip=0.35, pfr=0.18,
                sizes=[2.37, 3.1, 4.9, 2.8, 5.5] * 10)
    assert BotDetector().detect(1, s).p_is_bot < 0.4


def test_small_sample_shrinks_to_prior():
    s = summary(hands=1, sizes=[3.0])
    p = BotDetector().detect(1, s).p_is_bot
    assert abs(p - 0.3) < 0.25


def test_no_data_returns_prior():
    d = BotDetector().detect(1, None)
    assert d.p_is_bot == 0.3 and d.p_mirror == 0.1 and d.samples == 0


def result_with(actions, n=6):
    return HandResult(1, [200] * n, [200] * n, [[] for _ in range(n)], [],
                      actions, [0] * n, 2)


def test_mirror_same_actions():
    det = BotDetector()
    for _ in range(50):
        # identical action histograms: both bots raise preflop
        acts = [(0, "preflop", Action.raise_to(6, "r")), (1, "preflop", Action.raise_to(6, "r"))]
        det.observe(result_with(acts), my_seat=0)
    d = det.detect(1, summary(hands=50))
    assert d.p_mirror > 0.6


def test_no_mirror_for_different_actions():
    det = BotDetector()
    for _ in range(50):
        acts = [(0, "preflop", Action.raise_to(6, "r")), (1, "preflop", Action.fold("f"))]
        det.observe(result_with(acts), my_seat=0)
    d = det.detect(1, summary(hands=50))
    assert d.p_mirror < 0.4


def test_hellinger_identical_is_zero():
    a = {"flop:bet": 10, "preflop:raise": 5}
    assert _hellinger(a, dict(a)) == 0.0
