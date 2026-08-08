from pokr.cards import card_from_str
from pokr.engine import HandResult
from pokr.models import ModelManager, OpponentModel
from pokr.strategy import Action


def make_result(actions, community=(), bb=2):
    n = 6
    return HandResult(1, [200] * n, [200] * n, [[] for _ in range(n)],
                      [card_from_str(c) for c in community], actions, [0] * n, bb)


def test_vpip_pfr():
    m = OpponentModel()
    acts = [(1, "preflop", Action.raise_to(6, "r")), (0, "preflop", Action.fold("f"))]
    m.update(make_result(acts), observer_id=0, target_id=1)
    s = m.summary()
    assert s.vpip == 1.0
    assert s.pfr == 1.0


def test_vpip_excludes_blinds_only():
    m = OpponentModel()
    acts = [(1, "preflop", Action.fold("f"))]
    m.update(make_result(acts), observer_id=0, target_id=1)
    assert m.summary().vpip == 0.0


def test_aggression_freq():
    m = OpponentModel()
    acts = [
        (1, "flop", Action.bet(10, "b")),
        (1, "turn", Action.bet(20, "b")),
        (1, "river", Action.call(5, "c")),
    ]
    m.update(make_result(acts, community=["2c", "3c", "4c"]), observer_id=0, target_id=1)
    s = m.summary()
    assert s.aggression_freq == 2 / 3


def test_fold_to_cbet():
    m = OpponentModel()
    acts = [
        (1, "preflop", Action.raise_to(6, "r")),
        (2, "preflop", Action.call(4, "c")),
        (2, "flop", Action.bet(6, "b")),
        (1, "flop", Action.fold("f")),
    ]
    m.update(make_result(acts, community=["2c", "3c", "4c"]), observer_id=0, target_id=1)
    assert m.summary().fold_to_cbet == 1.0


def test_fold_to_cbet_no_fold():
    m = OpponentModel()
    acts = [
        (1, "preflop", Action.raise_to(6, "r")),
        (2, "preflop", Action.call(4, "c")),
        (2, "flop", Action.bet(6, "b")),
        (1, "flop", Action.call(6, "c")),
    ]
    m.update(make_result(acts, community=["2c", "3c", "4c"]), observer_id=0, target_id=1)
    assert m.summary().fold_to_cbet == 0.0


def test_fold_rate_postflop():
    m = OpponentModel()
    acts = [(1, "flop", Action.fold("f")), (1, "turn", Action.check("x"))]
    m.update(make_result(acts, community=["2c", "3c", "4c"]), observer_id=0, target_id=1)
    assert m.summary().fold_rate_postflop == 0.5


def test_raise_sizes_exact():
    m = OpponentModel()
    m.update(make_result([(1, "preflop", Action.raise_to(6, "r"))], bb=2), 0, 1)   # 3x
    m.update(make_result([(1, "preflop", Action.raise_to(10, "r"))], bb=2), 0, 1)  # 5x
    s = m.summary()
    assert s.raise_sizes_exact == [3.0, 5.0]


def test_manager_excludes_observer():
    mgr = ModelManager(6)
    mgr.observe(make_result([(1, "preflop", Action.call(1, "c"))]), observer_id=0)
    assert mgr.summary(1).hands_observed == 1
    assert mgr.summary(0).hands_observed == 0
