from pokr.strategy import Action, ActionType, BaseStrategy


def test_action_constructors():
    assert Action.fold().action_type == ActionType.FOLD
    assert Action.check().action_type == ActionType.CHECK
    a = Action.call(50, "call to win")
    assert a.amount == 50 and a.reason == "call to win"
    assert Action.bet(100).action_type == ActionType.BET
    assert Action.raise_to(300).amount == 300


def test_action_is_frozen():
    from dataclasses import FrozenInstanceError
    import pytest

    a = Action.fold()
    with pytest.raises(FrozenInstanceError):
        a.amount = 5


def test_base_strategy_defaults():
    b = BaseStrategy()
    assert b.on_hand_end(None, 0) is None
    try:
        b.decide(None, 0)
        assert False
    except NotImplementedError:
        pass
