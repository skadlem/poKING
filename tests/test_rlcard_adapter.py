import random

import pytest

rlcard = pytest.importorskip("rlcard")

from pokr.cards import Card, card_from_str  # noqa: E402
from pokr.connector import available_plugins, build_strategy  # noqa: E402
from pokr.engine import GameState, LegalAction, PlayerView, PokerGame  # noqa: E402
from pokr.opponents import CallingStation, RandomBot  # noqa: E402
from pokr.rlcard_adapter import (  # noqa: E402
    RlcardAdapter,
    _our_action,
    _raw_state,
    _rl_card_string,
    _rl_legal_actions,
    _STAGE_NAMES,
    _rl_refs,
)
from pokr.strategy import ActionType  # noqa: E402
from rlcard.games.nolimitholdem.round import Action as RlAction  # noqa: E402


def hs(*strs):
    return [card_from_str(s) for s in " ".join(strs).split()]


def _state(hole, community=(), pot=100, to_call=0, stack=200, street="flop",
           legal=None, committed=0):
    ps = [PlayerView(0, stack, hole=hole, street_committed=committed),
          PlayerView(1, stack, hole=hs("Ks Kd"), street_committed=committed + to_call)]
    if legal is None:
        legal = []
        if to_call > 0:
            legal += [LegalAction(ActionType.FOLD),
                      LegalAction(ActionType.CALL, to_call, to_call),
                      LegalAction(ActionType.RAISE, to_call + 4, stack)]
        else:
            legal += [LegalAction(ActionType.CHECK),
                      LegalAction(ActionType.BET, 2, stack)]
    return GameState(ps, list(community), pot, to_call, 2, street, 0, 0, legal)


# -- translation ---------------------------------------------------------

def test_card_string_translation():
    assert _rl_card_string(Card(14, 3)) == "SA"   # ace of spades
    assert _rl_card_string(Card(2, 0)) == "C2"
    assert _rl_card_string(Card(10, 2)) == "HT"


def test_card_index_matches_rlcard_json():
    card2index = _rl_refs()[1]
    assert card2index["SA"] == 0
    assert card2index["CK"] == 51


def test_street_mapping():
    assert _STAGE_NAMES == {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def test_raw_state_fields():
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=150, to_call=0, stack=200)
    raw = _raw_state(s, 0)
    assert raw["hand"] == ["SA", "HA"]
    assert raw["public_cards"] == ["C2", "C3", "C4"]
    assert raw["pot"] == 150
    assert raw["stage"] == 1
    assert raw["stakes"] == [200, 200]


def test_rl_legal_actions_when_checked_to():
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=100, to_call=0, stack=200)
    legal = set(_rl_legal_actions(s, s.players[0]))
    assert RlAction.CHECK_CALL in legal
    assert RlAction.FOLD not in legal
    assert RlAction.RAISE_POT in legal or RlAction.ALL_IN in legal


def test_rl_legal_actions_when_facing_bet():
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=150, to_call=50, stack=200)
    legal = set(_rl_legal_actions(s, s.players[0]))
    assert RlAction.FOLD in legal
    assert RlAction.CHECK_CALL in legal


def test_rl_legal_actions_all_in_call_no_raise():
    # Calling would put us all-in (to_call >= stack): rlcard removes the
    # raise actions.
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=300, to_call=250, stack=200)
    legal = set(_rl_legal_actions(s, s.players[0]))
    assert RlAction.FOLD in legal
    assert RlAction.CHECK_CALL in legal
    assert RlAction.ALL_IN not in legal


# -- action mapping back ---------------------------------------------------

def test_check_call_maps_to_check():
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=100, to_call=0, stack=200)
    a = _our_action(s, 0, RlAction.CHECK_CALL)
    assert a.action_type == ActionType.CHECK


def test_check_call_maps_to_call():
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=150, to_call=50, stack=200)
    a = _our_action(s, 0, RlAction.CHECK_CALL)
    assert a.action_type == ActionType.CALL
    assert a.amount == 50


def test_fold_maps_to_fold():
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=150, to_call=50, stack=200)
    a = _our_action(s, 0, RlAction.FOLD)
    assert a.action_type == ActionType.FOLD


def test_raise_maps_into_our_legal_range():
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=150, to_call=50, stack=200)
    la = [x for x in s.legal_actions if x.action_type == ActionType.RAISE][0]
    for rl in (RlAction.RAISE_HALF_POT, RlAction.RAISE_POT, RlAction.ALL_IN):
        a = _our_action(s, 0, rl)
        assert a.action_type == ActionType.RAISE
        assert la.min_amount <= a.amount <= la.max_amount


def test_raise_maps_to_bet_when_unopened():
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=100, to_call=0, stack=200)
    la = [x for x in s.legal_actions if x.action_type == ActionType.BET][0]
    a = _our_action(s, 0, RlAction.RAISE_POT)
    assert a.action_type == ActionType.BET
    assert la.min_amount <= a.amount <= la.max_amount


def test_integer_action_accepted():
    # Real rlcard agents return int action values; the adapter must coerce.
    s = _state(hs("As Ah"), hs("2c 3c 4c"), pot=150, to_call=50, stack=200)
    a = _our_action(s, 0, RlAction.FOLD.value)
    assert a.action_type == ActionType.FOLD


# -- integration ------------------------------------------------------------

def test_adapter_plays_six_max_hands():
    rng = random.Random(7)
    adapter = RlcardAdapter(rng=random.Random(1))
    lineup = [adapter] + [CallingStation() for _ in range(5)]
    for h in range(40):
        g = PokerGame(lineup, [200] * 6, rng=random.Random(100 + h),
                      initial_dealer=h % 6)
        g.play_hand()  # raises IllegalAction if the adapter acts illegally


def test_adapter_plays_heads_up_hands():
    rng = random.Random(7)
    adapter = RlcardAdapter(rng=random.Random(1))
    lineup = [adapter, RandomBot(random.Random(2))]
    for h in range(40):
        g = PokerGame(lineup, [200, 200], rng=random.Random(200 + h),
                      initial_dealer=h % 2)
        g.play_hand()


def test_connector_registration():
    assert "rlcard" in available_plugins()
    strat = build_strategy("rlcard")
    assert isinstance(strat, RlcardAdapter)
