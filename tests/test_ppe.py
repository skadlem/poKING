import pytest

pypokerengine = pytest.importorskip("pypokerengine")

from pokr.ppe import PokrPlayer, to_our_card, to_our_cards


def test_card_conversion():
    c = to_our_card("SA")
    assert c.rank == 14 and c.suit == 3
    assert to_our_card("H5").rank == 5 and to_our_card("H5").suit == 2
    assert to_our_card("C2").suit == 0
    assert to_our_card("DT").rank == 10 and to_our_card("DT").suit == 1
    assert len(to_our_cards(["SA", "HK", "C3"])) == 3


def test_heads_up_runs_in_pypokerengine():
    from pypokerengine.api.game import setup_config, start_poker

    from pokr.ppe_compare import _load_external

    fish_cls = _load_external("fish")
    ours = PokrPlayer(rng_seed=3, mc_iters=5)
    config = setup_config(max_round=20, initial_stack=200, small_blind_amount=1)
    config.register_player(name="pokr", algorithm=ours)
    config.register_player(name="FishPlayer", algorithm=fish_cls())
    result = start_poker(config, verbose=0)
    stacks = {p["name"]: p["stack"] for p in result["players"]}
    assert "pokr" in stacks
    assert "FishPlayer" in stacks
    # chip conservation across the match
    assert abs((stacks["pokr"] - 200) + (stacks["FishPlayer"] - 200)) <= 1
