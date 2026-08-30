"""Day-0 gate for the RL agent: the encode/decode layer must be deterministic
and every action it can emit must be legal in the engine.

The engine sandboxes a crashing strategy (engine.py: prints a WARNING and
treats it as a fold), so an illegal action never raises out of play_hand -- it
silently becomes a fold and quietly poisons the training signal. These tests
therefore assert legality directly at each decision point AND assert that a
long session produces no sandbox warning on stderr.
"""
import pathlib
import random

import numpy as np
import pytest

from pokr.cards import Card, all_cards, card_from_str
from pokr.engine import GameState, LegalAction, PlayerView, PokerGame
from pokr.models import OpponentSummary
from pokr.opponents import CallingStation, Maniac, RandomBot, TightAggressive
from pokr.rl.agent import RLStrategy
from pokr.rl.encode import (
    ACTION_ALL_IN,
    ACTION_CHECK_CALL,
    ACTION_FOLD,
    ACTION_NAMES,
    NUM_ACTIONS,
    OBS_DIM,
    OBS_SLICES,
    RAISE_FRACTIONS,
    action_mask,
    card_index,
    decode,
    encode_obs,
    raise_target,
)
from pokr.strategy import ActionType

# Instance only used for its (self-free) action validator.
_VALIDATOR = PokerGame([CallingStation()], [200])


def hs(*strs):
    return [card_from_str(s) for s in " ".join(strs).split()]


def make_state(hole="Ah Kd", board="", pot=10, current_bet=4, street="preflop",
               my_committed=2, stacks=(198, 196, 200), dealer=0, seats=3):
    players = [PlayerView(i, stacks[i]) for i in range(seats)]
    players[0].hole = hs(hole)
    players[0].committed = players[0].street_committed = my_committed
    players[1].committed = players[1].street_committed = current_bet
    state = GameState(
        players=players, community=hs(board) if board else [], pot=pot,
        current_bet=current_bet, min_raise=2, street=street, dealer=dealer,
        current_player=0, legal_actions=[],
    )
    state.legal_actions = _VALIDATOR._legal_actions(state, 0)
    return state


# -- layout ---------------------------------------------------------------


def test_obs_slices_tile_the_vector_without_gaps():
    covered = sorted(OBS_SLICES.values(), key=lambda s: s.start)
    assert covered[0].start == 0
    assert covered[-1].stop == OBS_DIM
    for a, b in zip(covered, covered[1:]):
        assert a.stop == b.start


def test_card_index_is_a_bijection_onto_0_51():
    assert sorted(card_index(c) for c in all_cards()) == list(range(52))


def test_action_names_cover_the_action_space():
    assert len(ACTION_NAMES) == NUM_ACTIONS == ACTION_ALL_IN + 1
    assert NUM_ACTIONS == 2 + len(RAISE_FRACTIONS) + 1


# -- encoding -------------------------------------------------------------


def test_encode_shape_and_dtype():
    obs = encode_obs(make_state(), 0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_hole_and_board_are_multi_hot_at_the_right_indices():
    state = make_state(hole="Ah Kd", board="2c 7s Ts", street="flop")
    obs = encode_obs(state, 0)
    hole, board = obs[OBS_SLICES["hole"]], obs[OBS_SLICES["board"]]
    assert hole.sum() == 2 and board.sum() == 3
    assert {int(i) for i in np.flatnonzero(hole)} == {card_index(c) for c in hs("Ah Kd")}
    assert {int(i) for i in np.flatnonzero(board)} == {card_index(c) for c in hs("2c 7s Ts")}


@pytest.mark.parametrize("street,idx", [("preflop", 0), ("flop", 1), ("turn", 2), ("river", 3)])
def test_street_one_hot(street, idx):
    onehot = encode_obs(make_state(street=street), 0)[OBS_SLICES["street"]]
    assert onehot.sum() == 1 and onehot[idx] == 1.0


def test_position_is_encoded_relative_to_the_button():
    for dealer in range(3):
        pos = encode_obs(make_state(dealer=dealer, seats=3), 0)[OBS_SLICES["position"]]
        assert pos[:6].sum() == 1
        assert pos[(0 - dealer) % 3] == 1.0


def test_pot_odds_and_to_call_scalars():
    state = make_state(pot=10, current_bet=4, my_committed=2)
    scalars = encode_obs(state, 0)[OBS_SLICES["scalars"]]
    start_stack = state.players[0].stack + state.players[0].committed
    assert scalars[0] == pytest.approx(10 / start_stack)     # pot
    assert scalars[3] == pytest.approx(2 / start_stack)      # to_call
    assert scalars[4] == pytest.approx(2 / 12)               # pot odds


def test_derived_features_default_to_zero_and_flag_when_present():
    obs = encode_obs(make_state(), 0)
    assert not obs[OBS_SLICES["equity"]].any()
    assert not obs[OBS_SLICES["opponent"]].any()
    summary = OpponentSummary(hands_observed=250, vpip=0.3, pfr=0.2,
                              aggression_freq=0.5, fold_to_cbet=0.4,
                              fold_rate_postflop=0.35, round_size_frac=0.1)
    obs = encode_obs(make_state(), 0, equity=0.62, opponent=summary)
    assert obs[OBS_SLICES["equity"]] == pytest.approx([0.62, 1.0])
    assert obs[OBS_SLICES["opponent"]][0] == pytest.approx(0.3)
    assert obs[OBS_SLICES["opponent"]][5] == pytest.approx(1.0)  # hands clipped to 1


def test_encoding_is_deterministic():
    a = encode_obs(make_state(board="2c 7s Ts", street="flop"), 0, equity=0.5)
    b = encode_obs(make_state(board="2c 7s Ts", street="flop"), 0, equity=0.5)
    assert np.array_equal(a, b)


# -- action mask ----------------------------------------------------------


def test_fold_is_masked_out_when_checking_is_free():
    state = make_state(current_bet=2, my_committed=2)
    mask = action_mask(state, 0)
    assert not mask[ACTION_FOLD]
    assert mask[ACTION_CHECK_CALL]


def test_fold_is_available_when_facing_a_bet():
    assert action_mask(make_state(current_bet=4, my_committed=2), 0)[ACTION_FOLD]


def test_all_in_is_available_whenever_any_raise_is():
    state = make_state()
    mask = action_mask(state, 0)
    assert mask[ACTION_ALL_IN]
    assert raise_target(state, 0, ACTION_ALL_IN) == max(
        la.max_amount for la in state.legal_actions
        if la.action_type in (ActionType.BET, ActionType.RAISE))


def test_raise_sizes_are_distinct_and_increasing_when_legal():
    state = make_state(pot=100, current_bet=10, my_committed=2, stacks=(500, 490, 500))
    targets = [raise_target(state, 0, i) for i in range(2, ACTION_ALL_IN)]
    live = [t for t in targets if t is not None]
    assert len(live) >= 3
    assert live == sorted(live)
    assert len(set(live)) == len(live)


def test_oversized_raises_are_masked_not_clamped():
    """A short stack cannot make a 2x-pot raise; that index must be masked
    rather than silently collapsed onto the all-in amount."""
    state = make_state(pot=100, current_bet=10, my_committed=2, stacks=(30, 490, 500))
    assert raise_target(state, 0, ACTION_ALL_IN) is not None
    assert raise_target(state, 0, ACTION_ALL_IN - 1) is None  # 2x pot > stack


def test_raises_are_masked_when_calling_is_all_in():
    state = make_state(current_bet=60, my_committed=2, stacks=(50, 140, 200))
    mask = action_mask(state, 0)
    assert mask[ACTION_FOLD] and mask[ACTION_CHECK_CALL]
    assert not mask[2:].any()


# -- decode legality ------------------------------------------------------


def test_decode_of_a_masked_out_index_falls_back_to_check_call():
    state = make_state(current_bet=2, my_committed=2)      # checking is free
    action = decode(state, 0, ACTION_FOLD)                 # fold is masked out
    assert action.action_type is ActionType.CHECK


def test_decode_call_uses_the_exact_engine_amount():
    state = make_state(current_bet=4, my_committed=2)
    action = decode(state, 0, ACTION_CHECK_CALL)
    assert action.action_type is ActionType.CALL
    assert action.amount == 2


def test_every_decoded_action_is_legal_across_random_play():
    """The core gate: at every real decision point, all NUM_ACTIONS indices --
    masked or not -- decode to something the engine accepts."""
    checked = 0

    class Probe(RLStrategy):
        def decide(self, state, player_id):
            nonlocal checked
            mask = action_mask(state, player_id)
            assert mask.any(), "mask must never be all-False"
            assert mask[ACTION_CHECK_CALL], "check/call is always available"
            for i in range(NUM_ACTIONS):
                action = decode(state, player_id, i)
                _VALIDATOR._validate_action(state, player_id, action, state.legal_actions)
                checked += 1
            return super().decide(state, player_id)

    rng = random.Random(11)
    lineup = [Probe(rng=random.Random(3))] + [
        f(random.Random(i)) for i, f in enumerate(
            [lambda r: CallingStation(), TightAggressive, Maniac, RandomBot,
             lambda r: CallingStation()])
    ]
    stacks = [200] * 6
    for h in range(400):
        game = PokerGame(lineup, stacks, 1, 2, rng, initial_dealer=h % 6)
        result = game.play_hand()
        stacks = [200 if s <= 0 else s for s in result.ending_stacks]
    assert checked > 5000


# -- session-level gate ---------------------------------------------------


def _play(agent, hands, seed=7, seats=6):
    rng = random.Random(seed)
    lineup = [agent] + [f(random.Random(seed + i)) for i, f in enumerate(
        [lambda r: CallingStation(), TightAggressive, Maniac, RandomBot,
         lambda r: CallingStation()][:seats - 1])]
    stacks = [200] * seats
    results = []
    for h in range(hands):
        game = PokerGame(lineup, stacks, 1, 2, rng, initial_dealer=h % seats)
        result = game.play_hand()
        stacks = [200 if s <= 0 else s for s in result.ending_stacks]
        results.append(result)
    return results


def test_random_policy_plays_a_long_session_without_tripping_the_sandbox(capfd):
    agent = RLStrategy(rng=random.Random(0), record=True)
    results = _play(agent, 3000)
    err = capfd.readouterr().err
    assert "WARNING: strategy" not in err, err[:500]
    assert len(agent.buffer.episodes) > 0
    assert all(len(e.actions) > 0 for e in agent.buffer.episodes)
    assert len(results) == 3000


def test_torch_policy_plays_without_tripping_the_sandbox(capfd):
    torch = pytest.importorskip("torch")
    from pokr.rl.net import PolicyValueNet
    torch.manual_seed(0)
    agent = RLStrategy(net=PolicyValueNet(), rng=random.Random(0), record=True,
                       mc_iters=8)
    _play(agent, 200)
    assert "WARNING: strategy" not in capfd.readouterr().err


# -- trajectory recording -------------------------------------------------


def test_episode_reward_matches_the_benchmark_metric():
    agent = RLStrategy(rng=random.Random(0), record=True)
    results = _play(agent, 200)
    assert len(agent.buffer.episodes) == sum(
        1 for r in results if any(pid == 0 for pid, _, _ in r.actions))
    played = [r for r in results if any(pid == 0 for pid, _, _ in r.actions)]
    for episode, result in zip(agent.buffer.episodes, played):
        assert episode.reward == pytest.approx(result.winnings[0] / result.big_blind)
        assert len(episode.actions) == sum(1 for pid, _, _ in result.actions if pid == 0)


def test_episode_arrays_are_aligned_and_well_shaped():
    agent = RLStrategy(rng=random.Random(0), record=True)
    _play(agent, 100)
    for e in agent.buffer.episodes:
        t = len(e.actions)
        assert e.obs.shape == (t, OBS_DIM) and e.obs.dtype == np.float32
        assert e.masks.shape == (t, NUM_ACTIONS) and e.masks.dtype == np.bool_
        assert e.masks[np.arange(t), e.actions].all(), "recorded action must be legal"


def test_recording_off_by_default():
    agent = RLStrategy(rng=random.Random(0))
    _play(agent, 50)
    assert agent.buffer.episodes == []


# -- connector plugin -----------------------------------------------------


def test_rl_plugin_is_registered_and_benchable():
    from pokr.bench import LINEUP_ABBREVS, LINEUP_NAMES
    from pokr.connector import available_plugins, build_strategy
    assert "rl" in available_plugins()
    assert "rl" in LINEUP_ABBREVS and "rl" in LINEUP_NAMES
    strategy = build_strategy("rl")
    assert strategy._inner is None, "checkpoint must not load until first decision"


def test_importing_the_connector_does_not_import_torch(tmp_path):
    """torch stays optional, like rlcard: pokr.connector imports the plugin at
    startup, so the plugin must not pull torch in until it actually decides."""
    import subprocess
    import sys
    code = ("import sys; import pokr.connector; "
            "print('torch' in sys.modules or 'rlcard' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(pathlib.Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout


def test_trained_plugin_plays_from_a_checkpoint(tmp_path, capfd):
    torch = pytest.importorskip("torch")
    from pokr.rl.net import PolicyValueNet, save
    from pokr.rl.plugin import TrainedRLStrategy
    ckpt = tmp_path / "ppo.pt"
    torch.manual_seed(0)
    save(PolicyValueNet(), str(ckpt), iteration=1, config={"mc_iters": 0, "fast": False})
    agent = TrainedRLStrategy(ckpt_path=str(ckpt), rng=random.Random(0))
    _play(agent, 100)
    assert agent._inner is not None
    assert "WARNING: strategy" not in capfd.readouterr().err
