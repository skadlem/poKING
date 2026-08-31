"""NFSPStrategy: Pi as an engine citizen.

The PPO day-0 gate (legality under the sandbox, no silent fold-poisoning)
is re-run here for the new agent, then the NFSP-specific contracts that
design note 3 says would fail SILENTLY — plausible curves, meaningless
results — are each pinned by one test:

  3.2  opponent features off by default (an average strategy over an
       induced opponent distribution is not an average strategy);
  3.3  one info state -> one observation row (deterministic equity);
  3.4  deployment samples; no greedy parameter exists anywhere;
  2    sigma is a PER-HAND coin flip and M_SL holds only behaviour rows.

Design note 5 proposed adding br_mode to Episode; the rows carry it
instead, because fit() is where the distinction is consumed and ladder B's
outer loop (step 9) hands whole episodes over explicitly. Episode stays the
PPO record it already is.
"""
import inspect
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pokr.bench import LINEUP_ABBREVS, LINEUP_NAMES
from pokr.cards import card_from_str
from pokr.engine import PokerGame
from pokr.opponents import CallingStation, RandomBot
from pokr.rl.agent import Episode
from pokr.rl.avg_policy import AvgPolicyNet, save as save_pi
from pokr.rl.encode import NUM_ACTIONS, OBS_DIM
from pokr.rl.nfsp import NFSPConfig, NFSPStrategy, _state_key, select_fit_rows
from pokr.strategy import Action, ActionType


def make_agent(seed=0, **over):
    """over splits: NFSPConfig fields go to the config, the rest to the
    constructor. fit_every defaults to 0 (manual fits) so tests never refit
    mid-play unless they ask."""
    cfg_fields = set(NFSPConfig.__dataclass_fields__)
    cfg_over = {k: over.pop(k) for k in list(over) if k in cfg_fields}
    defaults: dict = dict(capacity=20_000, fit_every=0, batch_size=16, epochs=3)
    defaults.update(cfg_over)
    cfg = NFSPConfig(**defaults)
    return NFSPStrategy(config=cfg, rng=random.Random(seed), **over)


def play(agent, hands, seed=7):
    """Heads-up vs scripted opponents through the real engine, alternating
    the agent's seat (the sandbox turns an illegal action into a silent
    fold — hence the stderr checks in the tests that use this)."""
    rng = random.Random(seed)
    results = []
    for h in range(hands):
        agent_lineup = [agent, RandomBot()] if h % 2 == 0 \
            else [CallingStation(), agent]
        game = PokerGame(agent_lineup, [200, 200], rng=rng,
                         initial_dealer=h % 2)
        results.append(game.play_hand())
    return results


# -- day-0 parity with the PPO agent ------------------------------------------


def test_nfsp_agent_plays_a_real_session_without_tripping_the_sandbox(capfd):
    agent = make_agent()
    hands = play(agent, 120)
    err = capfd.readouterr().err
    assert "WARNING: strategy" not in err, err[:500]
    assert agent.buffer.seen > 0, "a whole session recording nothing is the " \
        "quiet failure this gate exists for"
    assert len(hands) == 120


def test_every_recorded_action_is_legal_at_its_mask():
    agent = make_agent()
    play(agent, 60)
    assert agent.buffer.seen > 50
    for obs, mask, idx, _br in agent.buffer.contents():
        assert mask.shape == (NUM_ACTIONS,) and mask.dtype == bool
        assert obs.shape == (OBS_DIM,)
        assert mask[idx], "recorded an illegal action — poisons the CE fit"


# -- 3.4: the deployment contract ----------------------------------------------


def test_no_greedy_path_exists_anywhere_on_the_nfsp_surface():
    """Not 'defaults to False' — ABSENT. The argmax of an approximate
    equilibrium is maximally exploitable, and a default can be flipped by a
    caller who read the PPO plugin instead of the design note."""
    assert "greedy" not in inspect.signature(NFSPStrategy.__init__).parameters
    assert "greedy" not in inspect.signature(NFSPStrategy.decide).parameters
    from pokr.rl.plugin import TrainedNFSPStrategy
    assert "greedy" not in inspect.signature(TrainedNFSPStrategy.__init__).parameters


def test_net_act_never_returns_the_argmax_every_time():
    """Pi's act path is multinomial over masked probs: on a fixed legal
    mask, 300 draws must not all land on one action."""
    net = AvgPolicyNet(hidden=(16,))
    torch.manual_seed(1)
    rng = np.random.RandomState(2)
    obs = rng.rand(OBS_DIM).astype(np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[0, 2, 5]] = True
    seen = {net.act(obs, mask)[0] for _ in range(300)}
    assert len(seen) > 1


# -- 3.2 + 3.3: the observation contract ---------------------------------------


def test_opponent_model_is_off_by_default():
    """The DEFAULT is the contract (3.2): a silently-enabled model block
    makes Pi an average over an induced distribution, which is not an
    average strategy at all. Enabling must be explicit."""
    default = make_agent(record=False)
    assert default._view.models is None
    assert make_agent(record=False, model_opponents=True)._view.models is not None


def test_one_info_state_one_observation_row():
    """Deterministic equity (step 1) through the NFSP seam: two agents with
    different rng streams observing the same decision point must produce
    byte-identical observations — the reservoir would otherwise hold
    aliases of one info state and smear the average again."""
    from tests.test_rl_agent import make_state

    st = make_state(hole="Ah Kd", board="2c 7s Ts", seats=2)
    a1 = make_agent(seed=1, record=False)
    a2 = make_agent(seed=99, record=False)
    assert np.array_equal(a1._view.observe(st, 0), a2._view.observe(st, 0))


# -- design note 2: the sigma flip and what feeds M_SL --------------------------


def test_sigma_flips_once_per_hand_not_per_step():
    """eta is an EPISODE-level coin: a mid-hand re-flip would let one hand
    contribute both behaviour and Pi rows. The agent always faces at least
    one decision per hand in this lineup, so hands == flips is exact."""
    agent = make_agent(behaviour="epsilon", eta=0.5, q_table={})
    real = agent._play_br_this_hand
    calls = []

    def spy():
        val = real()
        calls.append(val)
        return val
    agent._play_br_this_hand = spy
    play(agent, 40)
    assert len(calls) == 40, f"{len(calls)} flips for 40 hands"
    br_rows = sum(1 for r in agent.buffer.contents() if r[3])
    assert 0 < br_rows < agent.buffer.seen, \
        "an eta=0.5 run with every row the same mode is a broken flip"


def test_ppo_behaviour_never_flips_and_rows_are_all_pi():
    agent = make_agent(behaviour="ppo")
    play(agent, 20)
    assert agent.buffer.seen > 0
    assert not any(r[3] for r in agent.buffer.contents())


def test_select_fit_rows_prefers_behaviour_and_bootstraps_on_pi_rows():
    """The M_SL contract (design note 2) as a pure function: BR rows win
    when present — mixing Pi rows in drags the fictitious average toward
    itself — and Pi rows serve ONLY as the documented bootstrap."""
    row = (np.zeros(OBS_DIM, np.float32), np.ones(NUM_ACTIONS, bool), 0)
    pi = [row + (False,) for _ in range(50)]
    br = [row + (True,) for _ in range(3)]
    assert select_fit_rows(pi + br) == br
    assert select_fit_rows(pi) == pi
    assert select_fit_rows([]) == []


def test_record_episode_fans_steps_out_with_the_flag():
    """The ladder-B seam: an oracle PPO hand enters as behaviour rows."""
    agent = make_agent(record=False)
    T = 4
    masks = np.zeros((T, NUM_ACTIONS), dtype=bool)
    masks[:, 0] = True                              # at least fold legal
    ep = Episode(obs=np.random.RandomState(4).rand(T, OBS_DIM).astype(np.float32),
                 masks=masks,
                 actions=np.zeros(T, dtype=np.int64),
                 logps=np.zeros(T, np.float32),
                 values=np.zeros(T, np.float32), reward=1.0)
    before = agent.buffer.seen
    agent.record_episode(ep)
    assert agent.buffer.seen == before + T
    assert all(r[3] for r in agent.buffer.contents()[-T:])
    agent.record_episode(ep, br_mode=False)
    assert all(not r[3] for r in agent.buffer.contents()[-T:])


def test_fit_learns_the_behaviours_frequencies():
    """End-to-end learner check without an engine: reservoir holds 4000
    rows from one state whose behaviour bets 70%; after fit, Pi's
    probability on that observation must sit near 0.7 — and the BR rows
    must be the ones that taught it, with 10000 contradicting Pi rows
    present."""
    agent = make_agent(record=False)
    rng = np.random.RandomState(3)
    obs = rng.rand(OBS_DIM).astype(np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[0, 5]] = True
    for a in rng.choice([0, 5], size=4000, p=[0.3, 0.7]):
        agent.buffer.add((obs, mask, int(a), True))
    decoy_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    decoy_mask[[0]] = True                          # fold-only decoys
    for _ in range(10_000):
        agent.buffer.add((obs + 1.0, decoy_mask, 0, False))  # distinct obs
    agent.fit(epochs=60, batch_size=256, lr=5e-3)
    p = agent.net.probs(obs, mask)[0]
    assert abs(p[5] - 0.7) < 0.08, f"Pi {p[5]:.3f} vs behaviour 0.70"


def test_auto_fit_gates_on_hand_count_and_row_count():
    agent = make_agent(fit_every=5)
    play(agent, 4)
    assert agent.last_fit_loss is None              # 4 hands < 5
    play(agent, 1)
    assert agent.buffer.seen >= 16
    assert agent.last_fit_loss is not None, "fit threshold reached but no fit"
    twin = agent.clone_for_evaluation()
    assert twin.config.fit_every == 0, \
        "an eval twin that refits mid-scoring invalidates its own numbers"


def test_fit_on_empty_reservoir_says_so():
    agent = make_agent(record=False)
    with pytest.raises(ValueError, match="empty reservoir"):
        agent.fit()


# -- plugin wiring (roadmap step 2) ----------------------------------------------


def test_nfsp_plugin_registered_and_plays_from_a_checkpoint(tmp_path, capfd):
    from pokr.connector import available_plugins, build_strategy
    from pokr.rl.plugin import TrainedNFSPStrategy

    assert "nfsp" in available_plugins()
    assert "nfsp" in LINEUP_ABBREVS and LINEUP_NAMES["nfsp"] == "PokrNFSP"

    ckpt = tmp_path / "pi.pt"
    save_pi(AvgPolicyNet(hidden=(16,)), str(ckpt))
    strat = build_strategy("nfsp")
    assert isinstance(strat, TrainedNFSPStrategy)   # the factory's contract
    strat.ckpt_path = str(ckpt)
    assert len(play(strat, 25)) == 25
    assert "WARNING: strategy" not in capfd.readouterr().err
    assert strat._inner is not None and not strat._inner.record
    assert strat._inner.buffer.seen == 0, "a deployment must not fill M_SL"


def test_plugin_is_lazy_about_a_missing_checkpoint():
    """Constructing must not touch disk: the default path names a file that
    does not exist until step 9 trains one, and the connector module is
    imported at pokr startup — an eager load would break every bench run."""
    from pokr.connector import build_strategy
    from pokr.rl.plugin import TrainedNFSPStrategy
    s = build_strategy("nfsp")
    assert isinstance(s, TrainedNFSPStrategy)
    assert s._inner is None and s.ckpt_path.endswith("models/nfsp/nfsp_final.pt")


def test_importing_the_connector_does_not_import_torch(tmp_path):
    """Same contract as the PPO plugin's day-1 test: torch stays optional."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import pokr.connector; import sys;"
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, check=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    assert out.stdout.strip() == "False"


def test_state_key_is_immutable_history_bearing_and_hashable():
    """Ladder A's table key: two preflop lines arriving at the same chip
    position (the 5917f88 alias) must NOT share a key; raw Action objects
    would be unhashable — the key stores the action TYPE NAME instead."""
    from tests.test_rl_agent import make_state

    st = make_state(hole="Ah Kd")
    k1 = _state_key(st, 0)
    assert isinstance(hash(k1), int)
    st2 = make_state(hole="Ah Kd")
    st2.action_history = [(0, "preflop", Action(ActionType.RAISE, 6, ""))]
    assert _state_key(st2, 0) != k1
    assert not any(isinstance(a, Action)
                   for t in _state_key(st2, 0) if isinstance(t, tuple)
                   for a in t)
