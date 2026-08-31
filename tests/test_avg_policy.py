"""AvgPolicyNet: the mask must actually mask, and the CE fit must actually
recover the empirical action distribution — the second property is the
entire justification for calling Pi "the average of past behaviours".

The frequency-recovery tests are scored against the empirical distribution
of the drawn actions (the exact CE minimiser), with tolerance from sampling
error; a fixed tolerance against the generator's p would be wrong in the
other direction, and the tests say so where they differ.
"""
import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pokr.rl.avg_policy import (
    AvgPolicyNet,
    fit_avg_policy,
    load,
    save,
    sl_loss,
)
from pokr.rl.avg_policy import _MASK_FILL

NUM_A = 9  # NUM_ACTIONS in encode.py; nets here use fewer for speed


def make_nets(seed=0, obs_dim=8, num_actions=NUM_A, hidden=(16,)):
    torch.manual_seed(seed)
    return AvgPolicyNet(obs_dim=obs_dim, num_actions=num_actions, hidden=hidden)


def legal_mask(num_actions, k):
    m = np.zeros(num_actions, dtype=bool)
    m[:k] = True
    return m


# -- masking ---------------------------------------------------------------


def test_illegal_slots_get_exactly_zero_probability_and_zero_gradient():
    """The -1e8 fill must underflow float32 softmax to EXACTLY 0. A tiny
    residual probability on fold-vs-all-in is the kind of bug that costs a
    whole training campaign; 'small' is not the contract, 0.0 is."""
    net = make_nets()
    obs = torch.randn(4, 8)
    mask = torch.zeros(4, NUM_A, dtype=torch.bool)
    mask[:, :3] = True
    probs = torch.softmax(net.masked_logits(obs, mask), dim=-1)
    assert (probs[:, 3:] == 0.0).all()
    assert torch.allclose(probs[:, :3].sum(-1), torch.ones(4))

    logits = net(obs).detach().requires_grad_(True)
    masked = logits.masked_fill(~mask, _MASK_FILL)
    sl_loss(masked, torch.tensor([0, 1, 2, 0])).backward()
    assert (logits.grad[:, 3:] == 0.0).all(), "illegal logits must not drift"
    assert (logits.grad[:, :3] != 0.0).any()


def test_probs_and_act_are_legal_only():
    net = make_nets()
    obs = np.random.RandomState(0).randn(8, 8).astype(np.float32)
    mask = np.stack([legal_mask(NUM_A, 4)] * 8)
    p = net.probs(obs, mask)
    assert (p[:, 4:] == 0.0).all()
    assert np.allclose(p[:, :4].sum(1), 1.0)
    single = np.random.RandomState(1).randn(8).astype(np.float32)
    assert net.probs(single, legal_mask(NUM_A, 4)).shape == (1, NUM_A)

    seen = set()
    for _ in range(200):
        a, logp = net.act(single, mask[0])
        assert a < 4 and math.isfinite(logp)
        seen.add(a)
    assert len(seen) > 1, "act() sampling must actually sample"


def test_act_has_no_greedy_path():
    """Design note 3.4: the argmax of an approximate equilibrium is maximally
    exploitable. The API omits the rope, so there is no flag to set wrong."""
    import inspect

    assert "greedy" not in inspect.signature(AvgPolicyNet.act).parameters


# -- the CE fit recovers empirical frequencies ------------------------------


def draw_state_data(n, p_true, num_actions=NUM_A, k=3, obs_dim=8, seed=0):
    """n samples of one state, actions ~ p_true over the k legal slots."""
    rng = np.random.RandomState(seed)
    obs = np.tile(rng.randn(obs_dim), (n, 1)).astype(np.float32)
    actions = rng.choice(k, size=n, p=p_true)
    masks = np.stack([legal_mask(num_actions, k)] * n)
    return obs, masks, actions


def test_fit_recovers_the_empirical_action_distribution():
    """The CE minimiser for fixed data IS the empirical distribution — that
    equivalence is the whole fictitious-play argument, so assert Pi lands on
    the drawn frequencies (not the generator's p: with n=4000 a 5-sigma
    sampling deviation of ~0.012 can exceed any tighter claim)."""
    net = make_nets(seed=3)
    obs, masks, actions = draw_state_data(4000, [0.2, 0.3, 0.5], seed=4)
    emp = np.bincount(actions, minlength=NUM_A)[:3] / len(actions)
    losses = fit_avg_policy(net, obs, masks, actions, epochs=150,
                           batch_size=256, lr=5e-3,
                           generator=torch.Generator().manual_seed(5))
    assert losses[0] > losses[-1], "loss must fall"
    p = net.probs(obs[:1], masks[:1])[0][:3]
    # 5 sigma of the empirical proportion at the most variable slot:
    tol = 5 * math.sqrt(0.5 * 0.5 / len(actions)) + 0.02  # + optimiser slack
    assert np.abs(p - emp).max() < tol, f"Pi {p} vs empirical {emp}"
    assert (net.probs(obs[:1], masks[:1])[0][3:] == 0.0).all(), \
        "illegal slots must stay exactly zero after training"


def test_fit_separates_two_states():
    """One shared net, two distinct info states, two different behaviours —
    the fit must learn both, i.e. Pi is a function of s, not a global mean."""
    net = make_nets(seed=6)
    obs_a, mask_a, act_a = draw_state_data(2000, [0.8, 0.1, 0.1], seed=7)
    obs_b, mask_b, act_b = draw_state_data(2000, [0.1, 0.1, 0.8], seed=8)
    obs_b += 5.0  # second distinct state (obs_dim 8 constant rows)
    obs = np.concatenate([obs_a, obs_b])
    masks = np.concatenate([mask_a, mask_b])
    actions = np.concatenate([act_a, act_b])
    fit_avg_policy(net, obs, masks, actions, epochs=200, batch_size=256,
                   lr=5e-3, generator=torch.Generator().manual_seed(9))
    pa = net.probs(obs_a[:1], mask_a[:1])[0][:3]
    pb = net.probs(obs_b[:1], mask_b[:1])[0][:3]
    assert pa[0] > 0.6 and pb[2] > 0.6, f"{pa} / {pb}"


def test_fit_accepts_ragged_masks_and_rejects_illegal_targets():
    net = make_nets(seed=10)
    obs, masks, actions = draw_state_data(64, [1 / 3] * 3, seed=11)
    out = fit_avg_policy(net, obs, list(masks), actions, epochs=1,
                         generator=torch.Generator().manual_seed(0))
    assert len(out) == 1

    bad_actions = actions.copy()
    bad_actions[0] = NUM_A - 1  # illegal (mask has only first 3 set)
    with pytest.raises(ValueError, match="illegal target actions"):
        fit_avg_policy(net, obs, masks, bad_actions, epochs=1)
    with pytest.raises(ValueError, match="empty training set"):
        fit_avg_policy(net, obs[:0], masks[:0], actions[:0], epochs=1)


def test_fit_is_reproducible_under_a_seeded_generator():
    def run():
        net = make_nets(seed=12)
        obs, masks, actions = draw_state_data(500, [0.25, 0.25, 0.5], seed=13)
        fit_avg_policy(net, obs, masks, actions, epochs=30, batch_size=64,
                       lr=5e-3, generator=torch.Generator().manual_seed(14))
        return net.probs(obs[:1], masks[:1])[0].copy()

    a, b = run(), run()
    assert np.array_equal(a, b)


# -- reservoir seam ----------------------------------------------------------


def test_buffer_records_feed_the_fit_directly():
    """The real pipeline shape: reservoir holds (obs, mask, action) tuples,
    a sampled minibatch feeds fit_avg_policy with no adapter glue. If this
    composition ever needs a translator, the two modules diverged."""
    from pokr.rl.memory import ExponentialReservoirBuffer

    rng = random.Random(4)
    buf = ExponentialReservoirBuffer(500, rng, min_replacement=0.25)
    for i in range(5000):
        obs, masks, actions = draw_state_data(1, [0.2, 0.3, 0.5], seed=i)
        buf.add((obs[0], masks[0], int(actions[0])))
    batch = buf.sample(128)
    obs = np.stack([r[0] for r in batch])
    masks = np.stack([r[1] for r in batch])
    actions = np.array([r[2] for r in batch])
    net = make_nets(seed=15)
    losses = fit_avg_policy(net, obs, masks, actions, epochs=20,
                            generator=torch.Generator().manual_seed(16))
    assert len(losses) == 20 and losses[-1] < losses[0]


# -- checkpoint contract ------------------------------------------------------


def test_save_load_roundtrip_and_reserved_key_guard(tmp_path):
    net = make_nets(seed=17)
    ckpt = tmp_path / "pi.pt"
    save(net, str(ckpt), iteration=42, config={"lr": 1e-3})
    loaded, meta = load(str(ckpt))
    assert isinstance(loaded, AvgPolicyNet)
    assert meta["iteration"] == 42 and meta["config"]["lr"] == 1e-3
    assert loaded.config() == net.config()
    obs, masks, _ = draw_state_data(1, [0.2, 0.3, 0.5], seed=18)
    assert np.allclose(loaded.probs(obs, masks), net.probs(obs, masks),
                       atol=0, rtol=0)  # byte-identical weights
    with pytest.raises(ValueError, match="reserved checkpoint keys"):
        save(net, str(ckpt), state_dict={})


def test_pi_is_distinct_from_the_value_net_class():
    """Guard against someone 'deduplicating' the two nets: PolicyValueNet
    has a value head PPO depends on; a shared class would train a head
    nobody uses and change a shipped checkpoint contract."""
    from pokr.rl.net import PolicyValueNet

    assert not issubclass(AvgPolicyNet, PolicyValueNet)
    assert not hasattr(AvgPolicyNet, "v")
