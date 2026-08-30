"""PPO layer: GAE arithmetic, batch assembly, masked-logit numerics, and a
does-the-optimizer-actually-learn check on a synthetic task.

The learning test is deliberately trivial (a one-state bandit): if PPO cannot
solve that, no amount of poker rollout will help, and the failure is isolated
to the update rather than the environment.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pokr.rl.agent import Episode, RolloutBuffer  # noqa: E402
from pokr.rl.encode import NUM_ACTIONS  # noqa: E402
from pokr.rl.net import PolicyValueNet  # noqa: E402
from pokr.rl.ppo import (  # noqa: E402
    PPOConfig,
    PPOTrainer,
    episode_advantages,
    make_batch,
)

OBS = 8


def episode(values, reward, t=None, obs_dim=OBS, actions=None):
    values = np.asarray(values, dtype=np.float32)
    t = t or len(values)
    return Episode(
        obs=np.zeros((t, obs_dim), np.float32),
        masks=np.ones((t, NUM_ACTIONS), bool),
        actions=(np.zeros(t, np.int64) if actions is None
                 else np.asarray(actions, np.int64)),
        logps=np.full(t, -np.log(NUM_ACTIONS), np.float32),
        values=values,
        reward=float(reward),
    )


# -- GAE ------------------------------------------------------------------


def test_gae_with_gamma_and_lambda_one_is_reward_minus_value():
    """Terminal-only reward + no discount: the recursion must telescope to
    (R - V(s_t)) at every step, and every return must equal R."""
    ep = episode([0.5, -0.25, 2.0], reward=20.0)
    adv, ret = episode_advantages(ep, gamma=1.0, lam=1.0, reward_scale=10.0)
    assert adv == pytest.approx(2.0 - ep.values)
    assert ret == pytest.approx([2.0, 2.0, 2.0])


def test_gae_with_lambda_zero_is_one_step_td():
    ep = episode([0.5, -0.25, 2.0], reward=20.0)
    adv, _ = episode_advantages(ep, gamma=1.0, lam=0.0, reward_scale=10.0)
    v = ep.values
    assert adv == pytest.approx([v[1] - v[0], v[2] - v[1], 2.0 - v[2]])


def test_reward_scale_is_a_pure_linear_rescale():
    """The scale must not reorder anything -- it is only conditioning."""
    ep = episode([0.0, 0.0], reward=50.0)
    a1, _ = episode_advantages(ep, 1.0, 1.0, reward_scale=1.0)
    a2, _ = episode_advantages(ep, 1.0, 1.0, reward_scale=10.0)
    assert a1 == pytest.approx(a2 * 10.0)


def test_single_step_episode():
    ep = episode([1.0], reward=-30.0)
    adv, ret = episode_advantages(ep, 1.0, 0.95, 10.0)
    assert adv == pytest.approx([-3.0 - 1.0])
    assert ret == pytest.approx([-3.0])


# -- batching -------------------------------------------------------------


def test_make_batch_concatenates_in_episode_order():
    buf = RolloutBuffer([episode([0.0, 0.0], 10.0), episode([0.0], -4.0)])
    batch = make_batch(buf, PPOConfig(reward_scale=1.0, lam=1.0))
    assert len(batch) == 3
    assert batch.obs.shape == (3, OBS)
    assert batch.masks.shape == (3, NUM_ACTIONS)
    assert batch.returns.tolist() == pytest.approx([10.0, 10.0, -4.0])


def test_make_batch_rejects_an_empty_buffer():
    with pytest.raises(ValueError):
        make_batch(RolloutBuffer(), PPOConfig())


# -- masked-logit numerics ------------------------------------------------


def test_entropy_is_finite_when_only_one_action_is_legal():
    """Masked logits use a finite fill precisely so that p*log(p) on a masked
    entry is 0 * -1e8 = -0.0 rather than 0 * -inf = NaN."""
    net = PolicyValueNet(obs_dim=OBS)
    mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
    mask[0, 3] = True
    with torch.no_grad():
        dist, value = net.distribution(torch.zeros(1, OBS), mask)
    assert torch.isfinite(dist.entropy()).all()
    assert torch.isfinite(value).all()
    assert dist.probs[0, 3] == pytest.approx(1.0)
    assert dist.probs[0, 0] == pytest.approx(0.0)


def test_gradients_stay_finite_through_a_heavily_masked_update():
    torch.manual_seed(0)
    net = PolicyValueNet(obs_dim=OBS)
    masks = np.zeros((4, NUM_ACTIONS), bool)
    masks[:, 1] = True
    ep = episode([0.0] * 4, reward=25.0)
    ep.masks = masks
    ep.actions = np.ones(4, np.int64)
    trainer = PPOTrainer(net, PPOConfig(epochs=2, minibatch=4))
    stats = trainer.update(RolloutBuffer([ep]))
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])
    assert all(torch.isfinite(p).all() for p in net.parameters())


# -- the update actually learns -------------------------------------------


def _bandit_buffer(net, rng, n=256):
    """One state, NUM_ACTIONS arms; arm 0 pays +100bb, everything else -100bb."""
    obs = np.ones(OBS, np.float32)
    mask = np.ones(NUM_ACTIONS, bool)
    episodes = []
    for _ in range(n):
        action, logp, value = net.act(obs, mask, generator=rng)
        episodes.append(Episode(
            obs=obs[None, :].copy(), masks=mask[None, :].copy(),
            actions=np.array([action], np.int64),
            logps=np.array([logp], np.float32),
            values=np.array([value], np.float32),
            reward=100.0 if action == 0 else -100.0))
    return RolloutBuffer(episodes)


def test_ppo_solves_a_one_state_bandit():
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)
    net = PolicyValueNet(obs_dim=OBS, hidden=(32,))
    trainer = PPOTrainer(net, PPOConfig(lr=3e-3, epochs=4, minibatch=64,
                                        reward_scale=100.0, target_kl=None),
                         generator=rng)
    obs = torch.ones(1, OBS)
    mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
    with torch.no_grad():
        before = float(net.distribution(obs, mask)[0].probs[0, 0])
    for _ in range(30):
        trainer.update(_bandit_buffer(net, rng))
    with torch.no_grad():
        after = float(net.distribution(obs, mask)[0].probs[0, 0])
    assert after > before + 0.3, f"policy did not learn: {before:.3f} -> {after:.3f}"
    assert after > 0.8, f"expected near-deterministic best arm, got {after:.3f}"


def test_value_head_learns_the_bandit_baseline():
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)
    net = PolicyValueNet(obs_dim=OBS, hidden=(32,))
    trainer = PPOTrainer(net, PPOConfig(lr=3e-3, epochs=4, minibatch=64,
                                        reward_scale=100.0, target_kl=None),
                         generator=rng)
    for _ in range(30):
        trainer.update(_bandit_buffer(net, rng))
    with torch.no_grad():
        value = float(net.distribution(torch.ones(1, OBS),
                                       torch.ones(1, NUM_ACTIONS, dtype=torch.bool))[1])
    # policy converges on the +1 arm, so the state value must approach +1
    assert value > 0.5, value


def test_target_kl_stops_the_epoch_loop_early():
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)
    net = PolicyValueNet(obs_dim=OBS, hidden=(32,))
    buf = _bandit_buffer(net, rng, n=128)
    loose = PPOTrainer(PolicyValueNet(obs_dim=OBS, hidden=(32,)),
                       PPOConfig(lr=1e-1, epochs=10, minibatch=32, target_kl=None))
    strict = PPOTrainer(PolicyValueNet(obs_dim=OBS, hidden=(32,)),
                        PPOConfig(lr=1e-1, epochs=10, minibatch=32, target_kl=1e-6))
    assert not loose.update(buf)["early_stop"]
    assert strict.update(buf)["early_stop"]


def test_update_reports_rollout_summary():
    net = PolicyValueNet(obs_dim=OBS, hidden=(32,))
    buf = RolloutBuffer([episode([0.0, 0.0], 10.0), episode([0.0], -30.0)])
    stats = PPOTrainer(net, PPOConfig(epochs=1, minibatch=8)).update(buf)
    assert stats["steps"] == 3
    assert stats["episodes"] == 2
    assert stats["mean_reward_bb"] == pytest.approx(-10.0)


# -- checkpoint round-trip ------------------------------------------------


def test_checkpoint_round_trips_with_training_metadata(tmp_path):
    """train_rl.py saves `config=vars(args)` alongside the weights; that must
    not collide with the architecture the loader needs."""
    from pokr.rl.net import load, save
    net = PolicyValueNet(obs_dim=OBS, hidden=(32, 16))
    path = str(tmp_path / "ckpt.pt")
    save(net, path, iteration=5, config={"lr": 3e-4, "hidden": [999]}, eval={"cs": 1.0})
    restored, ckpt = load(path)
    assert restored.hidden == (32, 16)
    assert restored.obs_dim == OBS
    assert ckpt["iteration"] == 5 and ckpt["config"]["lr"] == 3e-4
    with torch.no_grad():
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        a = net.distribution(torch.ones(1, OBS), mask)[0].probs
        b = restored.distribution(torch.ones(1, OBS), mask)[0].probs
    assert torch.allclose(a, b)


def test_save_rejects_a_reserved_key():
    from pokr.rl.net import save
    with pytest.raises(ValueError, match="reserved"):
        save(PolicyValueNet(obs_dim=OBS), "/dev/null", net_config={"nope": 1})
