"""Train an RLCard DQN agent heads-up on no-limit hold'em (vs random play).

RLCard ships no pretrained NLH agents (its model zoo only has Leduc CFR),
so this trains one for use as an external benchmark opponent through pokr's
rlcard adapter (pokr/rlcard_adapter.py). The agent learns against random
play in rlcard's own engine, is evaluated periodically vs random, and
checkpoints for later pokr head-to-head benchmarks.

Run:
    python train_rlcard_dqn.py --steps 2000000 --seed 7
    python train_rlcard_dqn.py --steps 2000000 --resume models/rlcard_dqn/dqn_final.pt
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import time

import torch

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import reorganize, set_seed, tournament


def _feed_quiet(agent, ts) -> None:
    """agent.feed prints '\\rINFO - Step ...' on every train step; swallow it
    so progress logs stay readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        agent.feed(ts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train RLCard DQN on no-limit hold'em (heads-up, vs random)")
    ap.add_argument("--steps", type=int, default=2_000_000,
                    help="total DQN training steps")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt-dir", default="models/rlcard_dqn")
    ap.add_argument("--save-every", type=int, default=100_000,
                    help="checkpoint every N train steps")
    ap.add_argument("--replay-init", type=int, default=10_000,
                    help="random transitions before training starts")
    ap.add_argument("--eval-episodes", type=int, default=500,
                    help="evaluate vs random every N episodes")
    ap.add_argument("--eval-games", type=int, default=500,
                    help="games per evaluation")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128],
                    help="MLP hidden sizes")
    ap.add_argument("--resume", default="",
                    help="checkpoint file (torch dict) to resume from")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    env = rlcard.make("no-limit-holdem", config={"seed": args.seed})
    num_actions, state_shape = env.num_actions, env.state_shape[0]

    if args.resume:
        agent = DQNAgent.from_checkpoint(
            checkpoint=torch.load(args.resume, map_location="cpu"))
        print(f"resumed from {args.resume} (total_t={agent.total_t})")
    else:
        agent = DQNAgent(
            num_actions=num_actions,
            state_shape=state_shape,
            mlp_layers=list(args.hidden),
            replay_memory_size=200_000,
            replay_memory_init_size=args.replay_init,
            epsilon_decay_steps=min(args.steps, 500_000),
            device="cpu",
            save_path=args.ckpt_dir,
            save_every=args.save_every,
        )
    os.makedirs(args.ckpt_dir, exist_ok=True)

    def random_agents():
        return [RandomAgent(num_actions) for _ in range(env.num_players - 1)]

    env.set_agents([agent] + random_agents())

    steps_done = agent.total_t
    episode = 0
    window: list[float] = []
    t0 = last_t = time.time()
    last_steps = steps_done
    print(f"env: no-limit-holdem {env.num_players}-player | actions {num_actions} "
          f"| state {state_shape} | target {args.steps} steps")
    while steps_done < args.steps:
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        for ts in trajectories[0]:
            _feed_quiet(agent, ts)
        window.append(payoffs[0])
        if len(window) > 200:
            window.pop(0)
        steps_done = agent.total_t
        episode += 1

        if episode % args.eval_episodes == 0:
            env.set_agents([agent] + random_agents())
            mean_payoff = tournament(env, args.eval_games)[0]
            now = time.time()
            sps = (steps_done - last_steps) / (now - last_t)
            eps = agent.epsilons[min(agent.total_t, agent.epsilon_decay_steps - 1)]
            avg_recent = sum(window) / len(window)
            print(f"ep {episode:>6} | steps {steps_done:>9}/{args.steps} "
                  f"| eps {eps:.3f} | recent payoff {avg_recent:+.1f} "
                  f"| vs random {mean_payoff:+.2f} chips/game | {sps:,.0f} steps/s")
            last_t, last_steps = now, steps_done
        if steps_done >= args.steps:
            break

    final_ckpt = os.path.join(args.ckpt_dir, "dqn_final.pt")
    torch.save(agent.checkpoint_attributes(), final_ckpt)
    env.set_agents([agent] + random_agents())
    mean_payoff = tournament(env, 1000)
    dt = time.time() - t0
    print(f"done: {steps_done} steps in {dt / 60:.1f} min "
          f"({steps_done / dt:,.0f} steps/s overall)")
    print(f"final vs random: {mean_payoff[0]:+.2f} chips/game")
    print(f"checkpoint: {final_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
