"""Train a PPO agent in PyTorch to play inside pokr's own engine.

Unlike train_rlcard_dqn.py (which trains an external baseline in rlcard's
engine and imports it back through an adapter), this agent learns in the same
PokerGame the benchmarks run on, so there is no train/eval distribution shift
and the reward IS the benchmark metric: bb won per hand.

Rollouts come from bench.play_session -- the agent records itself as it plays
(pokr/rl/agent.py), so the existing harness doubles as the rollout collector.
The opponent lineup is resampled from a pool between iterations rather than
per hand, which keeps each session's opponent models warm while still giving
the agent a varied, stationary-per-batch environment. Pure self-play is
avoided on purpose: in an imperfect-information game it cycles rather than
converging, and the pool is what we actually benchmark against anyway.

Run:
    python train_rl.py --iters 200 --hands-per-iter 2000 --fast
    python train_rl.py --iters 500 --pool tag,heuristic --resume models/rl/ppo_final.pt
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch

from pokr.bench import (
    bot_own_stats,
    calling_station_factory,
    maniac_factory,
    play_session,
    random_factory,
    run_matchup,
    tight_aggressive_factory,
)
from pokr.bot import PokerBot
from pokr.rl.agent import RLStrategy
from pokr.rl.league import League
from pokr.rl.net import PolicyValueNet, load, save
from pokr.rl.ppo import PPOConfig, PPOTrainer

# Opponents the agent trains against. The heuristic (PokerBot) is the bot we
# are ultimately trying to beat, so it earns a seat in the pool.
POOL = {
    "cs": calling_station_factory,
    "tag": tight_aggressive_factory,
    "maniac": maniac_factory,
    "random": random_factory,
}
# Sampling weights: weighted toward the opponents that actually punish bad play.
# "league" is a frozen past self (pokr/rl/league.py), not a name in POOL.
POOL_WEIGHTS = {"cs": 2, "tag": 4, "maniac": 1, "random": 1,
                "heuristic": 3, "league": 3}

# Matchups reported at every evaluation, mirroring bench's naming.
EVAL_MATCHUPS = ("cs", "tag", "random", "heuristic")


def parse_seats(spec: str) -> list[int]:
    """'6' or '2,6' -> table sizes to sample from, one per iteration.

    Mixing sizes beats training ring-then-heads-up in sequence: a second phase
    at a single table size overwrites what the first one learned. Heads-up is a
    genuinely different game (blind structure, opening ranges), and the first
    trained agent lost 225 bb/100 to the heuristic there having only ever
    played 6-max.
    """
    seats = [int(x) for x in spec.split(",") if x.strip()]
    if not seats or any(n < 2 for n in seats):
        raise ValueError(f"--seats must be comma-separated integers >= 2, got {spec!r}")
    return seats


def heuristic_factory(mc_iters: int, mc_fast: bool):
    def factory(rng: random.Random):
        return PokerBot(rng, mc_iters=mc_iters, mc_fast=mc_fast)
    factory.__name__ = "heuristic"
    return factory


def build_pool(names: list[str], opp_mc_iters: int, mc_fast: bool,
               league: League | None = None, rng: random.Random | None = None,
               agent_mc_iters: int = 0, num_players: int = 6) -> list:
    """Resolve pool names to opponent factories for one iteration.

    League seats get the agent's OWN equity budget, not the heuristic's: a
    frozen net must be fed observations encoded the way it was trained, or its
    equity feature is noise and the past self plays worse than it really did.
    An empty league (before the first snapshot) falls back to the heuristic.
    """
    out = []
    for name in names:
        if name == "heuristic":
            out.append(heuristic_factory(opp_mc_iters, mc_fast))
        elif name == "league":
            factory = None
            if league is not None:
                factory = league.opponent_factory(
                    rng or random.Random(), agent_mc_iters, mc_fast, num_players)
            out.append(factory or heuristic_factory(opp_mc_iters, mc_fast))
        else:
            out.append(POOL[name])
    return out


def evaluate(agent: RLStrategy, hands: int, seed: int, seats: int,
             opp_mc_iters: int, mc_fast: bool) -> dict:
    """Greedy head-to-head vs each eval matchup, reusing bench.run_matchup so
    the numbers are directly comparable to the README table (incl. SE)."""
    out = {}
    for i, name in enumerate(EVAL_MATCHUPS):
        factory = (heuristic_factory(opp_mc_iters, mc_fast) if name == "heuristic"
                   else POOL[name])
        twin = agent.clone(greedy=True, rng=random.Random(seed + i))
        report = run_matchup(twin, factory, hands, seed + i * 97,
                             num_seats=seats, name=name)
        out[name] = report
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a PPO poker agent in the pokr engine")
    ap.add_argument("--iters", type=int, default=200, help="PPO iterations")
    ap.add_argument("--hands-per-iter", type=int, default=2000,
                    help="hands collected per iteration (one rollout batch)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seats", default="2,6",
                    help="table sizes to sample from, one per iteration "
                         "(e.g. '6' or '2,6'); heads-up and ring are different "
                         "games and mixing them beats training them in sequence")
    ap.add_argument("--mc-iters", type=int, default=30,
                    help="Monte Carlo equity iterations for the AGENT's equity "
                         "feature (0 disables the feature)")
    ap.add_argument("--opp-mc-iters", type=int, default=150,
                    help="Monte Carlo iterations for PokerBot opponents; separate "
                         "from --mc-iters, and must stay > 0 (PokerBot divides by "
                         "it). 150 is PokerBot's own default: training against a "
                         "weakened heuristic (10) produced an agent that drew with "
                         "the weak version and lost 225 bb/100 to the real one.")
    ap.add_argument("--fast", action="store_true",
                    help="use the numba equity fast path")
    ap.add_argument("--pool", type=str, default="cs,tag,maniac,random,heuristic,league",
                    help="comma-separated opponent pool "
                         f"({', '.join(sorted(POOL))}, heuristic, league)")
    ap.add_argument("--league-every", type=int, default=25,
                    help="snapshot the net into the league every N iterations "
                         "(0 disables the league)")
    ap.add_argument("--league-size", type=int, default=30,
                    help="max frozen snapshots kept; old ones break cycles, so "
                         "prefer an interval that never fills this")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--reward-scale", type=float, default=100.0)
    ap.add_argument("--eval-every", type=int, default=10, help="iterations between evals")
    ap.add_argument("--eval-hands", type=int, default=2000)
    ap.add_argument("--ckpt-dir", default="models/rl")
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        seats_pool = parse_seats(args.seats)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    max_seats = max(seats_pool)
    pool_names = [n.strip() for n in args.pool.split(",") if n.strip()]
    unknown = [n for n in pool_names
               if n not in POOL and n not in ("heuristic", "league")]
    if unknown:
        print(f"error: unknown pool entries {unknown}; "
              f"use {sorted(POOL)} + 'heuristic', 'league'")
        return 2
    if "league" in pool_names and args.league_every < 1:
        print("error: --pool includes 'league' but --league-every is 0")
        return 2
    if args.opp_mc_iters < 1:
        print("error: --opp-mc-iters must be >= 1 (PokerBot divides by it)")
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    start_iter = 0
    if args.resume:
        net, ckpt = load(args.resume)
        start_iter = int(ckpt.get("iteration", 0))
        print(f"resumed from {args.resume} (iteration {start_iter})")
    else:
        net = PolicyValueNet(hidden=tuple(args.hidden))

    cfg = PPOConfig(lr=args.lr, clip=args.clip, epochs=args.epochs,
                    minibatch=args.minibatch, lam=args.lam,
                    ent_coef=args.ent_coef, reward_scale=args.reward_scale)
    trainer = PPOTrainer(net, cfg)
    # the opponent-model table is sized for the largest table; smaller ones
    # simply leave the high seat ids unused
    agent = RLStrategy(net=net, rng=random.Random(args.seed), num_players=max_seats,
                       mc_iters=args.mc_iters, mc_fast=args.fast, record=True)

    league = League(max_size=args.league_size) if args.league_every > 0 else None
    weights = [POOL_WEIGHTS.get(n, 1) for n in pool_names]
    params = sum(p.numel() for p in net.parameters())
    print(f"net {args.hidden} ({params:,} params) | pool {pool_names} | "
          f"seats {seats_pool} | {args.hands_per_iter} hands/iter | "
          f"mc_iters {args.mc_iters} (opp {args.opp_mc_iters})"
          f"{' fast' if args.fast else ''}"
          + (f" | league every {args.league_every}" if league is not None else ""))

    t0 = time.time()
    for it in range(start_iter, start_iter + args.iters):
        if league is not None and it % args.league_every == 0:
            league.snapshot(net)
        seats = random.choice(seats_pool)
        picks = random.choices(pool_names, weights=weights, k=seats - 1)
        factories = build_pool(picks, args.opp_mc_iters, args.fast, league,
                               random.Random(args.seed + it), args.mc_iters, max_seats)
        agent.reset_models()
        agent.buffer.clear()

        t_roll = time.time()
        per_hand_bb, results = play_session(agent, factories, args.hands_per_iter,
                                            args.seed + it * 131, num_seats=seats)
        roll_s = time.time() - t_roll
        stats = trainer.update(agent.buffer)
        own = bot_own_stats(results, seat=0)

        print(f"it {it:>4} | {sum(per_hand_bb) / args.hands_per_iter * 100:>+8.1f} bb/100 "
              f"| ent {stats['entropy']:.3f} kl {stats['approx_kl']:+.4f} "
              f"clip {stats['clip_frac']:.2f} vloss {stats['value_loss']:.3f} "
              f"| VPIP {own['vpip'] * 100:>4.1f}% PFR {own['pfr'] * 100:>4.1f}% "
              f"aggr {own['aggression_freq']:.2f} "
              f"| {args.hands_per_iter / roll_s:>5.0f} h/s {seats}max vs {','.join(picks)}"
              + (f" [L{len(league)}]" if league is not None else ""))

        if (it + 1) % args.eval_every == 0 or it == start_iter + args.iters - 1:
            summary = {}
            for n_seats in seats_pool:
                reports = evaluate(agent, args.eval_hands, args.seed + 50_000,
                                   n_seats, args.opp_mc_iters, args.fast)
                print(f"  eval {n_seats}max " + "  ".join(
                    f"{n}: {r.bb_per_100:+.1f}+-{2 * r.se_bb_per_100:.0f}"
                    for n, r in reports.items()))
                summary.update({f"{n}@{n_seats}max": r.bb_per_100
                                for n, r in reports.items()})
            save(net, os.path.join(args.ckpt_dir, "ppo_latest.pt"),
                 iteration=it + 1, config=vars(args), eval=summary)

    final = os.path.join(args.ckpt_dir, "ppo_final.pt")
    save(net, final, iteration=start_iter + args.iters, config=vars(args))
    print(f"done: {args.iters} iters in {(time.time() - t0) / 60:.1f} min -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
