"""Multiprocess rollout collection for PPO training.

A training iteration's rollout phase (bench.play_session driving RLStrategy)
is pure CPU and embarrassingly parallel across hands: each hand only depends
on its own RNG stream, never on another hand's outcome. On a 10-core machine
a single-process rollout uses one core while nine sit idle, so the fix is to
split num_hands across worker processes and concatenate what comes back.

Two things cannot cross a process boundary as-is:
  - Opponent factories (bench.LINEUP_ABBREVS / train_rl.POOL) are lambdas, so
    workers are handed opponent *names* and rebuild factories locally via
    _resolve_factory -- the same trick bench's own CLI uses for --lineup.
  - A PolicyValueNet *can* pickle (it's an nn.Module), but shipping the raw
    module ties the worker to however the module happens to be constructed in
    the parent. Sending state_dict() + config() and rebuilding is the same
    amount of data with none of that coupling, so that's what crosses.

fork vs spawn: this module uses fork (multiprocessing.get_context("fork")),
not spawn. Measured on this machine: a spawn pool of 4 workers takes ~1.9s
just to re-import torch in each child before doing any work, vs ~0.05s for a
fork pool of 4 -- since collect_parallel is called once per training
iteration (hundreds of iterations in a run), spawn's per-call tax would
dominate the rollout budget and eat most of the parallelism gain. Fork's
usual hazard is that a child inherits copies of the parent's threads/locks
mid-state, which can deadlock native thread pools (this is why PyTorch's own
docs lean spawn for CUDA multiprocessing); measured here with a CPU-only net
and torch already exercised multithreaded in the parent before forking, plain
fork completed without hanging or warning. The one hazard that DID reproduce:
torch's global RNG (used by PolicyValueNet.act's torch.multinomial, which
takes no explicit generator) is copied byte-for-byte into every forked child,
so sibling workers drew an *identical* action sequence from the net until
each worker called torch.manual_seed() itself right after forking. Every
worker now does that (and torch.set_num_threads(1), see below) as the first
thing it does -- both are load-bearing, not defensive boilerplate.

torch.set_num_threads(1) inside each worker matters a lot: PyTorch defaults
to using every logical core for its intra-op thread pool, so N worker
processes each grabbing all cores fight each other and run slower than a
single serial process. Confirmed on this box (see collect_parallel's report):
omitting the call makes workers=4 *slower* than workers=1.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Callable, Sequence

import torch

from ..bench import (
    bot_own_stats,
    calling_station_factory,
    leak_hunter_factory,
    maniac_factory,
    play_session,
    random_factory,
    tight_aggressive_factory,
)
from ..bot import PokerBot
from ..strategy import Strategy
from .agent import Episode, RLStrategy
from .net import PolicyValueNet

# Opponents cheap enough to reconstruct inside a worker with no extra deps.
# Mirrors bench.LINEUP_ABBREVS' non-RL entries plus train_rl's "heuristic",
# i.e. exactly the names train_rl.POOL / build_pool support for rollout
# opponents. "league" is deliberately not here: it hands out a frozen NET
# snapshot (state, not a name), which is a different shape of input than the
# picklable strings this module is built around -- a caller that wants league
# opponents resolves the snapshot to a net and drives it through workers=1,
# or this module grows a net-sequence parameter later if that's ever the
# bottleneck.
_NAMED_FACTORIES: dict[str, Callable[[random.Random], Strategy]] = {
    "cs": calling_station_factory,
    "tag": tight_aggressive_factory,
    "maniac": maniac_factory,
    "random": random_factory,
    "leak": leak_hunter_factory,
}


def _league_factory(state: dict, config: dict, mc_iters: int, mc_fast: bool,
                    num_players: int) -> Callable[[random.Random], Strategy]:
    """Rebuild one frozen past self inside the worker.

    Only ONE snapshot crosses the process boundary per call (the caller pins
    it), because shipping a whole league would mean pickling ~10MB per worker
    per training iteration. A frozen self gets the AGENT's equity budget, not
    the heuristic's: fed differently-encoded observations it plays worse than
    it really did, which would quietly weaken the league.
    """
    net = PolicyValueNet(**config)
    net.load_state_dict(state)
    net.eval()
    net.requires_grad_(False)

    def factory(rng: random.Random) -> Strategy:
        return RLStrategy(net=net, rng=rng, num_players=num_players,
                          mc_iters=mc_iters, mc_fast=mc_fast,
                          greedy=False, record=False)
    return factory


def _resolve_factory(name: str, opp_mc_iters: int, mc_fast: bool
                      ) -> Callable[[random.Random], Strategy]:
    """Name -> opponent factory, validated eagerly (before any worker spawns)
    so a typo raises a clear error in the caller's process instead of dying
    silently inside a child (multiprocessing's default error reporting for a
    worker exception is a wall of pickled traceback, not a clear message)."""
    if name == "heuristic":
        def factory(rng: random.Random) -> Strategy:
            return PokerBot(rng, mc_iters=opp_mc_iters, mc_fast=mc_fast)
        factory.__name__ = "heuristic"
        return factory
    try:
        return _NAMED_FACTORIES[name]
    except KeyError:
        known = sorted(_NAMED_FACTORIES) + ["heuristic"]
        raise ValueError(f"unknown opponent name {name!r}; use one of {known}") from None


def _collect_one(
    net: PolicyValueNet | None,
    opponent_names: Sequence[str],
    num_hands: int,
    seed: int,
    num_seats: int,
    mc_iters: int,
    mc_fast: bool,
    opp_mc_iters: int,
    buy_in: int,
    small_blind: int,
    big_blind: int,
    league_state: dict | None = None,
    league_config: dict | None = None,
    reset_stacks: bool = False,
) -> tuple[list[Episode], list, dict]:
    """The actual rollout: build one RLStrategy + its opponents and play
    num_hands through bench.play_session, exactly like train_rl's main loop
    does today. Shared by the workers=1 fallback (called directly, no
    subprocess) and by each forked worker (called after that worker has set
    up its own thread count and RNG -- see _worker_entry)."""
    agent = RLStrategy(net=net, rng=random.Random(seed), num_players=num_seats,
                       mc_iters=mc_iters, mc_fast=mc_fast, record=True)
    factories = []
    for name in opponent_names:
        if name == "league":
            if league_state is None:
                raise ValueError("opponent 'league' requires league_state")
            factories.append(_league_factory(league_state, league_config, mc_iters,
                                             mc_fast, num_seats))
        else:
            factories.append(_resolve_factory(name, opp_mc_iters, mc_fast))
    per_hand_bb, results = play_session(agent, factories, num_hands, seed,
                                        num_seats=num_seats, buy_in=buy_in,
                                        small_blind=small_blind, big_blind=big_blind,
                                        reset_stacks=reset_stacks)
    # own_stats travels back with the episodes: VPIP/PFR collapsing toward zero
    # is the classic PPO failure here (folding caps loss and has near-zero
    # variance), and it is invisible in the reward curve until far too late.
    return agent.buffer.episodes, per_hand_bb, bot_own_stats(results, seat=0)


@dataclass
class _WorkerJob:
    """Everything one worker needs, pre-flattened to picklable primitives --
    tensors and plain values, never the net module or an opponent lambda."""
    state_dict: dict | None
    net_config: dict | None
    opponent_names: tuple[str, ...]
    num_hands: int
    seed: int
    num_seats: int
    mc_iters: int
    mc_fast: bool
    opp_mc_iters: int
    buy_in: int
    small_blind: int
    big_blind: int
    league_state: dict | None = None
    league_config: dict | None = None
    reset_stacks: bool = False


def _worker_entry(job: _WorkerJob) -> tuple[list[Episode], list, dict]:
    """Pool.map target. Runs inside a forked child: reseed torch's global RNG
    and pin thread count before touching the net or playing a single hand --
    see the module docstring for why both are load-bearing here."""
    torch.set_num_threads(1)
    torch.manual_seed(job.seed)
    net = None
    if job.state_dict is not None:
        net = PolicyValueNet(**job.net_config)
        net.load_state_dict(job.state_dict)
        net.eval()
    return _collect_one(net, job.opponent_names, job.num_hands, job.seed, job.num_seats,
                        job.mc_iters, job.mc_fast, job.opp_mc_iters, job.buy_in,
                        job.small_blind, job.big_blind, job.league_state,
                        job.league_config, job.reset_stacks)


@dataclass
class RolloutResult:
    """What one training iteration collected."""
    episodes: list[Episode]
    per_hand_bb: list
    own_stats: dict

    def __iter__(self):
        """Unpack as (episodes, per_hand_bb) for call sites that predate
        own_stats."""
        return iter((self.episodes, self.per_hand_bb))


def _split_hands(num_hands: int, workers: int) -> list[int]:
    """Distribute num_hands across `workers` as evenly as possible, remainder
    to the first workers, so an odd hand count is never silently dropped."""
    base, rem = divmod(num_hands, workers)
    return [base + (1 if i < rem else 0) for i in range(workers)]


def collect_parallel(
    net: PolicyValueNet | None,
    opponent_names: Sequence[str],
    num_hands: int,
    seed: int,
    workers: int = 4,
    num_seats: int = 6,
    mc_iters: int = 0,
    mc_fast: bool = False,
    opp_mc_iters: int = 150,
    buy_in: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    league_state: dict | None = None,
    league_config: dict | None = None,
    reset_stacks: bool = False,
) -> "RolloutResult":
    """Collect num_hands of rollouts, split across `workers` processes.

    Returns a RolloutResult (episodes, per-hand bb, own_stats) matching what
    RLStrategy(net=net, record=True).buffer.episodes and play_session's
    per-hand bb would have produced from a single-process run -- this is a
    drop-in for train_rl's `play_session(agent, factories, hands, seed, ...)`
    call, minus the agent object itself: collect_parallel builds a fresh
    RLStrategy per call rather than taking one, since a persistent agent
    (its running equity-cache / opponent-model state) can't be shared across
    separate processes anyway.

    workers=1 bypasses multiprocessing entirely -- straight through
    _collect_one/play_session in-process -- both as a fast escape hatch and
    because it keeps this module testable without forking in every test.

    Determinism: same (seed, workers) -> identical episodes and rewards,
    because each worker's slice gets its own derived seed
    (seed + i * a stride clear of play_session's internal per-seat opponent
    seeding) and reseeds torch's RNG from it. Changing `workers` for the same
    seed changes which hands land in which worker and in what order they're
    played, so the *set* of hands collected differs across worker counts --
    expected and not a bug; only (seed, workers) together are reproducible.
    """
    if num_hands < 1:
        raise ValueError(f"num_hands must be >= 1, got {num_hands}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if len(opponent_names) != num_seats - 1:
        raise ValueError(f"need {num_seats - 1} opponent names for {num_seats} seats, "
                         f"got {len(opponent_names)}")
    for name in opponent_names:
        if name == "league":
            if league_state is None:
                raise ValueError("opponent 'league' requires league_state")
            continue
        _resolve_factory(name, opp_mc_iters, mc_fast)  # raise here, not inside a worker

    if workers == 1:
        return RolloutResult(*_collect_one(
            net, opponent_names, num_hands, seed, num_seats, mc_iters,
            mc_fast, opp_mc_iters, buy_in, small_blind, big_blind,
            league_state, league_config, reset_stacks))

    # play_session itself derives per-opponent-seat RNGs as seed + 1000*(i+1)
    # for up to num_seats-1 (<=5 in this codebase) seats; a stride well past
    # that keeps worker seeds from ever landing on an opponent's sub-stream.
    stride = 1_000_003
    hand_counts = [c for c in _split_hands(num_hands, workers) if c > 0]

    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()} if net is not None else None
    net_config = net.config() if net is not None else None
    names = tuple(opponent_names)
    jobs = [
        _WorkerJob(state_dict, net_config, names, hand_counts[i], seed + i * stride,
                   num_seats, mc_iters, mc_fast, opp_mc_iters, buy_in, small_blind,
                   big_blind, league_state, league_config, reset_stacks)
        for i in range(len(hand_counts))
    ]

    ctx = get_context("fork")
    with ctx.Pool(processes=len(jobs)) as pool:
        results = pool.map(_worker_entry, jobs)

    episodes: list[Episode] = []
    per_hand_bb: list = []
    stats: list[dict] = []
    for eps, bb, own in results:
        episodes.extend(eps)
        per_hand_bb.extend(bb)
        stats.append(own)
    # workers get near-equal hand counts, so a plain mean of the per-worker
    # rates is a fine diagnostic; these are for spotting a collapse, not for
    # reporting.
    merged = {k: sum(d[k] for d in stats) / len(stats) for k in stats[0]} if stats else {}
    return RolloutResult(episodes, per_hand_bb, merged)
