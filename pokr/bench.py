from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .bot import PokerBot
from .engine import PokerGame
from .opponents import (
    CallingStation,
    LeakHunter,
    Maniac,
    RandomBot,
    TightAggressive,
)
from .strategy import Strategy


@dataclass
class MatchupReport:
    name: str
    hands: int
    total_bb: float
    bb_per_100: float
    win_rate: float
    var_bb_per_100: float


def calling_station_factory(rng: random.Random) -> Strategy:
    return CallingStation()


def tight_aggressive_factory(rng: random.Random) -> Strategy:
    return TightAggressive(rng)


def maniac_factory(rng: random.Random) -> Strategy:
    return Maniac(rng)


def random_factory(rng: random.Random) -> Strategy:
    return RandomBot(rng)


def leak_hunter_factory(rng: random.Random) -> Strategy:
    return LeakHunter(rng, target_seat=0)


def _rebuy(stacks, buy_in):
    return [buy_in if s <= 0 else s for s in stacks]


def run_matchup(
    bot: Strategy,
    opponent_factory: Callable[[random.Random], Strategy],
    num_hands: int,
    seed: int,
    num_seats: int = 6,
    buy_in: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    name: str = "",
) -> MatchupReport:
    rng = random.Random(seed)
    lineup = [bot] + [opponent_factory(random.Random(seed + 1000 * (i + 1)))
                      for i in range(num_seats - 1)]
    stacks = [buy_in] * num_seats
    per_hand_bb: list[float] = []
    for h in range(num_hands):
        game = PokerGame(lineup, stacks, small_blind, big_blind, rng,
                         initial_dealer=h % num_seats)
        result = game.play_hand()
        stacks = _rebuy(result.ending_stacks, buy_in)
        per_hand_bb.append(result.winnings[0] / big_blind)
    total = sum(per_hand_bb)
    bb_per_100 = total / num_hands * 100.0
    win_rate = sum(1 for w in per_hand_bb if w > 0) / num_hands
    var = float(np.var(per_hand_bb)) if num_hands > 1 else 0.0
    return MatchupReport(name or getattr(opponent_factory, "__name__", "opponent"),
                         num_hands, total, bb_per_100, win_rate, var)


_DEFAULT_FACTORIES = [
    calling_station_factory,
    tight_aggressive_factory,
    maniac_factory,
    random_factory,
]


def run_benchmark(
    bot: Strategy,
    num_hands: int,
    seed: int,
    opponent_factories: list | None = None,
    include_self: bool = True,
    include_leak_hunter: bool = True,
    num_seats: int = 6,
    buy_in: int = 200,
    mc_iters: int | None = None,
) -> list[MatchupReport]:
    """Run the full benchmark. mc_iters, when given, is applied to self-play
    opponents (the primary bot is passed in as `bot` and already configured)."""
    factories = list(opponent_factories) if opponent_factories else list(_DEFAULT_FACTORIES)
    reports = [
        run_matchup(bot, f, num_hands, seed + i * 97, num_seats=num_seats, buy_in=buy_in,
                    name=f.__name__)
        for i, f in enumerate(factories)
    ]
    if include_self:
        def self_factory(rng):
            return PokerBot(rng, mc_iters=mc_iters) if mc_iters is not None else PokerBot(rng)
        reports.append(run_matchup(bot, self_factory, num_hands,
                                   seed + 5000, num_seats=num_seats, buy_in=buy_in,
                                   name="self_play"))
    if include_leak_hunter:
        reports.append(run_matchup(bot, leak_hunter_factory, num_hands,
                                   seed + 9000, num_seats=num_seats, buy_in=buy_in,
                                   name="leak_hunter"))
    return reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark the pokr bot")
    ap.add_argument("--hands", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seats", type=int, default=6)
    ap.add_argument("--buy-in", type=int, default=200)
    ap.add_argument("--mc-iters", type=int, default=150,
                    help="Monte Carlo equity iterations per decision (default 150; "
                         "lower for faster runs, e.g. 10-30)")
    args = ap.parse_args(argv)
    reports = run_benchmark(PokerBot(random.Random(args.seed), mc_iters=args.mc_iters),
                            args.hands, args.seed,
                            num_seats=args.seats, buy_in=args.buy_in, mc_iters=args.mc_iters)
    print(f"{'matchup':<16}{'hands':>6}{'total_bb':>10}{'bb/100':>10}{'win%':>8}{'var':>10}")
    for r in reports:
        print(f"{r.name:<16}{r.hands:>6}{r.total_bb:>10.2f}{r.bb_per_100:>10.2f}"
              f"{r.win_rate * 100:>7.1f}%{r.var_bb_per_100:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
