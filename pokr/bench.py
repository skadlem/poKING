from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .bot import PokerBot
from .cards import evaluate_hand, hand_name
from .connector import build_strategy
from .engine import HandResult, PokerGame
from .models import OpponentModel
from .opponents import (
    CallingStation,
    LeakHunter,
    Maniac,
    RandomBot,
    TightAggressive,
)
from .strategy import ActionType, Strategy


@dataclass
class MatchupReport:
    name: str
    hands: int
    total_bb: float
    bb_per_100: float
    win_rate: float
    var_bb_per_100: float

    @property
    def se_bb_per_100(self) -> float:
        """Standard error of bb_per_100 over the per-hand sample. Matchups
        whose |bb/100| is within ~2 SE of 0 are statistically unresolved."""
        return math.sqrt(self.var_bb_per_100 / self.hands) * 100.0 if self.hands > 0 else 0.0


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
    """Top up busted players. Note this never CAPS a winner, so total chips
    inflate over a session: measured with six CallingStations, 1200 -> 1600
    chips over 2000 hands with a max stack of 293bb. Late hands in every
    matchup are therefore played 2-3x deeper than the nominal 100bb, which is
    a real contributor to the fat tails in the benchmark table. Pass
    reset_stacks=True to play every hand at the buy-in instead."""
    return [buy_in if s <= 0 else s for s in stacks]


def _next_stacks(stacks, buy_in, reset_stacks):
    return [buy_in] * len(stacks) if reset_stacks else _rebuy(stacks, buy_in)


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
    reset_stacks: bool = False,
) -> MatchupReport:
    """reset_stacks=True plays every hand at the buy-in, making hands
    independent and fixed-depth; the default carries stacks over (see _rebuy),
    which is what the published README numbers were measured with."""
    rng = random.Random(seed)
    lineup = [bot] + [opponent_factory(random.Random(seed + 1000 * (i + 1)))
                      for i in range(num_seats - 1)]
    stacks = [buy_in] * num_seats
    per_hand_bb: list[float] = []
    for h in range(num_hands):
        game = PokerGame(lineup, stacks, small_blind, big_blind, rng,
                         initial_dealer=h % num_seats)
        result = game.play_hand()
        stacks = _next_stacks(result.ending_stacks, buy_in, reset_stacks)
        per_hand_bb.append(result.winnings[0] / big_blind)
    total = sum(per_hand_bb)
    bb_per_100 = total / num_hands * 100.0
    win_rate = sum(1 for w in per_hand_bb if w > 0) / num_hands
    var = float(np.var(per_hand_bb)) if num_hands > 1 else 0.0
    return MatchupReport(name or getattr(opponent_factory, "__name__", "opponent"),
                         num_hands, total, bb_per_100, win_rate, var)


def play_session(
    bot: Strategy,
    opponent_factories: Sequence[Callable[[random.Random], Strategy]],
    num_hands: int,
    seed: int,
    num_seats: int = 6,
    buy_in: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    reset_stacks: bool = False,
) -> tuple[list[float], list[HandResult]]:
    """Play num_hands of 6-max poker: `bot` at seat 0, one opponent factory per
    remaining seat (reused across hands so models persist). Returns per-hand bb
    for seat 0 and the full HandResult list (for replay/analysis)."""
    assert len(opponent_factories) == num_seats - 1, \
        f"need {num_seats - 1} opponent factories for {num_seats} seats"
    rng = random.Random(seed)
    lineup = [bot] + [f(random.Random(seed + 1000 * (i + 1)))
                      for i, f in enumerate(opponent_factories)]
    stacks = [buy_in] * num_seats
    per_hand_bb: list[float] = []
    results: list[HandResult] = []
    for h in range(num_hands):
        game = PokerGame(lineup, stacks, small_blind, big_blind, rng,
                         initial_dealer=h % num_seats)
        result = game.play_hand()
        stacks = _next_stacks(result.ending_stacks, buy_in, reset_stacks)
        per_hand_bb.append(result.winnings[0] / big_blind)
        results.append(result)
    return per_hand_bb, results


def run_mixed_matchup(
    bot: Strategy,
    opponent_factories: Sequence[Callable[[random.Random], Strategy]],
    num_hands: int,
    seed: int,
    num_seats: int = 6,
    buy_in: int = 200,
    name: str = "mixed_lineup",
) -> MatchupReport:
    """Game benchmark: the bot at a full 6-max table against a mixed lineup of
    other bots (one factory per opponent seat)."""
    per_hand_bb, _ = play_session(bot, opponent_factories, num_hands, seed,
                                  num_seats=num_seats, buy_in=buy_in)
    total = sum(per_hand_bb)
    bb_per_100 = total / num_hands * 100.0
    win_rate = sum(1 for w in per_hand_bb if w > 0) / num_hands
    var = float(np.var(per_hand_bb)) if num_hands > 1 else 0.0
    return MatchupReport(name, num_hands, total, bb_per_100, win_rate, var)


def bot_own_stats(results: Sequence[HandResult], seat: int = 0) -> dict:
    """How the bot itself played: VPIP/PFR/aggression/fold rates from its own
    actions across the session (observer id -1 so it never equals the target)."""
    m = OpponentModel()
    for r in results:
        m.update(r, observer_id=-1, target_id=seat)
    s = m.summary()
    return {
        "hands": s.hands_observed,
        "vpip": s.vpip,
        "pfr": s.pfr,
        "aggression_freq": s.aggression_freq,
        "fold_to_cbet": s.fold_to_cbet,
        "fold_rate_postflop": s.fold_rate_postflop,
    }


def format_hand(result: HandResult, seat_names: Sequence[str] | None = None,
                hand_label: str | None = None) -> str:
    """Human-readable replay of one hand: blinds, every action with reason,
    streets, showdown, and net winnings."""
    n = len(result.starting_stacks)
    names = seat_names or [f"seat{i}" for i in range(n)]
    label = hand_label or f"Hand #{result.hand_number}"
    lines = [f"--- {label} (dealer {names[result.dealer]}) ---"]
    if n == 2:
        sb, bb = result.dealer, (result.dealer + 1) % n
    else:
        sb, bb = (result.dealer + 1) % n, (result.dealer + 2) % n
    lines.append(f"  blinds: {names[sb]} {result.big_blind // 2} / {names[bb]} {result.big_blind}")
    for (pid, street, action) in result.actions:
        amt = "" if action.amount == 0 else f" {action.amount}"
        reason = f"  [{action.reason}]" if action.reason else ""
        lines.append(f"  {names[pid]:<12} {street:<8} {action.action_type.value}{amt}{reason}")
    if len(result.community) >= 3:
        lines.append(f"  board: {' '.join(map(str, result.community))}")
    if len(result.hole) > 0:
        showdown = [i for i, w in enumerate(result.winnings) if w > 0]
        for i in showdown:
            score = evaluate_hand(result.hole[i] + result.community)
            lines.append(f"  {names[i]} wins {result.winnings[i]} with "
                         f"{hand_name(score)} ({' '.join(map(str, result.hole[i]))})")
    lines.append(f"  net: {' '.join(f'{names[i]} {w:+d}' for i, w in enumerate(result.winnings))}")
    return "\n".join(lines)


# Lineup abbreviations for the CLI: name -> factory
LINEUP_ABBREVS = {
    "cs": calling_station_factory,
    "tag": tight_aggressive_factory,
    "maniac": maniac_factory,
    "random": random_factory,
    "leak": leak_hunter_factory,
    "self": lambda rng: PokerBot(rng),
    "rlcard": lambda rng: build_strategy("rlcard"),
    "rlcard-dqn": lambda rng: build_strategy("rlcard-dqn"),
    "rl": lambda rng: build_strategy("rl"),
    "nfsp": lambda rng: build_strategy("nfsp"),
}
LINEUP_NAMES = {
    "cs": "CallingStation", "tag": "TightAggressive", "maniac": "Maniac",
    "random": "RandomBot", "leak": "LeakHunter", "self": "PokerBot",
    "rlcard": "RlcardRandom", "rlcard-dqn": "RlcardDQN", "rl": "PokrPPO",
    "nfsp": "PokrNFSP",
}


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
    mc_fast: bool = False,
    reset_stacks: bool = False,
) -> list[MatchupReport]:
    """Run the full benchmark. mc_iters, when given, is applied to self-play
    opponents and to fresh clones of the primary bot. A FRESH bot is used per
    matchup: reusing one bot across matchups pollutes its per-seat opponent
    models (the same seat index holds different opponents in different
    matchups), which distorts results (measured: TAG matchup -210 with a clean
    bot vs -265 with a polluted one)."""
    def fresh_bot():
        if isinstance(bot, PokerBot):
            return PokerBot(random.Random(seed), risk_cfg=bot.policy.risk_cfg,
                            num_players=bot.num_players, mc_iters=bot.policy.mc_iters,
                            mc_fast=bot.policy.mc_fast)
        return bot

    factories = list(opponent_factories) if opponent_factories else list(_DEFAULT_FACTORIES)
    reports = [
        run_matchup(fresh_bot(), f, num_hands, seed + i * 97, num_seats=num_seats, buy_in=buy_in,
                    name=f.__name__, reset_stacks=reset_stacks)
        for i, f in enumerate(factories)
    ]
    if include_self:
        def self_factory(rng):
            return (PokerBot(rng, mc_iters=mc_iters, mc_fast=mc_fast) if mc_iters is not None
                    else PokerBot(rng, mc_fast=mc_fast))
        reports.append(run_matchup(fresh_bot(), self_factory, num_hands,
                                   seed + 5000, num_seats=num_seats, buy_in=buy_in,
                                   name="self_play", reset_stacks=reset_stacks))
    if include_leak_hunter:
        reports.append(run_matchup(fresh_bot(), leak_hunter_factory, num_hands,
                                   seed + 9000, num_seats=num_seats, buy_in=buy_in,
                                   name="leak_hunter", reset_stacks=reset_stacks))
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
    ap.add_argument("--fast", action="store_true",
                    help="use the numba equity fast path (~10-120x faster per "
                         "decision; RNG stream differs from the pure path)")
    ap.add_argument("--lineup", type=str, default=None,
                    help="Play a real 6-max game vs a mixed lineup, e.g. "
                         "'cs,tag,tag,maniac,random' (one abbr per opponent seat: "
                         "cs, tag, maniac, random, leak, self)")
    ap.add_argument("--reset-stacks", action="store_true",
                    help="play every hand at the buy-in instead of carrying "
                         "stacks over. Carrying over inflates chips across a "
                         "session (see _rebuy), so late hands run 2-3x deeper "
                         "than nominal; resetting keeps every hand at 100bb. "
                         "Off by default: the README table was measured with "
                         "carry-over.")
    ap.add_argument("--replay", type=int, default=None,
                    help="With --lineup: print a human-readable replay of hand N "
                         "(0-based) from the session")
    args = ap.parse_args(argv)

    if args.lineup:
        abbrs = [a.strip() for a in args.lineup.split(",")]
        if len(abbrs) != args.seats - 1:
            print(f"error: --lineup needs {args.seats - 1} abbreviations for "
                  f"{args.seats} seats, got {len(abbrs)}")
            return 2
        for a in abbrs:
            if a not in LINEUP_ABBREVS:
                print(f"error: unknown lineup abbr {a!r}; use one of {sorted(LINEUP_ABBREVS)}")
                return 2
        factories = [LINEUP_ABBREVS[a] for a in abbrs]
        names = ["You(pokr)"] + [LINEUP_NAMES[a] for a in abbrs]
        bot = PokerBot(random.Random(args.seed), mc_iters=args.mc_iters, mc_fast=args.fast)
        per_hand_bb, results = play_session(bot, factories, args.hands, args.seed,
                                            num_seats=args.seats, buy_in=args.buy_in,
                                            reset_stacks=args.reset_stacks)
        total = sum(per_hand_bb)
        bb = total / args.hands * 100.0
        win_rate = sum(1 for w in per_hand_bb if w > 0) / args.hands
        var = float(np.var(per_hand_bb)) if args.hands > 1 else 0.0
        print(f"game vs lineup [{', '.join(names[1:])}]  ({args.hands} hands)")
        print(f"  you: {total:+.1f} bb total, {bb:+.2f} bb/100, "
              f"won {win_rate * 100:.1f}% of hands, var {var:.1f}")
        stats = bot_own_stats(results, seat=0)
        print(f"  your play: VPIP {stats['vpip'] * 100:.1f}%  PFR {stats['pfr'] * 100:.1f}%  "
              f"postflop aggression {stats['aggression_freq']:.2f}  "
              f"fold-to-cbet {stats['fold_to_cbet'] * 100:.0f}%  "
              f"postflop fold {stats['fold_rate_postflop'] * 100:.0f}%")
        if args.replay is not None:
            if 0 <= args.replay < len(results):
                print()
                print(format_hand(results[args.replay], names,
                                  hand_label=f"Hand #{args.replay} of {args.hands}"))
            else:
                print(f"error: --replay {args.replay} out of range (0..{len(results) - 1})")
                return 2
        return 0

    reports = run_benchmark(PokerBot(random.Random(args.seed), mc_iters=args.mc_iters,
                                     mc_fast=args.fast),
                            args.hands, args.seed,
                            num_seats=args.seats, buy_in=args.buy_in,
                            mc_iters=args.mc_iters, mc_fast=args.fast,
                            reset_stacks=args.reset_stacks)
    print(f"{'matchup':<16}{'hands':>6}{'total_bb':>10}{'bb/100':>10}{'SE':>8}"
          f"{'win%':>8}{'var':>10}")
    for r in reports:
        print(f"{r.name:<16}{r.hands:>6}{r.total_bb:>10.2f}{r.bb_per_100:>10.2f}"
              f"{r.se_bb_per_100:>8.1f}{r.win_rate * 100:>7.1f}%{r.var_bb_per_100:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
