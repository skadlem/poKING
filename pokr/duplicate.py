"""Duplicate-deck evaluation: one deck, played twice with the seats swapped.

Poker results are dominated by card luck. bench.run_matchup measures a single
arm and pays the full variance for it -- the README's own maniac and random
rows are unresolved at 2000 hands because the standard error exceeds the
estimate. Duplicate scoring removes most of that.

Both heroes sit at the SAME table, in seats 0 and 1, and each deck is played
twice with them swapped:

    run 1:  [A, B, *opponents]
    run 2:  [B, A, *opponents]      <- same deck, same opponents

A therefore holds seat 0's cards once and seat 1's cards once, and so does B.
A hero's score for a deck is the AVERAGE of its two hands, which is where the
variance goes: if seat 0's cards dominate seat 1's, the hero wins the first
hand and loses the second, and the deal's luck cancels inside its own score.

HOW MUCH THIS ACTUALLY BUYS: 1.0x to 3.5x, depending entirely on how alike the
two players are -- not the flat 5-10x that duplicate scoring gives in bridge.
Measured:

    PokerBot vs TightAggressive, heads-up      1.1x
    trained PPO vs PokerBot, heads-up          1.5x
    trained PPO vs an earlier PPO checkpoint   3.5x  (12x fewer hands)
    trained PPO vs PokerBot, 6-max ring        1.0x

The reason is structural and worth not rediscovering. 85% of NLHE hands finish
under 4bb, and the top 1% of hands carry 42% of the total variance. Those big
pots exist only when BOTH players choose to build one, so they mirror across
the swap only to the extent the two players make similar decisions -- the deal
constrains them far less than the betting line does. Two similar agents mirror
well (correlation between a hero's two hands strongly negative, 3.5x); two
unlike strategies barely mirror at all (measured rho = -0.131, predicting
1/sqrt(1+rho) = 1.07x, which is what the estimator reports). Duplicate bridge
works because every deal is played to completion by both tables; duplicate
NLHE does not.

The practical consequence: do not count on pairing to resolve a close matchup.
Hands are cheap here (~1500/s), so resolve it with hand count instead -- SE
falls as 1/sqrt(n), and 200k hands runs in a couple of minutes. The known
technique that does target the real variance source is all-in EV adjustment
(replace a realized all-in outcome with its equity over all runouts, which
pokr._fastcards can already compute); that is not implemented here.

Two framings that look right and are not:

- Scoring the same deck twice without swapping seats (each hero in its own
  separate game). The heroes' outcomes are then near-uncorrelated -- measured
  corr +0.10, because how you play moves more chips than what you hold -- and
  the estimator is no tighter than an unpaired one. Measured: 1.0x.
- Scoring the DIFFERENCE (A - B) rather than each hero's own averaged rate.
  Heads-up the game is zero-sum, so b = -a exactly and the difference just
  doubles A's variance. The difference is reported for ring games, but the
  headline is each hero's own duplicate-averaged win rate.

The pairing is exact by construction. Hole cards are dealt to every seat
before any action, and board cards are drawn in a fixed order from the same
deck, so both runs see identical cards no matter how differently the heroes
play.

Stacks reset to the buy-in every hand, unlike bench.run_matchup (there,
bench._rebuy tops up busted players but never caps winners, so chips inflate
over a session and late hands are played 2-3x deeper than nominal). Resetting
is what makes each hand an independent, exactly-paired sample, and it keeps
every hand at the intended depth. The tradeoff: these numbers measure a
fixed-depth game and are not directly comparable to the README table.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .cards import Deck, all_cards
from .engine import PokerGame
from .strategy import Strategy

Factory = Callable[[random.Random], Strategy]


def _reseed(lineup: Sequence[Strategy], base: int) -> None:
    """Common random numbers: give every seat the same RNG state in both runs
    of a paired deck.

    Without this the pairing buys nothing (measured: 1.0x reduction). Bots draw
    a different number of random values once the heroes act differently, so
    their streams desynchronize and each run faces differently-behaved
    opponents -- noise that is uncorrelated between the runs and swamps the
    card luck the pairing was meant to cancel. Reseeding in place (rng.seed)
    rather than rebinding matters: PokerBot shares one Random with its Policy.
    """
    for i, strategy in enumerate(lineup):
        rng = getattr(strategy, "rng", None)
        if isinstance(rng, random.Random):
            rng.seed(base + i)


@dataclass
class DuplicateReport:
    name_a: str
    name_b: str
    decks: int                      # paired decks; each is two engine hands per hero
    bb_per_100_a: float
    bb_per_100_b: float
    se_a: float                     # SE of A's rate, duplicate-averaged over both seats
    se_b: float
    unpaired_se_a: float            # SE from the SAME hands scored individually
    unpaired_se_b: float
    diff_bb_per_100: float          # a - b (redundant heads-up: exactly 2a)
    se_diff: float
    per_deck_a: np.ndarray = field(repr=False, default=None)
    per_deck_b: np.ndarray = field(repr=False, default=None)

    @property
    def hands_per_hero(self) -> int:
        return 2 * self.decks

    @property
    def variance_reduction(self) -> float:
        """How many times tighter A's error bar is than scoring the same hands
        without the duplicate structure. Squared, it is how many times fewer
        hands are needed for equal resolution."""
        return self.unpaired_se_a / self.se_a if self.se_a > 0 else float("inf")

    @property
    def resolved(self) -> bool:
        """The gap clears 2 standard errors, matching the README's convention."""
        return abs(self.diff_bb_per_100) > 2 * self.se_diff

    def format(self) -> str:
        verdict = "resolved" if self.resolved else "UNRESOLVED (within 2 SE of zero)"
        ahead = self.name_a if self.diff_bb_per_100 > 0 else self.name_b
        return "\n".join([
            f"duplicate: {self.name_a} vs {self.name_b}  "
            f"({self.decks} decks = {self.hands_per_hero} hands per hero)",
            f"  {self.name_a:<14} {self.bb_per_100_a:+9.2f} bb/100  "
            f"+-{2 * self.se_a:.2f} (2 SE)",
            f"  {self.name_b:<14} {self.bb_per_100_b:+9.2f} bb/100  "
            f"+-{2 * self.se_b:.2f} (2 SE)",
            f"  gap            {self.diff_bb_per_100:+9.2f} bb/100  "
            f"+-{2 * self.se_diff:.2f} -> {verdict}"
            + (f", {ahead} ahead" if self.resolved else ""),
            f"  duplicate averaging tightened {self.name_a}'s bar "
            f"{self.variance_reduction:.1f}x "
            f"(same hands scored singly: +-{2 * self.unpaired_se_a:.2f}; "
            f"{self.variance_reduction ** 2:.0f}x fewer hands for equal resolution)",
        ])


def run_duplicate(
    factory_a: Factory,
    factory_b: Factory,
    opponent_factories: Sequence[Factory] = (),
    num_hands: int = 5000,
    seed: int = 7,
    buy_in: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    name_a: str = "a",
    name_b: str = "b",
    on_hand: Callable[[int, object, object], None] | None = None,
) -> DuplicateReport:
    """Play num_hands duplicate decks. Table size is 2 + len(opponent_factories).

    Each hero gets TWO instances, one per seat, and each run gets its own
    opponent instances. Stateful bots (PokerBot's per-seat opponent models,
    TightAggressive's rng) therefore never see a seat change identity, which
    would silently corrupt their reads.
    """
    num_seats = 2 + len(opponent_factories)
    deck_rng = random.Random(seed)

    def opponents():
        return [f(random.Random(seed + 1000 * (i + 1)))
                for i, f in enumerate(opponent_factories)]

    a0, a1 = factory_a(random.Random(seed)), factory_a(random.Random(seed + 1))
    b0, b1 = factory_b(random.Random(seed + 2)), factory_b(random.Random(seed + 3))
    lineups = ([a0, b1] + opponents(), [b0, a1] + opponents())

    base = all_cards()
    a_bb = np.empty(num_hands)
    b_bb = np.empty(num_hands)
    singles_a = np.empty(2 * num_hands)   # A's hands scored individually
    singles_b = np.empty(2 * num_hands)
    for h in range(num_hands):
        order = base[:]
        deck_rng.shuffle(order)
        dealer = h % num_seats
        results = []
        hand_seed = seed + h * 7919
        for lineup in lineups:
            _reseed(lineup, hand_seed)
            game = PokerGame(lineup, [buy_in] * num_seats, small_blind, big_blind,
                             random.Random(seed + h), initial_dealer=dealer,
                             deck=Deck(cards=order, shuffle=False))
            results.append(game.play_hand())
        r1, r2 = results
        # A held seat 0 in run 1 and seat 1 in run 2; B the mirror image.
        singles_a[2 * h] = r1.winnings[0] / big_blind      # A in seat 0
        singles_a[2 * h + 1] = r2.winnings[1] / big_blind  # A in seat 1
        singles_b[2 * h] = r1.winnings[1] / big_blind
        singles_b[2 * h + 1] = r2.winnings[0] / big_blind
        a_bb[h] = (singles_a[2 * h] + singles_a[2 * h + 1]) / 2
        b_bb[h] = (singles_b[2 * h] + singles_b[2 * h + 1]) / 2
        if on_hand is not None:
            on_hand(h, r1, r2)

    return _report(a_bb, b_bb, singles_a, singles_b, name_a, name_b)


def _report(a_bb, b_bb, singles_a, singles_b, name_a, name_b) -> DuplicateReport:
    n = len(a_bb)
    ddof = 1 if n > 1 else 0
    root_n = math.sqrt(n) if n else 1.0
    root_2n = math.sqrt(2 * n) if n else 1.0

    def se(x, root):
        return float(np.std(x, ddof=ddof)) / root * 100.0

    return DuplicateReport(
        name_a=name_a, name_b=name_b, decks=n,
        bb_per_100_a=float(a_bb.mean()) * 100.0,
        bb_per_100_b=float(b_bb.mean()) * 100.0,
        se_a=se(a_bb, root_n), se_b=se(b_bb, root_n),
        # the honest no-duplicate baseline: the very same 2n hands, each scored
        # on its own instead of averaged with its mirror
        unpaired_se_a=se(singles_a, root_2n),
        unpaired_se_b=se(singles_b, root_2n),
        diff_bb_per_100=float((a_bb - b_bb).mean()) * 100.0,
        se_diff=se(a_bb - b_bb, root_n),
        per_deck_a=a_bb, per_deck_b=b_bb,
    )


def main(argv: list[str] | None = None) -> int:
    from .bench import LINEUP_ABBREVS, LINEUP_NAMES
    from .bot import PokerBot

    ap = argparse.ArgumentParser(
        description="Head-to-head on duplicate decks (variance-reduced)")
    ap.add_argument("--a", default="rl", help=f"hero A ({', '.join(sorted(LINEUP_ABBREVS))})")
    ap.add_argument("--b", default="self", help="hero B ('self' is the heuristic PokerBot)")
    ap.add_argument("--lineup", default="tag,tag,cs,random",
                    help="opponent seats beyond the two heroes, one abbreviation "
                         "each; empty for a heads-up duplicate match")
    ap.add_argument("--hands", type=int, default=5000, help="duplicate decks (2 hands each)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--buy-in", type=int, default=200)
    ap.add_argument("--mc-iters", type=int, default=150,
                    help="Monte Carlo equity iterations for the heuristic bot "
                         "('self'); 10-30 for fast exploratory runs")
    ap.add_argument("--fast", action="store_true",
                    help="use the numba equity fast path for the heuristic bot")
    args = ap.parse_args(argv)

    def resolve(name):
        """'self' is a bare PokerBot in LINEUP_ABBREVS, so its equity budget is
        not tunable there; rebuild it here the way bench's own CLI does."""
        if name == "self":
            return lambda rng: PokerBot(rng, mc_iters=args.mc_iters, mc_fast=args.fast)
        return LINEUP_ABBREVS[name]

    abbrs = [x.strip() for x in args.lineup.split(",") if x.strip()]
    for name in [args.a, args.b] + abbrs:
        if name not in LINEUP_ABBREVS:
            print(f"error: unknown bot {name!r}; use one of {sorted(LINEUP_ABBREVS)}")
            return 2
    report = run_duplicate(
        resolve(args.a), resolve(args.b), [resolve(x) for x in abbrs],
        args.hands, args.seed, buy_in=args.buy_in,
        name_a=LINEUP_NAMES[args.a], name_b=LINEUP_NAMES[args.b])
    print(f"table: {2 + len(abbrs)} seats"
          + (f", opponents {', '.join(LINEUP_NAMES[x] for x in abbrs)}" if abbrs else " (heads-up)"))
    print(report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
