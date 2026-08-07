"""Scratch diagnostic for the self-play / TAG leak (step 2).

Reconstructs pot evolution from HandResult.action streams and reports where
seat 0's money goes: by position, by showdown, by street, and by facing
bet/raising. Run:

    python analyze_leak.py --matchup self    [or tag]
    python analyze_leak.py --matchup self tag
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from pokr.bench import bot_own_stats, play_session
from pokr.bot import PokerBot
from pokr.engine import HandResult
from pokr.opponents import Maniac, TightAggressive
from pokr.strategy import ActionType

SB, BB = 1, 2
POS_NAMES = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"}


def _trace(result: HandResult):
    """Yield (pid, street, action, pot_before, committed_before) per action.

    pot_before is the pot (incl. blinds and prior action chips) at the moment
    the action is made; committed_before is street-committed per player."""
    n = len(result.starting_stacks)
    committed = [0] * n
    if n == 2:
        committed[result.dealer] = SB
        committed[(result.dealer + 1) % n] = BB
    else:
        committed[(result.dealer + 1) % n] = SB
        committed[(result.dealer + 2) % n] = BB
    pot = SB + BB
    prev_street = "preflop"
    for (pid, street, action) in result.actions:
        if street != prev_street:  # engine resets street_committed per street
            committed = [0] * n
            prev_street = street
        yield pid, street, action, pot, list(committed)
        t = action.action_type
        if t == ActionType.CALL:
            inc = action.amount
        elif t in (ActionType.BET, ActionType.RAISE):
            inc = action.amount - committed[pid]
        else:
            inc = 0
        committed[pid] += inc
        pot += inc


def _showdown(result: HandResult) -> bool:
    folded = {t for (t, s, a) in result.actions if a.action_type == ActionType.FOLD}
    dealt = [i for i in range(len(result.starting_stacks)) if result.starting_stacks[i] > 0]
    return len([i for i in dealt if i not in folded]) >= 2


def _analyze(results, label):
    st = bot_own_stats(results, seat=0)
    print(f"\n=== {label}  ({len(results)} hands, seat0 total "
          f"{sum(r.winnings[0] for r in results) / 2:+.1f} bb) ===")
    print(f"-- seat0 own play: VPIP {st['vpip']*100:.1f}%  PFR {st['pfr']*100:.1f}%  "
          f"pfr/vpip {st['pfr']/st['vpip'] if st['vpip'] else 0:.2f}  "
          f"postflop aggr {st['aggression_freq']:.2f}  "
          f"fold-to-cbet {st['fold_to_cbet']*100:.0f}%  "
          f"postflop fold {st['fold_rate_postflop']*100:.0f}% --")
    n_pos = len(results[0].starting_stacks)
    pl_pos = Counter()            # seat 0 P&L by position (bb)
    hands_by_pos = Counter()
    pl_sd = {"showdown": 0.0, "nosd": 0.0}
    pl_street = Counter()         # P&L by street of hand's last action
    sd_stats = {"won": 0, "lost": 0, "pots": []}
    own_bets: dict[str, list[tuple[float, str]]] = defaultdict(list)
    facing = Counter()            # (street, response) when facing opp bet
    facing_pl = defaultdict(float)
    for r in results:
        pos = POS_NAMES[(0 - r.dealer) % n_pos]
        hands_by_pos[pos] += 1
        w = r.winnings[0] / BB
        pl_pos[pos] += w
        is_sd = _showdown(r)
        pl_sd["showdown" if is_sd else "nosd"] += w
        pl_street[r.actions[-1][1] if r.actions else "none"] += w
        if is_sd:
            sd_stats["pots"].append(sum(x for x in r.winnings if x > 0) / BB)
            if w > 0:
                sd_stats["won"] += 1
            else:
                sd_stats["lost"] += 1
        # per-street action log for facing analysis
        street_acts: dict[str, list] = defaultdict(list)
        for (pid, street, action, pot, committed) in _trace(r):
            street_acts[street].append((pid, action, pot, committed))
            if pid == 0 and action.action_type in (ActionType.BET, ActionType.RAISE):
                inc = action.amount - committed[0]
                own_bets[street].append((inc / max(pot, 1), action.reason))
        for street, acts in street_acts.items():
            # seat 0 faces a bet when an opponent bet/raise occurred since the
            # last seat-0 action (or street start) in the same street.
            opp_bet_since = False
            for (pid, action, pot, committed) in acts:
                if pid != 0 and action.action_type in (ActionType.BET, ActionType.RAISE):
                    opp_bet_since = True
                elif pid == 0:
                    if opp_bet_since and action.action_type in (ActionType.CALL, ActionType.FOLD):
                        key = f"{street}:{action.action_type.value}"
                        facing[key] += 1
                        facing_pl[key] += w
                    opp_bet_since = False
    total = sum(pl_pos.values())
    pots = sd_stats["pots"]
    print("-- P&L by position (bb) --")
    for p in ("BTN", "SB", "BB", "UTG", "MP", "CO"):
        if hands_by_pos[p]:
            print(f"  {p}: {pl_pos[p]:+8.1f}  ({hands_by_pos[p]} hands)")
    print(f"-- showdown {pl_sd['showdown']:+.1f} bb "
          f"({sd_stats['won']} won/{sd_stats['lost']} lost, "
          f"avg SD pot {sum(pots)/len(pots) if pots else 0:.1f} bb) | "
          f"non-showdown {pl_sd['nosd']:+.1f} bb --")
    print("-- P&L by ending street --")
    for s in ("preflop", "flop", "turn", "river"):
        if pl_street[s]:
            print(f"  {s:<8} {pl_street[s]:+8.1f}")
    print("-- seat0 bet fractions (incremental/pot) by street --")
    for s in ("preflop", "flop", "turn", "river"):
        fr = own_bets.get(s)
        if not fr:
            continue
        fracs = [f for f, _ in fr]
        over = sum(1 for f in fracs if f > 0.8)
        print(f"  {s:<8} n={len(fracs):>4}  mean {sum(fracs)/len(fracs):.2f}  "
              f">0.8pot {over} ({100*over/len(fracs):.0f}%)")
    print("-- seat0 response when facing opponent bet/raise --")
    for k in sorted(facing):
        print(f"  {k:<10} {facing[k]:>4}  pl {facing_pl[k]:+.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchup", nargs="+", default=["self", "tag"])
    ap.add_argument("--hands", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mc-iters", type=int, default=10)
    args = ap.parse_args()
    for m in args.matchup:
        if m == "self":
            factories = [lambda rng: PokerBot(rng, mc_iters=args.mc_iters)] * 5
            label = "self-play (6x pokr)"
        elif m == "tag":
            factories = [lambda rng: TightAggressive(rng)] * 5
            label = "vs 5x TightAggressive"
        elif m == "maniac":
            factories = [lambda rng: Maniac(rng)] * 5
            label = "vs 5x Maniac"
        else:
            raise SystemExit(f"unknown matchup {m!r}")
        bot = PokerBot(random.Random(args.seed), mc_iters=args.mc_iters)
        _, results = play_session(bot, factories, args.hands, args.seed, num_seats=6)
        _analyze(results, label)


if __name__ == "__main__":
    main()
