"""Compare pokr's bot against PyPokerEngine's official example bots.

Runs the fight inside PyPokerEngine's own engine (independent rules, dealer,
and side-pot logic), so neither bot has an engine advantage. `pokr` plays via
the PokrPlayer adapter; the opponents are the framework's own example players
from its repo (external/*.py).

Usage:
    python -m pokr.ppe_compare --hands 2000 --mc-iters 10
"""
from __future__ import annotations

import argparse
import sys

from pypokerengine.api.game import setup_config, start_poker

from .ppe import PokrPlayer


def _load_external(name):
    """Import an official example bot from external/."""
    import importlib.util
    path = f"external/{name}_player.py"
    spec = importlib.util.spec_from_file_location(name + "_player", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Pick the bot class defined in this module (not imported BasePokerPlayer)
    cls = [getattr(mod, n) for n in dir(mod)
           if isinstance(getattr(mod, n), type)
           and getattr(mod, n).__module__ == mod.__name__][0]
    return cls


def _session_result(config, target_rounds, initial_stack, bb):
    """Run one PyPokerEngine session, return (bb_by_name, hands_played).
    The engine stops early if a player busts."""
    result = start_poker(config, verbose=0)
    bb_by_name = {p["name"]: (p["stack"] - initial_stack) / bb for p in result["players"]}
    # rounds actually played is not exposed; approximate by max_round unless
    # someone busted (then the game ended early). We can't know exactly, so
    # we just count the configured rounds as played and rely on many small
    # sessions to amortize the truncation error.
    return bb_by_name, target_rounds


def run_heads_up(our_player, opponent, max_round, initial_stack=200, sb=1):
    """Heads-up with rebuy: PyPokerEngine ends the game at the first bust, so
    run many small sessions and reset stacks between them."""
    bb = sb * 2
    total_pokr = 0.0
    total_opp = 0.0
    remaining = max_round
    session = 200
    while remaining > 0:
        rounds = min(remaining, session)
        config = setup_config(max_round=rounds, initial_stack=initial_stack,
                              small_blind_amount=sb)
        config.register_player(name="pokr", algorithm=our_player)
        config.register_player(name=opponent.__class__.__name__, algorithm=opponent)
        result = start_poker(config, verbose=0)
        stacks = {p["name"]: p["stack"] for p in result["players"]}
        pokr_bb = (stacks["pokr"] - initial_stack) / bb
        opp_bb = (stacks[opponent.__class__.__name__] - initial_stack) / bb
        total_pokr += pokr_bb
        total_opp += opp_bb
        remaining -= rounds
    return total_pokr, total_opp, {"pokr": total_pokr, opponent.__class__.__name__: total_opp}


def run_6max(our_player, opponents, max_round, initial_stack=200, sb=1):
    """6-max with rebuy: short sessions, stack reset between them."""
    bb = sb * 2
    totals = {n: 0.0 for n in ["pokr"] + [f"{o.__class__.__name__}{i}"
                                          for i, o in enumerate(opponents)]}
    remaining = max_round
    session = 200
    while remaining > 0:
        rounds = min(remaining, session)
        config = setup_config(max_round=rounds, initial_stack=initial_stack,
                              small_blind_amount=sb)
        config.register_player(name="pokr", algorithm=our_player)
        for i, opp in enumerate(opponents):
            config.register_player(name=f"{opp.__class__.__name__}{i}", algorithm=opp)
        result = start_poker(config, verbose=0)
        for p in result["players"]:
            totals[p["name"]] += (p["stack"] - initial_stack) / bb
        remaining -= rounds
    return totals, {}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare pokr vs PyPokerEngine example bots")
    ap.add_argument("--hands", type=int, default=2000)
    ap.add_argument("--mc-iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    bots = {}
    for name in ("honest", "fish", "random"):
        bots[name] = _load_external(name)

    print(f"=== pokr vs PyPokerEngine example bots (in PyPokerEngine engine, "
          f"{args.hands} hands, {args.mc_iters} MC iters, seed {args.seed}) ===")
    print(f"{'opponent':<10}{'pokr bb':>10}{'opp bb':>10}{'pokr bb/100':>14}")
    for name, cls in bots.items():
        ours = PokrPlayer(rng_seed=args.seed, mc_iters=args.mc_iters)
        opp = cls()
        pokr_bb, opp_bb, _ = run_heads_up(ours, opp, args.hands)
        print(f"{name:<10}{pokr_bb:>+10.1f}{opp_bb:>+10.1f}{pokr_bb / args.hands * 100:>+14.1f}")

    print()
    print("=== 6-max: pokr + one of each external bot ===")
    ours = PokrPlayer(rng_seed=args.seed, mc_iters=args.mc_iters)
    opponents = [bots["honest"](), bots["fish"](), bots["random"](),
                 bots["fish"](), bots["random"]()]
    result_bb, _ = run_6max(ours, opponents, args.hands)
    for name, bb in result_bb.items():
        print(f"  {name:<14} {bb:>+10.1f} bb total  ({bb / args.hands * 100:>+8.1f} bb/100)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
