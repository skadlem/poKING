"""train_nfsp.py: ladder B, the outer loop (roadmap step 9). HEADS-UP ONLY.

Design note 4's decision, verbatim shape:

    loop: BR <- train_PPO_against(Pi);  Pi <- supervised fit on reservoir of BR data

The BR oracle is `exploit.py:best_response` — code already debugged, which
was the entire reason ladder B was chosen over DQN. Its per-iteration
episodes are harvested (br_mode=True) into Pi's reservoir; between rounds
Pi is refitted by cross-entropy on that reservoir (`NFSPStrategy.fit`,
which uses BR rows exclusively once any exist — design note 2).

Heads-up only, no flag: NFSP's convergence guarantee is 2p0s (design note
3.5), so a 6-max checkpoint would be a claim this method does not get to
make. The one NFSP instance sits BOTH seats — position is part of the
observation and the pair of nets is shared — via the per-seat recording
NFSPStrategy does natively; step 8's "both seats recording" is exactly
that, not a rollout.py change (ladder B's BR data arrives through
exploit.best_response's harvest hook, so _collect_one is not on this path).

What success is NOT (stated up front, design note 8): beating the shipped
PPO agent head-to-head. An equilibrium approximator should win LESS
against weak opposition than a max-exploit agent. The success metric is
step 10: `python -m pokr.rl.exploit --target nfsp` landing well below the
shipped agent's 670.6 bb/100, with `ExploitReport.converged` true on that
probe. The per-round BR column is the progress signal: as Pi improves, the
best-response oracle's own win rate against it should FALL — a BR that
gets better forever means Pi is not tracking it.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import time
from dataclasses import dataclass

import torch

from pokr.rl.avg_policy import AvgPolicyNet, save as save_pi
from pokr.rl.exploit import best_response
from pokr.rl.memory import ReservoirBuffer
from pokr.rl.nfsp import NFSPConfig, NFSPStrategy
from pokr.rl.ppo import PPOConfig


@dataclass
class RoundStat:
    round: int
    br_bb100: float          # mean of the BR's last 10 training curves
    br_first_bb100: float    # its first curve: how far it had to come up
    rows: int                # reservoir size after this round
    seen: int                # total rows ever harvested
    fit_loss: float


def train(
    rounds: int = 30,
    iters_per_round: int = 40,
    hands_per_iter: int = 2000,
    capacity: int = 2_000_000,
    fit_epochs: int = 20,
    fit_batch: int = 1024,
    lr: float = 1e-3,
    max_fit_rows: int = 500_000,
    hidden: tuple[int, ...] = (256, 256),
    ckpt_dir: str = "models/nfsp",
    seed: int = 7,
    quiet: bool = False,
    br_cfg: PPOConfig | None = None,
) -> list[RoundStat]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = AvgPolicyNet(hidden=hidden)
    buffer = ReservoirBuffer(capacity, random.Random(seed + 1))
    cfg = NFSPConfig(capacity=capacity, fit_every=0,     # fits are manual here
                     epochs=fit_epochs, batch_size=fit_batch, lr=lr,
                     max_fit_rows=max_fit_rows,
                     hidden=hidden)
    # ONE instance, both seats: the shared net is the point of ladder B.
    # record=False: in this loop every row enters via record_episode from
    # the harvested BRs. Pi's own decisions would land as br_mode=False
    # rows — the exact self-imitation data select_fit_rows excludes — and
    # dilute M_SL's effective capacity ~2x for nothing. The bootstrap path
    # stays for manual/epsilon use where no BR data exists.
    player = NFSPStrategy(net=net, config=cfg, rng=rng, num_players=2,
                          model_opponents=False, record=False, buffer=buffer)
    pdir = pathlib.Path(ckpt_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    stats: list[RoundStat] = []
    t0 = time.time()
    for r in range(rounds):
        def harvest(episodes, _p=player):
            for ep in episodes:
                _p.record_episode(ep)                   # br_mode=True rows

        # eval_hands=0: the expensive duplicate eval is step 10's job once,
        # not every round; the BR's TRAINING curve is this loop's signal.
        report = best_response(
            lambda rg, _a=player: _a,                   # both seats: same agent
            "PokrNFSP-Pi", iters_per_round, hands_per_iter,
            seed + r * 7919, mc_iters=30, mc_fast=True,
            eval_hands=0, cfg=br_cfg, harvest=harvest, model_opponents=False)
        loss = player.fit()
        curve = report.curve
        st = RoundStat(r, sum(curve[-10:]) / len(curve[-10:]), curve[0],
                       len(buffer), buffer.seen, loss)
        stats.append(st)
        if not quiet:
            print(f"round {r:>3} | BR curve {st.br_first_bb100:+8.1f} -> "
                  f"{st.br_bb100:+8.1f} bb/100 | rows {st.rows:>9,}/{st.seen:,}"
                  f" | fit loss {st.fit_loss:.4f} | {time.time() - t0:6.0f}s")

        save_pi(net, str(pdir / "pi_last.pt"), round=r, config={
            "rounds": rounds, "capacity": capacity, "seed": seed,
            "fit_epochs": fit_epochs, "fit_batch": fit_batch, "lr": lr},
            reservoir_rows=len(buffer), reservoir_seen=buffer.seen)
        if (r + 1) % 10 == 0:
            save_pi(net, str(pdir / f"pi_r{r + 1:03d}.pt"), round=r + 1)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ladder-B NFSP: PPO best responses in, average policy out "
                    "(heads-up only)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--iters", type=int, default=40, help="PPO iters per round")
    ap.add_argument("--hands-per-iter", type=int, default=2000)
    ap.add_argument("--capacity", type=int, default=2_000_000)
    ap.add_argument("--max-fit-rows", type=int, default=500_000,
                    help="uniform subsample cap per supervised fit "
                         "(bounds memory; 0 disables)")
    ap.add_argument("--fit-epochs", type=int, default=20)
    ap.add_argument("--fit-batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-dir", default="models/nfsp")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    train(rounds=args.rounds, iters_per_round=args.iters,
          hands_per_iter=args.hands_per_iter, capacity=args.capacity,
          fit_epochs=args.fit_epochs, fit_batch=args.fit_batch, lr=args.lr,
          max_fit_rows=args.max_fit_rows,
          ckpt_dir=args.ckpt_dir, seed=args.seed, quiet=args.quiet)
    print("next: python -m pokr.rl.exploit --target nfsp   (step 10; check "
          "'converged' before quoting the number)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
