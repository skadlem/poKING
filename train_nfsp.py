"""train_nfsp.py: ladder B, the outer loop (roadmap step 9). HEADS-UP ONLY.

Design note 4's decision, verbatim shape:

    loop: BR <- train_PPO_against(Pi);  Pi <- supervised fit on reservoir of BR data

The oracle is `exploit.py:best_response` — code already debugged, which
was the entire reason ladder B was chosen over DQN. Its episodes are
harvested into Pi's reservoir and between rounds Pi is refitted by
cross-entropy on it (`NFSPStrategy.fit`, which uses BR rows exclusively
once any exist — design note 2).

What gets harvested, and at what weight, is the algorithm — campaign #1
got both wrong and measured a Pi ~1000 bb/100 exploitable (worse than
the shipped PPO's 670.6) whose CE fit loss of 1.64 sat next to 1.71 for
coin-flipping the legal mask: it had averaged itself into noise. Two
mechanisms, specific to ladder B:

1. best_response RESTARTS the BR from random init every round. A uniform
   reservoir folds every round's near-random opening iterations into the
   average forever. Fix: harvest only the trained tail of each round
   (--burn-in, default 0.75 — the first 75% of each round's iterations
   are discarded, not down-weighted).
2. The fictitious-play sequence is the LIST OF ROUNDS (each round's
   trained BR is one move), not the 1200-iteration concatenation — so
   the weight is `round + 1`, linear over rounds (the CFR averaging the
   Kuhn gate validated, applied at the right level). Global-iteration
   weighting would be wrong here: it would make each round's random-
   init phase weigh as much as its trained phase, since a round spans
   only 1/30th of the iteration range.

3. (Campaign #3, after the oracle-starvation diagnosis — HANDOFF 0.6
   step 10): campaign #2's Pi got WORSE with rounds (r020 probe 204.4,
   r030 probe 737.5, both converged, both seeds). Cause: from ~round 18
   the in-loop BR's own curve ENDED below break-even — 40 PPO iters from
   a random restart stopped being enough to exploit Pi — and those
   diffuse losing "best responses" entered the fictitious average at the
   reservoir's HIGHEST weights. A policy that loses to Pi is not a best
   response; it is a non-move, and the fictitious-play sequence may not
   contain it. Fix: `round_weight` — a round harvests only if its BR's
   final curve is positive, and --iters defaults to 80 because the
   starvation was in the oracle's training budget, not in the weighting
   alone.

The reservoir is WeightedReservoir (A-Res, fsp.py) — proportional
inclusion needs the stream to OVERFLOW capacity; tests assert it does.

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
from pokr.rl.fsp import WeightedReservoir
from pokr.rl.nfsp import NFSPConfig, NFSPStrategy
from pokr.rl.ppo import PPOConfig


@dataclass
class RoundStat:
    round: int
    br_bb100: float          # mean of the BR's last 10 training curves
    br_first_bb100: float    # its first curve: how far it had to come up
    rows: int                # reservoir size after this round
    seen: int                # total WEIGHT mass ever harvested (A-Res)
    fit_loss: float
    skipped: bool = False    # gate: br_bb100 <= 0 -> zero rows harvested


def round_weight(br_tail_bb100: float, r: int, gate: bool = True,
                 margin: float = 0.0) -> float:
    """The round's fictitious-play weight (campaign #3 gate + #4 margin).

    A policy that loses to Pi is not a best response — the oracle
    was starved (HANDOFF 0.6 step 10: campaign #2's losing rounds 18-29
    entered the average at the HIGHEST linear weights and r030 measured
    3x worse than r020). Those rounds are non-moves: weight 0, which
    WeightedReservoir skips outright (documented, never a silent
    down-weight). Positive rounds keep the Kuhn-validated linear
    weight = round + 1 — continuous re-scaling by curve size was
    considered and rejected: the gate validated LINEAR-over-rounds
    averaging, and the fictitious-play sequence is a sequence of MOVES,
    not of magnitudes.

    `margin` (campaign #4): the matched-budget re-probe showed campaign
    #3's late rounds still degraded Pi (r020 375.2 -> pi_last 673.9 at
    240-iter probes) because BRs whose tails barely cleared zero (+3..
    +52) are diffuse mid-training policies — a win that small is not a
    best response either. Rounds with tail <= margin contribute nothing.

    Round 0 is exempt: an empty reservoir cannot be fitted (fit() raises
    on it by contract), and a round-0 BR failing to beat a random-init Pi
    is a broken campaign, not a policy to protect the average from."""
    if r == 0:
        return 1.0
    if gate and br_tail_bb100 <= margin:
        return 0.0
    return float(r + 1)


def train(
    rounds: int = 30,
    iters_per_round: int = 80,
    hands_per_iter: int = 2000,
    capacity: int = 250_000,
    burn_in: float = 0.75,
    fit_epochs: int = 20,
    fit_batch: int = 1024,
    lr: float = 1e-3,
    max_fit_rows: int = 500_000,
    hidden: tuple[int, ...] = (256, 256),
    ckpt_dir: str = "models/nfsp",
    seed: int = 7,
    quiet: bool = False,
    gate: bool = True,
    margin: float = 0.0,
    patience: int = 0,
    br_cfg: PPOConfig | None = None,
) -> list[RoundStat]:
    if not 0.0 <= burn_in < 1.0:
        raise ValueError(f"burn_in must be in [0, 1), got {burn_in}")
    burn_iters = min(int(iters_per_round * burn_in), iters_per_round - 1)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = AvgPolicyNet(hidden=hidden)
    buffer = WeightedReservoir(capacity, random.Random(seed + 1))
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
    streak = 0                       # consecutive skips (for --patience)
    for r in range(rounds):
        # The gate reads the round's FINAL curve, which does not exist until
        # best_response returns — so the burn-in-passed episodes are held
        # for the round (~55k rows, tens of MB) and enter the reservoir in
        # one weighted flush, or not at all.
        pending: list = []

        def harvest(episodes, it, _p=pending):
            # Burn-in: this round's BR starts from RANDOM net (the oracle
            # restarts per round), so iterations below burn_iters have not
            # best-responded to anything yet -- discarded, not down-weighted.
            if it < burn_iters:
                return
            _p.extend(episodes)

        # eval_hands=0: the expensive duplicate eval is step 10's job once,
        # not every round; the BR's TRAINING curve is this loop's signal.
        report = best_response(
            lambda rg, _a=player: _a,                   # both seats: same agent
            "PokrNFSP-Pi", iters_per_round, hands_per_iter,
            seed + r * 7919, mc_iters=30, mc_fast=True,
            eval_hands=0, cfg=br_cfg, harvest=harvest, model_opponents=False)
        curve = report.curve
        tail = sum(curve[-10:]) / len(curve[-10:])
        w = round_weight(tail, r, gate=gate, margin=margin)
        if w > 0:
            for ep in pending:
                player.record_episode(ep, weight=w)    # br_mode=True rows
        loss = player.fit()
        st = RoundStat(r, tail, curve[0],
                       len(buffer), int(buffer.weight_sum), loss,
                       skipped=(w <= 0.0))
        stats.append(st)
        if not quiet:
            print(f"round {r:>3} | BR curve {st.br_first_bb100:+8.1f} -> "
                  f"{st.br_bb100:+8.1f} bb/100 | rows {st.rows:>9,}"
                  f" | fit loss {st.fit_loss:.4f}"
                  f" | {'SKIP' if st.skipped else f'w={w:g}'}"
                  f" | {time.time() - t0:6.0f}s")
        # Patience (campaign #4): after the margin, a long skip streak means
        # the oracle can no longer produce moves worth averaging — every
        # further round pays full BR cost to harvest nothing. Stopping is
        # honest; the best checkpoint was already saved at the last w>0.
        if patience > 0 and st.skipped:
            streak += 1
            if streak >= patience:
                if not quiet:
                    print(f"early stop: {streak} consecutive skips "
                          f"(--patience {patience}) at round {r}")
                break
        else:
            streak = 0

        save_pi(net, str(pdir / "pi_last.pt"), round=r, config={
            "rounds": rounds, "capacity": capacity, "seed": seed,
            "fit_epochs": fit_epochs, "fit_batch": fit_batch, "lr": lr,
            "iters_per_round": iters_per_round, "burn_in": burn_in,
            "gate": gate, "margin": margin},
            reservoir_rows=len(buffer), reservoir_weight_sum=buffer.weight_sum)
        if (r + 1) % 10 == 0:
            save_pi(net, str(pdir / f"pi_r{r + 1:03d}.pt"), round=r + 1)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ladder-B NFSP: PPO best responses in, average policy out "
                    "(heads-up only)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--iters", type=int, default=80,
                    help="PPO iters per round (campaign #1/#2 used 40; the "
                         "oracle-starvation diagnosis says that budget went "
                         "below what Pi needs to be exploited -> losing "
                         "'BRs' poisoned the average)")
    ap.add_argument("--hands-per-iter", type=int, default=2000)
    ap.add_argument("--capacity", type=int, default=250_000,
                    help="M_SL slots (WeightedReservoir; the stream must "
                         "overflow this or weighting is inert)")
    ap.add_argument("--burn-in", type=float, default=0.75,
                    help="fraction of each round's BR iterations discarded "
                         "before harvesting (the oracle restarts random)")
    ap.add_argument("--max-fit-rows", type=int, default=500_000,
                    help="uniform subsample cap per supervised fit "
                         "(bounds memory; 0 disables)")
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="campaign-#1/#2 behaviour: harvest every round, "
                         "losing BRs included (reproduction flag only — the "
                         "oracle-starvation result is why this is not the "
                         "default)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="skip rounds whose BR tail is <= this, not just "
                         "<= 0 (campaign #4: barely-winning BRs are still "
                         "diffuse mid-training policies; 100 is the tested "
                         "value)")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many consecutive skips (0 = run "
                         "all rounds; a skip streak means the oracle can no "
                         "longer produce moves worth averaging)")
    ap.add_argument("--fit-epochs", type=int, default=20)
    ap.add_argument("--fit-batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-dir", default="models/nfsp")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    train(rounds=args.rounds, iters_per_round=args.iters,
          hands_per_iter=args.hands_per_iter, capacity=args.capacity,
          burn_in=args.burn_in, gate=args.gate, margin=args.margin,
          patience=args.patience,
          fit_epochs=args.fit_epochs, fit_batch=args.fit_batch, lr=args.lr,
          max_fit_rows=args.max_fit_rows,
          ckpt_dir=args.ckpt_dir, seed=args.seed, quiet=args.quiet)
    print("next: python -m pokr.rl.exploit --target nfsp   (step 10; check "
          "'converged' before quoting the number)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
