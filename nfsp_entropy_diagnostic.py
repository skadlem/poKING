"""nfsp_entropy_diagnostic.py — settle campaign #2's open question (step 10).

ANSWER (measured 2026-09-01, logs: models/nfsp_entropy_diag.log,
models/nfsp_probe_r0{10,20}{,_s11}.log): neither hypothesis. The three
numbers this printed killed the binary:

  1. The component BRs are SHARP: per-state entropy 0.257 (BR vs pi_r010)
     and 0.842 (BR vs pi_r030) vs the 1.587 coin-flip floor. The oracle is
     not the weak link in the data.
  2. Averaging IS entropy-generating: mean TV between the two components is
     0.58, and their 2-component mixture already sits at entropy 1.351 —
     near the floor. The DESIGN premise is real.
  3. But the decisive measurement was the per-round RE-PROBE (this script
     motivated it; exploit --target nfsp on the saved checkpoints, both
     seeds, all converged): exploitability FALLS r010 -> r020, then SPIKES
     ~3x at r030 (seed 7: 264.6 -> 204.4 -> 737.5; seed 11: 498.8 -> 468.3
     -> 1011.1). pi_last is not the best checkpoint — the late campaign
     made Pi WORSE. That is not "30 rounds is too few" (monotone in rounds,
     what DESIGN predicts) and not the fitter (a fresh 20-epoch fit on the
     2-comp slice beat pi_last by only 0.127 nats there, a confounded
     margin: pi_last legitimately fits a 30-component average scored on 2
     components).

The mechanism the campaign log already contained: from ~round 18 on, each
round's BR training ENDS below break-even (curve -315 -> -25 at r18, -447
-> -13 at r24, -262 -> -27 at r29). The harvest takes the TRAINED TAIL of a
runaway-worse "best response" — a diffuse mid-training policy, not a best
response — and linear-over-rounds weighting gives those poison rows the
HIGHEST weights in the reservoir (w=20..30). Oracle starvation: 40 PPO
iters stopped being enough to exploit Pi once the exploitable structure
moved to rare states, while the post-hoc probe gets 120 iters and finds
737.5 bb/100 of it. A converged 120-iter probe says a strong BR exists;
the in-loop 40-iter oracle fails to find it and its average pollutes Pi.

Consequences, in order: (a) pi_r020 at 204.4 +-59.9 (seed 7) is the least-bad
NFSP artifact on the probe axis (a lower bound; and it still LOSES duplicate
heads-up to the heuristic, -202.4 +-33.9, so not promotable — the seed-7
PPO-vs-Pi comparison stands as the verdict: NFSP failed step 10); (b) campaign
#3 must fix the ORACLE per round (more iters, early-stop on the curve, or
weight rows by BR final-curve strength clipped at 0 — never let a losing "BR"
hold the top weight), not add rounds; (c) more rounds WITHOUT that fix makes
it worse — that is the r020->r030 spike, measured, on two seeds.

Original method (kept for reproduction): rebuild reservoir structure from
the saved round checkpoints — train two fresh oracle BRs (one vs pi_r010,
one vs pi_r030) with the campaign's own harvest path (40 iters, 75%
burn-in, model_opponents=False) and measure on the union of rows, grouped
per info state (obs bytes; one info state -> one observation is the 3.3
determinism contract):

  H_beh(1)   entropy of the late component alone            (sharpness of a BR)
  H_beh(2)   entropy of the weight-combined mixture         (what averaging does)
  loss_pi30  CE(mixture, pi_last)                           (where Pi actually is)
  loss_fresh CE(mixture, fresh 20-epoch fit)                (what the fitter CAN reach)
  tv         mean total-variation distance between the two components'
             per-state action distributions                 (how much averaging spreads)

Caveat carried from step 10: `exploit` reports a LOWER bound, and probe
strength varies by seed (the seed-11 PPO exploiter reached only curve +143
and printed 176.3 — weaker evidence, not a better agent). Same-seed
comparisons only.
"""
from __future__ import annotations

import math
import pathlib
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pokr.rl.avg_policy import AvgPolicyNet, fit_avg_policy, load as load_pi
from pokr.rl.exploit import best_response
from pokr.rl.nfsp import NFSPStrategy

ITERS = 40
BURN = 30            # 0.75 * 40 — the campaign's own burn-in
HANDS = 2000
CAP = 300_000        # subsample cap for the fresh fit (campaign used 500k max)


def harvest_round(pi_path: str, seed: int) -> tuple[list, float]:
    """Replay one campaign round's harvest against a saved Pi. Returns rows
    [(obs_bytes, mask, action), ...] and the weight that round carried."""
    net, ck = load_pi(pi_path)
    player = NFSPStrategy(net=net, rng=random.Random(0), num_players=2,
                          model_opponents=False, record=False)
    rows: list = []

    def harvest(episodes, it, _w=ck.get("round", 0) + 1):
        if it < BURN:
            return
        for ep in episodes:
            for t in range(ep.obs.shape[0]):
                rows.append((ep.obs[t].copy(), ep.masks[t].copy(),
                             int(ep.actions[t])))

    best_response(lambda rg, _a=player: _a, f"Pi({pathlib.Path(pi_path).stem})",
                  ITERS, HANDS, seed, mc_iters=30, mc_fast=True,
                  eval_hands=0, harvest=harvest, model_opponents=False)
    return rows, float(ck.get("round", 0) + 1)


def dists_by_state(rows, weights):
    """Group rows per info state (obs bytes). Returns per-state dicts:
    counts[action] -> weight mass, legal actions from the mask, visit mass."""
    per = defaultdict(lambda: [np.zeros(9), 0.0])
    for (obs, mask, a), w in zip(rows, weights):
        key = obs.tobytes()
        per[key][0][a] += w
        per[key][1] += w
    return per


def H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main() -> int:
    torch.set_num_threads(8)
    print("harvesting component 1 (BR vs pi_r010, weight 11)...", flush=True)
    rows_a, w_a = harvest_round("models/nfsp/pi_r010.pt", seed=101)
    print(f"  {len(rows_a):,} rows  (w={w_a:g})", flush=True)
    print("harvesting component 2 (BR vs pi_r030, weight 30)...", flush=True)
    rows_b, w_b = harvest_round("models/nfsp/pi_r030.pt", seed=102)
    print(f"  {len(rows_b):,} rows  (w={w_b:g})", flush=True)

    # per-state conditional action distributions for each component, on the
    # state keys BOTH sides visited (the shared support; the campaign's
    # reservoir mixed all 30, this is the two-point proxy)
    pa = dists_by_state(rows_a, [w_a] * len(rows_a))
    pb = dists_by_state(rows_b, [w_b] * len(rows_b))
    shared = [k for k in pa if k in pb and pa[k][1] >= 20 and pb[k][1] >= 20]
    print(f"states: {len(pa):,} / {len(pb):,}, shared (>=20 visits both): "
          f"{len(shared):,}", flush=True)

    h_a = h_b = h_mix = unif = tv = mass = 0.0
    mix_w = []
    for k in shared:
        da, ma = pa[k]
        db, mb = pb[k]
        mask = (da + db) > 0                      # legal set (both share state)
        qa = da / ma
        qb = db / mb
        mix = (qa * ma + qb * mb) / (ma + mb)
        obs = np.frombuffer(k, dtype=np.float32)
        mix_w.append((obs, mask.astype(bool), mix, ma + mb))
        h_a += H(qa[mask]) * ma
        h_b += H(qb[mask]) * mb
        h_mix += H(mix[mask]) * (ma + mb)
        unif += math.log(int(mask.sum())) * (ma + mb)
        tv += 0.5 * np.abs(qa - qb)[mask].sum() * (ma + mb)
        mass += ma + mb

    print(f"\nH_beh(BR@r010) = {h_a / mass:.4f}   H_beh(BR@r030) = "
          f"{h_b / mass:.4f}   H_mix(2-comp) = {h_mix / mass:.4f}")
    print(f"E[log k] (coin-flip floor)      = {unif / mass:.4f}")
    print(f"mean TV between components      = {tv / mass:.4f}", flush=True)

    # Pi_last's CE on the mixture rows (the quantity the verdict turns on)
    net, _ = load_pi("models/nfsp/pi_last.pt")
    obs_mat = np.stack([w[0] for w in mix_w]).astype(np.float32)
    mask_mat = np.stack([w[1] for w in mix_w])
    probs_pi = net.probs(obs_mat, mask_mat)
    loss_pi = h_loss_fresh = None
    ce_terms = []
    for i, (_o, m, mix, v) in enumerate(mix_w):
        p = probs_pi[i][m]
        q = mix[m]
        ce_terms.append(float((q * -np.log(np.clip(p, 1e-9, None))).sum()) * v)
    loss_pi = sum(ce_terms) / mass
    ent_pi = sum(H(probs_pi[i][mix_w[i][1]]) * mix_w[i][3]
                 for i in range(len(mix_w))) / mass
    print(f"\nloss_pi30  CE(mix, pi_last)     = {loss_pi:.4f}")
    print(f"E[H(pi_last(s))]                = {ent_pi:.4f}")

    # fresh 20-epoch fit on the same mixture, weighted by visit mass
    rng = np.random.default_rng(7)
    psample = np.array([w[3] for w in mix_w])
    psample = psample / psample.sum()
    idx = rng.choice(len(mix_w), size=min(CAP, len(mix_w)), replace=True,
                     p=psample)
    obs_t = obs_mat[idx]
    mask_t = mask_mat[idx]
    acts_t = np.array([int(np.argmax(mix_w[i][2])) for i in idx])
    # sample target actions from the mixture, not argmax (CE of a distribution)
    acts_t = np.array([rng.choice(9, p=mix_w[i][2] / mix_w[i][2].sum())
                       for i in idx])
    assert mask_t[np.arange(len(idx)), acts_t].all()
    fresh = AvgPolicyNet(hidden=(256, 256))
    losses = fit_avg_policy(fresh, obs_t, mask_t, acts_t,
                            epochs=20, batch_size=1024, lr=1e-3,
                            generator=torch.Generator().manual_seed(7))
    probs_f = fresh.probs(obs_mat, mask_mat)
    loss_fresh = sum(
        float((mix_w[i][2][mix_w[i][1]] *
               -np.log(np.clip(probs_f[i][mix_w[i][1]], 1e-9, None))).sum())
        * mix_w[i][3] for i in range(len(mix_w))) / mass
    print(f"fresh-fit final epoch loss      = {losses[-1]:.4f}")
    print(f"loss_fresh CE(mix, fresh fit)   = {loss_fresh:.4f}")

    gap = loss_pi - loss_fresh
    verdict = ("UNDERFIT" if gap > 0.05 else
               "DESIGN (Pi already at the mixture floor)"
               if abs(gap) <= 0.05 else "Pi BELOW the fresh fit?? recheck")
    print(f"\ngap loss_pi30 - loss_fresh      = {gap:+.4f}  -> {verdict}")
    print(f"H_mix(2-comp) vs loss_pi30      = {h_mix / mass:.4f} vs "
          f"{loss_pi:.4f}  (a converged fit of the FULL 30-comp mixture "
          "should sit at its own H, which is >= this 2-comp proxy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
