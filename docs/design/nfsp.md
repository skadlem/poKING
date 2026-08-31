# NFSP in pokr — design note

**Date:** 2026-08-31
**Status:** Design accepted. Prerequisites 3.1 (betting history) and the Kuhn
gate (section 6) are implemented, tested, and 3.1 is now measured as a 2.9x
exploitability reduction. 3.2-3.4 are not done. No NFSP code exists yet.
Section 10 records what has been measured.
**Scope:** heads-up only. Everything below is a two-player zero-sum claim.

## 1. Why NFSP, and why here

The README's honest summary is that every bot in this repo is 670–1020 bb/100
exploitable — a best response wins ~7-10 big blinds per hand — and that
"approximating equilibrium needs Deep CFR or NFSP". This note is the plan for
the NFSP half.

Deep CFR is the better algorithm on the merits: it reaches lower exploitability
with roughly three orders of magnitude fewer episodes (37 vs 47 mbb/g on the
Deep CFR paper's benchmark). It is *not* the better algorithm for this
codebase, for one structural reason:

- **Deep CFR needs tree traversal.** It resets to a decision node and evaluates
  every action from there. `PokerGame.play_hand()` is a single straight-line
  call with no step interface, no state cloning, and no undo. The `deck=`
  injection parameter (`engine.py:71`, already used by `duplicate.py`) makes
  replay-with-a-scripted-prefix *possible*, but that is O(depth x branching)
  replays of a pure-Python engine per traversal, on top of refactoring the
  engine into a steppable or immutable-state form. The engine is the single
  source of rules in this project and it is deliberately callback-driven.
- **NFSP only ever plays hands front to back.** That is exactly what
  `bench.play_session` already does.

Roughly 60% of NFSP is already written here: the encode/decode layer and its
masked action space, the multiprocess rollout collector (with its fork, RNG and
thread-count lessons already paid for), the frozen-snapshot league, the
duplicate-deck evaluator, and — most importantly — `pokr/rl/exploit.py`, which
is the only metric that can say whether any of this worked.

Deep CFR stays the right long-term answer, and the engine surgery it needs is
worth doing eventually. NFSP is what can be built against the engine as it
stands.

## 2. The algorithm, as specified

From Heinrich & Silver 2016 (arXiv:1603.01121). Per player: two networks, two
memories.

| Component | Role |
| --- | --- |
| `Q(s,a)` | best response β, learned by DQN off-policy from `M_RL` |
| `Pi(s)` | average strategy, learned by *supervised* classification from `M_SL` |
| `M_RL` | circular buffer of `(s, a, r, s')` — **all** play goes here |
| `M_SL` | reservoir-sampled `(s, a)` — **only** actions taken while following β |
| `eta` | anticipatory parameter: each *episode*, play `eps-greedy(Q)` w.p. eta, else `Pi` |

Reservoir sampling is the load-bearing part. A uniform sample over all past
best-response experience *is* the average of past best responses, which is what
makes this fictitious play rather than self-play. **`Pi` is the output**, not
`Q`. Evaluating the Q-net is the classic implementation bug.

Paper hyperparameters, Limit Hold'em column (the one to port from):

- `eta = 0.1`; `eps` from 0.08, decayed to 0 proportional to inverse square root
- `M_RL` 600k circular; `M_SL` 30M exponentially-averaged reservoir, minimum
  replacement probability 0.25
- 4 hidden layers (1024, 512, 1024, 512), ReLU
- vanilla SGD, no momentum, lr 0.1 (RL) / 0.01 (SL)
- minibatch 256, **2 updates per 256 game steps**, target net refit every 1000
  updates
- ~10^8 game states visited

Leduc column, for the validation gate in section 6: single 64-unit hidden
layer, lr 0.1/0.005, minibatch 128, 2 updates per 128 steps, target refit every
300 updates, `eps` from 0.06, `M_RL` 200k circular / `M_SL` 2M reservoir,
reaching exploitability 0.06.

## 3. Five things in the current code that would break NFSP silently

These are prerequisites, not implementation details. Each one produces a
plausible-looking training curve and a meaningless result.

### 3.1 The observation is not an information state

`encode_obs` never touches `state.action_history`, which *is* carried on
`GameState` (`engine.py:47`), so this needs no engine change. The seat block
carries `folded / all_in / committed / street_committed / is_dealer /
acted_round` — no last action, no aggressor, no raise count.

Concretely: heads-up, "SB limped, BB raised to 6, SB called" and "SB raised to
6, BB called" produce **byte-identical flop observations**. Same pot, same
committed, `acted_round` reset by the street. Who was the preflop aggressor is
the most range-informative bit in poker and it is not encoded.

For PPO this is a leak. For NFSP it is fatal: the average strategy would be
averaging over aliased information states, and the resulting "equilibrium" is
an equilibrium of a game nobody is playing.

**Fix:** a betting-history block appended to the observation (section 5.1).
Worth measuring against the current PPO agent on its own, independently of
NFSP.

### 3.2 Opponent-model features make the strategy history-dependent across hands

`OBS_SLICES["opponent"]` feeds VPIP / PFR / aggression / fold-to-cbet into the
net. A Nash strategy is a function of the information state alone. Conditioning
on cross-hand opponent statistics means `Pi` is averaging over whatever
opponent-model distribution the training run happened to induce, and the
average strategy is not well defined.

**Fix:** `model_opponents=False` for NFSP. This gives up exploitation ability,
which is the entire point of the exercise.

### 3.3 Monte Carlo equity is stochastic, so one info state maps to many observations

`RLStrategy._equity` caches within a hand but reseeds across hands from
`self.rng`. The same information state yields a different observation each
time, and the average policy smears across the difference.

**Fix:** seed the equity RNG deterministically from
`hash((hole, board, n_opponents))`. This keeps the feature — which is
legitimately a function of the hero's own private and public information —
while making the encoding a true function of the info state. Dropping equity
entirely is the alternative, and it is the difference between a 110k-parameter
net at 10^6 hands and something much larger at 10^8.

### 3.4 Greedy deployment destroys the equilibrium

`plugin.py` sets `greedy=True`. The argmax of an approximate equilibrium is a
pure strategy and is maximally exploitable. NFSP must deploy by *sampling*
`Pi`. `exploit.py` already measures both deployments and keeps the better one
(lines 126-134), so the probe will not be fooled — but the plugin needs a
sampled path.

### 3.5 Two-player zero-sum only

NFSP's convergence guarantee does not extend to 6-max. The `--seats 2,6` mixing
that helped the PPO agent is wrong here. NFSP produces a heads-up checkpoint,
and the 6-max column is not a claim this method gets to make.

### Already in our favour

`PokerGame.play_hand` calls `on_hand_end(result, i)` for **every** seat
(`engine.py:136`), so both seats can record with correct per-seat rewards.
`play_session` alternates the button via `initial_dealer=h % num_seats`, so
there is no positional bias to correct for.

## 4. Two ladders

**Ladder A — faithful NFSP.** DQN best-response head plus reservoir-SL average
net, both seats learning continuously, `eta` sampled per hand. Correct, and it
is what the literature's numbers refer to. But DQN is a new learner in this
repo with its own target-network, replay and epsilon-schedule failure modes,
and it would be debugged against a game with no ground truth.

**Ladder B — fictitious self-play with a PPO best response.** The
best-response oracle already exists: `exploit.py:best_response` trains a fresh
PPO agent against a frozen target and reports what it wins. FSP is then

    loop: BR <- train_PPO_against(Pi);  Pi <- supervised fit on reservoir of BR data

No DQN, roughly 150 lines, reusing code that has already been debugged. It is
honestly a *policy-space* fictitious play, a close cousin of PSRO — and
`pokr/rl/league.py` is most of a PSRO population already. It is slower per unit
of compute than online NFSP, because each outer iteration retrains a best
response from scratch, but far cheaper to get correct.

**Decision: B first**, as a working and measurable baseline. A only if B's
exploitability curve flattens above the target. B also produces a number to
compare against early, which is how the rest of this repo was built.

## 5. Implementation map

New modules, in dependency order:

    pokr/rl/memory.py    ReservoirBuffer (+ exponentially-averaged variant), CircularBuffer
    pokr/rl/qnet.py      QNet + AvgPolicyNet  (do NOT touch PolicyValueNet; PPO depends on it)
    pokr/rl/nfsp.py      NFSPStrategy: the agent, the sigma coin flip, the two learners
    train_nfsp.py        outer loop

Changes to existing files, all small:

- **`encode.py`** — the betting-history block (5.1).
- **`agent.py`** — add `br_mode: bool` to `Episode`. Because sigma is chosen
  **per episode**, one flag per hand is enough to split `M_SL` from `M_RL`
  downstream. And because consecutive steps within an episode already give
  `s'`, the existing `Episode` shape (obs, masks, actions, terminal reward) is
  *already* a complete transition record — both memories are derived in the
  parent. This halves what crosses the process boundary versus storing explicit
  `next_obs`.
- **`rollout.py`** — `_collect_one` records seat 0 only. Heads-up NFSP has the
  same agent in both seats (position is in the observation, one shared pair of
  nets), so build the second agent outside and seat it with
  `lambda rng: agent1`, then return both buffers. Fork context, per-worker
  `torch.manual_seed` and `set_num_threads(1)` all carry over unchanged.
- **`plugin.py`** — register `"nfsp"`, sampled not greedy, so `bench`,
  `duplicate` and `exploit` accept it as a target with no further work.

### 5.1 The betting-history block

Sixteen dimensions appended to the end of the layout, so `obs[:OBS_DIM_V1]` is
byte-identical to the previous encoding and old checkpoints keep working:

| Field | Dims | Encoding |
| --- | --- | --- |
| preflop aggressor | 7 | one-hot, relative to hero; index 6 = limped pot |
| current-street aggressor | 7 | one-hot, relative to hero; index 6 = no bet yet |
| raises this street | 1 | `min(n, 4) / 4` |
| raises preflop | 1 | `min(n, 4) / 4` |

Relative indexing matches the existing seats block: index 0 is the hero, index
`j` is the player at `(player_id + j) % n`. Blinds are not in
`state.action_history` (`_post_blind` goes through `_chips_in`, which does not
append), so a preflop raise count is a true raise count.

Per-seat last-action one-hots were considered and dropped: given the aggressor
one-hot, `acted_round` and `street_committed` already separate check from call
from raise for every seat.

### 5.2 Batched versus online

Textbook NFSP is one online loop: act, store, two SGD steps per 256 steps. This
repo's infrastructure is batched and multiprocess. The reconciliation is
rounds: collect K hands with frozen nets across W workers, ship episodes back,
run the proportional number of SGD updates in the parent, push new weights.

The deviation is that the behaviour policy lags one round. At K = 2000 hands
(~20k steps, ~156 updates) that is mild, and it is what distributed NFSP
implementations do anyway. Recorded here so it is a known deviation rather than
a later discovery.

### 5.3 Memory budget

`M_SL` at the paper's 30M entries x 160-dim float32 is ~19 GB. Not available.
In order of preference:

1. Cap `M_SL` at 1-2M and store observations as float16: 320-640 MB.
2. Store cards as indices (2 hole + 5 board bytes) plus the ~56 non-card
   floats: ~120 B/entry, so 2M entries in 240 MB.

A smaller reservoir is a shorter-memory average, i.e. a worse approximation to
fictitious play. That is a real cost, not a free optimization, and it belongs
in whatever the run reports.

## 6. Validation gate: Kuhn poker before NLHE

**Do not point NFSP at pokr before it converges on a game with a known answer.**

Kuhn poker is three cards and about fifty lines. Exploitability is exact — with
six information sets per player there are only 2^6 = 64 pure strategies per
player, so a best response is a maximum over an enumeration, with no CFR
machinery and nothing to get subtly wrong. The known Nash family (a
one-parameter family in `alpha` in [0, 1/3], game value -1/18 to the first
player) gives a profile whose exploitability must come out at zero.

A test asserting that NFSP reaches exploitability < 0.05 on Kuhn within N
iterations is a genuine correctness test of the implementation, and it costs an
afternoon. Without it, hours of NLHE get read through a noisy PPO-based lower
bound in order to decide whether the reservoir sampler has an off-by-one.

Given that this project's headline finding is that its own exploitability proxy
under-reported by 70x, running an unvalidated equilibrium algorithm against an
approximate metric is precisely the trap already documented in the README.

Leduc is the second gate: tabular exploitability, and the paper reports 0.06,
so it confirms the *neural* path rather than just the algorithm.

## 7. Compute

Measured in this repo: 3,305 hands/s at 8 workers, ~5 decisions/hand heads-up,
so ~16k game states/s.

| Phase | Estimate |
| --- | --- |
| 10^8 states ~ 2x10^7 hands of rollout | ~1.7 h |
| ~780k minibatches x 2 nets at 1-2 ms | ~1-2 h |

An overnight run end to end, which is affordable. The caveat is that the
paper's 10^8 was for *limit* hold'em, a far smaller game than 100bb NLHE with
nine discretized actions.

Note the direction of the usual tradeoff is reversed here. The Deep CFR
literature observes that both methods are SGD-bound in wallclock, which softens
Deep CFR's sample-efficiency advantage. In this repo the environment is a
pure-Python engine and the network is 110k parameters, so **sampling
dominates** and sample efficiency matters more, not less. That is the strongest
argument for eventually doing the engine surgery and moving to Deep CFR.

## 8. What success looks like, stated before the run

The win condition is `python -m pokr.rl.exploit --target nfsp` dropping well
below the shipped agent's 670.6 bb/100.

It is **not** beating the PPO agent head-to-head. An equilibrium-approximating
agent should lose to a max-exploit agent against weak opposition, and should
win *less* against the calling station and the maniac. If NFSP arrives at, say,
250 bb/100 exploitable while winning less against the scripted pool, that is a
success — and the README should say so before the run rather than after.

Report `Pi` sampled, never `Q`, and never greedy.

## 9. Stop conditions

- **Kuhn does not converge.** The bug is in NFSP, not in poker. Do not proceed.
- **The probe on `Pi` reads worse than on the PPO agent after a full run.**
  Look at aliasing (3.1) and greedy deployment (3.4) before touching
  hyperparameters.
- **Exploitability plateaus above ~400 bb/100.** That is the action abstraction
  talking — nine discretized sizes — and more compute will not move it. The fix
  is the abstraction, or Deep CFR.

## 10. Measured so far

### 10.1 The betting-history block: loses on win rate, wins 2.9x on exploitability

Section 3.1 argued the history block on information-state grounds and noted it
was "worth measuring against the current PPO agent on its own". Measured on
both axes, it produces exactly the split this document predicted in section 8:
**it wins less and is much harder to beat.**

| arm | vs heuristic (bb/100) | exploitability (bb/100) |
| --- | --- | --- |
| no history (obs 160, shipped) | +639.51 +- 31.56 | 565.6 +- 60.4 |
| history (obs 176) | +461.76 +- 24.73 | **192.3 +- 59.1** |

The exploitability gap is 373.4 bb/100, or 8.8 SE -- decisively resolved, and a
2.94x reduction. At 192.3 the history arm is by a wide margin the least
exploitable agent measured in this project (the heuristic is 1019.7, the
shipped PPO agent 670.6).

Both probes trained 120 iterations (240k hands) at seed 7 and scored 5,000
duplicate decks at both deployments, so the comparison is like for like.

**Caveat that limits the absolute number, not the comparison.** Exploitability
here is a LOWER bound from a probe that tried a fixed amount, so a small number
is evidence of absence only in proportion to how hard the exploiter tried. 192.3
may partly mean "harder for a 120-iteration probe to crack" rather than "3x
closer to equilibrium". A longer probe would firm the absolute up; it would not
change the ranking, since both arms got the same budget.

Note also that this run reads the shipped agent at 565.6 where the README
reports 670.6 +- 68.8 -- a different probe seed and 5,000 decks rather than
4,000, about 1.5 combined SE apart. Not chased down.

#### The win-rate half, which lost



Both arms trained with the shipped checkpoint's exact config (600 iters x 2000
hands, seats 2,6, `--fast --reset-stacks`, seed 7, 8 workers), then scored on
20k duplicate decks heads-up against the heuristic at 150 MC iterations:

| arm | bb/100 vs heuristic |
| --- | --- |
| no history (obs 160, shipped checkpoint) | +639.51 +- 31.56 |
| history (obs 176) | +461.76 +- 24.73 |

-178 bb/100, resolved well outside the bars. One seed per arm, so one sample
rather than a measurement.

Read together with the exploitability column, this is not a regression: it is
the max-exploit / robustness tradeoff, and it is the trade this document exists
to make. An agent that gives up 178 bb/100 against a weak scripted opponent to
be 2.9x harder to counter is moving in the direction NFSP is aimed at.

The likely mechanism is that 16 extra input dimensions, with an unchanged
network, unchanged hyperparameters and the same 1.2M-hand budget, cost more in
credit assignment than the aliasing fix pays back. Note also that "beats the
heuristic harder" is the max-exploit metric, which is the opposite of what this
block is for.

**Kept, and now on evidence rather than principle.** The block was always a
correctness requirement for the method in this document -- an average over
aliased information states is an equilibrium of no game -- and it survives on
that ground alone. The exploitability column says it also pays for itself.
What should still not be claimed is that it makes PPO win more; it does not.

**Ruled out:** the history arm ends training at entropy 0.188 and PFR 0.0%,
which resembles the entropy collapse recorded in HANDOFF section 7c. It is not
one. The shipped arm ends at 0.220 / 1.7%, and in both logs the low-entropy
lines are the 6-max iterations while the heads-up lines sit at 0.6-0.7. It is a
table-size artifact of the per-iteration statistics.

**Still open:** one seed per arm on both axes. 2-3 seeds each is the only way
the -178 and the 373.4 become facts rather than anecdotes. Reproduce with:

    POKR_RL_CKPT=models/rl_history/ppo_final.pt python -m pokr.rl.exploit --target rl --iters 120

### 10.2 The Kuhn harness certifies itself

`pokr/rl/kuhn.py`, `tests/test_kuhn.py` (39 tests):

| profile | exploitability | value to player 0 |
| --- | --- | --- |
| `nash(alpha)`, alpha in {0, 1/6, 1/3} | < 1e-17 | -1/18 exactly |
| `uniform()` | 11/24 | +1/8 |

The episode generator is pinned against the analytic tree walk over 40k hands,
so a learner cannot train on one game and be scored on another.

## 11. References

- Heinrich & Silver, *Deep Reinforcement Learning from Self-Play in
  Imperfect-Information Games*, arXiv:1603.01121
- Brown et al., *Deep Counterfactual Regret Minimization*, arXiv:1811.00164
- Kovarik et al., *Revisiting Game Representations*, arXiv:2112.10890
