# Handoff — pokr Poker Bot

**Repo:** https://github.com/skadlem/poKING (public)
**Latest session:** 2026-08-31 (section 0 below). Sections 1-8 are the
2026-08-08 handoff and are still accurate except where section 0 says
otherwise.

---

## 0. Session 2026-08-31 — NFSP prerequisites

**Branch:** `feat/nfsp-prereqs`, pushed to origin, working tree clean, no PR
opened. Branched off `main` at `170c46c`.

    d9516aa fix(exploit): a probe that never cleared break-even bounds nothing
    fb3799e docs: the betting-history block is 2.9x less exploitable   <- HEADLINE RETRACTED by d9516aa
    5d19efc docs: record the encoding A/B and the road to NFSP
    d78a8af feat(rl): Kuhn poker harness with exact exploitability
    5917f88 feat(rl): encode betting history, so the observation is an information state
    3053711 docs: design note for NFSP, and why it fits this engine

`fb3799e`'s message claims a 2.9x exploitability win from one seed. It does not
replicate; `d9516aa` retracts it. Both are kept in history on purpose. **Read
section 0.4 before quoting any number from `fb3799e`.**

Full suite: **296 passed** (was 250). ~1 min idle, ~3.5 min if a training or
duplicate run is competing for CPU. The venv is `.venv/` and is NOT on the
default `python3` — use `.venv/bin/python`.

**Nothing was promoted and the README is untouched** (section 0.5).

### 0.1 What this session was

Research + design for NFSP, then the two prerequisites that do not depend on
any NFSP code existing. **The design note is `docs/design/nfsp.md` and it is
the document to read first** — this section is only what happened, not why.

Headline of the design note: NFSP is the right *next* algorithm here not
because it beats Deep CFR (it does not — lower sample efficiency, higher
exploitability in the literature) but because **it needs no engine changes**.
It only plays hands front to back, which `bench.play_session` already does.
Deep CFR needs tree traversal and `PokerGame.play_hand` is a straight-line
call with no step interface, no cloning and no undo.

### 0.2 Betting-history encoding (`5917f88`)

`encode_obs` never touched `state.action_history`, so the observation was a
snapshot of chip positions only. Two different preflop lines that arrive at the
same chip position encoded **byte-identically** — heads-up, "SB limps, BB
raises to 6, SB calls" versus "SB raises to 6, BB calls". Who raised preflop
was absent. `test_two_preflop_lines_alias_in_the_old_layout_and_separate_in_the_new`
asserts the aliasing rather than describing it.

Added 16 dims: preflop aggressor + current-street aggressor (one-hot over
relative seat, last slot = nobody), plus street and preflop raise counts capped
at 4. OBS_DIM 160 -> 176.

**The layout contract, which matters for everything downstream:** the block is
appended LAST, so `obs[:OBS_DIM_V1]` is byte-identical to the old layout and
`models/rl/ppo_final.pt` (and every README number) still runs. `RLStrategy`
infers the layout from `net.obs_dim` rather than a flag — the layout is part of
a checkpoint's contract and obs_dim already records it. An obs_dim matching
neither layout raises at construction. **Do not insert new observation fields
anywhere but the end**, or pre-history checkpoints die.

`train_rl.py --no-history` trains the old layout; the banner prints which.

### 0.3 Kuhn harness (`d78a8af`)

`pokr/rl/kuhn.py` — the gate an equilibrium algorithm passes before it sees
NLHE. Best response is an exhaustive max over 2**6 pure strategies per player,
so `exploitability()` is **exact**: not a bound, not an estimate, nothing to
converge. Self-validating — the Nash family is at machine epsilon for every
alpha in [0, 1/3], game value -1/18, uniform play 11/24 exploitable.

No torch, no numpy, no rng in the pure functions.

Rationale (design note section 6): `pokr/rl/exploit.py` grades a bot by
training a PPO best response, which is a lower bound off a noisy run. Fine for
ranking bots, useless for deciding whether a reservoir sampler has an
off-by-one. This project's headline finding is that its own exploitability
proxy under-reported by ~70x; running an unvalidated equilibrium algorithm
against an approximate metric is that same trap.

### 0.4 A/B of the encoding — three seeds, and a retracted headline

Seeds 7, 11, 23 x two arms. Each arm: 600 iters x 2000 hands, seats 2,6,
`--fast --reset-stacks`, 8 workers (~34 min); scored on 20k duplicate decks vs
the heuristic and by a best-response probe (120 iters, probe seed fixed at 7,
5,000 decks). Checkpoints in `models/ab/s{seed}_{hist,nohist}/` (gitignored).

**Win rate vs the heuristic (bb/100)** — all three seeds usable:

| seed | history | no history | diff |
|---|---|---|---|
| 7 | +461.76 | +639.51 | −177.75 |
| 11 | +514.46 | +581.80 | −67.34 |
| 23 | +297.64 | +658.72 | −361.08 |
| **mean** | **+424.62** | **+626.68** | **−202.06** |

Settled: the block costs ~200 bb/100 against the heuristic on every seed. It
also destabilises training — between-seed spread 217 for history vs 77 without.

**Exploitability (bb/100)** — only 2 of 3 seeds produced a usable probe:

| seed | history | no history | diff |
|---|---|---|---|
| 7 | 192.3 | 565.6 | −373.3 |
| 11 | 553.2 | 609.7 | −56.5 |
| 23 | *probe failed* | *probe failed* | — |

**RETRACTED: the "2.9x less exploitable" headline in commit `fb3799e` does not
replicate.** Seed 11 gives a 56.5 gap, inside a single probe's ±60 bar. Both
usable seeds favour history directionally, but at 373 and 57 from n=2 the
magnitude is unestablished, and seed 7's 192.3 is the outlier rather than the
effect. Commit `fb3799e`'s message is wrong on this point and is corrected by
the commit that added this section — the git history keeps both, deliberately.

**Where that leaves the block.** It is kept on the section 3.1 correctness
argument alone (an average over aliased information states is an equilibrium of
no game), which does not depend on either measurement. It has NOT earned a
performance claim. Do not quote 192.3 or 2.9x anywhere.

### 0.5 The probe fails ~1/3 of the time and says "unexploitable"

Both seed-23 probes finished with the exploiter below break-even:

    seed 23, history:    curve -341 -> +27,  eval  +14.95 ± 7.15
    seed 23, no history: curve -410 -> -91,  eval  -61.67 ± 41.91

`ExploitReport.exploitability` clamps negatives to zero, so these printed as
`exploitability lower bound: 14.9` and `0.0` — which read as *the least
exploitable agents ever measured in this project*, against targets whose
sibling seeds measured 553.2 and 609.7. Two of six probes failed this way.

Same failure shape as the leak hunter under-reporting by 70x, one layer up: a
broken instrument returning a small number that looks like good news.

**Fixed:** `ExploitReport` now has `converged`, and `format()` prints
`PROBE FAILED ... bounds NOTHING` rather than a lower bound when the exploiter
never cleared break-even. Tests in `tests/test_exploit.py`. **Check `converged`
before believing any exploitability number, including future NFSP ones.**

**Not promoted, and README not rewritten — an explicit decision, 2026-08-31.**
Promotion was raised and declined after the three-seed table came in.
`plugin.py` still loads `models/rl/ppo_final.pt` (the obs-160, seed-7,
no-history agent) and the README's Exploitability section is unchanged, so
every published number still describes the agent it was measured on.

If promotion is ever revisited, the selection rule that avoids re-selecting on
the noise this session already retracted: **require a converged probe, then
take the best duplicate win rate.** That picks `models/ab/s11_hist`
(+514.46 vs the heuristic, 553.2 exploitable) — NOT `models/rl_history`, whose
192.3 is the outlier. Note that any such pick is a max over three seeds and so
is itself optimistically biased. And re-open only with a probe budget large
enough that failures are rare (more `--iters`) and more than three seeds.

Promotion would also make the README's headline numbers **worse**, not better:
the history arm wins less against every scripted opponent. That is the expected
direction (section 8 of the design note) and not a reason on its own to avoid
it — but it should be a deliberate choice, not a surprise.

### 0.6 Roadmap to NFSP implemented

Strict order. Nothing below step 6 starts until step 6 is green.

| # | Work | Size |
|---|---|---|
| 1 | ~~Deterministic equity~~ **DONE `44238bf`**: `RLStrategy._equity` seeds `random.Random(hash(key))` per info state; key is ints-only (asserted stable across worker processes). Re-measured the shipped agent on this code: duplicate +633.8 ±31.7 (published +604 ±36) and probe 668.7 ±56.4 (published 670.6) — inside the old bars, no re-anchor needed. Suite 296 -> 300. | ~10 lines + test |
| 2 | Register `"nfsp"` in `plugin.py` with `greedy=False` — the argmax of an approximate equilibrium is maximally exploitable (3.4). Needs `NFSPStrategy` (step 7) to exist; do it there, not as a stub | ~15 lines |
| 3 | Opponent-model features off (3.2) | already there: `model_opponents=False` zeroes the block |
| 4 | ~~`pokr/rl/memory.py`~~ **DONE `06f753b`**: `ReservoirBuffer` (Algorithm R) + `ExponentialReservoirBuffer` (0.25 floor), torch/numpy-free. Uniformity asserted via chi-square (calibrated null ~174±19 over 40 seeds; a 5-sigma per-position band was tried, and produced a false failure at position 50 on the unbiased sampler — do not reintroduce it). Exact inclusion product for the floored variant is in the test and brute-force verified | small |
| 5 | ~~`AvgPolicyNet`~~ **DONE `1e61456`**: `pokr/rl/avg_policy.py` (not `qnet.py` — ladder B has no QNet). Pi-only net + `fit_avg_policy` masked CE, `save`/`load` with the same reserved-key contract as `net.py`. Illegal slots verified to have exactly 0.0 probability AND 0.0 gradient; `act()` deliberately has no greedy flag (3.4). CE-recover-frequencies property is asserted, which is what makes Pi the fictitious average | small |
| 6 | **GATE: GREEN** — ~~FSP on Kuhn~~ **DONE**: `pokr/rl/fsp.py` + `tests/test_fsp_kuhn.py`. `tabular_cfr` (ground truth) and `neural_avg_cfr` (the step 4+5 pipeline: `WeightedReservoir` A-Res harvest -> `AvgPolicyNet` CE fit) both asserted on exact `kuhn.exploitability()`; 1000 iters ~4 s, net 0.009-0.014 on five seeds, bar 0.05. Statistical bars calibrated on the real sampler before trusted (test_memory's lesson). Leduc remains the optional second gate for the neural path's feature generalisation | the real work |
| 7 | `pokr/rl/nfsp.py` — `NFSPStrategy`, sigma coin-flip per hand in `on_hand_end`, `br_mode` on `Episode` | medium |
| 8 | `rollout.py` — both seats recording; build agent 1 outside, seat it with `lambda rng: agent1` | small |
| 9 | `train_nfsp.py` — round loop, **heads-up only** (NFSP is a 2p0s guarantee) | medium |
| 10 | Measure: `exploit.py --target nfsp` against the shipped 670.6 | — |

Ladder choice (design note section 4): **ladder B first** — fictitious self-play
with a PPO best response, reusing `exploit.py:best_response` as the oracle. No
DQN, ~150 lines, code that is already debugged. Ladder A (faithful DQN +
reservoir NFSP) only if B's exploitability curve flattens too high.

Success condition, stated before the run: `exploit.py --target nfsp` well below
670.6. It is **not** beating the PPO agent head-to-head — an equilibrium
approximator should lose to a max-exploit agent against weak opposition and win
less against the calling station and the maniac.

### 0.7 Local artifacts — none of this is in git

`models/` is gitignored in full. On this machine:

| path | what |
|---|---|
| `models/rl/` | **the shipped agent**: obs 160, no history, seed 7. What `plugin.py` loads. |
| `models/rl_history/` | history arm, seed 7 (+461.76 / 192.3). The retracted-outlier one. |
| `models/ab/s11_hist`, `s11_nohist` | seed 11 pair (+514.46 / 553.2 and +581.80 / 609.7) |
| `models/ab/s23_hist`, `s23_nohist` | seed 23 pair (+297.64 and +658.72; both probes FAILED) |
| `models/rl_carry`, `rl_leak`, `rl_v1`, `rl_v2` | older agents from previous sessions |
| `models/rlcard_dqn/` | the external DQN baseline |

A fresh clone has none of them. Re-training one arm is ~34 min at 8 workers;
the full three-seed campaign was ~2h45m end to end (train 34 min, duplicate
1.5 min, probe 4-6 min per arm).

The campaign script is not in the repo either — it was a scratch file. Its
shape is in section 0.4 and it is ~20 lines to rewrite.

### 0.8 Gotchas found this session

- **`models/` is gitignored.** `models/rl_history/` (the history arm, 11 MB)
  exists only on this machine. Re-training it is 35 min.
- Two `tests/test_league.py` tests build a real `RLStrategy` from a stub net and
  needed a real `obs_dim`; the other seven stubs never encode and were left at
  `obs_dim=16`.
- Do not wait on a background job with `until ! pgrep -f "pokr.duplicate"` —
  the waiter's own command line contains the pattern, so it matches itself and
  never exits.
- The venv is `.venv/` and is NOT on the default `python3`. Use
  `.venv/bin/python`.

---

## 1. What this project is

A research-grade 6-max No-Limit Texas Hold'em poker bot (`pokr`, Python 3.14,
numpy/numba/pytest) built through the full gated process: spec → plan →
subagent-driven implementation (17 tasks, each TDD + reviewed) → whole-branch
review → merged.

Core features (all implemented and tested, 243 tests passing):

- **Deterministic NLHE engine** (`pokr/engine.py`): betting rounds, side pots,
  all-ins, chip-conservation invariants, bot-exception sandboxing, dealer
  rotation. Known simplification: all-in short raises reopen action.
- **Opponent modeling** (`pokr/models.py`): per-opponent VPIP/PFR/aggression/
  fold-to-cbet/raise-size stats.
- **Bot detection** (`pokr/botdetect.py`): logistic P(is-bot) over stats +
  round-bet-size features, and P(mirror) via Hellinger distance of action
  histograms; both shrink toward priors on small samples.
- **Dynamic risk** (`pokr/risk.py`): Kelly-style sizing; `RiskConfig` defaults
  `max_bet_as_pot_fraction=0.66` (evidence-tuned sweet spot), bankroll-manager
  seam (Protocol, `@runtime_checkable`).
- **Bankroll seam** (`pokr/risk.py` `BankrollManager` Protocol +
  `PokerBot.begin_session`): per-session Kelly budget hook. The spec scopes
  cross-session management out ("seam only"); `SimpleBankrollManager`
  (session budget + stop-loss/stop-win) was removed as unused in 2026-08-18 —
  re-add it when a client wires the seam.
- **Policy** (`pokr/policy.py`): Monte Carlo equity (150 iters default,
  configurable `mc_iters`), softmax randomization over EV candidates (never
  deterministic), balanced bluffing tied to opponent fold freq, mirror mode
  (larger bets when P(mirror) ≥ 0.6), range-aware equity discount vs
  tight-aggressive betting/raising ranges (preflop via pfr/vpip, postflop via
  aggression), fold-equity scaling vs tight preflop openers (no bluff-reraises
  into ~5% premium ranges), risk caps with fallback that folds capped -EV
  bluffs instead of calling.
- **Canned opponents** (`pokr/opponents.py`): CallingStation, TightAggressive,
  Maniac, RandomBot, LeakHunter (adaptive exploitability proxy).
- **Benchmarks** (`pokr/bench.py`): per-matchup reports (bb/100, win%, var),
  fresh bot per matchup (prevents cross-matchup model pollution), mixed-lineup
  game mode (`--lineup`, `--replay N`) with the bot's own play stats, CLI
  `--mc-iters` flag, and `--fast` (opt-in numba equity path).
- **External comparison** (`pokr/ppe.py`, `pokr/ppe_compare.py`): `PokrPlayer`
  adapter plays inside **PyPokerEngine** (independent third-party engine)
  against its official example bots (external/*.py: Honest/Fish/Random).
- **Plugin connector** (`pokr/connector.py`): registry for external bots, with
  an **RLCard adapter** (`pokr/rlcard_adapter.py`): translates our GameState to
  RLCard's no-limit-holdem state/action model and back. Plugins: "rlcard"
  (uniform random over RLCard's action set) and "rlcard-dqn" (`TrainedDQNPolicy`
  — greedy policy from a trained DQN checkpoint, lazy torch load, path via
  `RLCARD_DQN_CKPT`, default `models/rlcard_dqn/dqn_final.pt`). Works in
  `bench --lineup`. RLCard is optional (lazy imports; rlcard 1.2.0, termcolor,
  setuptools-for-distutils-shim, torch 2.13.0+cpu installed in the venv).
  RLCard ships no pretrained NLH agent, so one is trained heads-up vs random
  play with `train_rlcard_dqn.py` (DQN, ~500 steps/s CPU). Measured 2026-08-18
  (heads-up, 2000 hands, seed 7, mc_iters=10): pokr **+1,545** bb/100 vs
  RlcardRandom, **+911** vs the 2.1M-step DQN — the trained agent is ~40%
  harder than random but still shallow (DQN-vs-random never defends blind
  steals). Eval curve plateaus once epsilon hits its 0.1 floor (~500k steps):
  more steps do not buy a stronger agent; NFSP/self-play would, at ~3-5x the
  CPU cost.

## 2. Key files

| File | Role |
|---|---|
| `pokr/engine.py` | rules engine (single source of truth) |
| `pokr/policy.py` | decision brain (EV, softmax, range-fold, risk caps) |
| `pokr/bot.py` | PokerBot composition (models+detector+policy) |
| `pokr/bench.py` | benchmarks + CLI (`python -m pokr.bench`) |
| `pokr/ppe.py` / `ppe_compare.py` | PyPokerEngine adapter + comparison CLI |
| `pokr/rlcard_adapter.py` | RLCard translation adapter (plugins "rlcard", "rlcard-dqn") |
| `analyze_leak.py` | scratch diagnostic: per-position/street P&L, facing-bet stats |
| `pokr/connector.py` | external-bot plugin registry |
| `docs/superpowers/specs/2026-08-06-poker-bot-design.md` | approved spec |
| `docs/superpowers/plans/2026-08-07-poker-bot.md` | implementation plan |

## 3. How to run

```bash
pip install -r requirements.txt        # numpy, numba, pytest, pypokerengine
python -m pytest -q                    # 243 tests, ~1 min (227 + 1 skip without rlcard)

# Internal benchmark (per-matchup)
python -m pokr.bench --hands 2000 --seed 7 --mc-iters 10

# Game vs mixed lineup with hand replay
python -m pokr.bench --lineup tag,tag,maniac,cs,random --hands 1000 --mc-iters 10 --replay 17

# External comparison vs PyPokerEngine bots (heads-up + 6-max)
python -m pokr.ppe_compare --hands 2000 --mc-iters 10 --seed 7

# RLCard benchmark: train a DQN (optional; checkpoint ships in models/, gitignored)
python train_rlcard_dqn.py --steps 5000000 --seed 7
# Train the in-engine PPO agent (PyTorch), then benchmark it vs the heuristic
python train_rl.py --iters 600 --hands-per-iter 2000 --seats 2,6 --fast   # ~35 min CPU
python -m pokr.duplicate --a rl --b self --lineup "" --hands 20000 --mc-iters 150 --fast

# pokr vs RLCard random / trained DQN (heads-up)
python -m pokr.bench --lineup rlcard --seats 2 --hands 2000 --mc-iters 10 --seed 7
python -m pokr.bench --lineup rlcard-dqn --seats 2 --hands 2000 --mc-iters 10 --seed 7
```

## 4b. Long-run reference (50k hands, seed 7, mc_iters=10, --fast, post-fixes + OOP blind fold)

| matchup | bb/100 | SE | verdict |
|---|---|---|---|
| calling station | **+636** | 21.3 | strongly profitable (~30 SE) |
| tight-aggressive | **-10.2** | 1.3 | resolved negative (~8 SE); now below the blind cost (was -18.6) |
| maniac | **+18,755** | 7899 | directionally positive (~2.4 SE) but unresolved; huge showdown pots |
| random | **+6,185** | 1051 | strongly profitable (~6 SE) |
| self-play | **-140** | 274 | unresolved (~0.5 SE); was -426±149 (2.9 SE) before the OOP blind fold — the blind leak is fixed structurally, but mirror-pots variance grew (375k vs 111k) |
| leak hunter | **+11.5** | 1.5 | small positive edge vs the exploitability proxy (~8 SE) |

The 50k run takes ~10 min with `--fast`. Run:
`python -m pokr.bench --hands 50000 --seed 7 --mc-iters 10 --fast`.
(It was unrunnable before the incremental round-size stat, commit 308ef38:
bot-detection rescanned full raise-size history per decision, O(hands) per
decision — a 50k run stalled past 75 min.)

Notes: the maniac row is seed-lottery at any sample size (SE ~8k). The OOP
blind fold (marginal preflop calls from SB/BB, 6-max+ only) is what moved
self-play from genuinely negative to unresolved; TAG improved -18.6 → -10.2.
Heads-up is excluded from the rule (it regressed the PyPokerEngine
HonestPlayer matchup).

Note the maniac verdict changed from the 2000-hand "-224" estimate (that was
seed-lottery at SE ~1300); at 50k hands the direction flips positive, though the
variance is still huge. Self-play is genuinely negative, not noise around 0.

## 4. Current measured results (seed 7, 2000 hands, mc_iters=10)

NOTE: the README's main table was re-anchored with `--reset-stacks` (fixed
100bb depth). The numbers in this section predate that and are the old
carry-over figures; maniac and random in particular were inflated by
stack drift (+3,411 → +290, +5,426 → +352) and their variance fell 2,800x
and 338x respectively. Trust the README.

Internal (own engine, fresh bot per matchup):

| matchup | bb/100 | var |
|---|---|---|
| calling station | +832 | 2.9k |
| tight-aggressive | **−20.5** | 12 |
| maniac | −224 (newly measurable; was a 7.4M-var lottery) | 16k |
| random | +5687 (still fat-tail) | 386k |
| self-play | −358 (EV noise: SE ~570/100; consistent with 0) | 64k |
| leak hunter | +9.0 | 21 |

External (PyPokerEngine engine, its official bots):
heads-up Honest **+1.9**, Fish **+27.5**, Random **+29.9** bb/100;
6-max: pokr +37.6 (2nd of 6, behind HonestPlayer +47.6).

Honest takeaway: profitable vs all external bots; thin edge vs the only
equity-aware one. Internally, TAG went from −165 to −20.5 (≈ the unavoidable
blind cost: SB/BB pay 25 bb/100), and self-play variance dropped 472k → 64k
(−7.4x) with EV consistent with 0.

Session progression (TAG / self-play var):
baseline −165 / 472k → preflop range fix −27 / 224k → capped-bluff fold fix
−20.5 / 64k.

A preflop-raise-floor experiment (making the risk cap respect the legal
minimum raise so blind steals / premium opens become possible) was tried and
reverted: it re-triggered the mirror re-raise escalation (TAG −20 → −90,
self-play var 64k → 1.4M). Mirror mode does not reliably activate in self-play
anymore (P(mirror) < 0.6 after the preflop fix), so gating steals on mirror
mode did not help. Stealing remains future work (needs position-aware preflop
logic, not a cap change).

## 5. Known issues / deferred items

- **High-variance matchups** (maniac/random/self-play) need 50k+ hands to
  resolve; 2000-hand numbers there are seed-lottery.
- **Self-play variance (64k per-hand) is still ~20x a normal matchup** — the
  mirrors still build big showdown pots (avg ~360 bb), but the 680 bb
  all-in-flip regime is gone. EV is noise around 0.
- **Maniac: the 2000-hand "−224" estimate was seed lottery.** The 50k-hand
  long run (section 4b) resolves it to +17,323 ± 9,401 bb/100: positive but
  unresolved. The blind-folding habit still shows in the replay, but the
  matchup is not a net leak.
- **Opponent targeting fixed** (session 2026-08-08): `bot.decide` used to read
  `opponents[0]` always (lowest-seat live opponent), applying one seat's stats to
  whoever acted. Now `PokerBot._target_opponent` picks the last bettor/raiser on the
  current street, falling back to the last aggressor in the hand, then the first live
  opponent. Mixed-lineup reference (seed 7, 1000 hands, mc_iters=10, pure path):
  total -5703.5 bb -> -2800.0 bb (+4903 bb), VPIP 52.8% -> 14.7%.
- The bot cannot steal blinds: the 0.66xpot risk cap sits below the legal
  preflop raise-to, making preflop raises impossible in small pots (PFR ~0.1%
  vs TAG). Attempted fix re-triggered mirror wars — see section 4.
- `mirror_mode` is sticky (never reset once triggered) — by design for now.
- numba fast path for equity is done (`pokr/_fastcards.py`, opt-in via `bench --fast` /
  `PokerBot(mc_fast=True)`): int-encoded cards + `@njit` evaluator, order-equivalent to
  `cards.evaluate_hand` (0 mismatches in 200k cross-checked hands), ~10-120x faster per
  decision. End-to-end: the 200-hand x 6-matchup benchmark at mc_iters=150 went
  361.8s -> 2.3s. Pure path remains the default so tuned numbers stay comparable
  (the fast path draws a different RNG stream).
- RLCard: DONE (2026-08-18) — rlcard 1.2.0 + torch 2.13.0+cpu work on Python
  3.14 (needed: termcolor + setuptools for the distutils shim that
  `rlcard.agents` requires). Plugins "rlcard" and "rlcard-dqn" benchmarked;
  `train_rlcard_dqn.py` trains the DQN. Remaining: NFSP/self-play if a
  stronger opponent is ever wanted.
- OpenSpiel: not installed (no Windows wheels); RLCard adapter covers the
  connector story for now.
- PyPokerEngine comparison uses rebuy sessions (engine ends at first bust);
  per-session hands played is approximate.
- `Action.reason` is populated everywhere — useful for tracing/debugging.

## 6. What happened late in the session (context for next steps)

- Found and fixed a benchmark bug: reusing one bot across matchups polluted
  its per-seat models (TAG −210 clean vs −265 polluted).
- Evidence-based tweaks: range-fold rule (TAG −266 → −165), 0.66×pot default
  cap (mirror-signal sweet spot; 0.4 killed the mirror signal).
- PyPokerEngine comparison initially gave garbage (game ends at first bust)
  → fixed with short rebuy sessions.
- (this session) TAG leak diagnosed: the bot called a ~5% premium open range
  as if random (401 calls vs 60 folds over 2000 hands) and 4-bet-bluffed into
  opens that never fold. Fixed with a preflop range discount (tight x
  pfr/vpip) + fold-equity scaling for tight openers + a capped-bluff fallback
  that folds instead of calling. TAG −165 → −20.5, self-play var 472k → 64k.
- (this session) Mirror-detection subtlety: the preflop fold-equity scaling
  initially weakened the P(mirror) signal (identical bots drift apart in a
  feedback loop); weighting the scaling by opener tightness (loose mirrors
  unaffected) preserved it.
- (2026-08-18) OOP blind fold: marginal preflop calls from SB/BB now fold
  (6-max+ only, equity < 0.42 — both the main call path and the risk-cap
  fallback). Measured: preflop wide calls 531 → 270 and BB P&L −5988 → −625
  bb at 2000 self-play hands (seed 7); TAG 50k −18.6 → −10.2; self-play 50k
  −426 → −140 (unresolved). Heads-up excluded after a measured regression vs
  PyPokerEngine's HonestPlayer.

## 7. Suggested next steps (in order)

1. **Self-play variance** (the blind leak is fixed; self-play is now -140 ± 274
   at 50k hands, unresolved instead of genuinely negative): the mirrors still
   build big showdown pots (avg ~230 bb) and mirror mode still does not
   reliably activate (P(mirror) ~0.20 vs the 0.6 threshold). Two paths:
   (a) lower the mirror bet-sizing threshold so the documented river exploit
   fires (risk: re-raise wars — see section 4), or (b) tighter postflop
   call-downs with marginal pairs (showdown ratio ~1:4.5).
2. **Preflop stealing done right**: recover the blind cost vs TAG (currently
   −10.2 at 50k, down from −18.6 after the OOP blind fold). Requires
   position-aware preflop logic (the cap-floor shortcut re-triggers mirror
   wars).
3. Run a long (50k+ hand) benchmark to resolve maniac/random/self-play.
   Now affordable in minutes with `bench --fast`.
4. RLCard: done (DQN trained + benchmarked; plugins "rlcard"/"rlcard-dqn").
5. In-engine PPO agent: done (`pokr/rl/`, plugin "rl", `train_rl.py`). Beats
   the heuristic heads-up (+180.4 ± 32.6 bb/100) and 6-max (+753.7 vs +121.1),
   both resolved at ~10 SE on duplicate decks, and is LESS exploitable than
   the heuristic by best-response probe (879.4 ± 101.5 vs 1019.7 ± 89.9).
   Depth mismatch is FIXED: training and scoring both run at a fixed 100bb
   via `--reset-stacks`, which took the agent from +180.4 to +604.0 bb/100
   against the heuristic and cut exploitability 879.4 → 670.6.
   Open follow-ups, in order:
   a. Everything is still ~670-1020 bb/100 exploitable — nowhere near
      equilibrium. Deep CFR / NFSP is the real answer if that matters.
   b. Do NOT put the leak hunter in the training pool. Measured: it flips the
      leak-hunter column (−136 → +1203) while making the agent more
      exploitable by a real best response (879 → 1294). The leak hunter
      under-reports exploitability by ~70x and is not a safe training signal.
      It is available in `--pool` but off by default.
   c. Hyperparameters have never been swept; entropy collapsed to 0.59
      mid-run once, so `--ent-coef` is the first thing to look at.
   d. Observation features (equity, opponent model) are unvalidated — an
      ablation would say whether the Monte Carlo cost earns its place.
   e. The PyPokerEngine and RLCard comparison tables were measured under the
      old carry-over depth and have NOT been re-anchored.
6. Re-add a bankroll manager (seam stays; SimpleBankrollManager dropped as
   unused 2026-08-18) when a client wires `begin_session` end-to-end.
7. Consider making `--fast` the default once tuned numbers are re-anchored
   (the fast path draws a different RNG stream than the pure path).

## 8. Git state

- `main` at `38b41f0`: `bc51def` (opt-in numba equity fast path + cleanups:
  dead code removal, .gitignore, faster e2e smoke) and `38b41f0` (aggressor
  targeting fix), on top of the previous handoff's:
  `33571fd` (preflop range fix, TAG −165 → −27), `7b24685` (capped-bluff fold,
  TAG −27 → −20.5, self-play var 64k), `066cf56` (BankrollManager),
  `97be9e5` (RLCard adapter), plus the docs update.
- PR #1 was merged and branch deleted; `origin/feat/poker-bot` still exists
  (remote only) if you want the pre-merge history.
- Untracked: `bench_fix_final.txt`, `ppe_results.txt` (old run artifacts),
  `analyze_leak.py` (scratch diagnostic tool — keep, it is useful for leak
  work), `bench_fix1.txt` (latest benchmark output).
