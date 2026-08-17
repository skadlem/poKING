# Handoff — pokr Poker Bot (session 2026-08-08)

**Repo:** https://github.com/skadlem/poKING (public)
**Branch:** everything merged to `main` (PR #1 merged + one cherry-pick)
**Date:** 2026-08-08

---

## 1. What this project is

A research-grade 6-max No-Limit Texas Hold'em poker bot (`pokr`, Python 3.14,
numpy/numba/pytest) built through the full gated process: spec → plan →
subagent-driven implementation (17 tasks, each TDD + reviewed) → whole-branch
review → merged.

Core features (all implemented and tested, 124 tests passing):

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
python -m pytest -q                    # 89 tests, ~3-4 min (drift test is slow)

# Internal benchmark (per-matchup)
python -m pokr.bench --hands 2000 --seed 7 --mc-iters 10

# Game vs mixed lineup with hand replay
python -m pokr.bench --lineup tag,tag,maniac,cs,random --hands 1000 --mc-iters 10 --replay 17

# External comparison vs PyPokerEngine bots (heads-up + 6-max)
python -m pokr.ppe_compare --hands 2000 --mc-iters 10 --seed 7

# RLCard benchmark: train a DQN (optional; checkpoint ships in models/, gitignored)
python train_rlcard_dqn.py --steps 5000000 --seed 7
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
   Optional follow-up: NFSP/self-play for a stronger opponent.
5. Re-add a bankroll manager (seam stays; SimpleBankrollManager dropped as
   unused 2026-08-18) when a client wires `begin_session` end-to-end.
6. Consider making `--fast` the default once tuned numbers are re-anchored
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
