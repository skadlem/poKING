# Handoff — pokr Poker Bot (session 2026-08-06/07)

**Repo:** https://github.com/skadlem/poKING (public)
**Branch:** everything merged to `main` (PR #1 merged + one cherry-pick)
**Date:** 2026-08-07

---

## 1. What this project is

A research-grade 6-max No-Limit Texas Hold'em poker bot (`pokr`, Python 3.14,
numpy/numba/pytest) built through the full gated process: spec → plan →
subagent-driven implementation (17 tasks, each TDD + reviewed) → whole-branch
review → merged.

Core features (all implemented and tested, 89 tests passing):

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
  seam (Protocol, no impl yet).
- **Policy** (`pokr/policy.py`): Monte Carlo equity (150 iters default,
  configurable `mc_iters`), softmax randomization over EV candidates (never
  deterministic), balanced bluffing tied to opponent fold freq, mirror mode
  (larger bets when P(mirror) ≥ 0.6), range-aware fold rule vs tight-aggressive
  betting ranges (postflop only), risk caps with check/call fallback.
- **Canned opponents** (`pokr/opponents.py`): CallingStation, TightAggressive,
  Maniac, RandomBot, LeakHunter (adaptive exploitability proxy).
- **Benchmarks** (`pokr/bench.py`): per-matchup reports (bb/100, win%, var),
  fresh bot per matchup (prevents cross-matchup model pollution), mixed-lineup
  game mode (`--lineup`, `--replay N`) with the bot's own play stats, CLI
  `--mc-iters` flag.
- **External comparison** (`pokr/ppe.py`, `pokr/ppe_compare.py`): `PokrPlayer`
  adapter plays inside **PyPokerEngine** (independent third-party engine)
  against its official example bots (external/*.py: Honest/Fish/Random).
- **Plugin connector** (`pokr/connector.py`): registry for external bots.

## 2. Key files

| File | Role |
|---|---|
| `pokr/engine.py` | rules engine (single source of truth) |
| `pokr/policy.py` | decision brain (EV, softmax, range-fold, risk caps) |
| `pokr/bot.py` | PokerBot composition (models+detector+policy) |
| `pokr/bench.py` | benchmarks + CLI (`python -m pokr.bench`) |
| `pokr/ppe.py` / `ppe_compare.py` | PyPokerEngine adapter + comparison CLI |
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
```

## 4. Current measured results (seed 7, 2000 hands, mc_iters=10)

Internal (own engine, fresh bot per matchup):

| matchup | bb/100 | var |
|---|---|---|
| calling station | +832 | 2.9k |
| tight-aggressive | **−165** | 180 |
| maniac / random | unmeasurable | 1-7M |
| self-play | −1515 | 472k |
| leak hunter | −1.4 | 19 |

External (PyPokerEngine engine, its official bots):
heads-up Honest **+1.9**, Fish **+27.5**, Random **+29.9** bb/100;
6-max: pokr +37.6 (2nd of 6, behind HonestPlayer +47.6).

Honest takeaway: profitable vs all external bots; thin edge vs the only
equity-aware one; the TAG/equity-aware matchup is the known weakness.

## 5. Known issues / deferred items

- **High-variance matchups** (maniac/random/self-play) need 50k+ hands to
  resolve; 2000-hand numbers there are seed-lottery.
- **Self-play is negative at high variance** — the bot coin-flips stacks vs a
  copy of itself; 0.66×pot cap helped (66k→11k var) but didn't fix the EV.
- `mirror_mode` is sticky (never reset once triggered) — by design for now.
- numba fast path for equity is deferred (`ponytail:` comment in cards.py).
- Cross-session bankroll manager: seam exists (`BankrollManager` Protocol),
  no implementation.
- RLCard/OpenSpiel pretrained agents: connector exists, adapters not written.
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

## 7. Suggested next steps (in order)

1. Run a long (50k+ hand) benchmark for maniac/random/self-play if you want
   those numbers resolved (slow: ~1 hr+ at mc_iters=10).
2. Attack the self-play / TAG leak — the biggest measurable weaknesses.
3. Wire an RLCard/OpenSpiel pretrained agent via the connector for a stronger
   external baseline than PyPokerEngine's example bots.
4. Implement the BankrollManager for cross-session bankroll management.
5. Consider the numba fast path if simulation speed becomes a bottleneck.

## 8. Git state

- `main` at `0f94852` (merge of PR #1 + cherry-picked rebuy fix).
- PR #1 was merged and branch deleted; `origin/feat/poker-bot` still exists
  (remote only) if you want the pre-merge history.
- Clean tree (two untracked run artifacts: `bench_fix_final.txt`,
  `ppe_results.txt` — can be deleted or gitignored).
