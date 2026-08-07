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

Core features (all implemented and tested, 126 tests passing):

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
- **Bankroll management** (`pokr/bankroll.py`): `SimpleBankrollManager`
  implements the seam (session budget = clamp(fraction x bankroll); stop-loss /
  stop-win in session budgets); `PokerBot.begin_session(bankroll)` feeds it
  into the Kelly sizing.
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
  `--mc-iters` flag.
- **External comparison** (`pokr/ppe.py`, `pokr/ppe_compare.py`): `PokrPlayer`
  adapter plays inside **PyPokerEngine** (independent third-party engine)
  against its official example bots (external/*.py: Honest/Fish/Random).
- **Plugin connector** (`pokr/connector.py`): registry for external bots, with
  an **RLCard adapter** (`pokr/rlcard_adapter.py`): translates our GameState to
  RLCard's no-limit-holdem state/action model and back (plugin "rlcard", works
  in `bench --lineup`; bundled policy is uniform random; torch-based pretrained
  agents can plug in later). RLCard is optional (lazy import, rlcard 1.2.0
  installed here; its `rlcard.agents` module is broken on Python 3.14 - no
  distutils/setuptools/torch).

## 2. Key files

| File | Role |
|---|---|
| `pokr/engine.py` | rules engine (single source of truth) |
| `pokr/policy.py` | decision brain (EV, softmax, range-fold, risk caps) |
| `pokr/bot.py` | PokerBot composition (models+detector+policy) |
| `pokr/bench.py` | benchmarks + CLI (`python -m pokr.bench`) |
| `pokr/ppe.py` / `ppe_compare.py` | PyPokerEngine adapter + comparison CLI |
| `pokr/bankroll.py` | SimpleBankrollManager (session budget + stop rules) |
| `pokr/rlcard_adapter.py` | RLCard translation adapter (plugin "rlcard") |
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
```

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
- **Maniac is now measurable and negative** (−224 ± 28 at 2000 hands): the
  bot folds its blinds to maniac's 60% opens and pays off its value bets. New
  finding from the capped-bluff fix; needs a longer run + a defense strategy.
- The bot cannot steal blinds: the 0.66xpot risk cap sits below the legal
  preflop raise-to, making preflop raises impossible in small pots (PFR ~0.1%
  vs TAG). Attempted fix re-triggered mirror wars — see section 4.
- `mirror_mode` is sticky (never reset once triggered) — by design for now.
- numba fast path for equity is deferred (`ponytail:` comment in cards.py).
- RLCard pretrained agents need torch + a working `rlcard.agents` (broken on
  this Python 3.14 env: distutils removed, no setuptools). The adapter's
  policy callable is the plug point.
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

## 7. Suggested next steps (in order)

1. **Maniac defense** (newly visible, −224): the bot needs to defend its
   blinds vs wide opens and call down maniac's bluff-heavy pot bets.
2. **Preflop stealing done right**: recover the blind cost vs TAG (currently
   −20.5 ≈ blinds). Requires position-aware preflop logic (the cap-floor
   shortcut re-triggers mirror wars).
3. Run a long (50k+ hand) benchmark to resolve maniac/random/self-play
   (slow: ~1 hr+ at mc_iters=10).
4. RLCard pretrained agent: install torch + setuptools (for rlcard.agents) and
   plug an NFSP/DQN agent into `RlcardAdapter.policy`; or wire an OpenSpiel
   agent when Windows wheels exist.
5. Wire `begin_session`/`should_stop` into a client or bench mode to exercise
   the BankrollManager end-to-end.
6. Consider the numba fast path if simulation speed becomes a bottleneck.

## 8. Git state

- `main` at `97be9e5`, 5 commits ahead of the previous handoff:
  `33571fd` (preflop range fix, TAG −165 → −27), `7b24685` (capped-bluff fold,
  TAG −27 → −20.5, self-play var 64k), `066cf56` (BankrollManager),
  `97be9e5` (RLCard adapter), plus the docs update.
- PR #1 was merged and branch deleted; `origin/feat/poker-bot` still exists
  (remote only) if you want the pre-merge history.
- Untracked: `bench_fix_final.txt`, `ppe_results.txt` (old run artifacts),
  `analyze_leak.py` (scratch diagnostic tool — keep, it is useful for leak
  work), `bench_fix1.txt` (latest benchmark output).
