# Poker Bot with Dynamic Risk Assessment — Design

**Date:** 2026-08-06
**Status:** Approved at decision gate (all five sections)

## 1. Purpose and scope

A research-grade poker bot (primary goal) that may later be pointed at real-money
play (secondary goal). It must:

1. Play 6-max No-Limit Texas Hold'em cash games in a deterministic simulation engine.
2. Model each opponent's strategy and statistics, and use those models in decisions.
3. Estimate, per opponent, the probability that the opponent is a bot, including the
   specific case that the opponent is the same bot (a mirror). These probabilities
   influence decisions and are also reported for study.
4. Maximize expected profit across the games it plays, per-session for now, with a
   clean seam for cross-session bankroll management later.
5. Be comparable against other bots: beat our canned opponent archetypes and expose a
   plugin interface so external bots can be dropped in later.

Decisions recorded during brainstorming:

| Question | Decision |
| --- | --- |
| Where does the bot play? | Simulation core first; adapter seam reserved, no real-platform adapter yet. |
| Primary purpose | Research/learning now; real-money possible later. Keep bankroll seam honest. |
| Variant / format | 6-max No-Limit Texas Hold'em, cash game. |
| What does bot detection feed into? | Both: it influences decisions AND is reported as a study metric. |
| Profit-maximization scope | Per-session optimization now, with a seam for cross-session bankroll management. |
| Benchmark target | Beat canned opponents AND expose a plugin interface for external bots. |
| Language / stack | Python 3 with NumPy/Numba for vectorized hand evaluation and Monte Carlo. |
| Mirror response | Exploit our own known leaks against the mirror. |

### Competitive landscape findings (2026-08-07)

Reviewed leading open-source poker systems: OpenSpiel (DeepMind), PokerRL (Deep
CFR/NFSP), RLCard (no-limit Hold'em + CFR), rs-poker (Rust/CFR), a Pluribus
reimplementation (6-max NLHE), and rosbo's classical bot (closest to our design).
Four things from them could beat our current strategy, so the design incorporates
all four as deltas:

1. **Exploitability measurement.** Strong bots measure how beatable an opponent is
   (PokerRL Best Response/LBR; OpenSpiel Nash-distance exploitability). Win rate vs.
   canned archetypes can be gamed. Delta: an exploitability proxy in `bench.py`.
2. **Deterministic play is death.** A pure EV-threshold policy is learnable and
   exploitable by anyone modeling us (which is exactly what our own bot does). Every
   serious bot randomizes. Delta: explicit mixed-strategy randomization in `policy.py`.
3. **Unbalanced bluffing.** CFR opponents play balanced bluff-to-value frequencies;
   EV-threshold bots bluff too little or with tellable frequency. Delta: explicit
   bluff ranges in the decision layer.
4. **Subgame solving** (Pluribus method) is state of the art for 6-max; static
   policies lose on the river. Not v1, but `policy.py` is structured so a river solver
   can slot in later.

Also validated: RLCard/OpenSpiel ship pretrained CFR/NFSP agents. The `BotPlugin`
connector can import those later, making "comparable to other bots" a measured
number. None of these bots do opponent bot detection or mirror detection; that is
our differentiator.

## 2. Architecture

Approach B (approved): modular pipeline. One-way dependency layering; nothing below a
layer imports from above it. Package `pokr`.

```
pokr/
  cards.py      # Card, Deck; vectorized hand evaluator (NumPy) -> HandRank
  engine.py     # deterministic 6-max NLHE cash game: betting, side pots, showdown
  strategy.py   # Strategy protocol: decide(state, player_id) -> Action
  opponents.py  # canned archetypes: calling station, TAG, maniac, random
  models.py     # per-opponent stats: VPIP, PFR, aggression, bet-size patterns
  botdetect.py  # P(is bot) per opponent + mirror-match detector
  risk.py       # per-session risk sizing (Kelly-style), seam for bankroll mgmt
  policy.py     # combines hand EV + opponent model + bot detection + risk -> Action
  bot.py        # PokerBot: composes the above, implements Strategy
  bench.py      # thousands of games, reports BB/100, win rate, variance, matchups
  connector.py  # BotPlugin protocol: seam for external/third-party bots
tests/          # pytest per module
```

Key properties:

- The engine is the single source of truth for rules; bots only ever request legal
  actions.
- `Strategy` is the boundary between the game and any bot. Canned opponents, `PokerBot`,
  and future third-party bots all implement it.
- `connector.py` is a thin protocol now, not an adapter for any real platform.
- Everything is deterministic given a seed, so benchmarks are reproducible.

## 3. Components

- **`cards.py`** — `Card`, `Deck`, `evaluate_hand(cards) -> HandRank`. NumPy-vectorized
  7-card evaluator with a precomputed rank table (JIT-cached via Numba); deterministic,
  no global mutable state.
- **`engine.py`** — deterministic 6-max NLHE cash game loop. Owns `GameState` (seats,
  stacks, community, pot, betting round, current player, min-raise, legal actions).
  Enforces rules: betting order, minimum raise, side pots, all-ins, showdown. Emits
  `HandResult` (actions, hole cards, community, winnings) at hand end.
- **`strategy.py`** — `Strategy` protocol: `decide(state, player_id) -> Action`. The only
  boundary bots cross.
- **`opponents.py`** — canned archetypes implementing `Strategy`: calling station,
  tight-aggressive, maniac, random, plus a **leak hunter** opponent that models our
  action frequencies and counter-adjusts (the exploitability proxy).
- **`models.py`** — per-opponent stat tracker fed by `HandResult`: VPIP, PFR, aggression
  frequency, preflop raise-size distribution, fold-to-cbet. Emits a summary object per
  hand for detection.
- **`botdetect.py`** — two outputs per opponent: `P(is_bot)` (logistic regression over
  stats + pattern features like pot-fraction betting, implemented in NumPy directly, no
  scikit-learn) and mirror probability (statistical distance between the opponent's action
  distribution and our own). Both carry uncertainty when sample size is small.
- **`risk.py`** — per-hand risk sizing: Kelly-style fraction over win probability and pot
  odds, capped by stack depth and a per-session bankroll budget; exposes the seam where
  cross-session bankroll management will plug in later.
- **`policy.py`** — decision layer. Takes hand EV, opponent model, bot detection, and
  risk-adjusted sizing, returns the chosen `Action` with an explanation record. Two
  competitive-landscape deltas: explicit mixed-strategy randomization over candidate
  actions (never fully deterministic), and balanced bluff-to-value ranges (a bluff
  frequency tied to the value-bet frequency of the same spot, so a river solver can
  slot in later).
- **`bot.py`** — `PokerBot` composes models + botdetect + risk + policy; implements
  `Strategy`; includes mirror-exploitation mode (attack our own known leaks when a mirror
  is detected).
- **`bench.py`** — runs thousands of seeded games across seat rotations and opponent
  lineups; reports BB/100, win rate, variance, and per-matchup results including
  self-play. Includes the exploitability proxy: every benchmark run also pits the bot
  against the leak-hunter opponent and reports the resulting BB/100 as its
  exploitability score.
- **`connector.py`** — `BotPlugin` protocol so external bots can be dropped in later; no
  real-platform adapter. Documented future use: RLCard/OpenSpiel pretrained CFR/NFSP
  agents as benchmark opponents.

## 4. Data flow

One hand of poker:

```
engine deals hole cards, posts blinds
  └> for each decision point:
       state ──> Strategy.decide(state, player_id) ──> Action
       engine validates & applies, advances betting
  └> showdown: engine evaluates hands (cards.py), awards pots
  └> engine emits HandResult (actions, hole cards, community, winnings)
       └> each bot's models.py consumes HandResult ─> per-opponent stats
       └> botdetect.py reads stats ─> P(is_bot), P(mirror)
       └> policy.py reads stats + detection + risk next decision point
       └> bench.py aggregates HandResults ─> performance report
(future) connector.py adapts this same flow to a real platform
```

Key properties:

- **Stateless decisions.** Each `decide()` call takes the current `GameState` and returns
  an `Action`; bots keep their own memory (stats, detection) in their own instance, not in
  the engine. This makes bots swappable and benchmark comparisons fair.
- **Deterministic seeds.** `bench.py` runs the same seed sequence for every bot so
  head-to-head and self-play comparisons are reproducible.
- **Explanations travel with actions.** `Action` carries a `reason` field so the research
  goal (understanding why the bot acted) is supported without a separate trace subsystem.

## 5. Error handling

- **Engine is the rule enforcer.** Every action is validated before it applies: illegal
  actions raise `IllegalAction` (fold when no bet, raise below min-raise, out-of-turn, act
  with no chips). Bots can't corrupt the game.
- **Bots are sandboxed.** A bot that raises an unexpected exception is treated as a fold
  and logged with a warning; the game continues. A misbehaving bot can never crash the
  simulation or break the invariants.
- **Chip-exact invariants.** The engine asserts after every hand: sum of all stacks + pots
  == starting chip count (no chips created or destroyed), no negative stacks, pot totals
  match contributions. Side-pot/all-in logic gets the most tests since it's the
  highest-risk path.
- **Small-sample detection.** Bot and mirror detection return a prior plus uncertainty
  instead of confident garbage when sample size is small; policy degrades gracefully
  (falls back to neutral play).
- **Seedable RNG everywhere.** Engine shuffle, opponents' randomness, and the bot's own
  randomization all take an `rng` so any failure is reproducible with the same seed.

## 6. Testing

- **`cards`** — exhaustive-ish hand-rank tests: every hand category (high card through
  royal flush), ties, kickers, wheel straights, 7-card best-5 selection. Determinism and
  shuffle correctness.
- **`engine`** — rule-compliance property tests: illegal actions rejected, betting order,
  min-raise rules, blinds, all-ins, side pots (multi-way), split pots, chop scenarios,
  chip conservation invariant after every hand, heads-up and full-ring rotations.
- **`opponents`** — each canned archetype behaves as named: calling station never folds to
  a callable bet, maniac raises more than it calls, TAG raises a narrow range preflop,
  random is uniform within legal actions, and the leak hunter reacts to our observed
  frequencies (tightens against a tight player, loosens against a loose one).
- **`models`** — stats update correctly from scripted `HandResult`s: VPIP/PFR/aggression
  denominators and numerators, bet-size distribution bins, reset behavior.
- **`botdetect`** — known-bot classification: a scripted robot-like action stream scores
  high on `P(is_bot)`; a humanlike action stream scores low; mirror detection flags two
  identical `PokerBot` instances and clears distinct opponents; both return uncertainty
  priors on tiny samples.
- **`risk`** — sizing scales with win probability and pot odds, respects stack and
  bankroll caps, Kelly fraction never exceeds 1, seam accepts a bankroll-manager for
  cross-session behavior.
- **`policy`** — decision tests: EV-positive vs EV-negative situations, opponent-model
  influence (exploits loose caller, respects tight player), bot detection influence
  (mirror triggers exploit-leak mode), risk cap respected, `Action.reason` populated.
  Deltas: randomization is statistically present over repeated identical states (not
  deterministic), and bluff frequency tracks value frequency (balanced ranges).
- **`bench`** — determinism (same seed, same report), seat-rotation fairness (equal
  positional distribution), reporting math (BB/100, win rate, variance), exploitability
  score reported for the leak-hunter matchup, and a smoke test running a small number of
  hands end to end.

## 7. Explicit non-goals (v1)

- No real-money platform adapter.
- No cross-session bankroll management (seam only).
- No RL/CFR learning; classical explainable decision engine.
- No subgame solving on the river; `policy.py` keeps a seam for it.
- No multi-table play or table selection.
