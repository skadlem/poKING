# pokr

A research-grade 6-max No-Limit Texas Hold'em poker bot with dynamic risk
assessment, opponent modeling, and bot/mirror detection. Spec:
`docs/superpowers/specs/2026-08-06-poker-bot-design.md`.

## Install

    pip install -r requirements.txt

## Run a benchmark

    python -m pokr.bench --hands 2000 --seed 7 --mc-iters 10

Reports BB/100, win rate, and variance for each matchup: calling station,
tight-aggressive, maniac, random, self-play, and the leak hunter
(exploitability proxy). `--mc-iters` sets the Monte Carlo equity iterations
per decision (default 150; use 10-30 for fast exploratory runs).

## Benchmark results

Reference run: seed 7, 2000 hands per matchup, `mc_iters=10`, 6-max, 200bb
buy-in, no rake, fresh bot per matchup.

| matchup | bb/100 | win% | variance (bb²) | read |
|---|---|---|---|---|
| calling station | **+832** | 8.6% | 2.9k | strong edge vs a never-folding opponent |
| tight-aggressive | **−165** | 47.2% | 180 | loses to tight value-betting (best-measured leak) |
| maniac | +14,847 | 2.4% | 7.4M | unmeasurable at 2000 hands (huge variance) |
| random | +6,050 | 12.2% | 1.2M | unmeasurable at 2000 hands |
| self-play | −1,515 | 17.8% | 472k | 6 copies coin-flip stacks; seed-sensitive |
| leak hunter | −1.4 | 46.2% | 19 | exploitability proxy: near break-even |

Caveats: with `mc_iters=10` equity estimates are coarse, and matchups whose
variance exceeds ~10⁵ bb² need 50k+ hands to resolve. Only the calling-station,
tight-aggressive, and leak-hunter rows are statistically meaningful at 2000
hands. Fixes that measurably improved the TAG matchup: a range-aware fold rule
for marginal calls into tight betting ranges (−266 → −165 bb/100) and a default
bet cap of 0.66× pot (cut stack-shove variance ~6× without killing the mirror
signal).

## Comparison with other bots

The bot is benchmarked against three reference classes:

1. **Scripted archetypes** (proxies for human styles): calling station,
   tight-aggressive, maniac, random. These are simple but deterministic
   baselines for regression testing; beating them is necessary, not sufficient.
2. **Itself (self-play)**: the hardest honest test — a strategy that can't hold
   its own against a copy is exploitable by anyone who models it. Current
   self-play is negative at high variance, the main known weakness.
3. **Leak hunter** (exploitability proxy): an adaptive opponent that models the
   bot's frequencies and counter-adjusts. Near break-even (−1.4 bb/100) means
   the bot is not trivially exploitable by a simple adaptive counter.

External bots (e.g. RLCard/OpenSpiel pretrained CFR/NFSP agents) can be dropped
in through the plugin connector and benchmarked head-to-head:

```python
from pokr.connector import register_plugin
from pokr.bench import run_matchup, run_benchmark
from pokr.bot import PokerBot

register_plugin("rlcard_agent", lambda: MyRLCardAdapter())  # implements Strategy
run_matchup(PokerBot(), lambda rng: MyRLCardAdapter(), num_hands=2000, seed=7)
```

The `Strategy` interface (decide/on_hand_end) is the only boundary a third-party
bot must implement to be comparable.

## Module map

- `pokr/cards.py` — Card, Deck, hand evaluator, Monte Carlo equity
- `pokr/strategy.py` — Action and the Strategy interface
- `pokr/engine.py` — the deterministic NLHE game loop (single source of rules)
- `pokr/opponents.py` — canned archetypes + leak hunter
- `pokr/models.py` — per-opponent statistics (VPIP, PFR, aggression, cbet folds)
- `pokr/botdetect.py` — P(is bot) and P(mirror) per opponent
- `pokr/risk.py` — Kelly-style risk sizing, bankroll seam
- `pokr/policy.py` — decision layer: EV, randomization, balanced bluffing
- `pokr/bot.py` — PokerBot composing the above
- `pokr/bench.py` — benchmark harness
- `pokr/connector.py` — plugin registry for external bots

## Tests

    python -m pytest -q
