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

## Play a game against a mixed lineup

The bot can sit at a real 6-max table against a chosen mix of other bots,
report how it actually played, and replay individual hands:

    python -m pokr.bench --lineup tag,tag,maniac,cs,random --hands 1000 --mc-iters 10 --replay 17

`--lineup` takes one abbreviation per opponent seat (`cs`, `tag`, `maniac`,
`random`, `leak`, `self`). The report shows your profit (bb/100), hands won,
the bot's own VPIP/PFR/aggression/fold stats, and with `--replay N` a
human-readable transcript of hand N (blinds, every action with its reason,
board, showdown, net result). Example:

    game vs lineup [TightAggressive, TightAggressive, Maniac, CallingStation, RandomBot]  (1000 hands)
      you: -5703.5 bb total, -570.35 bb/100, won 10.2% of hands, var 608196.0
      your play: VPIP 52.8%  PFR 16.1%  postflop aggression 0.65  fold-to-cbet 40%  postflop fold 23%

    --- Hand #17 of 1000 (dealer RandomBot) ---
      blinds: You(pokr) 1 / TightAggressive 2
      TightAggressive preflop  fold  [tag folds junk]
      Maniac       preflop  raise 100  [maniac raise]
      ...
      CallingStation wins 203 with two pair (Ah Qc)
      net: You(pokr) -1 TightAggressive -2 TightAggressive +0 Maniac -200 CallingStation +203 RandomBot +0

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

### Measured: pokr vs PyPokerEngine's official example bots

`pokr` also plays inside the external **PyPokerEngine** engine (its own dealer,
rules, and side-pot logic) against the framework's own example bots, via the
`PokrPlayer` adapter (`pokr/ppe.py`). Run it with:

    python -m pokr.ppe_compare --hands 2000 --mc-iters 10 --seed 7

Heads-up, 2000 hands each, 10 MC iters, 200bb stacks (rebuy sessions since
PyPokerEngine ends a game at the first bust):

| opponent | pokr bb/100 | read |
|---|---|---|
| HonestPlayer | **+1.9** | marginal win vs the equity-aware bot |
| FishPlayer | **+27.5** | strong edge vs a never-folding fish |
| RandomPlayer | **+29.9** | solid edge vs random play |

6-max table (pokr + one of each), 2000 hands:

| player | bb/100 |
|---|---|
| HonestPlayer | +47.6 |
| **pokr** | **+37.6** |
| FishPlayer | +38.9 / −24.3 |
| RandomPlayer | −50.0 ×2 |

Takeaway: pokr is profitable against all three external bots, heads-up and at
a full table. The edge is comfortable against the weak bots (fish, random) and
thin against the equity-aware one (+1.9), which matches the internal finding
that tight/equity-aware opponents are the bot's hardest matchup.

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
