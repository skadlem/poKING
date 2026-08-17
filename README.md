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

Reference run: seed 7, 2000 hands per matchup, `mc_iters=10`, `--fast` (numba
equity path), 6-max, 200bb buy-in, no rake, fresh bot per matchup, after the
aggressor-targeting fix. SE = std error of bb/100 (sqrt(var/hands) x 100);
rows whose |bb/100| is within ~2 SE of 0 are statistically unresolved.

| matchup | bb/100 | SE | win% | variance (bb²) | read |
|---|---|---|---|---|---|
| calling station | **+647** | 110 | 8.8% | 2.4k | strong edge vs a never-folding opponent |
| tight-aggressive | **−15.4** | 7.9 | 22.1% | 12 | ≈ blind cost (SB/BB pay 25 bb/100); tight |
| maniac | −190 | 1314 | 1.4% | 345k | unresolved at 2000 hands (SE ≈ 7x estimate) |
| random | +5,795 | 1798 | 7.5% | 647k | positive but unresolved (fat tails) |
| self-play | −380 | 181 | 15.6% | 6.5k | ≈ 2.1 SE below 0; variance cut ~10x by targeting fix |
| leak hunter | **+15.0** | 5.1 | 34.8% | 5 | exploitability proxy: small positive edge |

Caveats: with `mc_iters=10` equity estimates are coarse, and matchups whose
variance exceeds ~10⁵ bb² need 50k+ hands to resolve (see HANDOFF.md for the
long-run reference). Only the calling-station, tight-aggressive, self-play,
and leak-hunter rows are statistically meaningful at 2000 hands. Fixes that
measurably improved results: a range-aware fold rule for marginal calls into
tight betting ranges (−266 → −165 bb/100 vs TAG), a default bet cap of
0.66× pot (cut stack-shove variance ~6× without killing the mirror signal),
and aggressor targeting (the bot now reads the model of whoever actually bet;
mixed-lineup total −5703.5 → −2800.0 bb at seed 7/1000 hands).

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

External bots (e.g. RLCard/OpenSpiel agents, trained or pretrained) can be
dropped in through the plugin connector and benchmarked head-to-head:

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

Heads-up, 1000 hands each, 10 MC iters, 200bb stacks, seed 7 (rebuy sessions
since PyPokerEngine ends a game at the first bust), after the
aggressor-targeting fix:

| opponent | pokr bb/100 | read |
|---|---|---|
| HonestPlayer | **+5.1** | win vs the equity-aware bot |
| FishPlayer | **+40.5** | strong edge vs a never-folding fish |
| RandomPlayer | **+49.9** | solid edge vs random play |

6-max table (pokr + honest, fish, random, fish, random), 1000 hands, seed 7:

| player | bb/100 |
|---|---|
| **pokr** | **+3.9** |
| HonestPlayer | −45.8 |
| FishPlayer (seat 1) | −50.0 |
| RandomPlayer (seat 2) | +10.0 |
| FishPlayer (seat 3) | +131.8 |
| RandomPlayer (seat 4) | −50.0 |

Takeaway: pokr is profitable against all three external bots heads-up. The
6-max table swings heavily with seat draw and the rebuy truncation (each
200-round session resets stacks), so treat that table as directional. The old
reference numbers (+1.9 / +27.5 / +29.9 heads-up at 2000 hands) predate the
aggressor-targeting fix, which improved all three matchups.

### Measured: pokr vs RLCard (random + self-trained DQN)

`pokr` also plays against **RLCard** agents (a third RL card-game framework,
CSIRO Data61) through the plugin connector: `pokr/rlcard_adapter.py`
translates our engine states into RLCard's no-limit-holdem state/action model
and back, so any RLCard-style policy can sit in a pokr lineup. Two policies
are registered:

- `rlcard` — RLCard's action set played uniformly at random (the translation
  layer's smoke test).
- `rlcard-dqn` — a **self-trained DQN**: RLCard ships no pretrained NLH agent
  (its model zoo only has Leduc-hold'em CFR), so one was trained heads-up vs
  random play in RLCard's own engine. Train/reuse it with:

      python train_rlcard_dqn.py --steps 5000000 --seed 7   # ~2h CPU, checkpoints every 100k steps
      RLCARD_DQN_CKPT=models/rlcard_dqn/dqn_final.pt python -m pokr.bench --lineup rlcard-dqn --seats 2 --hands 2000 --mc-iters 10 --seed 7

Heads-up, 2000 hands each, 10 MC iters, 200bb stacks, seed 7 (checkpoint at
2.1M training steps, `models/rlcard_dqn/`, gitignored):

| opponent | pokr bb/100 | SE | win% | variance (bb²) | read |
|---|---|---|---|---|---|
| RlcardRandom | **+1,545** | 518 | 42.5% | 54k | huge edge vs uniform-shove play |
| RlcardDQN | **+911** | 413 | 27.1% | 34k | solid edge vs the trained agent |

Takeaway: pokr wins decisively against both, but the trained DQN is
meaningfully harder than RLCard's random policy — it cuts pokr's edge by
~40% and drops pokr's hand win rate from 42.5% to 27.1%. It learned to fold
junk and call/shove premiums (it beats random play by ~+8-10 chips/game in
its own evals). Caveats: DQN-vs-random is a shallow agent — it never learns
to defend blind steals, which is exactly what pokr exploits (81% VPIP) — so
this is an honest "pokr vs a genuinely trained (if simple) third-party RL
agent" row, not a test against a strong NLH solver. At 2 SE the DQN edge is
statistically solid (+911 ± ~826); the random row is marginal by SE but
decisive by win rate.

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
