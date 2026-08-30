# pokr

A research-grade 6-max No-Limit Texas Hold'em poker bot with dynamic risk
assessment, opponent modeling, and bot/mirror detection. Spec:
`docs/superpowers/specs/2026-08-06-poker-bot-design.md`.

## Benchmark results

Reference run: seed 7, 2000 hands per matchup, `mc_iters=10`, `--fast` (numba
equity path), 6-max, 200bb buy-in, no rake, fresh bot per matchup, after the
aggressor-targeting fix. SE = std error of bb/100 (sqrt(var/hands) x 100);
rows whose |bb/100| is within ~2 SE of 0 are statistically unresolved.

| matchup | bb/100 | SE | win% | variance (bb²) | read |
|---|---|---|---|---|---|
| calling station | **+683** | 124 | 7.5% | 3.1k | strong edge vs a never-folding opponent |
| tight-aggressive | **−9.9** | 6.9 | 18.4% | 9.5 | below blind cost after the OOP blind fold (50k: −10.2) |
| maniac | +3,411 | 7020 | 1.5% | 9.9M | unresolved at 2000 hands (SE ≈ 2x estimate); 50k: +18.8k |
| random | +5,426 | 2001 | 6.2% | 801k | positive but unresolved (fat tails) |
| self-play | −340 | 175 | 14.9% | 6.1k | ≈ 1.9 SE below 0 at 2000 hands; 50k: −140, unresolved |
| leak hunter | **+8.8** | 5.7 | 32.2% | 6.4 | exploitability proxy: small positive edge |
| ppo (self-trained) | −231 | 1092 | 11.2% | 239k | unresolved, and stays that way: at 20k hands it reads +2,812 ± 1,763 — the sign flips. Five learned agents at one table make this row meaningless; the resolved head-to-head is the duplicate-deck run below |

Caveats: with `mc_iters=10` equity estimates are coarse, and matchups whose
variance exceeds ~10⁵ bb² need 50k+ hands to resolve (see HANDOFF.md for the
long-run reference). Only the calling-station, tight-aggressive, self-play,
and leak-hunter rows are statistically meaningful at 2000 hands. Fixes that
measurably improved results: a range-aware fold rule for marginal calls into
tight betting ranges (−266 → −165 bb/100 vs TAG), a default bet cap of
0.66× pot (cut stack-shove variance ~6× without killing the mirror signal),
aggressor targeting (the bot now reads the model of whoever actually bet;
mixed-lineup total −5703.5 → −2800.0 bb at seed 7/1000 hands), and an
out-of-position blind fold (marginal preflop calls from SB/BB in 6-max+:
TAG −15.4 → −9.9, self-play 50k −426 → −140; heads-up excluded — it
regressed PyPokerEngine's HonestPlayer matchup there).

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

## Head-to-head on duplicate decks

`pokr/duplicate.py` compares two bots with less variance than a plain matchup:
each deck is played twice with the two heroes swapping seats, so a hero holds
both sets of hole cards and the deal's luck cancels inside its own score.

    python -m pokr.duplicate --a rl --b self --lineup "" --hands 20000 --mc-iters 150 --fast

`--a`/`--b` take the same abbreviations as `--lineup` (`--lineup ""` is a
heads-up match). The report gives each side's bb/100 with a 2 SE bar, the gap,
whether it is resolved, and how much the pairing actually tightened the
estimate.

That last number is worth reading rather than assuming. Duplicate scoring buys
1.0x-3.5x here, not the 5-10x it buys in duplicate bridge, and how much depends
on how alike the two bots are: 85% of NLHE hands finish under 4bb while the top
1% carry 42% of all variance, and those big pots only happen when *both*
players choose to build one. Two similar agents mirror well (3.5x measured, so
12x fewer hands for the same resolution); a heuristic against a scripted
archetype barely mirrors at all (1.1x). Hands are cheap (~40k in 12s on the
numba path), so resolve close matchups with hand count rather than expecting
the pairing to do it.

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
4. **Learned agents**: a PPO agent trained in PyTorch inside this engine
   (`pokr/rl/`, see below), plus RLCard's DQN as an external reference. These
   test the heuristic against opponents that were optimized against it rather
   than hand-written.

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

### Measured: pokr vs its own PPO agent (PyTorch, trained in-engine)

The RLCard DQN above is trained in *RLCard's* engine and imported through an
adapter. `pokr/rl/` instead trains an agent **inside this engine**, so there is
no train/eval mismatch and the reward is literally the benchmark metric: the
hand's net bb (`HandResult.winnings[seat] / big_blind`), the same number
`bench.run_matchup` reports.

The agent plays as an ordinary `Strategy` and records itself — `decide()`
appends (observation, action mask, action) and `on_hand_end()` stamps the
hand's result on the trajectory as the terminal reward — so `bench.play_session`
doubles as the rollout collector and no gym wrapper or thread inversion is
needed. The 160-dim observation reuses the numba Monte Carlo equity
(`pokr/_fastcards.py`) and the opponent models (`pokr/models.py`) as features,
which is what lets a 109k-parameter MLP learn from ~1M hands instead of ~100M.
Nine discrete actions (fold, check/call, six pot-fraction raises, all-in) are
masked against the engine's own legal ranges rather than clamped into them.

Opponents each iteration are drawn from a pool of {calling station,
tight-aggressive, maniac, random, PokerBot, **frozen past selves**}, with table
size sampled from {2, 6}. The frozen snapshots (`pokr/rl/league.py`, one every
25 iterations) are what make the difference; see the progression below. This is
not live self-play, which cycles rather than converging in an
imperfect-information game — the league opponents are frozen, so each iteration
still faces a stationary environment.

    python train_rl.py --iters 600 --hands-per-iter 2000 --seats 2,6 --fast   # ~35 min CPU
    python -m pokr.duplicate --a rl --b self --lineup "" --hands 20000 --mc-iters 150 --fast

Trained 600 iterations x 2000 hands (1.2M hands, 35 min on one CPU core).
Head-to-head vs the heuristic on duplicate decks, PokerBot at its default 150
MC iterations:

| table | PokrPPO bb/100 | PokerBot bb/100 | gap (2 SE) | read |
|---|---|---|---|---|
| heads-up (40k hands each) | **+180.4 ± 32.6** | −180.4 | +360.9 ± 65.1 | **PPO wins**, ~11 SE |
| 6-max vs tag, tag, cs, random (16k each) | **+753.7 ± 127.6** | +121.1 ± 46.4 | +632.6 ± 132.5 | **PPO wins**, ~10 SE |

What actually moved the number, measured heads-up against the same
full-strength heuristic over 40k hands each:

| training setup | heads-up vs heuristic |
|---|---|
| 6-max only, opponents at 10 MC iters | −225.1 ± 44.2 |
| + tables sampled from {2, 6}, opponents at 150 MC iters | −122.2 ± 35.8 |
| + frozen past selves in the pool | **+180.4 ± 32.6** |

Each change is worth more than it looks. Training against `PokerBot` weakened
to 10 MC iterations produced an agent that drew with the weak version and lost
to the real one — `--opp-mc-iters` now defaults to PokerBot's own 150. Table
size has to be trained rather than assumed: the first agent only ever played
6-max. And the league is what broke a cycle — before it, the second agent beat
its predecessor's record against the heuristic while *losing* to that
predecessor head-to-head (−288.5 ± 37.8), ordinary rock-paper-scissors
non-transitivity. The league agent beats both (+586.8 ± 46.8 vs the second,
+262.2 ± 61.0 vs the first) as well as the heuristic.

The league agent also plays a recognizably different game: VPIP 50.1% / PFR
41.7% / postflop aggression 0.95, against the heuristic's 52.8% / 16.1% / 0.65.
Similar looseness, but it raises almost everything it plays instead of calling.

Caveats, in order of how much they matter:

- **It is more exploitable than the heuristic.** Against the leak hunter (the
  adaptive exploitability proxy) over 20k hands it runs −136.4 ± 199.8 —
  unresolved, but negative and with ~50x the variance of the heuristic's
  +14.8 ± 3.9. Beating a fixed pool is not the same as being hard to counter,
  and this is the number to fix next.
- **The ring win is narrow in what it proves.** Two of the four opponent seats
  are a calling station and a random bot; the agent is substantially
  out-extracting weak players, which is what its training pool rewarded.
- **"Better" is not a total order.** The non-transitivity above was resolved
  among these four checkpoints, not in general; a single matchup is not a
  ranking.
- Beating the heuristic head-to-head is not the same as playing well. Both are
  far from equilibrium, and neither has been tested against a strong NLH
  solver.

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
- `pokr/duplicate.py` — duplicate-deck head-to-head (variance-reduced A vs B)
- `pokr/connector.py` — plugin registry for external bots
- `pokr/rl/encode.py` — GameState → observation vector, action mask, decode
- `pokr/rl/net.py` — policy/value MLP (PyTorch)
- `pokr/rl/agent.py` — the agent as a Strategy, recording its own trajectories
- `pokr/rl/ppo.py` — PPO update (GAE, clipped surrogate, KL early-stop)
- `pokr/rl/plugin.py` — connector plugin for a trained checkpoint
- `train_rl.py` — training loop (rollouts via the benchmark harness)

## Tests

    python -m pytest -q
