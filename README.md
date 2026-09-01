# pokr

[![tests](https://github.com/skadlem/poKING/actions/workflows/tests.yml/badge.svg)](https://github.com/skadlem/poKING/actions/workflows/tests.yml)
[![writeup: the 2,800x variance bug](https://img.shields.io/badge/writeup-the_2%2C800%C3%97_variance_bug-d8a24a)](https://skadlem.github.io/poKING/variance-bug.html)

A 6-max No-Limit Hold'em engine, a hand-tuned bot, a PPO agent trained inside
that engine, and an NFSP (average-policy) pipeline validated against exact
exploitability on Kuhn — built mainly as an exercise in measuring a strategy's
edge honestly under very heavy noise.

## The numbers (all reproducible, all seeded)

| claim | measurement | how |
|---|---|---|
| PPO agent beats the heuristic | **+604.0 ± 36.1 bb/100** heads-up, 30k duplicate hands | `python -m pokr.duplicate --a rl --b self --lineup "" --hands 20000 --mc-iters 150 --fast` |
| least exploitable bot measured | **670.6 ± 68.8** bb/100 (heuristic: 1019.7) | `python -m pokr.rl.exploit --target rl --iters 200` |
| 4.4x the heuristic's edge vs a third-party trained DQN | **+979.5 ± 97.9** (heuristic: +221.6) | duplicate vs the RLCard DQN, 3000 decks |
| profitable vs an external engine's bots | +5.1 / +40.5 / +49.9 bb/100 (Honest/Fish/Random, heads-up) | `python -m pokr.ppe_compare --hands 2000 --mc-iters 10 --seed 7` |
| NFSP average policy, 30-round ladder-B campaign | 737.5 ± 86.0 bb/100 exploitable (seed 7) — the honest result: **not yet competitive** with the PPO's 670.6 | `python -m pokr.rl.exploit --target nfsp --iters 120` |

**What it is not.** 670 bb/100 exploitable is catastrophic in absolute terms —
a best response wins ~6.7 big blinds per hand, taking a 100bb stack every ~15
hands. Nothing here is close to equilibrium; the scripted opponents are weak,
and the honest summary is "a good exploiter of weak static opponents, measured
carefully." Closing that gap is what the NFSP workstream (below) is for.

**Claim discipline.** Every headline number here is seed-replicated or
withdrawn. When a result failed to replicate across seeds, the README says so
and the number is gone — see the retracted "2.9x less exploitable" note in
`HANDOFF.md` 0.4. This repo treats a measurement bug as more interesting than
a win: [The 2,800× Variance Bug](https://skadlem.github.io/poKING/variance-bug.html)
is the full story of a rebuy rule that inflated variance ~2,800x and left two
matchups statistically unresolved for the project's entire history.

**What this project is really about.** Poker win rates are dominated by
variance, and most of the interesting work here turned out to be measurement
rather than modelling:

- **A benchmark bug inflated measurement variance 2,800x.** The rebuy rule
  topped up busted players without ever capping a winner, so stacks drifted
  from 100bb to ~300bb across a session and late hands were played at the
  wrong depth. Two matchups had been statistically unresolved for the
  project's entire history; both resolve at 2,000 hands once depth is fixed.
  The point estimates moved too — the headline "+3,411 bb/100 vs a maniac"
  was mostly compounding deep-stack pots, not a win rate (it is +290).
- **The project's own exploitability proxy under-reported by ~70x.** A
  hand-written adaptive opponent rated the heuristic bot as nearly
  unexploitable (+14.8 bb/100 in the bot's favour). A best response trained
  from scratch takes 1,019.7 bb/100 off it.
- **Training against that proxy made things worse.** Adding it to the
  training pool flipped its own column (−136 → +1,203 bb/100) while making
  the agent nearly twice as exploitable by a real best response
  (879 → 1,294). It is not in the default pool.
- **Two variance-reduction techniques were implemented and measured honestly
  rather than assumed.** Duplicate decks buy 1.0-3.5x here, not the 5-10x
  they buy in bridge, and all-in EV buys 1.0-1.23x — both far less than the
  depth fix above. The reasoning and the dead ends are recorded in the
  modules so they are not rediscovered.

**The result.** A PPO agent (110k parameters, ~15 min of CPU training) and a
30-round NFSP campaign against it — both in the table above, both reproducible
with the commands there.

Spec: `docs/superpowers/specs/2026-08-06-poker-bot-design.md`. The NFSP
design note is `docs/design/nfsp.md`; session history and the roadmap state
are in `HANDOFF.md`.

## Benchmark results

Reference run: seed 7, 2000 hands per matchup, `mc_iters=10`, `--fast` (numba
equity path), `--reset-stacks`, 6-max, 100bb stacks, no rake, fresh bot per
matchup. SE = std error of bb/100 (sqrt(var/hands) x 100); rows whose |bb/100|
is within ~2 SE of 0 are statistically unresolved.

| matchup | bb/100 | SE | win% | variance (bb²) | read |
|---|---|---|---|---|---|
| calling station | **+518** | 83 | 7.4% | 1.4k | strong edge vs a never-folding opponent |
| tight-aggressive | −13.7 | 6.9 | 17.9% | 9.6 | at ~2 SE; the one archetype that beats us |
| maniac | **+290** | 132 | 1.6% | 3.5k | resolved at 2000 hands |
| random | **+352** | 109 | 4.7% | 2.4k | resolved at 2000 hands |
| self-play | −52 | 83 | 16.9% | 1.4k | unresolved; no longer clearly negative |
| leak hunter | **+12.1** | 5.5 | 32.4% | 6.1 | weak exploitability proxy — see Exploitability |
| ppo (self-trained) | see below | | | | five learned agents at one table; the resolved comparison is the duplicate-deck run |

**These numbers were re-anchored with `--reset-stacks`.** Every hand now starts
at the nominal 100bb. The previous table carried stacks between hands, and
because `bench._rebuy` tops up busted players without ever capping a winner,
chips inflated across a session (measured: 1200 → 1600 chips over 2000 hands,
max stack 293bb), so late hands were played 2-3x deeper than intended. That
was the single largest source of variance in this project — far larger than
anything duplicate decks or all-in EV recovers:

| matchup | carry-over (old) | fixed depth (new) | variance |
|---|---|---|---|
| maniac | +3,411 ± 14,041 *(unresolved)* | **+290 ± 264** | 9.86M → 3.5k (**2,800x**) |
| random | +5,426 ± 4,002 *(unresolved)* | **+352 ± 218** | 801k → 2.4k (**338x**) |
| self-play | −340 ± 349 | −52 ± 167 | 6.1k → 1.4k (4.4x) |
| calling station | +683 ± 247 | +518 ± 165 | 3.1k → 1.4k (2.2x) |

The maniac and random rows had been unresolved since the project started;
both resolve at 2000 hands once depth is held fixed. Note the estimates move
too, not just the error bars — the old +3,411 vs a maniac was mostly the
compounding of deep-stack pots, not a real win rate. Pass `--reset-stacks` to
`python -m pokr.bench` to reproduce; omit it for the historical numbers.

Fixes that measurably improved results: a range-aware fold rule for marginal
calls into tight betting ranges, a default bet cap of 0.66x pot (cut
stack-shove variance ~6x without killing the mirror signal), aggressor
targeting (the bot reads the model of whoever actually bet), and an
out-of-position blind fold (marginal preflop calls from SB/BB in 6-max+;
heads-up excluded — it regressed PyPokerEngine's HonestPlayer matchup there).

## Install

    pip install -r requirements.txt
    pytest -q                              # 358 tests, ~1.5 min

Torch is only needed for the PPO agent — the engine, the heuristic bot and all
benchmarks import it lazily and run without it. For a CPU-only build (the
agent is a 110k-parameter MLP; the bottleneck is the pure-Python engine, not
matmuls):

    pip install torch --index-url https://download.pytorch.org/whl/cpu

`rlcard` is optional and used only for the external DQN benchmark; its tests
skip when it is absent.

## Run a benchmark

    python -m pokr.bench --hands 2000 --seed 7 --mc-iters 10

Reports BB/100, win rate, and variance for each matchup: calling station,
tight-aggressive, maniac, random, self-play, and the leak hunter
(exploitability proxy). `--mc-iters` sets the Monte Carlo equity iterations
per decision (default 150; use 10-30 for fast exploratory runs).

`--reset-stacks` plays every hand at the buy-in instead of carrying stacks
over. Carrying over (the default, and what the table above was measured with)
tops up busted players but never caps winners, so chips inflate across a
session: measured with six calling stations, 1200 → 1600 chips over 2000
hands, with a max stack of 293bb. Late hands in every matchup therefore run
2-3x deeper than the nominal 100bb, which is part of why the high-variance
rows never resolve.

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

`--allin-ev` stacks a second, independent reduction on top: when betting
locked up before the board was complete, the realized runout is replaced by
its expectation over every completion (`pokr/allin_ev.py`, exact enumeration
under 200k board completions, Monte Carlo above). The two compose — pairing
removes which cards you were dealt, all-in EV removes how they ran out. Its
value depends entirely on how often a matchup produces pre-river all-ins:
0% of hands in heuristic-vs-tight-aggressive heads-up (the heuristic caps
bets at 0.66x pot and simply never gets there, so 1.00x), 10% in PPO-vs-
heuristic heads-up (1.04x), 26% in a shove-heavy 6-max lineup (1.23x). It is
unbiased — over 4000 hands the adjusted and realized means differ by 39 bb/100
against a 2 SE of 240.

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

The PPO agent (`pokr/rl/`) was measured against the same DQN on duplicate
decks, heads-up, 3000 decks — it takes roughly 4.4x the edge the heuristic
does off the same third-party opponent:

| | vs RlcardDQN (bb/100, 2 SE) |
|---|---|
| **PPO agent** | **+979.5 ± 97.9** |
| PokerBot (heuristic) | +221.6 ± 103.1 |

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

    python train_rl.py --iters 600 --hands-per-iter 2000 --seats 2,6 --fast
    python -m pokr.duplicate --a rl --b self --lineup "" --hands 20000 --mc-iters 150 --fast
    python -m pokr.rl.exploit --target rl --iters 200        # exploitability

Rollouts are collected across processes (`--workers`, default 8, via
`pokr/rl/rollout.py`): measured 553 hands/s single-process against 3,305 at 8
workers on a 10-core box, which takes a 600-iteration run from ~35 minutes to
~19. Workers rebuild the network from a state dict and their opponents from
names, since neither a live module nor a factory lambda survives pickling;
each pins `torch.set_num_threads(1)` and reseeds torch's RNG, because `fork`
copies the parent's global RNG state byte-for-byte into every child.

Trained 600 iterations x 2000 hands (1.2M hands, ~15 min across 8 workers).
Head-to-head vs the heuristic on duplicate decks, PokerBot at its default 150
MC iterations:

| table | PokrPPO bb/100 | PokerBot bb/100 | gap (2 SE) | read |
|---|---|---|---|---|
| heads-up (40k hands each) | **+180.4 ± 32.6** | −180.4 | +360.9 ± 65.1 | **PPO wins**, ~11 SE |
| 6-max vs tag, tag, cs, random (16k each) | **+753.7 ± 127.6** | +121.1 ± 46.4 | +632.6 ± 132.5 | **PPO wins**, ~10 SE |

What actually moved the number, each measured heads-up against the same
full-strength heuristic on duplicate decks:

| training setup | heads-up vs heuristic | exploitability |
|---|---|---|
| 6-max only, opponents at 10 MC iters | −225.1 ± 44.2 | |
| + tables sampled from {2, 6}, opponents at 150 MC iters | −122.2 ± 35.8 | |
| + frozen past selves in the pool | +180.4 ± 32.6 | 879.4 ± 101.5 |
| + `--reset-stacks` (train at the depth you score at) | **+604.0 ± 36.1** | **670.6 ± 68.8** |
| (+ leak hunter in the pool — rejected, see Exploitability) | +238.1 ± 33.0 | 1294.3 ± 86.9 |

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

### Exploitability

Beating a fixed pool is not the same as being hard to counter, so
`pokr/rl/exploit.py` measures the latter directly: it trains a fresh PPO agent
from scratch against a frozen target and reports what that exploiter wins.
That is a lower bound — a real best response does at least as well — so a big
number proves a leak, while a small one is only evidence of absence in
proportion to how hard the exploiter tried.

    python -m pokr.rl.exploit --target rl --iters 200

Each exploiter here trained 120 iterations (240k hands) heads-up, then was
scored over 4,000 duplicate decks:

| target | exploitability lower bound | leak hunter says |
|---|---|---|
| **PPO agent (shipped: league pool, fixed depth)** | **670.6 ± 68.8** | +468.1 ± 58.0 |
| PPO agent (league pool, carry-over depth) | 879.4 ± 101.5 | −136.4 ± 199.8 |
| PokerBot (the heuristic) | 1019.7 ± 89.9 | +14.8 ± 3.9 |
| PPO agent (leak hunter in pool) | 1294.3 ± 86.9 | +1202.7 ± 268.7 |

Three things fall out of that table, and two of them are uncomfortable.

**The leak hunter is a very weak proxy.** It rates the heuristic as nearly
unexploitable (+14.8 bb/100 in its favour) while a best response trained for
240k hands takes 1019.7 bb/100 off it — under-reporting by ~70x. It only knows
a handful of counter-rules, so passing it means little. The rows above are the
number to trust.

**Training against the proxy corrupts it.** Adding the leak hunter to the
opponent pool did flip that column (−136 → +1203) — and made the agent
*more* exploitable by a real best response (879 → 1294). It learned to beat
one specific counter-strategy at the cost of general robustness, which is
what training on your own test set looks like. The shipped checkpoint is the
league-pool agent, not the leak-pool one, on the strength of this table rather
than the leak-hunter column.

**Every bot here is hugely exploitable.** ~900-1300 bb/100 means none of them
is remotely near equilibrium; the trained agent is somewhat better than the
heuristic on this axis, and that is the whole claim.

Caveats, in order of how much they matter:

- **The ring win is narrow in what it proves.** Two of the four opponent seats
  are a calling station and a random bot; the agent is substantially
  out-extracting weak players, which is what its training pool rewarded.
- **"Better" needs the exploitability column, not just the win rate.** The
  leak-pool agent beats the shipped one against the heuristic on some
  measures and is nearly twice as exploitable. Read both.
- **"Better" is not a total order.** The shipped agent beats the heuristic
  3.3x harder than its predecessor and is 24% less exploitable, yet loses to
  that predecessor head-to-head by −35.8 ± 34.7. A single matchup is not a
  ranking.
- Beating the heuristic head-to-head is not the same as playing well. Neither
  has been tested against a strong NLH solver.

### NFSP: the average-policy pipeline (and its honest negative result)

Every bot above is ~670-1020 bb/100 exploitable, which is far from
equilibrium. `pokr/rl/` grew a second algorithm family to attack that axis
directly: NFSP, an average-policy method whose convergence guarantee is
provable — so the pipeline was built in strict order with a validation gate
that cannot lie:

1. **Kuhn gate (`pokr/rl/fsp.py`)** — the same reservoir-sampling + masked-CE
   fitting that NFSP uses, validated against Kuhn poker's *exact*
   exploitability (a maximum over 64 pure strategies; ground truth, not an
   estimate). The neural path must drive exploitability below 0.05 — and does,
   on five seeds.
2. **A second diagnostic earned its keep**: the first NLHE campaign
   (30 rounds, ~2.4M hands) trained a policy measurably indistinguishable from
   a coin flip over its legal-action mask — CE loss 1.641 vs the analytic
   uniform-mask floor 1.712 — while every component test AND the Kuhn gate
   stayed green. The gate validated the algorithm; the bug was wiring: a
   uniform reservoir had silently replaced the linearly-weighted one the gate
   validated, so each round's random-init restart of the best-response oracle
   was folded into the average forever. Fixed with a weighted reservoir,
   round-level weights, and 75% per-round burn-in (post-mortem: `HANDOFF.md`
   0.6 step 10; lesson: *a green component gate does not validate
   integration*).
3. **Campaign #2, measured end to end** (`train_nfsp.py`, 30 rounds x 40
   iterations): the loop is healthy — per-round best-response curves fall from
   +1696 to negative by round 18, fit loss holds ~0.4 nats under the
   coin-flip floor. The converged probe says Pi_last is **737.5 ± 86.0**
   bb/100 exploitable (seed 7; 1011.1 ± 86.5 on seed 11) versus the shipped
   PPO's 670.6 — and it loses duplicate heads-up to the heuristic by −285.9 ±
   34.7. **The success condition ("well below 670.6") failed honestly.** At 30
   fictitious-play moves the average is still dominated by weak early best
   responses; NFSP's ceiling only rises with rounds, and 30 is a fraction of
   what the method needs. The checkpoints and logs are kept locally as
   negative evidence.

What this buys the project: an equilibrium-approximation path with a validated
core, a diagnostic (loss-vs-entropy-floor) that turns "converged" claims into
arithmetic, and a number that says how many more rounds this scale of compute
buys — which is exactly the question the next campaign runs against.

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
- `pokr/allin_ev.py` — all-in EV adjustment (expectation over board runouts)
- `pokr/connector.py` — plugin registry for external bots
- `pokr/rl/encode.py` — GameState → observation vector, action mask, decode
- `pokr/rl/net.py` — policy/value MLP (PyTorch)
- `pokr/rl/agent.py` — the agent as a Strategy, recording its own trajectories
- `pokr/rl/ppo.py` — PPO update (GAE, clipped surrogate, KL early-stop)
- `pokr/rl/league.py` — frozen past selves used as training opponents
- `pokr/rl/rollout.py` — multiprocess rollout collection
- `pokr/rl/exploit.py` — best-response probe (exploitability lower bound)
- `pokr/rl/kuhn.py` — exact-exploitability Kuhn harness (the NFSP gate)
- `pokr/rl/fsp.py` — WeightedReservoir (A-Res) + the Kuhn gate itself
- `pokr/rl/memory.py` — uniform/floored reservoirs
- `pokr/rl/avg_policy.py` — the average-policy net (masked-CE fit)
- `pokr/rl/nfsp.py` — NFSPStrategy (Pi as an engine citizen)
- `pokr/rl/plugin.py` — connector plugin for a trained checkpoint
- `train_rl.py` — PPO training loop (rollouts via the benchmark harness)
- `train_nfsp.py` — ladder-B NFSP outer loop (BR harvest -> average fit)

## Tests

    python -m pytest -q
