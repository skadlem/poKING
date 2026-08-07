# pokr

A research-grade 6-max No-Limit Texas Hold'em poker bot with dynamic risk
assessment, opponent modeling, and bot/mirror detection. Spec:
`docs/superpowers/specs/2026-08-06-poker-bot-design.md`.

## Install

    pip install -r requirements.txt

## Run a benchmark

    python -m pokr.bench --hands 2000 --seed 7

Reports BB/100, win rate, and variance for each matchup: calling station,
tight-aggressive, maniac, random, self-play, and the leak hunter
(exploitability proxy).

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
