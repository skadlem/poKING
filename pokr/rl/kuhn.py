"""Kuhn poker: the gate an equilibrium algorithm passes before it sees NLHE.

NFSP (and Deep CFR, and anything else aiming at equilibrium) is graded here by
`pokr/rl/exploit.py`, which trains a PPO best response and reports what it
wins. That is a LOWER BOUND read off a noisy training run -- fine for ranking
bots, useless for deciding whether a reservoir sampler has an off-by-one. This
project's headline finding is that its own exploitability proxy under-reported
by ~70x; running an unvalidated equilibrium algorithm against an approximate
metric is that same trap.

So: a game with a known answer. Kuhn poker is three cards, two players, one
betting round, and twelve information sets. Exploitability here is EXACT and
needs no CFR machinery -- with six information sets per player there are only
2**6 = 64 pure strategies each, so a best response is a maximum over an
enumeration. Nothing in this module can be subtly wrong in the way a sampled
estimator can.

Rules. Deck {J, Q, K}. Each player antes 1 and is dealt one card. Player 0 acts
first; each player may pass or bet 1. Pass-pass and bet-call go to showdown
(high card wins the pot); a pass facing a bet folds. Payoffs below are to
player 0, in antes.

The reference equilibrium is a one-parameter family in `alpha` in [0, 1/3]
(`nash`), with game value -1/18 to player 0 (`GAME_VALUE`). Both facts are
asserted in tests/test_kuhn.py, so the harness checks itself before it is asked
to check anything else.

No torch, no numpy, no rng in the pure functions: the whole module is
deterministic and importable anywhere.
"""
from __future__ import annotations

import random
from typing import Callable, Iterable, Mapping, Sequence

PASS, BET = 0, 1
ACTIONS = (PASS, BET)
CARDS = (0, 1, 2)          # J, Q, K
CARD_NAMES = ("J", "Q", "K")

# Every deal is equally likely; the pair is (player 0's card, player 1's card).
DEALS: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in CARDS for b in CARDS if a != b)

# Histories where the hand is over, as "who acts next never happens".
_TERMINALS = frozenset({"pp", "pbp", "pbb", "bp", "bb"})

Strategy = Mapping[str, Sequence[float]]   # info-set key -> (P(pass), P(bet))

# Info-set key is card index + history, e.g. "2pb" is "I hold the king and I am
# facing a bet after checking". Player to act at history h is len(h) % 2.
INFO_SETS_BY_PLAYER: tuple[tuple[str, ...], tuple[str, ...]] = (
    tuple(f"{c}{h}" for h in ("", "pb") for c in CARDS),
    tuple(f"{c}{h}" for h in ("p", "b") for c in CARDS),
)
INFO_SETS: tuple[str, ...] = INFO_SETS_BY_PLAYER[0] + INFO_SETS_BY_PLAYER[1]

GAME_VALUE = -1.0 / 18.0   # to player 0, under any Nash profile

_PURE = ((1.0, 0.0), (0.0, 1.0))


def player_to_act(history: str) -> int:
    return len(history) % 2


def is_terminal(history: str) -> bool:
    return history in _TERMINALS


def info_set(history: str, cards: Sequence[int]) -> str:
    """The acting player's information set: their own card plus the history."""
    return f"{cards[player_to_act(history)]}{history}"


def terminal_payoff(history: str, cards: Sequence[int]) -> float:
    """Payoff to PLAYER 0, in antes. Raises on a non-terminal history."""
    c0, c1 = cards[0], cards[1]
    win = 1.0 if c0 > c1 else -1.0
    if history == "pp":
        return win
    if history == "pbp":
        return -1.0          # player 0 folded
    if history == "bp":
        return 1.0           # player 1 folded
    if history in ("pbb", "bb"):
        return 2.0 * win
    raise ValueError(f"not a terminal history: {history!r}")


def _walk(sigma: Strategy, history: str, cards: Sequence[int]) -> float:
    if is_terminal(history):
        return terminal_payoff(history, cards)
    probs = sigma[info_set(history, cards)]
    return (probs[PASS] * _walk(sigma, history + "p", cards)
            + probs[BET] * _walk(sigma, history + "b", cards))


def expected_value(sigma: Strategy) -> float:
    """Expected payoff to player 0 under a full strategy profile."""
    return sum(_walk(sigma, "", d) for d in DEALS) / len(DEALS)


def best_response_value(sigma: Strategy, player: int) -> float:
    """Value to `player` of their best response to `sigma`.

    Exhaustive over that player's 2**6 pure strategies. A best response can
    always be taken pure, so the maximum over pure strategies IS the best
    response value -- exact, with no iteration to converge and no tolerance to
    tune. `sigma` only needs to be correct at the OTHER player's info sets.
    """
    keys = INFO_SETS_BY_PLAYER[player]
    best = float("-inf")
    profile = dict(sigma)
    for bits in range(1 << len(keys)):
        for i, key in enumerate(keys):
            profile[key] = _PURE[(bits >> i) & 1]
        ev0 = expected_value(profile)
        best = max(best, ev0 if player == 0 else -ev0)
    return best


def exploitability(sigma: Strategy) -> float:
    """NashConv / 2, in antes per hand: the average a best responder gains.

    Zero exactly at a Nash equilibrium. This is the number an NFSP run on Kuhn
    has to drive down, and unlike pokr/rl/exploit.py's probe it is not a bound
    or an estimate.
    """
    return (best_response_value(sigma, 0) + best_response_value(sigma, 1)) / 2.0


def nash(alpha: float = 1.0 / 3.0) -> dict[str, tuple[float, float]]:
    """The known equilibrium family, parameterised by alpha in [0, 1/3].

    Player 0 bluffs the jack with probability alpha and bets the king with 3
    alpha, keeping the bluff-to-value ratio fixed; every alpha in range is an
    equilibrium with the same value. Player 1's strategy does not depend on
    alpha.
    """
    if not 0.0 <= alpha <= 1.0 / 3.0 + 1e-12:
        raise ValueError(f"alpha must lie in [0, 1/3], got {alpha}")
    return {
        # player 0, opening
        "0": (1 - alpha, alpha),                    # J: bluff at alpha
        "1": (1.0, 0.0),                            # Q: always check
        "2": (1 - 3 * alpha, 3 * alpha),            # K: bet at 3 alpha
        # player 0, facing a bet after checking
        "0pb": (1.0, 0.0),                          # J: fold
        "1pb": (2 / 3 - alpha, 1 / 3 + alpha),      # Q: call at alpha + 1/3
        "2pb": (0.0, 1.0),                          # K: call
        # player 1, facing a check
        "0p": (2 / 3, 1 / 3),                       # J: bluff 1/3
        "1p": (1.0, 0.0),                           # Q: check
        "2p": (0.0, 1.0),                           # K: bet
        # player 1, facing a bet
        "0b": (1.0, 0.0),                           # J: fold
        "1b": (2 / 3, 1 / 3),                       # Q: call 1/3
        "2b": (0.0, 1.0),                           # K: call
    }


def uniform() -> dict[str, tuple[float, float]]:
    """Coin-flip everywhere. The trivial baseline an algorithm must beat."""
    return {key: (0.5, 0.5) for key in INFO_SETS}


# -- episode generation (what a learner consumes) -------------------------

Policy = Callable[[str, int], int]   # (info-set key, player) -> action


def play(policy0: Policy, policy1: Policy, rng: random.Random,
         cards: Sequence[int] | None = None
         ) -> tuple[list[tuple[int, str, int]], tuple[float, float]]:
    """One hand. Returns (steps, payoffs), steps as (player, info_set, action).

    `cards` forces a deal, which is what makes a test deterministic without
    reaching into the rng. Payoffs are (to player 0, to player 1) and always
    sum to zero.
    """
    deal = tuple(cards) if cards is not None else rng.choice(DEALS)
    history = ""
    steps: list[tuple[int, str, int]] = []
    while not is_terminal(history):
        player = player_to_act(history)
        key = info_set(history, deal)
        action = (policy0 if player == 0 else policy1)(key, player)
        if action not in ACTIONS:
            raise ValueError(f"policy returned {action!r}, expected 0 or 1")
        steps.append((player, key, action))
        history += "pb"[action]
    u0 = terminal_payoff(history, deal)
    return steps, (u0, -u0)


def sampling_policy(sigma: Strategy, rng: random.Random) -> Policy:
    """Turn a strategy table into a Policy that samples from it."""
    def policy(key: str, player: int) -> int:
        return BET if rng.random() < sigma[key][BET] else PASS
    return policy


def average_of(strategies: Iterable[Strategy]) -> dict[str, tuple[float, float]]:
    """Uniform average over strategy tables -- the fictitious-play average, and
    the shape an NFSP reservoir is approximating. Handy for testing that the
    averaging itself is right before a neural net is involved."""
    tables = list(strategies)
    if not tables:
        raise ValueError("no strategies to average")
    return {key: (sum(t[key][PASS] for t in tables) / len(tables),
                  sum(t[key][BET] for t in tables) / len(tables))
            for key in INFO_SETS}
