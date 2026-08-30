"""Observation encoding and action decoding for the RL agent.

Pure functions over the engine's GameState: no torch, no rng, no I/O, so the
whole layer is deterministic and testable without a training run. The agent
(pokr/rl/agent.py) supplies the two derived inputs -- Monte Carlo equity and
the opponent-model summary -- rather than computing them here, which keeps
encoding side-effect free and lets tests pin exact vectors.

Chip quantities are normalized by the hero's stack at the start of the hand
(stack + committed), not by the big blind: GameState does not carry the blind
level, and the ratio is scale-invariant, so one encoding works at any stakes.
"""
from __future__ import annotations

import numpy as np

from ..engine import GameState, LegalAction
from ..models import OpponentSummary
from ..strategy import Action, ActionType

# -- action space ---------------------------------------------------------

# Raise sizes as a fraction of the pot the hero would face after calling.
RAISE_FRACTIONS = (0.33, 0.5, 0.66, 1.0, 1.5, 2.0)

ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
_FIRST_RAISE = 2
ACTION_ALL_IN = _FIRST_RAISE + len(RAISE_FRACTIONS)
NUM_ACTIONS = ACTION_ALL_IN + 1

ACTION_NAMES: tuple[str, ...] = (
    ("fold", "check/call")
    + tuple(f"raise {f:g}p" for f in RAISE_FRACTIONS)
    + ("all-in",)
)

# -- observation layout ---------------------------------------------------

MAX_SEATS = 6
_STREETS = ("preflop", "flop", "turn", "river")
_SEAT_FEATURES = 6

_LAYOUT: tuple[tuple[str, int], ...] = (
    ("hole", 52),                                 # multi-hot hole cards
    ("board", 52),                                # multi-hot community cards
    ("street", len(_STREETS)),                    # one-hot street
    ("scalars", 7),                               # pot/stack/odds/spr geometry
    ("seats", _SEAT_FEATURES * (MAX_SEATS - 1)),  # opponents, relative order
    ("position", MAX_SEATS + 1),                  # seat vs button + to-act
    ("equity", 2),                                # MC equity + present flag
    ("opponent", 6),                              # target opponent model
)
OBS_DIM = sum(n for _, n in _LAYOUT)

_offsets: dict[str, slice] = {}
_pos = 0
for _name, _n in _LAYOUT:
    _offsets[_name] = slice(_pos, _pos + _n)
    _pos += _n
OBS_SLICES: dict[str, slice] = dict(_offsets)


def card_index(card) -> int:
    """0..51 index for a Card: (rank - 2) * 4 + suit."""
    return (card.rank - 2) * 4 + card.suit


# -- action legality ------------------------------------------------------


def _raise_legal_action(state: GameState) -> LegalAction | None:
    for la in state.legal_actions:
        if la.action_type in (ActionType.BET, ActionType.RAISE):
            return la
    return None


def raise_target(state: GameState, player_id: int, action_idx: int) -> int | None:
    """Raise-to amount for a sizing action, or None if that size is illegal.

    Pot-fraction convention: call first, then raise by `frac` of the resulting
    pot, so raise_to = street_committed + to_call + frac * (pot + to_call).
    With to_call == 0 this reduces to a plain pot-fraction bet.

    A size is legal only if it already lands inside the engine's
    [min_amount, max_amount] raise-to range -- it is never clamped in. Clamping
    would collapse several sizes onto the same amount and make the policy's
    choice ambiguous; masking keeps every live action distinct, and ALL_IN
    always covers the top of the range.
    """
    if action_idx < _FIRST_RAISE or action_idx > ACTION_ALL_IN:
        return None
    la = _raise_legal_action(state)
    if la is None:
        return None
    if action_idx == ACTION_ALL_IN:
        # For a BET the engine's max_amount is the stack rather than
        # street_committed + stack, so this is all-in up to a BB-option chip.
        return la.max_amount
    p = state.players[player_id]
    to_call = state.current_bet - p.street_committed
    frac = RAISE_FRACTIONS[action_idx - _FIRST_RAISE]
    target = p.street_committed + to_call + int(frac * (state.pot + to_call))
    if la.min_amount <= target <= la.max_amount:
        return target
    return None


def action_mask(state: GameState, player_id: int) -> np.ndarray:
    """Boolean mask over NUM_ACTIONS. Never all-False: the engine always offers
    CHECK or CALL, so ACTION_CHECK_CALL is always available."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    types = {la.action_type for la in state.legal_actions}
    if ActionType.FOLD in types:
        mask[ACTION_FOLD] = True
    if types & {ActionType.CHECK, ActionType.CALL}:
        mask[ACTION_CHECK_CALL] = True
    for i in range(_FIRST_RAISE, NUM_ACTIONS):
        mask[i] = raise_target(state, player_id, i) is not None
    return mask


def decode(state: GameState, player_id: int, action_idx: int) -> Action:
    """Action index -> an engine Action that is legal by construction.

    Any index the mask excluded falls back to check/call, so a policy bug can
    never reach the engine's exception handler (which would silently fold and
    quietly poison the training signal).
    """
    p = state.players[player_id]
    to_call = state.current_bet - p.street_committed
    types = {la.action_type for la in state.legal_actions}
    if action_idx == ACTION_FOLD and ActionType.FOLD in types:
        return Action.fold("rl fold")
    target = raise_target(state, player_id, action_idx)
    if target is not None:
        name = ACTION_NAMES[action_idx]
        if to_call > 0:
            return Action.raise_to(target, f"rl {name}")
        return Action.bet(target, f"rl {name}")
    if to_call > 0:
        return Action.call(min(to_call, p.stack), "rl call")
    return Action.check("rl check")


# -- observation ----------------------------------------------------------


def encode_obs(
    state: GameState,
    player_id: int,
    equity: float | None = None,
    opponent: OpponentSummary | None = None,
) -> np.ndarray:
    """GameState -> float32 vector of length OBS_DIM.

    equity: Monte Carlo equity vs the live opponents (pokr.cards /
        pokr._fastcards), or None when the caller skipped it.
    opponent: model summary of the opponent whose range we are facing (see
        PokerBot._target_opponent), or None when unmodelled.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    me = state.players[player_id]
    n = len(state.players)
    start_stack = float(me.stack + me.committed) or 1.0

    hole = obs[OBS_SLICES["hole"]]
    for c in me.hole:
        hole[card_index(c)] = 1.0
    board = obs[OBS_SLICES["board"]]
    for c in state.community:
        board[card_index(c)] = 1.0
    obs[OBS_SLICES["street"]][_STREETS.index(state.street)] = 1.0

    to_call = max(0, state.current_bet - me.street_committed)
    live = [q for q in state.players if not q.folded]
    obs[OBS_SLICES["scalars"]] = [
        state.pot / start_stack,
        me.stack / start_stack,
        me.committed / start_stack,
        to_call / start_stack,
        to_call / (state.pot + to_call) if state.pot + to_call > 0 else 0.0,
        np.log1p(me.stack / state.pot) / 5.0 if state.pot > 0 else 1.0,
        len(live) / MAX_SEATS,
    ]

    # Opponents in acting order starting left of the hero, so the encoding is
    # position-relative: seat 0 of this block is always the next player.
    seats = obs[OBS_SLICES["seats"]].reshape(MAX_SEATS - 1, _SEAT_FEATURES)
    for k in range(min(n - 1, MAX_SEATS - 1)):
        q = state.players[(player_id + 1 + k) % n]
        seats[k] = [
            0.0 if q.folded else 1.0,
            1.0 if q.all_in else 0.0,
            q.committed / start_stack,
            q.street_committed / start_stack,
            1.0 if q.id == state.dealer else 0.0,
            1.0 if q.acted_round else 0.0,
        ]

    pos = obs[OBS_SLICES["position"]]
    pos[(player_id - state.dealer) % n] = 1.0
    to_act = sum(
        1 for q in state.players
        if q.id != player_id and not q.folded and not q.all_in
        and (not q.acted_round or q.street_committed < state.current_bet)
    )
    pos[MAX_SEATS] = to_act / MAX_SEATS

    if equity is not None:
        obs[OBS_SLICES["equity"]] = [equity, 1.0]
    if opponent is not None:
        obs[OBS_SLICES["opponent"]] = [
            opponent.vpip,
            opponent.pfr,
            opponent.aggression_freq,
            opponent.fold_to_cbet,
            opponent.fold_rate_postflop,
            min(opponent.hands_observed / 100.0, 1.0),
        ]
    return obs
