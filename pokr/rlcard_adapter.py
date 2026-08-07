"""RLCard adapter for pokr (handoff step 3).

Translates pokr engine states into RLCard's no-limit-holdem state/action
model and back, so an RLCard-style policy (a callable that takes an
extracted env state and returns an RLCard action) can play inside our
engine. Registered in the connector as plugin "rlcard".

The bundled policy is uniform random over RLCard's legal actions; it
exercises the full translation layer. Pretrained torch-based agents
(NFSP/DQN from rlcard.agents) can plug in later once that module imports
on this platform (it currently fails: distutils was removed in Python
3.12+ and setuptools/torch are not installed here).

RLCard is imported lazily so pokr imports and the rest of the suite work
without it installed; tests use pytest.importorskip.

Translation approximations (documented):
- RLCard's raise actions are pot-fraction based (raise BY half-pot or BY
  pot, or all-in). Our engine has continuous raise-to ranges, so the
  computed raise-to is clamped into our legal [min, max] raise-to range.
- RLCard's raise-by-pot uses the total hand pot (its known simplification);
  we mirror that using our state.pot.
- RLCard's legal-action gating (no raises when calling is all-in, pot
  fraction vs remaining chips) is mirrored exactly.
"""
from __future__ import annotations

import os
import random
from collections import OrderedDict
from typing import Callable

import numpy as np

from .connector import register_plugin
from .strategy import Action, ActionType, BaseStrategy

_STAGE_NAMES = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
# our suit encoding 0..3 = c,d,h,s -> rlcard suit letters S,H,D,C
_RL_SUITS = {3: "S", 2: "H", 1: "D", 0: "C"}
_RL_RANKS = {r: str(r) for r in range(2, 10)}
_RL_RANKS.update({10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"})

_refs = None  # (RlAction enum class, card2index dict), resolved lazily


def _rl_refs():
    """Lazily import the rlcard pieces we need (cheap, no torch/agents)."""
    global _refs
    if _refs is None:
        import json
        import rlcard
        from rlcard.games.nolimitholdem.round import Action as RlAction
        with open(os.path.join(rlcard.__path__[0],
                               "games/limitholdem/card2index.json")) as f:
            card2index = json.load(f)
        _refs = (RlAction, card2index)
    return _refs


def _rl_card_string(card) -> str:
    return _RL_SUITS[card.suit] + _RL_RANKS[card.rank]


def _card_index(card_str: str) -> int:
    return _rl_refs()[1][card_str]


def _rl_legal_actions(state: GameState, p: PlayerView):
    """Map our legal actions onto RLCard's action set, mirroring its gating."""
    RlAction, _ = _rl_refs()
    to_call = state.current_bet - p.street_committed
    out = []
    if to_call > 0:
        out.append(RlAction.FOLD)
        out.append(RlAction.CHECK_CALL)
        if to_call >= p.stack:
            # calling would be all-in: rlcard removes all raise actions
            return out
    else:
        out.append(RlAction.CHECK_CALL)
    # rlcard's gating for the raise actions
    if state.pot <= p.stack:
        out.append(RlAction.RAISE_POT)
    if int(state.pot / 2) <= p.stack and \
            p.street_committed + int(state.pot / 2) > state.current_bet:
        out.append(RlAction.RAISE_HALF_POT)
    out.append(RlAction.ALL_IN)
    return out


def _raw_state(state: GameState, player_id: int) -> dict:
    """Our GameState -> rlcard-style raw state (mirrors its get_state)."""
    p = state.players[player_id]
    committed = [q.committed for q in state.players]
    return {
        "hand": [_rl_card_string(c) for c in p.hole],
        "public_cards": [_rl_card_string(c) for c in state.community],
        "all_chips": committed,
        "my_chips": committed[player_id],
        "stakes": [q.stack for q in state.players],
        "pot": state.pot,
        "stage": _STAGE_NAMES[state.street],
        "current_player": player_id,
        "legal_actions": _rl_legal_actions(state, p),
    }


def _extract(raw: dict) -> dict:
    """Build the env-style extracted state an rlcard agent consumes."""
    RlAction, card2index = _rl_refs()
    legal = OrderedDict({a.value: None for a in raw["legal_actions"]})
    cards = raw["public_cards"] + raw["hand"]
    obs = np.zeros(54)
    for c in cards:
        obs[card2index[c]] = 1.0
    obs[52] = float(raw["my_chips"])
    obs[53] = float(max(raw["all_chips"]))
    return {
        "legal_actions": legal,
        "raw_legal_actions": list(raw["legal_actions"]),
        "obs": obs,
        "raw_obs": raw,
    }


def _our_action(state: GameState, player_id: int, rl_action) -> Action:
    """RLCard action -> our Action, clamped into our legal ranges."""
    RlAction, _ = _rl_refs()
    if isinstance(rl_action, int):
        rl_action = RlAction(rl_action)
    p = state.players[player_id]
    to_call = state.current_bet - p.street_committed
    if rl_action == RlAction.FOLD:
        return Action.fold("rlcard fold")
    if rl_action == RlAction.CHECK_CALL:
        if to_call > 0:
            return Action.call(min(to_call, p.stack), "rlcard call")
        return Action.check("rlcard check")
    # raise actions (RAISE_HALF_POT / RAISE_POT / ALL_IN)
    raise_la = [x for x in state.legal_actions
                if x.action_type in (ActionType.BET, ActionType.RAISE)]
    if not raise_la:
        if to_call > 0:
            return Action.call(min(to_call, p.stack), "rlcard raise fallback call")
        return Action.check("rlcard raise fallback check")
    la = raise_la[0]
    if rl_action == RlAction.ALL_IN:
        raise_to = p.street_committed + p.stack
    elif rl_action == RlAction.RAISE_POT:
        raise_to = p.street_committed + state.pot
    else:
        raise_to = p.street_committed + int(state.pot / 2)
    raise_to = min(max(raise_to, la.min_amount), la.max_amount)
    if to_call > 0:
        return Action.raise_to(raise_to, "rlcard raise")
    return Action.bet(raise_to, "rlcard bet")


class RandomRlcardPolicy:
    """Uniform random over RLCard's legal actions (bundled policy)."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: dict):
        return self.rng.choice(list(state["raw_legal_actions"]))


class RlcardAdapter(BaseStrategy):
    """Plays our engine by delegating decisions to an rlcard-style policy.

    policy: callable(extracted_state: dict) -> rlcard Action (enum member or
    int). Defaults to RandomRlcardPolicy.
    """

    def __init__(
        self,
        policy: Callable[[dict], object] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.policy = policy if policy is not None else RandomRlcardPolicy(rng or random.Random())

    def decide(self, state: GameState, player_id: int) -> Action:
        raw = _raw_state(state, player_id)
        extracted = _extract(raw)
        rl_action = self.policy(extracted)
        return _our_action(state, player_id, rl_action)


def rlcard_factory() -> RlcardAdapter:
    return RlcardAdapter()


register_plugin("rlcard", rlcard_factory)
