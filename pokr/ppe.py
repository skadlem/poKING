"""Adapter: play pokr's PokerBot inside the external PyPokerEngine.

PyPokerEngine (https://github.com/ishikota/PyPokerEngine) is an independent
no-limit hold'em engine with its own rules, dealer, and side-pot logic. This
module adapts our Strategy into its BasePokerPlayer interface so the bot can
sit at a real PyPokerEngine table against the framework's own example bots
(HonestPlayer, FishPlayer, RandomPlayer from examples/players).

Action semantics (PyPokerEngine): 'raise' amount is the raise-to total for the
street; 'call' amount is the agree amount (total street commitment level);
'call' with 0 is a check. Card strings are like 'SA' (suit letter + rank).
"""
from __future__ import annotations

import random

from pypokerengine.players import BasePokerPlayer

from .bot import PokerBot
from .cards import Card
from .engine import GameState, HandResult, LegalAction, PlayerView
from .strategy import Action, ActionType

# PyPokerEngine suit letters -> our suit ids (c=0, d=1, h=2, s=3)
_SUIT_MAP = {"C": 0, "D": 1, "H": 2, "S": 3}
_RANK_MAP = {c: i for i, c in enumerate("23456789TJQKA", start=2)}


def to_our_card(s: str) -> Card:
    """'SA' (Ace of spades) -> Card(14, 3)."""
    return Card(_RANK_MAP[s[1]], _SUIT_MAP[s[0]])


def to_our_cards(strs) -> list[Card]:
    return [to_our_card(s) for s in strs]


class PokrPlayer(BasePokerPlayer):
    """Wraps a pokr PokerBot so it can play inside PyPokerEngine."""

    def __init__(self, rng_seed: int = 0, mc_iters: int = 150,
                 risk_cfg=None, num_players: int = 6):
        super().__init__()
        self.bot = PokerBot(random.Random(rng_seed), risk_cfg=risk_cfg,
                            num_players=num_players, mc_iters=mc_iters)
        self._my_idx = -1
        self._round_actions: list[tuple[int, str, Action]] = []
        self._round_hole: list[list[Card]] = []
        self._round_start: list[int] = []
        self._round_num = 0

    # -- BasePokerPlayer interface ---------------------------------------

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self._build_state(valid_actions, hole_card, round_state)
        action = self.bot.decide(state, self._my_idx)
        self._round_actions.append((self._my_idx, state.street, action))
        return self._to_ppe_action(action, state)

    def receive_game_start_message(self, game_info):
        self._my_idx = 0  # registration order; seat 0 is ours in our runners

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._round_num = round_count
        self._round_start = [s["stack"] for s in seats]
        self._round_actions = []
        # Hole cards per seat: we only know our own; opponents are empty lists
        # so HandResult.hole stays index-aligned with seats.
        self._round_hole = [to_our_cards(hole_card) if s["uuid"] == self.uuid else []
                            for s in seats]

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        # track opponents' actions for the bot's models: new_action is
        # {"player_uuid": ..., "action": "call|raise|fold", "amount": int}
        seats = round_state["seats"]
        pid = next((i for i, s in enumerate(seats)
                    if s["uuid"] == new_action["player_uuid"]), -1)
        if pid < 0:
            return
        street = round_state["street"]
        a = new_action["action"]
        amt = new_action.get("amount", 0)
        if a == "fold":
            action = Action.fold("opponent fold")
        elif a == "call":
            action = Action.check("opponent call/check") if amt == 0 else Action.call(amt, "opponent call")
        elif a == "raise":
            action = Action.raise_to(amt, "opponent raise")
        else:
            return
        self._round_actions.append((pid, street, action))

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Build a HandResult so the bot's models/detector learn from the hand.
        seats = round_state["seats"]
        n = len(seats)
        ending = [s["stack"] for s in seats]
        winnings = [e - s for e, s in zip(ending, self._round_start)]
        community = to_our_cards(round_state.get("community_card", []))
        result = HandResult(
            hand_number=self._round_num,
            starting_stacks=list(self._round_start),
            ending_stacks=ending,
            hole=self._round_hole,
            community=community,
            actions=list(self._round_actions),
            winnings=winnings,
            big_blind=round_state.get("small_blind_amount", 1) * 2,
            dealer=round_state.get("dealer_btn", 0),
        )
        self.bot.on_hand_end(result, self._my_idx)
        self._round_actions = []

    # -- helpers ----------------------------------------------------------

    def _build_state(self, valid_actions, hole_card, round_state) -> GameState:
        seats = round_state["seats"]
        n = len(seats)
        self._my_idx = next(i for i, s in enumerate(seats) if s["uuid"] == self.uuid)
        fold_info, call_info, raise_info = valid_actions[0], valid_actions[1], valid_actions[2]
        current_bet = call_info["amount"]  # agree amount = street commitment level
        my_stack = seats[self._my_idx]["stack"]
        # raise max = my_stack + my street commitment -> derive street_committed
        rmax = raise_info["amount"]["max"]
        my_committed = max(0, rmax - my_stack) if rmax >= 0 else 0
        to_call = max(0, current_bet - my_committed)

        players = []
        for i, s in enumerate(seats):
            hole = to_our_cards(hole_card) if i == self._my_idx else []
            players.append(PlayerView(
                id=i, stack=s["stack"], hole=hole,
                folded=s["state"] in ("folded", "out"),
                all_in=s["state"] == "allin",
                street_committed=my_committed if i == self._my_idx else 0,
            ))

        legal: list[LegalAction] = []
        if to_call > 0:
            legal.append(LegalAction(ActionType.FOLD))
            legal.append(LegalAction(ActionType.CALL, to_call, to_call))
            if rmax >= 0 and raise_info["amount"]["min"] >= 0:
                legal.append(LegalAction(ActionType.RAISE,
                                         raise_info["amount"]["min"],
                                         raise_info["amount"]["max"]))
        else:
            legal.append(LegalAction(ActionType.CHECK))
            if rmax >= 0:
                min_bet = max(round_state.get("small_blind_amount", 1) * 2, 1)
                legal.append(LegalAction(ActionType.BET, min_bet, rmax))
        community = to_our_cards(round_state.get("community_card", []))
        pot_total = round_state["pot"]["main"]["amount"] + \
            sum(sp["amount"] for sp in round_state["pot"]["side"])
        return GameState(
            players=players, community=community, pot=pot_total,
            current_bet=current_bet, min_raise=0, street=round_state["street"],
            dealer=round_state.get("dealer_btn", 0),
            current_player=self._my_idx, legal_actions=legal,
        )

    @staticmethod
    def _to_ppe_action(action: Action, state: GameState):
        t = action.action_type
        p = state.players[state.current_player]
        if t == ActionType.FOLD:
            return "fold", 0
        if t in (ActionType.CHECK, ActionType.CALL):
            # PyPokerEngine: 'call' amount = agree amount (0 = check)
            return "call", state.current_bet
        # BET / RAISE: amount is raise-to total (our semantics match theirs)
        return "raise", action.amount
