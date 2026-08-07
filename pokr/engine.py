from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from .cards import Card, Deck, HandScore, evaluate_hand
from .strategy import Action, ActionType, Strategy


class IllegalAction(Exception):
    pass


@dataclass(frozen=True)
class LegalAction:
    action_type: ActionType
    min_amount: int = 0
    max_amount: int = 0


@dataclass
class PlayerView:
    id: int
    stack: int
    hole: list[Card] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    committed: int = 0
    street_committed: int = 0
    last_action: Action | None = None
    acted_round: bool = False


@dataclass
class GameState:
    players: list[PlayerView]
    community: list[Card]
    pot: int
    current_bet: int
    min_raise: int
    street: str
    dealer: int
    current_player: int
    legal_actions: list[LegalAction]
    action_history: list[tuple[int, str, Action]] = field(default_factory=list)


@dataclass
class HandResult:
    hand_number: int
    starting_stacks: list[int]
    ending_stacks: list[int]
    hole: list[list[Card]]
    community: list[Card]
    actions: list[tuple[int, str, Action]]
    winnings: list[int]
    big_blind: int


class PokerGame:
    """Plays one hand of 6-max (or n-max) NLHE. Single source of truth for rules."""

    def __init__(
        self,
        strategies: Sequence[Strategy],
        stacks: Sequence[int],
        small_blind: int = 1,
        big_blind: int = 2,
        rng: random.Random | None = None,
        initial_dealer: int = 0,
        deck: Deck | None = None,
    ) -> None:
        assert len(strategies) == len(stacks)
        self.strategies = list(strategies)
        self.stacks = list(stacks)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rng = rng or random.Random()
        self.initial_dealer = initial_dealer % len(strategies)
        self._deck = deck
        self.hands_played = 0

    # -- public ----------------------------------------------------------

    def play_hand(self) -> HandResult:
        self.hands_played += 1
        n = len(self.strategies)
        starting = list(self.stacks)
        players = [PlayerView(i, self.stacks[i]) for i in range(n)]
        deck = self._deck or Deck(self.rng)
        for p in players:
            p.hole = deck.draw(2)
        state = GameState(
            players=players, community=[], pot=0, current_bet=0,
            min_raise=self.big_blind, street="preflop", dealer=self.initial_dealer,
            current_player=-1, legal_actions=[],
        )
        sb = self.initial_dealer if n == 2 else (self.initial_dealer + 1) % n
        bb = (self.initial_dealer + 1) % n if n == 2 else (self.initial_dealer + 2) % n
        self._post_blind(state, players[sb], self.small_blind)
        self._post_blind(state, players[bb], self.big_blind)
        state.current_bet = players[bb].street_committed
        state.min_raise = self.big_blind

        preflop_first = self.initial_dealer if n == 2 else (self.initial_dealer + 3) % n
        self._run_street(state, preflop_first, reset_street=False)

        postflop_first = (self.initial_dealer + 1) % n
        for street, count in (("flop", 3), ("turn", 1), ("river", 1)):
            if len(self._not_folded(players)) < 2:
                break
            state.community += deck.draw(count)
            state.street = street
            if len(self._active(players)) >= 2:
                self._run_street(state, postflop_first, reset_street=True)

        self._award(state, players)
        ending = [p.stack for p in players]
        result = HandResult(
            hand_number=self.hands_played,
            starting_stacks=starting,
            ending_stacks=ending,
            hole=[list(p.hole) for p in players],
            community=list(state.community),
            actions=list(state.action_history),
            winnings=[e - s for e, s in zip(ending, starting)],
            big_blind=self.big_blind,
        )
        for i, strat in enumerate(self.strategies):
            strat.on_hand_end(result, i)
        return result

    # -- internals -------------------------------------------------------

    def _post_blind(self, state: GameState, p: PlayerView, amount: int) -> None:
        self._chips_in(state, p, min(amount, p.stack))

    def _chips_in(self, state: GameState, p: PlayerView, amount: int) -> None:
        amount = max(0, min(amount, p.stack))
        p.stack -= amount
        p.street_committed += amount
        p.committed += amount
        state.pot += amount
        if p.stack == 0:
            p.all_in = True

    def _run_street(self, state: GameState, first_actor: int, reset_street: bool) -> None:
        if reset_street:
            for p in state.players:
                p.street_committed = 0
                p.acted_round = False
            state.current_bet = 0
            state.min_raise = self.big_blind
        if len(self._active(state.players)) < 2:
            return
        n = len(state.players)
        current = first_actor
        while True:
            p = state.players[current]
            if p.folded or p.all_in:
                current = (current + 1) % n
                if current == first_actor:
                    break
                continue
            if p.acted_round and p.street_committed == state.current_bet:
                break
            state.current_player = current
            state.legal_actions = self._legal_actions(state, current)
            action = self.strategies[current].decide(state, current)
            self._validate_action(state, current, action)
            state.action_history.append((current, state.street, action))
            self._apply_action(state, current, action)
            p.acted_round = True
            current = (current + 1) % n

    def _legal_actions(self, state: GameState, pid: int) -> list[LegalAction]:
        p = state.players[pid]
        to_call = state.current_bet - p.street_committed
        out: list[LegalAction] = []
        if to_call > 0:
            out.append(LegalAction(ActionType.FOLD))
            call = min(to_call, p.stack)
            out.append(LegalAction(ActionType.CALL, call, call))
            if p.stack > to_call:
                min_raise_to = state.current_bet + state.min_raise
                max_raise_to = p.street_committed + p.stack
                if min_raise_to > max_raise_to:
                    out.append(LegalAction(ActionType.RAISE, max_raise_to, max_raise_to))
                else:
                    out.append(LegalAction(ActionType.RAISE, min_raise_to, max_raise_to))
        else:
            out.append(LegalAction(ActionType.CHECK))
            if p.stack > 0:
                min_bet = min(self.big_blind, p.stack)
                out.append(LegalAction(ActionType.BET, min_bet, p.stack))
        return out

    def _validate_action(self, state: GameState, pid: int, action: Action) -> None:
        legal = [la for la in self._legal_actions(state, pid)
                 if la.action_type == action.action_type]
        if not legal:
            raise IllegalAction(f"{action.action_type} not legal for player {pid}")
        la = legal[0]
        if action.action_type in (ActionType.BET, ActionType.RAISE):
            if not (la.min_amount <= action.amount <= la.max_amount):
                raise IllegalAction(
                    f"amount {action.amount} out of range {la.min_amount}..{la.max_amount}"
                )
        elif action.action_type == ActionType.CALL:
            if action.amount != la.min_amount:
                raise IllegalAction(f"call amount {action.amount} != {la.min_amount}")

    def _apply_action(self, state: GameState, pid: int, action: Action) -> None:
        p = state.players[pid]
        t = action.action_type
        prev_bet = state.current_bet
        if t == ActionType.FOLD:
            p.folded = True
        elif t == ActionType.CHECK:
            pass
        elif t == ActionType.CALL:
            self._chips_in(state, p, action.amount)
        elif t == ActionType.BET:
            self._chips_in(state, p, action.amount - p.street_committed)
            state.current_bet = p.street_committed
            state.min_raise = max(self.big_blind, state.current_bet - prev_bet)
        elif t == ActionType.RAISE:
            self._chips_in(state, p, action.amount - p.street_committed)
            state.current_bet = p.street_committed
            inc = state.current_bet - prev_bet
            if inc >= state.min_raise:
                state.min_raise = inc
        p.last_action = action

    def _not_folded(self, players: list[PlayerView]) -> list[PlayerView]:
        return [p for p in players if not p.folded]

    def _active(self, players: list[PlayerView]) -> list[PlayerView]:
        return [p for p in players if not p.folded and not p.all_in]

    def _award(self, state: GameState, players: list[PlayerView]) -> None:
        from .cards import HandScore  # noqa: F401  (used for annotation)

        eligible = self._not_folded(players)
        if len(eligible) == 1:
            eligible[0].stack += state.pot
            return
        levels = sorted({p.committed for p in players})
        prev = 0
        for level in levels:
            slice_pot = (level - prev) * sum(1 for p in players if p.committed >= level)
            prev = level
            if slice_pot == 0:
                continue
            contenders = [p for p in eligible if p.committed >= level]
            if not contenders:
                continue
            best: HandScore | None = None
            winners: list[PlayerView] = []
            for p in contenders:
                score = evaluate_hand(p.hole + state.community)
                if best is None or score > best:
                    best = score
                    winners = [p]
                elif score == best:
                    winners.append(p)
            share, rem = divmod(slice_pot, len(winners))
            for w in winners:
                w.stack += share
            # odd chips to winners in seat order starting after the button
            order = sorted(winners, key=lambda w: (w.id - state.dealer - 1) % len(players))
            for k in range(rem):
                order[k % len(order)].stack += 1
