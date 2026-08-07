from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass(frozen=True)
class Action:
    """A poker action.

    Amount semantics: for BET/RAISE, amount is the raise-to (total street
    commitment after the action). For CALL, amount is chips added. 0 otherwise.
    """
    action_type: ActionType
    amount: int = 0
    reason: str = ""

    @staticmethod
    def fold(reason: str = "") -> "Action":
        return Action(ActionType.FOLD, 0, reason)

    @staticmethod
    def check(reason: str = "") -> "Action":
        return Action(ActionType.CHECK, 0, reason)

    @staticmethod
    def call(amount: int, reason: str = "") -> "Action":
        return Action(ActionType.CALL, amount, reason)

    @staticmethod
    def bet(amount: int, reason: str = "") -> "Action":
        return Action(ActionType.BET, amount, reason)

    @staticmethod
    def raise_to(amount: int, reason: str = "") -> "Action":
        return Action(ActionType.RAISE, amount, reason)


class Strategy:
    """Interface every bot implements. Engine only talks to this.

    GameState and HandResult are defined in engine.py; annotations are strings
    to avoid a circular import.
    """

    def decide(self, state: "GameState", player_id: int) -> Action:
        raise NotImplementedError

    def on_hand_end(self, result: "HandResult", my_seat: int) -> None:
        return


class BaseStrategy(Strategy):
    """Convenience base with a no-op on_hand_end; concrete bots override decide."""

    def decide(self, state: "GameState", player_id: int) -> Action:
        raise NotImplementedError
