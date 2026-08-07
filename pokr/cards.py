from __future__ import annotations

import random
from typing import Sequence

RANKS = "23456789TJQKA"
SUITS = "cdhs"


class Card:
    """A playing card. rank is 2..14 (14 = Ace), suit is 0..3 (c,d,h,s)."""
    __slots__ = ("rank", "suit")

    def __init__(self, rank: int, suit: int) -> None:
        assert 2 <= rank <= 14, rank
        assert 0 <= suit <= 3, suit
        self.rank = rank
        self.suit = suit

    def __repr__(self) -> str:
        return f"{RANKS[self.rank - 2]}{SUITS[self.suit]}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


def all_cards() -> list[Card]:
    return [Card(r, s) for r in range(2, 15) for s in range(4)]


def card_from_str(s: str) -> Card:
    assert len(s) == 2, s
    rank = RANKS.index(s[0].upper()) + 2
    suit = SUITS.index(s[1].lower())
    return Card(rank, suit)


class Deck:
    """A shuffled 52-card deck. Deterministic given rng."""

    def __init__(
        self,
        rng: random.Random | None = None,
        cards: Sequence[Card] | None = None,
        shuffle: bool = True,
    ) -> None:
        self._rng = rng or random.Random()
        self._cards: list[Card] = list(cards) if cards is not None else all_cards()
        if shuffle:
            self._rng.shuffle(self._cards)

    def draw(self, n: int = 1) -> list[Card]:
        assert n <= len(self._cards)
        out = self._cards[:n]
        self._cards = self._cards[n:]
        return out

    @property
    def remaining(self) -> int:
        return len(self._cards)
