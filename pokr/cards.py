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

from itertools import combinations

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)

_CATEGORY_NAMES = [
    "high card", "one pair", "two pair", "three of a kind",
    "straight", "flush", "full house", "four of a kind", "straight flush",
]

HandScore = tuple[int, ...]


def evaluate_5(cards: Sequence[Card]) -> HandScore:
    """Score exactly 5 cards. Higher tuple wins; lexicographic comparison is hand comparison."""
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1
    distinct = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(distinct) == 5:
        if distinct[0] - distinct[4] == 4:
            is_straight = True
            straight_high = distinct[0]
        elif distinct == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5
    if is_flush and is_straight:
        return (STRAIGHT_FLUSH, straight_high)
    counts = sorted(((ranks.count(r), r) for r in set(ranks)), reverse=True)
    if counts[0][0] == 4:
        return (QUADS, counts[0][1], counts[1][1])
    if counts[0][0] == 3 and counts[1][0] == 2:
        return (FULL_HOUSE, counts[0][1], counts[1][1])
    if is_flush:
        return (FLUSH, *ranks)
    if is_straight:
        return (STRAIGHT, straight_high)
    if counts[0][0] == 3:
        return (TRIPS, counts[0][1], *[r for _, r in counts[1:]])
    if counts[0][0] == 2 and counts[1][0] == 2:
        return (TWO_PAIR, counts[0][1], counts[1][1], counts[2][1])
    if counts[0][0] == 2:
        return (PAIR, counts[0][1], *[r for r in ranks if r != counts[0][1]])
    return (HIGH_CARD, *ranks)


def evaluate_hand(cards: Sequence[Card]) -> HandScore:
    """Best 5-card hand from 5..7 cards."""
    cards = list(cards)
    if len(cards) == 5:
        return evaluate_5(cards)
    best: HandScore | None = None
    for combo in combinations(cards, 5):
        s = evaluate_5(combo)
        if best is None or s > best:
            best = s
    assert best is not None
    return best


def hand_name(score: HandScore) -> str:
    return _CATEGORY_NAMES[score[0]]


# ponytail: pure-python equity; njit the sample loop when benches are too slow
def monte_carlo_equity(
    hole: Sequence[Card],
    board: Sequence[Card],
    num_opponents: int,
    iterations: int,
    rng: random.Random,
) -> float:
    """Fraction of runouts where hole+board wins (ties count half) vs num_opponents random hands."""
    dead = set(hole) | set(board)
    deck = [c for c in all_cards() if c not in dead]
    need_board = 5 - len(board)
    total = 0.0
    for _ in range(iterations):
        sample = rng.sample(deck, num_opponents * 2 + need_board)
        full_board = list(board) + sample[num_opponents * 2 :]
        mine = evaluate_hand(list(hole) + full_board)
        best_opp = max(
            evaluate_hand(sample[2 * i : 2 * i + 2] + full_board)
            for i in range(num_opponents)
        )
        if mine > best_opp:
            total += 1.0
        elif mine == best_opp:
            total += 0.5
    return total / iterations
