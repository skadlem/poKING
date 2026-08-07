import random

from pokr.cards import Card, Deck, all_cards, card_from_str


def test_deck_has_52_unique():
    d = Deck(random.Random(1))
    cards = d.draw(52)
    assert len(cards) == 52
    assert len(set(cards)) == 52


def test_draw_is_deterministic_for_seed():
    a = Deck(random.Random(7)).draw(10)
    b = Deck(random.Random(7)).draw(10)
    assert a == b


def test_draw_consumes():
    d = Deck(random.Random(1))
    assert d.remaining == 52
    d.draw(5)
    assert d.remaining == 47


def test_card_roundtrip():
    c = card_from_str("As")
    assert c.rank == 14 and c.suit == 3
    assert repr(c) == "As"


def test_all_cards_distinct():
    cards = all_cards()
    assert len(cards) == 52
    assert len(set(cards)) == 52
