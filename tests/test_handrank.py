from pokr.cards import card_from_str, evaluate_hand, hand_name


def hs(*strs):
    return [card_from_str(s) for s in " ".join(strs).split()]


def test_categories():
    cases = [
        ("As Ks Qs Js Ts", "straight flush"),
        ("9h 8c 7d 6s 5h", "straight"),
        ("5h 4c 3d 2s Ah", "straight"),        # wheel
        ("As Ah Ac Ad Kd", "four of a kind"),
        ("Kh Kd Ks 2c 2h", "full house"),
        ("Ah 2h 5h 9h Jh", "flush"),
        ("As Kd Qc Jh 9d", "high card"),
        ("As Ad Kh Qc Jd", "one pair"),
        ("As Ad Kh Kc Qd", "two pair"),
        ("As Ad Ah Kc Qd", "three of a kind"),
    ]
    for cards, name in cases:
        assert hand_name(evaluate_hand(hs(*cards.split()))) == name, cards


def test_wheel_is_low_straight():
    wheel = evaluate_hand(hs("5h 4c 3d 2s Ah"))
    six = evaluate_hand(hs("6h 5c 4d 3s 2h"))
    assert six > wheel


def test_seven_card_picks_best():
    score = evaluate_hand(hs("As Ah Kd Kc Qd 2s 3h"))
    assert hand_name(score) == "two pair"


def test_kickers():
    a = evaluate_hand(hs("As Ad Kh Qc Jd"))
    b = evaluate_hand(hs("As Ad Kh Qc Td"))
    assert a > b


def test_royal_is_straight_flush():
    score = evaluate_hand(hs("As Ks Qs Js Ts"))
    assert hand_name(score) == "straight flush"


def test_two_pair_ordering():
    a = evaluate_hand(hs("As Ad Kd Kc 2s"))
    b = evaluate_hand(hs("As Ad Qd Qc 3s"))
    assert a > b
