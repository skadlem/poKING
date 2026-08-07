from pokr.risk import RiskConfig, kelly_fraction, risk_adjusted_bet_size


def test_kelly_basic():
    assert kelly_fraction(0.5, 1.0) == 0.0
    assert abs(kelly_fraction(0.6, 1.0) - 0.2) < 1e-9
    assert abs(kelly_fraction(0.5, 2.0) - 0.25) < 1e-9


def test_kelly_clamped():
    # full Kelly at even money: f* = (b*p - q)/b = 0.9 - 0.1 = 0.8 (< 1)
    assert abs(kelly_fraction(0.9, 1.0) - 0.8) < 1e-9
    assert kelly_fraction(0.1, 1.0) == 0.0
    assert kelly_fraction(0.0, 0.0) == 0.0


def test_kelly_monotonic_in_win_prob():
    a = kelly_fraction(0.51, 1.0)
    b = kelly_fraction(0.7, 1.0)
    assert b > a


def test_bet_size_caps_by_stack_fraction():
    cfg = RiskConfig(max_bet_fraction_of_stack=0.2)
    size = risk_adjusted_bet_size(0.9, 100, 200, 1000, cfg)
    assert size <= 40
    assert size >= 0


def test_bet_size_never_exceeds_stack():
    cfg = RiskConfig(max_bet_fraction_of_stack=0.9)
    size = risk_adjusted_bet_size(0.99, 1000, 200, 10000, cfg)
    assert size <= 200


def test_bet_size_zero_for_losing():
    size = risk_adjusted_bet_size(0.01, 100, 200, 1000, RiskConfig())
    assert size == 0
