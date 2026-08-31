"""The reservoir's whole job is a statistical guarantee, so the tests measure
it. Uniformity is scored as one chi-square over the stream positions, not as
per-position bands: each trial holds exactly `capacity` items, so the
position counts are correlated and a pointwise band over 200 positions is a
multiple-comparisons trap (it failed at position 50 on a sampler that is
provably unbiased). Every band used is 5 sigma on the trial-to-trial sample
variance — no magic margins.
"""
import math
import random

import pytest

from pokr.rl.memory import ExponentialReservoirBuffer, ReservoirBuffer


def fill_contents(capacity, n_items, rng, cls=ReservoirBuffer, **kwargs):
    buf = cls(capacity, rng, **kwargs)
    buf.add_many(range(n_items))
    return buf.contents()


def chi2_uniform(capacity, n_items, trials, seed=0, cls=ReservoirBuffer,
                 **kwargs):
    """sum (hits - expected)^2 / expected over stream positions, where
    expected is capacity*trials/n_items at every position (the uniform null)."""
    hits = [0] * n_items
    rng = random.Random(seed)
    for _ in range(trials):
        for i in fill_contents(capacity, n_items,
                               random.Random(rng.randrange(1 << 40)),
                               cls=cls, **kwargs):
            hits[i] += 1
    assert sum(hits) == capacity * trials
    e = capacity * trials / n_items
    return sum((h - e) ** 2 / e for h in hits)


def region_rate(capacity, n_items, region, trials, seed, cls=ReservoirBuffer,
                **kwargs):
    """Mean, per trial, of the fraction of the buffer drawn from `region`,
    plus 5 sigma of that mean from the trial-to-trial sample variance."""
    rs = random.Random(seed)
    vals = []
    for _ in range(trials):
        c = fill_contents(capacity, n_items,
                          random.Random(rs.randrange(1 << 40)),
                          cls=cls, **kwargs)
        vals.append(len(set(c) & region) / len(region))
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, 5 * sd / math.sqrt(len(vals))


def expected_inclusion(capacity, floor, n_items, positions):
    """Exact P(position i is in the final buffer) for a floored reservoir:
    enter with probability 1 if before capacity else max(capacity/t, floor),
    then survive every later arrival with probability 1 - p_t/capacity (the
    replacement must also pick this item's slot). A direct product over the
    stream, not a closed form — verified empirically by
    test_expected_formula_is_not_a_dud."""
    def p(t):                                  # arrival t's replacement prob
        return max(capacity / t, floor) if t > capacity else 0.0

    out = []
    for i in positions:                        # i = 0-based stream position
        enter = 1.0 if i < capacity else p(i + 1)
        surv = 1.0
        for t in range(i + 2, n_items + 1):    # arrivals strictly after i
            surv *= 1.0 - p(t) / capacity
        out.append(enter * surv)
    return out


# -- Algorithm R: uniform over the stream ---------------------------------


def test_fills_to_capacity_and_reports_the_stream_length():
    buf = ReservoirBuffer(3)
    assert buf.seen == 0 and len(buf) == 0
    buf.add_many([1, 2, 3, 4])
    assert len(buf) == 3 and buf.seen == 4


def test_first_capacity_items_are_kept_verbatim():
    buf = ReservoirBuffer(4, rng=random.Random(9))
    buf.add_many("abcd")
    assert buf.contents() == ["a", "b", "c", "d"]
    buf.add("e")  # only from here on does sampling start
    assert len(buf) == 4


def test_inclusion_is_uniform_over_the_stream():
    """The defining property: every position, first or last, lands in the
    final buffer with probability capacity/n. This is what makes the average
    of reservoir-sampled best responses the average OVER the best responses,
    not a recency blur.

    Threshold, calibrated on 40 seeds before being trusted: under H0 the
    stat runs ~174 +-19 (the fixed-per-trial total gives a negative
    within-trial correlation, pulling it under the chi2(199) null); the
    observed value here is 191. A sampler decaying old items to half the
    uniform rate scores several hundred.
    """
    stat = chi2_uniform(20, 200, 8000)
    assert stat < 320, f"chi2={stat:.0f} on 199 dof, empirical null ~174"


def test_exponential_variant_with_zero_floor_is_exactly_uniform():
    """min_replacement=0 must collapse the variant onto Algorithm R — the
    control that says any decay found below comes from the floor."""
    stat = chi2_uniform(16, 160, 6000, cls=ExponentialReservoirBuffer,
                        min_replacement=0.0)
    assert stat < 260, (f"chi2={stat:.0f} on 159 dof: with floor 0 the sampler "
                        "must reduce exactly to Algorithm R")


# -- the exponential floor: recency weighting ------------------------------


def test_floor_matches_the_exact_inclusion_curve():
    """Early items are annihilated and late items sit at ~the floor — both
    against the exact expected curve, in a 5-sigma band of the observed
    trial-to-trial spread, not hand-set margins."""
    cap, n, floor, trials = 100, 5000, 0.25, 400
    early = set(range(100))
    late = set(range(n - 100, n))
    exp_e = sum(expected_inclusion(cap, floor, n, sorted(early))) / len(early)
    exp_l = sum(expected_inclusion(cap, floor, n, sorted(late))) / len(late)

    obs_e, tol_e = region_rate(cap, n, early, trials, seed=2,
                               cls=ExponentialReservoirBuffer,
                               min_replacement=floor)
    obs_l, tol_l = region_rate(cap, n, late, trials, seed=3,
                               cls=ExponentialReservoirBuffer,
                               min_replacement=floor)
    # the sample band degenerates when a region empties completely every
    # trial (zero variance, real Poisson noise); take whichever band is wider
    def band(exp, tol):
        return max(tol, 5 * math.sqrt(exp * (1 - exp) / (trials * 100)))
    assert abs(obs_e - exp_e) <= band(exp_e, tol_e), \
        f"early {obs_e} vs {exp_e} +- {band(exp_e, tol_e)}"
    assert abs(obs_l - exp_l) <= band(exp_l, tol_l), \
        f"late {obs_l} vs {exp_l} +- {band(exp_l, tol_l)}"
    # directional restatement so a failure says WHAT the floor is for:
    uniform = cap / n
    assert exp_e < uniform / 4, "the formula itself must decay early items"
    assert obs_l > uniform * 5, "the sampler must lift late items"


def test_expected_formula_is_not_a_dud():
    """Guard the guard: expected_inclusion against a direct empirical
    measurement on a small case where trials are cheap. If the formula and
    the sampler disagree, the test above asserts nothing meaningful."""
    cap, n, floor, trials = 10, 200, 0.25, 4000
    positions = [0, 9, 12, 60, 150, 199]
    exp = dict(zip(positions, expected_inclusion(cap, floor, n, positions)))
    rng = random.Random(5)
    hits = {i: 0 for i in positions}
    for _ in range(trials):
        c = fill_contents(cap, n, random.Random(rng.randrange(1 << 40)),
                          cls=ExponentialReservoirBuffer, min_replacement=floor)
        for i in positions:
            if i in c:
                hits[i] += 1
    for i in positions:
        o = hits[i] / trials
        sd = math.sqrt(max(exp[i], 1e-9) * (1 - exp[i]) / trials)
        assert abs(o - exp[i]) <= 5 * sd, \
            f"pos {i}: formula {exp[i]:.5f} vs sampled {o:.5f}"


def test_replacement_probability_is_the_max_of_uniform_and_floor():
    """Unit-level: at seen = cap+1 the arrival enters with probability
    max(cap/(cap+1), floor) — strictly above plain Algorithm R there."""
    cap, floor, trials = 4, 0.5, 4000
    enters = 0
    rs = random.Random(11)
    for _ in range(trials):
        b = ExponentialReservoirBuffer(cap, random.Random(rs.randrange(1 << 40)),
                                       min_replacement=floor)
        b.add_many([0, 1, 2, 3])
        b.add(99)
        enters += 99 in b.contents()
    p = max(cap / (cap + 1), floor)              # = 0.8
    plain = cap / (cap + 1)                      # = 0.8 -> identical here; see next line
    sd = math.sqrt(p * (1 - p) / trials)
    assert abs(enters / trials - p) <= 5 * sd, f"{enters/trials} vs {p}"
    assert p == 0.8 and plain == 0.8             # floor active only past seen=8
    enters2 = 0
    rs = random.Random(12)
    for _ in range(trials):                       # seen=9: R gives 4/9, floor 1/2
        b = ExponentialReservoirBuffer(cap, random.Random(rs.randrange(1 << 40)),
                                       min_replacement=floor)
        b.add_many([0, 1, 2, 3, 4, 5, 6, 7])
        b.add(99)
        enters2 += 99 in b.contents()
    sd2 = math.sqrt(0.5 * 0.5 / trials)
    assert abs(enters2 / trials - 0.5) <= 5 * sd2, \
        f"floor not lifting: {enters2/trials} vs Algorithm R's {4/9}"


# -- minibatch seam ---------------------------------------------------------


def test_sample_is_without_replacement_and_reproducible():
    buf = ReservoirBuffer(50)
    buf.add_many(range(100))
    b1 = ReservoirBuffer(50, random.Random(7)); b1.add_many(range(100))
    b2 = ReservoirBuffer(50, random.Random(7)); b2.add_many(range(100))
    assert b1.contents() == b2.contents()
    s1, s2 = b1.sample(10), b2.sample(10)
    assert s1 == s2 and len(set(s1)) == 10          # deterministic, no dupes
    assert set(s1) <= set(b1.contents())
    assert b1.sample(0) == []
    with pytest.raises(ValueError):
        b1.sample(51)


def test_clear_resets_both_counters():
    buf = ReservoirBuffer(5)
    buf.add_many(range(9))
    buf.clear()
    assert len(buf) == 0 and buf.seen == 0


def test_bad_construction_is_rejected():
    with pytest.raises(ValueError):
        ReservoirBuffer(0)
    with pytest.raises(ValueError):
        ReservoirBuffer(-3)
    with pytest.raises(ValueError):
        ExponentialReservoirBuffer(4, min_replacement=1.5)
    with pytest.raises(ValueError):
        ExponentialReservoirBuffer(4, min_replacement=-0.1)
