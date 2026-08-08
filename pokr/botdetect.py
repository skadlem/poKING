from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .models import OpponentSummary

_FEATURE_WEIGHTS = [1.0, 2.0, 1.5, 1.0, 4.0]
_INTERCEPT = -4.0
_PRIOR_BOT = 0.3
_PRIOR_MIRROR = 0.1
_BOT_SAMPLE_K = 25.0
_MIRROR_SAMPLE_K = 15.0
_MIRROR_SCALE = 0.15

@dataclass
class DetectionResult:
    p_is_bot: float
    p_mirror: float
    samples: int


def _features(s: OpponentSummary):
    pfr_ratio = (s.pfr / s.vpip) if s.vpip > 0 else 0.0
    return [s.vpip, pfr_ratio, s.aggression_freq, s.fold_to_cbet, s.round_size_frac]


def _logistic(x) -> float:
    z = sum(w * xi for w, xi in zip(_FEATURE_WEIGHTS, x)) + _INTERCEPT
    return 1.0 / (1.0 + math.exp(-z))


def _shrink(p: float, n: int, prior: float, k: float) -> float:
    w = n / (n + k)
    return p * w + prior * (1 - w)


def _hellinger(a: Counter, b: Counter) -> float:
    """Hellinger distance between two count histograms, normalized to [0, 1]."""
    na = sum(a.values())
    nb = sum(b.values())
    if na == 0 or nb == 0:
        return 1.0
    keys = set(a) | set(b)
    s = 0.0
    for k in keys:
        pa = a.get(k, 0) / na
        pb = b.get(k, 0) / nb
        s += (math.sqrt(pa) - math.sqrt(pb)) ** 2
    return math.sqrt(s) / math.sqrt(2.0)


class BotDetector:
    """Estimates P(opponent is a bot) and P(opponent is a mirror of us)."""

    def __init__(self) -> None:
        self._own_hist: Counter[str] = Counter()
        self._opp_hists: dict[int, Counter[str]] = {}

    def observe(self, result, my_seat: int) -> None:
        for (t, s, a) in result.actions:
            key = f"{s}:{a.action_type.value}"
            if t == my_seat:
                self._own_hist[key] += 1
            else:
                self._opp_hists.setdefault(t, Counter())[key] += 1

    def detect(self, opponent_id: int, summary: OpponentSummary | None) -> DetectionResult:
        if summary is None or summary.hands_observed == 0:
            return DetectionResult(_PRIOR_BOT, _PRIOR_MIRROR, 0)
        n = summary.hands_observed
        p_bot = _shrink(_logistic(_features(summary)), n, _PRIOR_BOT, _BOT_SAMPLE_K)
        hist = self._opp_hists.get(opponent_id, Counter())
        own = self._own_hist
        m_n = sum(hist.values())
        if m_n > 0 and sum(own.values()) > 0:
            h = _hellinger(hist, own)
            p_mirror = math.exp(-h / _MIRROR_SCALE)
            p_mirror = _shrink(p_mirror, m_n, _PRIOR_MIRROR, _MIRROR_SAMPLE_K)
        else:
            p_mirror = _PRIOR_MIRROR
        return DetectionResult(p_bot, p_mirror, n)
