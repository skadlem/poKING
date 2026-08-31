"""Replay memories for NFSP (design note 5).

Two flavours, both keyed to the Lanctot et al. 2017 NFSP buffers:

- Reservoir: uniform sample over the ENTIRE stream seen so far, at fixed
  memory (Algorithm R). This is the "average of all past behaviours" that
  makes fictitious play fictitious rather than a recency-biased self-play
  blur: the contents are exchangeable, so the supervised fit sees each past
  hand with probability capacity/seen, no matter when it happened.
- Exponential: identical sampler with a floor on the replacement probability
  (the paper's 0.25). Once the stream is long, every new item lands in the
  buffer with probability >= min_replacement regardless of how full it is,
  which decays old contents geometrically -- an exponentially-weighted
  average of recent behaviour instead of a uniform average of all of it.
  The paper uses this for M_SL.

What goes IN the buffer is the caller's business: both are used for
(s, a, mask) records derived from `Episode`s. These buffers stay torch-free
and numpy-free so the Kuhn gate (design note 6, tabular) can reuse them
against a game with no features at all. Keep records small -- an NFSP-grade
reservoir is sized in millions and a list of Episode objects will not fit.

sample() is the trainer's seam: uniform without replacement, deterministic
under a seeded rng, so a failed run is reproducible line by line.
"""
from __future__ import annotations

import random


class ReservoirBuffer:
    """Algorithm R reservoir sampling: exactly `capacity` items, each of the
    `seen` items ever added present with probability capacity/seen.

    Items must be hash-free lightweight records (the buffer keeps whatever
    object it is handed); determinism of the contents follows from the rng.
    """

    def __init__(self, capacity: int, rng: random.Random | None = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.rng = rng or random.Random()
        self._items: list = []
        self._seen = 0

    # -- contract ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    @property
    def seen(self) -> int:
        """How many items the stream has delivered in total."""
        return self._seen

    def add(self, item) -> None:
        self._seen += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        j = self._replacement_index(self._seen)
        if j is not None:
            self._items[j] = item

    def add_many(self, items) -> None:
        for item in items:
            self.add(item)

    def clear(self) -> None:
        self._items.clear()
        self._seen = 0

    def contents(self) -> list:
        """Snapshot of what is held now (order is the buffer's own; not a
        stream order -- uniform reservoirs are sets, not queues)."""
        return list(self._items)

    def sample(self, n: int) -> list:
        """Uniform minibatch without replacement."""
        if not 0 <= n <= len(self._items):
            raise ValueError(f"cannot sample {n} of {len(self._items)}")
        return self.rng.sample(self._items, n)

    # -- internals --------------------------------------------------------

    def _replacement_index(self, seen: int) -> int | None:
        """Which slot the `seen`-th item takes, or None if it is dropped.

        Algorithm R: with probability capacity/seen replace a uniform slot.
        """
        if self.rng.random() >= self.capacity / seen:
            return None
        return self.rng.randrange(self.capacity)


class ExponentialReservoirBuffer(ReservoirBuffer):
    """Reservoir with a floor on the replacement probability.

    Once capacity/seen falls below `min_replacement`, every arrival still
    enters with probability min_replacement -- so the buffer stops being a
    uniform sample of all history and becomes an exponentially-weighted
    sample of recent history (a step of the old contents survives each
    arrival with probability 1 - min_replacement/capacity). The paper's
    M_SL: the average policy should forget ancient, near-random behaviour.
    min_replacement=0 recovers exact uniformity, which is what the tests
    use as the control.
    """

    def __init__(self, capacity: int, rng: random.Random | None = None,
                 min_replacement: float = 0.25) -> None:
        if not 0.0 <= min_replacement <= 1.0:
            raise ValueError(f"min_replacement must be in [0, 1], got {min_replacement}")
        super().__init__(capacity, rng)
        self.min_replacement = min_replacement

    def _replacement_index(self, seen: int) -> int | None:
        p = max(self.capacity / seen, self.min_replacement)
        if self.rng.random() >= p:
            return None
        return self.rng.randrange(self.capacity)
