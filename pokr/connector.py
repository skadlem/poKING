from __future__ import annotations

from collections.abc import Callable

from .strategy import Strategy

# Registry of external bot factories. A future RLCard or OpenSpiel adapter
# (e.g. a pretrained CFR/NFSP agent) registers here and then participates in
# benchmarks like any canned opponent.
_plugins: dict[str, Callable[[], Strategy]] = {}


def register_plugin(name: str, factory: Callable[[], Strategy]) -> None:
    _plugins[name] = factory


def build_strategy(name: str) -> Strategy:
    if name not in _plugins:
        raise KeyError(f"no plugin named {name!r}; available: {sorted(_plugins)}")
    return _plugins[name]()


def available_plugins() -> list[str]:
    return sorted(_plugins)
