"""Numba fast path for Monte Carlo equity (opt-in; see cards.monte_carlo_equity).

Cards are encoded as ints 0..51: code = (rank - 2) * 4 + suit. The hand
evaluator here is order-equivalent to cards.evaluate_hand (cross-checked on
200k random 5/6/7-card matchups) but ~100x faster inside the sampling loop.
Sampling uses a partial Fisher-Yates over a preallocated deck buffer instead
of rng.sample, so it does not draw the same RNG stream as the pure path;
equity values are statistically equivalent but not bit-identical.
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from numba import njit

from .cards import Card, all_cards


def _code(c: Card) -> int:
    return (c.rank - 2) * 4 + c.suit


@njit(cache=True)
def _straight_high(mask):
    for r in range(12, 3, -1):
        need = 31 << (r - 4)
        if (mask & need) == need:
            return r
    if (mask & 0b1000000001111) == 0b1000000001111:  # wheel A-2-3-4-5
        return 3
    return -1


@njit(cache=True)
def _top_ranks(mask, k, out):
    """Top-k set bits of mask (rank order, high first) into out."""
    i = 0
    for r in range(12, -1, -1):
        if mask & (1 << r):
            out[i] = r
            i += 1
            if i == k:
                return


@njit(cache=True)
def _score(cat, v0, v1, v2, v3, v4):
    return ((((cat * 13 + v0) * 13 + v1) * 13 + v2) * 13 + v3) * 13 + v4


@njit(cache=True)
def eval_n(cards):
    """Best-hand score of 5..7 cards; higher is better (same order as
    cards.evaluate_hand's tuple comparison)."""
    n = cards.shape[0]
    counts = np.zeros(13, np.int64)
    suit_counts = np.zeros(4, np.int64)
    for i in range(n):
        counts[cards[i] >> 2] += 1
        suit_counts[cards[i] & 3] += 1
    fs = -1
    for s in range(4):
        if suit_counts[s] >= 5:
            fs = s
    fmask = 0
    if fs >= 0:
        for i in range(n):
            if (cards[i] & 3) == fs:
                fmask |= 1 << (cards[i] >> 2)
        sh = _straight_high(fmask)
        if sh >= 0:
            return _score(8, sh, 0, 0, 0, 0)
    # ranks ordered by (count desc, rank desc)
    order_r = np.zeros(13, np.int64)
    order_c = np.zeros(13, np.int64)
    m = 0
    for c in range(4, 0, -1):
        for r in range(12, -1, -1):
            if counts[r] == c:
                order_r[m] = r
                order_c[m] = c
                m += 1
    if order_c[0] == 4:
        kmask = 0
        for r in range(13):
            if counts[r] > 0 and r != order_r[0]:
                kmask |= 1 << r
        out = np.zeros(1, np.int64)
        _top_ranks(kmask, 1, out)
        return _score(7, order_r[0], out[0], 0, 0, 0)
    if order_c[0] == 3 and order_c[1] >= 2:
        return _score(6, order_r[0], order_r[1], 0, 0, 0)
    if fs >= 0:
        out = np.zeros(5, np.int64)
        _top_ranks(fmask, 5, out)
        return _score(5, out[0], out[1], out[2], out[3], out[4])
    mask = 0
    for r in range(13):
        if counts[r] > 0:
            mask |= 1 << r
    sh = _straight_high(mask)
    if sh >= 0:
        return _score(4, sh, 0, 0, 0, 0)
    if order_c[0] == 3:
        kmask = 0
        for r in range(13):
            if counts[r] > 0 and r != order_r[0]:
                kmask |= 1 << r
        out = np.zeros(2, np.int64)
        _top_ranks(kmask, 2, out)
        return _score(3, order_r[0], out[0], out[1], 0, 0)
    if order_c[0] == 2 and order_c[1] == 2:
        kmask = 0
        for r in range(13):
            if counts[r] > 0 and r != order_r[0] and r != order_r[1]:
                kmask |= 1 << r
        out = np.zeros(1, np.int64)
        _top_ranks(kmask, 1, out)
        return _score(2, order_r[0], order_r[1], out[0], 0, 0)
    if order_c[0] == 2:
        kmask = 0
        for r in range(13):
            if counts[r] > 0 and r != order_r[0]:
                kmask |= 1 << r
        out = np.zeros(3, np.int64)
        _top_ranks(kmask, 3, out)
        return _score(1, order_r[0], out[0], out[1], out[2], 0)
    out = np.zeros(5, np.int64)
    _top_ranks(mask, 5, out)
    return _score(0, out[0], out[1], out[2], out[3], out[4])


@njit(cache=True)
def _equity_loop(hero, board, avail, rands, iters, num_opps, need_board):
    k = num_opps * 2 + need_board
    navail = avail.shape[0]
    total = 0.0
    deck = np.empty(navail, np.int64)
    hero_full = np.empty(7, np.int64)
    full_board = np.empty(5, np.int64)
    vh = np.empty(7, np.int64)
    nb = board.shape[0]
    hero_full[0] = hero[0]
    hero_full[1] = hero[1]
    for i in range(nb):
        hero_full[2 + i] = board[i]
        full_board[i] = board[i]
    for it in range(iters):
        for i in range(navail):
            deck[i] = avail[i]
        base = it * k
        for j in range(k):
            idx = j + np.int64(rands[base + j] * (navail - j))
            if idx > navail - 1:
                idx = navail - 1
            tmp = deck[j]
            deck[j] = deck[idx]
            deck[idx] = tmp
        for i in range(need_board):
            c = deck[num_opps * 2 + i]
            hero_full[2 + nb + i] = c
            full_board[nb + i] = c
        my = eval_n(hero_full)
        best = -1
        for o in range(num_opps):
            vh[0] = deck[2 * o]
            vh[1] = deck[2 * o + 1]
            for i in range(5):
                vh[2 + i] = full_board[i]
            s = eval_n(vh)
            if s > best:
                best = s
        if my > best:
            total += 1.0
        elif my == best:
            total += 0.5
    return total / iters


def monte_carlo_equity_fast(
    hole: Sequence[Card],
    board: Sequence[Card],
    num_opponents: int,
    iterations: int,
    rng: random.Random,
) -> float:
    """Same contract as cards.monte_carlo_equity, numba-accelerated."""
    dead = set(hole) | set(board)
    avail = np.array([_code(c) for c in all_cards() if c not in dead], dtype=np.int64)
    hero = np.array([_code(c) for c in hole], dtype=np.int64)
    b = np.array([_code(c) for c in board], dtype=np.int64)
    need_board = 5 - len(board)
    k = num_opponents * 2 + need_board
    total_draws = iterations * k
    rands = np.fromiter((rng.random() for _ in range(total_draws)),
                        np.float64, total_draws)
    return _equity_loop(hero, b, avail, rands, iterations, num_opponents, need_board)
