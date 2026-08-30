"""All-in EV adjustment: a variance-reduction technique for poker benchmarking.

When players get all-in before the board is complete, the realized result of
the hand depends heavily on the runout — which cards happen to fall on the
turn and river. That luck is real chip movement, but it says nothing about
decision quality, and it dominates the variance of any benchmark built from
realized results: a measurement on this project's own hands found the top 1%
of hands (almost all of them all-in pots) carry 42% of total variance, even
though 85% of hands finish under 4bb. Any win-rate comparison that keeps
realized all-in results is, in practice, mostly measuring who ran better —
not who played better.

The fix is the standard one from poker tracking software ("all-in EV" / "EV
adjusted winnings"): once betting has locked in with 2+ players all-in (or
committed such that no more decisions remain before showdown), replace the
realized outcome with the *expected* outcome averaged over every possible
completion of the board. Decisions made are left untouched; only the luck
after the last decision is replaced by its expectation.

This module is deliberately a pure post-processing step over `HandResult`:
it does not touch `PokerGame`, it just replays what the engine recorded
(actions + blinds) to reconstruct each player's total commitment, then reuses
the exact side-pot algorithm from `PokerGame._award` (levels of commitment,
one slice of the pot per level, split among the best hand(s) at that level)
against every possible board completion.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations

from .cards import Card, all_cards, evaluate_hand
from .engine import HandResult
from .strategy import ActionType

# Board cards known to be public once betting has reached (but not yet
# finished) each street. Betting on "river" means the board is already
# complete, so there is nothing left to adjust for.
_KNOWN_BOARD_LEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

# Below this many possible completions, enumerate exactly; above it, the
# combinatorics get expensive enough (preflop all-ins run into the millions)
# that we fall back to Monte Carlo sampling instead.
_EXACT_ENUMERATION_LIMIT = 200_000
_DEFAULT_SAMPLE_ITERATIONS = 10_000


@dataclass(frozen=True)
class CompletionPlan:
    """How allin_adjusted_winnings will complete the board for one hand.

    Exposed as its own function mainly for testability: sample-based results
    are inherently noisy, so tests need a way to assert "this hand enumerates
    exactly" / "this hand falls back to sampling" without reverse-engineering
    it from the (approximate, for sampling) winnings output.
    """
    street: str
    known_board_len: int
    cards_needed: int
    remaining_deck_size: int
    combo_count: int
    method: str  # "exact" or "sample"
    iterations: int  # combos enumerated (method="exact") or samples drawn (method="sample")


def _folded_ids(result: HandResult) -> set[int]:
    """A player folded iff they have a FOLD action recorded (per spec)."""
    return {pid for pid, _street, action in result.actions if action.action_type == ActionType.FOLD}


def _blind_seats(result: HandResult) -> tuple[int, int]:
    """Small/big blind seats, mirroring PokerGame.play_hand's own formula.

    The n==2 case is special-cased in the engine (dealer posts SB, the other
    seat posts BB) because heads-up play swaps the usual button/blind
    relationship.
    """
    n = len(result.starting_stacks)
    dealer = result.dealer % n
    if n == 2:
        return dealer, (dealer + 1) % n
    return (dealer + 1) % n, (dealer + 2) % n


def _reconstruct_committed(result: HandResult) -> list[int]:
    """Total chips each seat put into the pot over the whole hand.

    HandResult does not store PlayerView.committed directly, so we replay
    the action log the same way PokerGame._apply_action does: BET/RAISE
    amount is a raise-TO (total street commitment after the action), CALL
    amount is chips added, blinds are posted before any action and are
    capped at the poster's starting stack (covers a short all-in blind).
    Street-committed amounts reset to 0 at the start of each new street,
    exactly like PokerGame._run_street(reset_street=True) does for every
    street after preflop.

    HandResult only carries big_blind, not small_blind — per spec we assume
    the standard small_blind = big_blind // 2 used throughout this repo.
    """
    n = len(result.starting_stacks)
    sb_id, bb_id = _blind_seats(result)
    small_blind = result.big_blind // 2

    committed = [0] * n
    street_committed = [0] * n

    sb_amt = min(small_blind, result.starting_stacks[sb_id])
    bb_amt = min(result.big_blind, result.starting_stacks[bb_id])
    committed[sb_id] += sb_amt
    street_committed[sb_id] += sb_amt
    committed[bb_id] += bb_amt
    street_committed[bb_id] += bb_amt

    current_street = "preflop"
    for pid, street, action in result.actions:
        if street != current_street:
            street_committed = [0] * n
            current_street = street
        if action.action_type == ActionType.CALL:
            committed[pid] += action.amount
            street_committed[pid] += action.amount
        elif action.action_type in (ActionType.BET, ActionType.RAISE):
            delta = action.amount - street_committed[pid]
            committed[pid] += delta
            street_committed[pid] = action.amount
        # FOLD / CHECK move no chips.
    return committed


def allin_street(result: HandResult) -> str | None:
    """The street on which betting ceased with 2+ players still live and at
    least one all-in, or None if this hand needs no adjustment.
    """
    folded = _folded_ids(result)
    n = len(result.starting_stacks)
    live = [i for i in range(n) if i not in folded]
    if len(live) < 2:
        return None

    # No actions at all happens when both/all remaining stacks are covered
    # entirely by blinds (e.g. heads-up, both stacks shorter than the big
    # blind) — betting "ceased" before the board was ever touched.
    last_street = result.actions[-1][1] if result.actions else "preflop"
    if last_street == "river":
        return None

    committed = _reconstruct_committed(result)
    if not any(committed[i] == result.starting_stacks[i] for i in live):
        return None
    return last_street


def _dead_cards(result: HandResult, known_board_len: int) -> set[Card]:
    """Cards no longer available to complete the board: every seat's hole
    cards (HandResult stores them for folded seats too) plus the known
    board prefix. Community cards past the prefix are the real dealt cards
    for the ACTUAL runout — they carry the luck we're averaging away, so
    they must not leak into the "known" set.
    """
    dead: set[Card] = set(result.community[:known_board_len])
    for hole in result.hole:
        dead.update(hole)
    return dead


def completion_plan(result: HandResult, iterations: int = 0) -> CompletionPlan | None:
    """Decide how allin_adjusted_winnings would complete the board, without
    doing the (possibly expensive) work. Returns None when the hand needs no
    adjustment at all.
    """
    street = allin_street(result)
    if street is None:
        return None
    known_len = _KNOWN_BOARD_LEN[street]
    need = 5 - known_len
    dead = _dead_cards(result, known_len)
    remaining = 52 - len(dead)
    combo_count = math.comb(remaining, need)
    if combo_count <= _EXACT_ENUMERATION_LIMIT:
        return CompletionPlan(street, known_len, need, remaining, combo_count, "exact", combo_count)
    eff_iterations = iterations if iterations > 0 else _DEFAULT_SAMPLE_ITERATIONS
    return CompletionPlan(street, known_len, need, remaining, combo_count, "sample", eff_iterations)


def _split_pot(committed: list[int], folded: set[int], hole: list[list[Card]],
                board: list[Card]) -> list[float]:
    """The exact side-pot algorithm from PokerGame._award (commitment-level
    slices, best hand per slice, split among ties), reused verbatim against
    one candidate board.

    One deliberate difference: _award hands odd chips to winners in seat
    order starting after the button, because a real hand has to settle in
    whole chips. That rule is noise for an EV computation — dividing a tied
    slice evenly (fractionally) is the more correct expectation, so that's
    what this does instead.
    """
    n = len(committed)
    eligible = [i for i in range(n) if i not in folded]
    payouts = [0.0] * n
    if len(eligible) == 1:
        payouts[eligible[0]] = float(sum(committed))
        return payouts

    levels = sorted(set(committed))
    prev = 0
    for level in levels:
        slice_pot = (level - prev) * sum(1 for c in committed if c >= level)
        prev = level
        if slice_pot == 0:
            continue
        contenders = [i for i in eligible if committed[i] >= level]
        if not contenders:
            continue
        best = None
        winners: list[int] = []
        for i in contenders:
            score = evaluate_hand(hole[i] + board)
            if best is None or score > best:
                best = score
                winners = [i]
            elif score == best:
                winners.append(i)
        share = slice_pot / len(winners)
        for w in winners:
            payouts[w] += share
    return payouts


def allin_adjusted_winnings(result: HandResult, iterations: int = 0,
                             rng: random.Random | None = None) -> list[float] | None:
    """Expected net winnings per seat over all completions of the board from
    the point betting ceased. Returns None when the hand needs no adjustment
    (in which case callers should keep result.winnings unchanged).

    Returned values are plain floats (not rounded), zero-sum to within
    floating-point rounding error. Callers that need integer chip counts
    should round themselves (e.g. via banker's rounding plus a residual fix
    up, since naive per-seat rounding is not guaranteed to stay zero-sum).
    """
    plan = completion_plan(result, iterations)
    if plan is None:
        return None

    folded = _folded_ids(result)
    committed = _reconstruct_committed(result)
    known_board = list(result.community[:plan.known_board_len])
    dead = _dead_cards(result, plan.known_board_len)
    deck = [c for c in all_cards() if c not in dead]

    n = len(committed)
    totals = [0.0] * n

    if plan.method == "exact":
        count = 0
        for combo in combinations(deck, plan.cards_needed):
            board = known_board + list(combo)
            payouts = _split_pot(committed, folded, result.hole, board)
            for i in range(n):
                totals[i] += payouts[i] - committed[i]
            count += 1
        denom = count
    else:
        sampler = rng or random.Random()
        denom = plan.iterations
        for _ in range(denom):
            combo = sampler.sample(deck, plan.cards_needed)
            board = known_board + combo
            payouts = _split_pot(committed, folded, result.hole, board)
            for i in range(n):
                totals[i] += payouts[i] - committed[i]

    return [t / denom for t in totals]
