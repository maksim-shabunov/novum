"""Selection policies, and which of them need to think.

Every policy reduces to the same two steps -- assign a value per candidate,
then hand the values to `core.budgets.select_two_budget` -- so the comparison
between them is a comparison of *values*, not of selection machinery.

    fifo          value = capture order. Transmits oldest first until the bits
                  run out. Needs no scores, so it pays no cycle tax: the
                  closest stand-in for non-intelligent operations, and the bar
                  everything else has to clear.
    random        value = uniform noise. The floor.
    score_first   value = novelty. Ignores that frames cost different numbers
                  of bits.
    greedy_ratio  value = novelty, selected by novelty-per-bit.
    oracle        value = 1 for natural-class frames, 0 otherwise, selected per
                  bit. NOT ACHIEVABLE -- it reads the labels. Reported as the
                  upper bound so the remaining gap is visible.

`needs_scores` is what makes the compute budget bite asymmetrically: policies
that rank by novelty must pay to compute novelty, and under a tight cycle
budget some frames never get scored at all. FIFO never has that problem, which
is exactly why it is the honest baseline rather than a strawman.
"""

from __future__ import annotations

import numpy as np

from core.budgets import BudgetPlan, BudgetSpec, select_two_budget

METHODS: tuple[str, ...] = ("fifo", "random", "score_first", "greedy_ratio", "oracle")

#: Policies that require a novelty score, and therefore pay the cycle tax.
SCORE_BASED: frozenset[str] = frozenset({"score_first", "greedy_ratio"})

#: Policies that cannot run onboard: they read ground-truth labels.
ORACLE_METHODS: frozenset[str] = frozenset({"oracle"})


def needs_scores(method: str) -> bool:
    return method in SCORE_BASED


def is_oracle(method: str) -> bool:
    return method in ORACLE_METHODS


def policy_values(
    method: str,
    *,
    scores: np.ndarray,
    capture_order: np.ndarray,
    is_natural: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Value per candidate, plus the core.budgets method to select with."""
    n = len(capture_order)
    if method == "fifo":
        # Strictly decreasing in capture order, so "highest value" == "oldest".
        # score_first on this value is exactly first-in-first-out.
        return (-capture_order.astype(np.float64), "score_first")
    if method == "random":
        return (rng.random(n), "score_first")
    if method == "score_first":
        return (np.asarray(scores, dtype=np.float64), "score_first")
    if method == "greedy_ratio":
        return (np.asarray(scores, dtype=np.float64), "greedy_ratio")
    if method == "oracle":
        # Maximise natural frames per bit: the best any selector could do.
        return (is_natural.astype(np.float64), "greedy_ratio")
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def select(
    method: str,
    *,
    scores: np.ndarray,
    bits: np.ndarray,
    cycles: np.ndarray,
    capture_order: np.ndarray,
    is_natural: np.ndarray,
    budget: BudgetSpec,
    rng: np.random.Generator,
) -> BudgetPlan:
    """Run one policy over the candidates under both budgets."""
    if len(capture_order) == 0:
        return BudgetPlan(
            selected=np.zeros(0, dtype=np.int64),
            total_value=0.0,
            used_bits=0.0,
            used_cycles=0.0,
            budget=budget,
            method=method,
            binding_constraint="neither",
        )

    values, selector = policy_values(
        method, scores=scores, capture_order=capture_order, is_natural=is_natural, rng=rng
    )

    # A value floor keeps the density ordering well-defined for greedy_ratio
    # when every candidate scores zero (an all-rover window under the oracle).
    if selector == "greedy_ratio" and not np.any(values > 0):
        values = values + 1e-12

    plan = select_two_budget(
        values, bits, cycles, budget, method=selector, seed=int(rng.integers(0, 2**31 - 1))
    )
    plan.method = method  # report the policy the caller asked for, not the mechanism
    return plan
