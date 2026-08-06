"""Two-budget selection: feasibility first, then quality."""

from __future__ import annotations

import numpy as np
import pytest

from core.budgets import (
    METHODS,
    BudgetSpec,
    estimate_bits_from_frames,
    estimate_frame_bits,
    estimate_frame_cycles,
    fractional_upper_bound,
    select_two_budget,
)


@pytest.fixture
def candidates():
    rng = np.random.default_rng(7)
    n = 200
    values = rng.gamma(2.0, 1.0, size=n)
    bits = rng.uniform(1_000, 10_000, size=n)
    cycles = rng.uniform(1e5, 1e6, size=n)
    return values, bits, cycles


@pytest.mark.parametrize("method", METHODS)
def test_every_method_respects_both_budgets(candidates, method: str) -> None:
    values, bits, cycles = candidates
    budget = BudgetSpec(bits=100_000, cycles=1e7)
    plan = select_two_budget(values, bits, cycles, budget, method=method)

    assert plan.used_bits <= budget.bits + 1e-9
    assert plan.used_cycles <= budget.cycles + 1e-9
    assert plan.n_selected > 0
    assert plan.total_value == pytest.approx(values[plan.selected].sum())


def test_greedy_sweep_is_at_least_as_good_as_the_naive_baselines(candidates) -> None:
    values, bits, cycles = candidates
    budget = BudgetSpec(bits=100_000, cycles=1e7)

    sweep = select_two_budget(values, bits, cycles, budget, method="greedy_sweep")
    naive = select_two_budget(values, bits, cycles, budget, method="score_first")
    random_pick = select_two_budget(values, bits, cycles, budget, method="random")

    assert sweep.total_value >= naive.total_value
    assert sweep.total_value > random_pick.total_value


def test_selection_does_not_stop_at_the_first_unaffordable_item() -> None:
    """One huge frame must not block the small ones behind it."""
    values = np.array([100.0, 1.0, 1.0, 1.0])
    bits = np.array([1_000_000.0, 10.0, 10.0, 10.0])
    cycles = np.array([1.0, 1.0, 1.0, 1.0])
    budget = BudgetSpec(bits=100.0, cycles=100.0)

    plan = select_two_budget(values, bits, cycles, budget, method="score_first")
    assert plan.n_selected == 3
    assert 0 not in plan.selected


def test_the_second_budget_actually_binds() -> None:
    """With cycles scarce, a bits-only view would overcommit."""
    values = np.ones(10)
    bits = np.full(10, 10.0)
    cycles = np.full(10, 100.0)
    budget = BudgetSpec(bits=1000.0, cycles=350.0)

    plan = select_two_budget(values, bits, cycles, budget, method="greedy_sweep")
    assert plan.n_selected == 3           # cycles cap at 3, not the 10 bits allow
    assert plan.binding_constraint == "cycles"


def test_upper_bound_is_admissible(candidates) -> None:
    values, bits, cycles = candidates
    budget = BudgetSpec(bits=100_000, cycles=1e7)
    bound = fractional_upper_bound(values, bits, cycles, budget)
    for method in METHODS:
        plan = select_two_budget(values, bits, cycles, budget, method=method)
        assert plan.total_value <= bound + 1e-6


def test_optimality_gap_is_reported(candidates) -> None:
    values, bits, cycles = candidates
    plan = select_two_budget(
        values, bits, cycles, BudgetSpec(bits=100_000, cycles=1e7), method="greedy_sweep"
    )
    assert plan.optimality_gap is not None
    assert 0.0 <= plan.optimality_gap < 1.0


def test_random_is_reproducible(candidates) -> None:
    values, bits, cycles = candidates
    budget = BudgetSpec(bits=50_000, cycles=5e6)
    a = select_two_budget(values, bits, cycles, budget, method="random", seed=42)
    b = select_two_budget(values, bits, cycles, budget, method="random", seed=42)
    np.testing.assert_array_equal(a.selected, b.selected)


def test_free_items_are_taken_first() -> None:
    values = np.array([1.0, 5.0])
    bits = np.array([0.0, 100.0])
    cycles = np.array([0.0, 100.0])
    plan = select_two_budget(
        values, bits, cycles, BudgetSpec(bits=1.0, cycles=1.0), method="greedy_ratio"
    )
    assert list(plan.selected) == [0]


def test_rejects_a_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="bit budget"):
        BudgetSpec(bits=0, cycles=1)
    with pytest.raises(ValueError, match="cycle budget"):
        BudgetSpec(bits=1, cycles=-5)


def test_rejects_mismatched_candidate_arrays() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        select_two_budget([1, 2], [1], [1, 2], BudgetSpec(bits=1, cycles=1))


def test_rejects_an_unknown_method(candidates) -> None:
    values, bits, cycles = candidates
    with pytest.raises(ValueError, match="unknown method"):
        select_two_budget(values, bits, cycles, BudgetSpec(bits=1, cycles=1), method="magic")


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------
def test_estimate_frame_bits() -> None:
    # 64*64*6 samples * 8 bits / 4x compression
    assert estimate_frame_bits((64, 64, 6)) == pytest.approx(49_152.0)


def test_estimate_frame_cycles() -> None:
    assert estimate_frame_cycles(1_000, cycles_per_flop=3.0) == pytest.approx(3_000.0)


def test_texture_model_charges_busy_frames_more() -> None:
    rng = np.random.default_rng(1)
    flat = np.full((5, 64, 64, 6), 100.0, dtype=np.float32)
    busy = rng.normal(100.0, 40.0, size=(5, 64, 64, 6)).astype(np.float32)
    frames = np.concatenate([flat, busy])

    bits = estimate_bits_from_frames(frames)
    assert bits[:5].mean() < bits[5:].mean()
    assert np.isfinite(bits).all()


def test_texture_model_survives_a_constant_batch() -> None:
    frames = np.full((4, 64, 64, 6), 42.0, dtype=np.float32)
    bits = estimate_bits_from_frames(frames)
    assert np.allclose(bits, bits[0])
    assert np.isfinite(bits).all()
