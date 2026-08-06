"""The two-budget selection problem.

A rover cannot transmit everything it photographs, and it also cannot afford to
think hard about every frame. Those are two genuinely different scarcities, and
optimising for either one alone gives the wrong answer:

  * **Downlink budget (bits)** -- the relay pass is a fixed number of seconds at
    a fixed rate. Spend it on frames that are worth a scientist's attention.
  * **Compute budget (cycles)** -- the flight processor is shared with driving,
    thermal management and comms. Cycles spent scoring imagery are cycles not
    spent elsewhere, and they are consumed whether or not the frame is
    ultimately transmitted.

Selecting under both constraints at once is a 2-dimensional knapsack, which is
NP-hard, so NOVUM ships fast heuristics plus an upper bound to report the gap.

Convention: `value` is the novelty score (higher = more worth transmitting).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Method = Literal["greedy_sweep", "greedy_ratio", "score_first", "random"]
METHODS: tuple[str, ...] = ("greedy_sweep", "greedy_ratio", "score_first", "random")


@dataclass(frozen=True)
class BudgetSpec:
    """The two budgets for a single downlink opportunity."""

    bits: float
    cycles: float

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError(f"bit budget must be positive, got {self.bits}")
        if self.cycles <= 0:
            raise ValueError(f"cycle budget must be positive, got {self.cycles}")


@dataclass
class BudgetPlan:
    """What the onboard selector decided, and what it cost."""

    selected: np.ndarray  # int64 indices into the candidate arrays
    total_value: float
    used_bits: float
    used_cycles: float
    budget: BudgetSpec
    method: str
    upper_bound: float | None = None
    #: "bits" | "cycles" | "both" | "neither". Which budget stopped the selector.
    binding_constraint: str = "neither"
    meta: dict = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return int(self.selected.size)

    @property
    def bit_utilisation(self) -> float:
        return self.used_bits / self.budget.bits

    @property
    def cycle_utilisation(self) -> float:
        return self.used_cycles / self.budget.cycles

    @property
    def optimality_gap(self) -> float | None:
        """Fraction of the (loose) upper bound left on the table. 0.0 is ideal."""
        if self.upper_bound is None or self.upper_bound <= 0:
            return None
        return float(max(0.0, 1.0 - self.total_value / self.upper_bound))

    def to_json(self) -> dict:
        return {
            "method": self.method,
            "n_selected": self.n_selected,
            "total_value": self.total_value,
            "used_bits": self.used_bits,
            "used_cycles": self.used_cycles,
            "budget_bits": self.budget.bits,
            "budget_cycles": self.budget.cycles,
            "bit_utilisation": self.bit_utilisation,
            "cycle_utilisation": self.cycle_utilisation,
            "binding_constraint": self.binding_constraint,
            "upper_bound": self.upper_bound,
            "optimality_gap": self.optimality_gap,
            **self.meta,
        }


def _validate_candidates(
    values: Sequence[float] | np.ndarray,
    bits: Sequence[float] | np.ndarray,
    cycles: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64).ravel()
    b = np.asarray(bits, dtype=np.float64).ravel()
    c = np.asarray(cycles, dtype=np.float64).ravel()
    if not (v.shape == b.shape == c.shape):
        raise ValueError(f"values/bits/cycles length mismatch: {v.shape}, {b.shape}, {c.shape}")
    if v.size == 0:
        raise ValueError("no candidate frames")
    if not (np.isfinite(v).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        raise ValueError("candidate values/bits/cycles contain non-finite entries")
    if (b < 0).any() or (c < 0).any():
        raise ValueError("costs must be non-negative")
    return v, b, c


def _take_feasible(order: np.ndarray, v, b, c, budget: BudgetSpec) -> tuple[np.ndarray, float, float, float]:
    """Walk `order`, taking anything that still fits under BOTH budgets.

    The walk does not stop at the first item that does not fit: a large frame
    being unaffordable says nothing about the small ones behind it, and
    stopping early leaves a measurable fraction of the downlink unused.
    """
    chosen: list[int] = []
    used_b = used_c = total = 0.0
    for i in order:
        bi, ci = b[i], c[i]
        if used_b + bi <= budget.bits and used_c + ci <= budget.cycles:
            chosen.append(int(i))
            used_b += bi
            used_c += ci
            total += v[i]
    return np.asarray(chosen, dtype=np.int64), total, used_b, used_c


def _binding_constraint(v, b, c, budget: BudgetSpec, selected: np.ndarray, used_b: float, used_c: float) -> str:
    """Which budget actually stopped the selector.

    Utilisation alone cannot answer this. A cycle budget of 350 against items
    costing 100 each is fully binding at 300 used -- 86% utilisation, and not
    one more frame will fit. So ask the real question instead: of the frames
    left behind, which budget was too small to admit them?
    """
    remaining = np.ones(v.size, dtype=bool)
    if selected.size:
        remaining[selected] = False
    if not remaining.any():
        return "neither"  # everything fit; neither budget was reached

    headroom_bits = budget.bits - used_b
    headroom_cycles = budget.cycles - used_c
    blocked_by_bits = bool((b[remaining] > headroom_bits + 1e-9).any())
    blocked_by_cycles = bool((c[remaining] > headroom_cycles + 1e-9).any())

    if blocked_by_bits and blocked_by_cycles:
        return "both"
    if blocked_by_bits:
        return "bits"
    if blocked_by_cycles:
        return "cycles"
    return "neither"


def _density_order(v, b, c, budget: BudgetSpec, weight: float) -> np.ndarray:
    """Order by value per unit of blended, budget-normalised cost.

    Normalising each cost by its own budget is what makes bits and cycles
    comparable at all -- they have no common unit otherwise.
    """
    cost = weight * (b / budget.bits) + (1.0 - weight) * (c / budget.cycles)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(cost > 0, v / cost, np.inf)
    # Free items first, then by descending density; index breaks ties for determinism.
    return np.lexsort((np.arange(v.size), -density))


def select_two_budget(
    values,
    bits,
    cycles,
    budget: BudgetSpec,
    *,
    method: Method = "greedy_sweep",
    weight_grid: int = 21,
    seed: int = 0,
    compute_upper_bound: bool = True,
) -> BudgetPlan:
    """Choose which frames to downlink under both budgets.

    Methods:
      greedy_sweep  -- greedy_ratio swept over the bits/cycles blend, best kept.
                       Default: it costs a few milliseconds and never loses.
      greedy_ratio  -- single 50/50 blend of the two normalised costs.
      score_first   -- pure novelty order. The naive baseline, and the one the
                       simulator exists to beat.
      random        -- seeded random order. The floor.
    """
    v, b, c = _validate_candidates(values, bits, cycles)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")

    meta: dict = {}
    if method == "score_first":
        order = np.lexsort((np.arange(v.size), -v))
        selected, total, used_b, used_c = _take_feasible(order, v, b, c, budget)
    elif method == "random":
        order = np.random.default_rng(seed).permutation(v.size)
        selected, total, used_b, used_c = _take_feasible(order, v, b, c, budget)
    elif method == "greedy_ratio":
        order = _density_order(v, b, c, budget, 0.5)
        selected, total, used_b, used_c = _take_feasible(order, v, b, c, budget)
    else:  # greedy_sweep
        best: tuple[np.ndarray, float, float, float] | None = None
        best_w = 0.5
        for w in np.linspace(0.0, 1.0, max(2, int(weight_grid))):
            cand = _take_feasible(_density_order(v, b, c, budget, float(w)), v, b, c, budget)
            if best is None or cand[1] > best[1]:
                best, best_w = cand, float(w)
        assert best is not None
        selected, total, used_b, used_c = best
        meta["best_weight"] = best_w

    return BudgetPlan(
        selected=selected,
        total_value=float(total),
        used_bits=float(used_b),
        used_cycles=float(used_c),
        budget=budget,
        method=method,
        upper_bound=(fractional_upper_bound(v, b, c, budget) if compute_upper_bound else None),
        binding_constraint=_binding_constraint(v, b, c, budget, selected, used_b, used_c),
        meta=meta,
    )


def fractional_upper_bound(values, bits, cycles, budget: BudgetSpec) -> float:
    """An admissible upper bound on achievable value.

    Relaxing the 2-D knapsack to each single constraint separately can only
    enlarge the feasible set, so the fractional optimum of either relaxation
    bounds the true optimum. Taking the tighter of the two gives a bound that
    is valid, cheap, and good enough to report an honest optimality gap.
    """
    v, b, c = _validate_candidates(values, bits, cycles)

    def one_dim(cost: np.ndarray, cap: float) -> float:
        order = np.lexsort((np.arange(v.size), -np.where(cost > 0, v / np.maximum(cost, 1e-12), np.inf)))
        remaining = cap
        total = 0.0
        for i in order:
            if remaining <= 0:
                break
            ci = cost[i]
            if ci <= 0:
                total += max(0.0, v[i])
                continue
            take = min(1.0, remaining / ci)
            total += take * v[i]
            remaining -= take * ci
        return total

    return float(min(one_dim(b, budget.bits), one_dim(c, budget.cycles)))


# ---------------------------------------------------------------------------
# Cost models. Deliberately simple and explicit -- the simulator will refine
# them, but every number the selector uses should be traceable to an assumption
# written down here rather than buried in a magic constant.
# ---------------------------------------------------------------------------

def estimate_frame_bits(
    frame_shape: Sequence[int] = (64, 64, 6),
    *,
    bits_per_sample: int = 8,
    compression_ratio: float = 4.0,
) -> float:
    """Downlink cost of one frame after onboard compression.

    compression_ratio 4.0 is a conservative stand-in for the lossless-ish
    ratios ICER achieves on Mastcam imagery; override it per config.
    """
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")
    n_samples = int(np.prod(frame_shape))
    return float(n_samples * bits_per_sample / compression_ratio)


def estimate_bits_from_frames(
    frames: np.ndarray,
    *,
    bits_per_sample: int = 8,
    compression_ratio: float = 4.0,
    texture_exponent: float = 0.5,
    clip: tuple[float, float] = (0.25, 4.0),
) -> np.ndarray:
    """Per-frame downlink cost under a texture-dependent compression model.

    ILLUSTRATIVE. A flat per-frame bit cost makes the two-budget problem
    degenerate -- if every frame costs the same, selection collapses to "take
    the top k" and the second budget never binds. Real compressors spend more
    bits on busy scenes, so this scales the nominal cost by each frame's mean
    absolute gradient relative to the batch median. The downlink simulator will
    replace this with a real rate model; until then it is a stand-in that at
    least has the right sign.
    """
    x = np.asarray(frames, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"expected (N, H, W, C), got shape {x.shape}")

    gradient = np.abs(np.diff(x, axis=1)).mean(axis=(1, 2, 3)) + np.abs(
        np.diff(x, axis=2)
    ).mean(axis=(1, 2, 3))
    median = float(np.median(gradient))
    if median <= 0:
        factor = np.ones(len(x), dtype=np.float64)
    else:
        factor = np.clip((gradient / median) ** texture_exponent, *clip)

    base = estimate_frame_bits(
        x.shape[1:], bits_per_sample=bits_per_sample, compression_ratio=compression_ratio
    )
    return (base * factor).astype(np.float64)


def estimate_frame_cycles(flops_per_inference: float, *, cycles_per_flop: float = 3.0) -> float:
    """Compute cost of scoring one frame on a flight processor.

    cycles_per_flop ~3 reflects a radiation-hardened core with no SIMD and a
    software float path -- a RAD750 is not a laptop. Tune per tier config.
    """
    if flops_per_inference < 0:
        raise ValueError("flops_per_inference must be non-negative")
    return float(flops_per_inference * cycles_per_flop)
