"""Downlink window simulator.

=============================================================================
PARTIAL STUB. `replay()` -- the actual simulation loop -- raises
NotImplementedError. The scaffolding around it (chronological ordering, window
construction) is implemented and tested, because preprocessing already produces
the sol metadata those need and there is no reason to leave them guessed at.
=============================================================================

What this will do
-----------------
Replay the mission in sol order. On each sol the rover captures frames; every
so often an orbiter passes overhead and opens a downlink window with a finite
number of bits. Between windows the flight processor has a finite number of
cycles to decide what is worth sending.

The ordering matters, and it is the reason the manifest carries the sol. A
novelty model that has already seen the whole mission is not doing novelty
detection -- it is doing anomaly detection on a closed set. Replaying in sol
order means "novel" keeps its real meaning: unlike the terrain seen *so far*.

Open questions to settle before implementing `replay()`:
  * Does the model refit as new typical terrain arrives, or stay frozen after
    a training sol cutoff? Both are defensible; they are different missions.
  * Are unselected frames discarded or retained for a later window?
  * Does a downlinked frame's ground-truth label feed back into the model?
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from core.budgets import BudgetSpec
from core.manifest import ManifestRow


@dataclass(frozen=True)
class DownlinkWindow:
    """One relay pass: a sol range and the two budgets available for it."""

    index: int
    first_sol: int
    last_sol: int
    budget: BudgetSpec
    frame_indices: tuple[int, ...] = field(default=())

    @property
    def n_candidates(self) -> int:
        return len(self.frame_indices)

    @property
    def sol_span(self) -> int:
        return self.last_sol - self.first_sol + 1

    def __repr__(self) -> str:
        return (
            f"DownlinkWindow(#{self.index}, sols {self.first_sol}-{self.last_sol}, "
            f"{self.n_candidates} candidates, {self.budget.bits:.0f} bits)"
        )


def chronological_order(rows: Sequence[ManifestRow]) -> np.ndarray:
    """Row indices in mission order: ascending sol, then manifest index.

    Frames whose filename carried no parseable sol sort **last**. Sorting them
    first would seed the "terrain seen so far" baseline with frames of unknown
    date, which is exactly the kind of quiet mistake that makes a replay look
    better than it is.
    """
    sols = np.array(
        [np.iinfo(np.int64).max if r.sol is None else r.sol for r in rows], dtype=np.int64
    )
    return np.lexsort((np.arange(len(rows)), sols))


def plan_windows(
    rows: Sequence[ManifestRow],
    *,
    sols_per_window: int = 10,
    bits_per_window: float = 8_000_000.0,
    cycles_per_window: float = 2_000_000_000.0,
) -> list[DownlinkWindow]:
    """Group frames into fixed-length sol windows, in mission order.

    Frames with no parseable sol are collected into a final window rather than
    dropped, so nothing silently disappears from the replay.
    """
    if sols_per_window < 1:
        raise ValueError(f"sols_per_window must be >= 1, got {sols_per_window}")
    if not rows:
        return []

    order = chronological_order(rows)
    budget = BudgetSpec(bits=bits_per_window, cycles=cycles_per_window)

    dated: list[tuple[int, int]] = []  # (sol, row index)
    undated: list[int] = []
    for position in order:
        row = rows[int(position)]
        if row.sol is None:
            undated.append(int(position))
        else:
            dated.append((int(row.sol), int(position)))

    windows: list[DownlinkWindow] = []
    if dated:
        first_sol = dated[0][0]
        current: list[int] = []
        current_start = first_sol
        for sol, position in dated:
            if sol - current_start >= sols_per_window and current:
                windows.append(
                    DownlinkWindow(
                        index=len(windows),
                        first_sol=current_start,
                        last_sol=rows[current[-1]].sol or current_start,
                        budget=budget,
                        frame_indices=tuple(current),
                    )
                )
                current = []
                current_start = sol
            current.append(position)
        if current:
            windows.append(
                DownlinkWindow(
                    index=len(windows),
                    first_sol=current_start,
                    last_sol=rows[current[-1]].sol or current_start,
                    budget=budget,
                    frame_indices=tuple(current),
                )
            )

    if undated:
        windows.append(
            DownlinkWindow(
                index=len(windows),
                first_sol=-1,
                last_sol=-1,
                budget=budget,
                frame_indices=tuple(undated),
            )
        )
    return windows


def replay(
    frames: np.ndarray,
    rows: Sequence[ManifestRow],
    model,
    windows: Iterable[DownlinkWindow] | None = None,
    **kwargs,
):
    """Replay the mission window by window. NOT IMPLEMENTED.

    Intended signature once written: score each window's candidate frames with
    `model`, select under both budgets via `core.budgets.select_two_budget`,
    and return per-window transmitted sets plus cumulative science yield.
    """
    raise NotImplementedError(
        "sim.replay is not implemented yet. The pieces it will compose already exist: "
        "sim.plan_windows for the schedule, core.budgets.select_two_budget for the "
        "selection, and core.scoring for the yield metrics. "
        "See the module docstring for the design questions still open."
    )
